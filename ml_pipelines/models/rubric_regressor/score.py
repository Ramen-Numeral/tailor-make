import json
import logging
from time import monotonic, sleep

import polars as pl
from pydantic import BaseModel, ConfigDict

from app.infrastructure.llm import LLMRoutedClient
from config.llm import judge_route
from config.settings import get_settings
from config.train_data import TrainDataConfig
from ml_pipelines.models.rubric_regressor.rubric import (
    AXES_DEFINITIONS,
    make_sys_prompt,
)


data_cfg = TrainDataConfig()
logger = logging.getLogger(__name__)


class RubricBatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    scores: dict[str, int]


class RubricBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RubricBatchItem]


def score_text_batch(
    texts: list[str],
    *,
    max_attempts: int = 5,
) -> dict[int, dict[str, int]]:
    """Score one batch with one client and one shared rubric prompt."""
    if not texts:
        return {}

    model = LLMRoutedClient(
        judge_route,
        api_key=get_settings().runtime.require_groq_api_key(),
    )
    writings = [
        {"id": f"r{index}", "text": text}
        for index, text in enumerate(texts)
    ]
    prompt = f"""
{make_sys_prompt()}

Score each writing independently.
Return {{"items":[{{"id":"r0","scores":{{...}}}}]}}.
Copy each input id exactly. Omit an item only if it cannot be scored.

Writings:
{json.dumps(writings, ensure_ascii=False, separators=(",", ":"))}
""".strip()

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = model.invoke_structured(
                prompt=prompt,
                schema=RubricBatchResponse,
                temperature=0.2,
                max_tokens=max(300, len(texts) * 100),
                trace_context=f"rubric_batch size={len(texts)}",
            )
            expected_ids = {f"r{index}" for index in range(len(texts))}
            returned_ids = [item.id for item in response.items]
            if len(returned_ids) != len(set(returned_ids)):
                raise ValueError("Rubric batch returned duplicate ids")
            unknown_ids = set(returned_ids) - expected_ids
            if unknown_ids:
                raise ValueError(
                    f"Rubric batch returned unknown ids: {sorted(unknown_ids)}"
                )
            batch_scores = {
                int(item.id[1:]): item.scores
                for item in response.items
            }
            for scores in batch_scores.values():
                _validate_rubric_scores(scores, require_all=True)
            return batch_scores
        except Exception as error:
            last_error = error
            logger.warning(
                "Rubric batch attempt %d/%d failed: %s",
                attempt,
                max_attempts,
                error,
            )

    raise RuntimeError(
        f"Rubric batch failed after {max_attempts} attempts: {last_error}"
    ) from last_error


def score_text(
    text: str,
    *,
    max_attempts: int = 5,
) -> dict[str, int]:
    """Score text on rubric axes using LLM judge.

    Instantiates a fresh LLM client on initial attempt to avoid contamination.
    Retries with the same client to avoid unnecessary instantiation.

    Args:
        text: Text to score
        max_attempts: Max retry attempts

    Returns:
        Dict mapping axis name to score (1-5)

    Raises:
        RuntimeError: If max_attempts exceeded without getting all axes
    """
    scores: dict[str, int] = {}
    attempts = 0
    model = None

    while set(scores) != set(AXES_DEFINITIONS):
        if attempts >= max_attempts:
            missing = [
                axis
                for axis in AXES_DEFINITIONS
                if axis not in scores
            ]
            raise RuntimeError(
                f"Missing rubric axes after {max_attempts} attempts: {missing}"
            )

        # Fresh instantiation on first attempt; reuse on retries
        if model is None:
            logger.debug("Instantiating fresh LLM client for scoring")
            model = LLMRoutedClient(
                judge_route,
                api_key=get_settings().runtime.require_groq_api_key(),
            )

        missing = [
            axis
            for axis in AXES_DEFINITIONS
            if axis not in scores
        ]

        if not scores:
            prompt = f"""
{make_sys_prompt()}

Writing to evaluate:
{text}
""".strip()
        else:
            prompt = f"""
{make_sys_prompt()}

Writing to evaluate:
{text}

The following rubric axes are still missing:
{chr(10).join(f"- {axis}" for axis in missing)}

Return scores only for those missing axes.
""".strip()

        try:
            response = model.invoke(prompt)
            new_scores = parse_rubric_response(response)

            scores.update({
                axis: score
                for axis, score in new_scores.items()
                if axis in missing
            })
        except Exception as e:
            logger.warning(f"Attempt {attempts + 1}/{max_attempts} failed: {e}")
            # On timeout or other transient errors, instantiate a fresh client for next attempt
            if "timeout" in str(e).lower():
                logger.info("Timeout detected; creating fresh client for retry")
                model = None

        attempts += 1

    logger.debug(f"Scoring complete after {attempts} attempt(s)")
    return scores





def parse_rubric_response(response: str) -> dict[str, int]:
    try:
        scores = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Rubric response is not valid JSON: {response!r}"
        ) from exc

    if not isinstance(scores, dict):
        raise TypeError("Rubric response must be a JSON object.")

    _validate_rubric_scores(scores)
    return scores


def _validate_rubric_scores(
    scores: dict[str, int],
    *,
    require_all: bool = False,
) -> None:
    unknown_axes = set(scores) - set(AXES_DEFINITIONS)
    if unknown_axes:
        raise ValueError(
            f"Unknown rubric axes: {sorted(unknown_axes)}"
        )

    invalid_scores = {
        axis: score
        for axis, score in scores.items()
        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or not 1 <= score <= 5
        )
    }

    if invalid_scores:
        raise ValueError(
            f"Invalid rubric scores: {invalid_scores}"
        )
    if require_all and set(scores) != set(AXES_DEFINITIONS):
        missing = set(AXES_DEFINITIONS) - set(scores)
        raise ValueError(f"Missing rubric axes: {sorted(missing)}")


def score_multitext(
    X: list[str],
    y: list[int],
    max_attempts_per: int = 5,
    min_interval_seconds: float = 0.0,
    batch_size: int = 10,
    max_failures: int = 100,
) -> pl.DataFrame:
    """Score multiple texts on rubric axes.

    Each text gets a fresh LLM client to prevent cross-contamination in the dataset.

    Args:
        X: List of texts to score
        y: List of labels (parallel to X)
        max_attempts_per: Max retry attempts per text
        min_interval_seconds: Minimum time between starting model requests

    Returns:
        DataFrame with text, target, and all rubric axis scores
    """
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length: {len(X)} != {len(y)}"
        )
    if min_interval_seconds < 0:
        raise ValueError("min_interval_seconds cannot be negative")

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_failures < 0:
        raise ValueError("max_failures cannot be negative")

    rows: list[dict] = []
    failures = 0
    next_progress_report = 50
    last_request_started: float | None = None

    for start in range(0, len(X), batch_size):
        texts = X[start:start + batch_size]
        targets = y[start:start + batch_size]
        if last_request_started is not None:
            remaining = (
                min_interval_seconds
                - (monotonic() - last_request_started)
            )
            if remaining > 0:
                logger.info(
                    "Pacing rubric request %d/%d for %.2f seconds",
                    start // batch_size + 1,
                    (len(X) + batch_size - 1) // batch_size,
                    remaining,
                )
                sleep(remaining)

        last_request_started = monotonic()
        logger.info(
            "Scoring rubric batch %d-%d/%d",
            start + 1,
            start + len(texts),
            len(X),
        )
        try:
            batch_scores = score_text_batch(
                texts,
                max_attempts=max_attempts_per,
            )

            for offset, scores in batch_scores.items():
                rows.append({
                    data_cfg.text_column: texts[offset],
                    data_cfg.target_column: targets[offset],
                    **scores,
                })

            failures += len(texts) - len(batch_scores)
        except Exception as e:
            logger.error(
                "Failed to score rubric batch %d-%d: %s",
                start + 1,
                start + len(texts),
                e,
            )
            failures += len(texts)

        while len(rows) >= next_progress_report:
            logger.info("Successfully scored dataframe rows: %d", len(rows))
            next_progress_report += 50

        if failures > max_failures:
            raise RuntimeError(
                f"Stopping after {failures} failed examples; "
                f"maximum allowed is {max_failures}"
            )

    schema = {
        data_cfg.text_column: pl.String,
        data_cfg.target_column: pl.Int64,
        **{
            axis: pl.Int64
            for axis in AXES_DEFINITIONS
        },
    }

    logger.info(f"Successfully scored {len(rows)}/{len(X)} texts")
    return pl.DataFrame(
        rows,
        schema=schema,
        strict=True,
    )
