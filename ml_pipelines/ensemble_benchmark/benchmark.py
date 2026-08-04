"""Benchmark AI-detection components and their weighted ensemble.

This module is deliberately isolated from application runtime code. It loads the
existing artifacts, evaluates them on an external labelled dataset, and writes
all generated reports beneath this package's ``reports/`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep
from typing import Callable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "reports" / ".matplotlib"),
)

import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import polars as pl
import torch
from catboost import CatBoostClassifier
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.features.ai_detection.config import ai_detection_config
from app.infrastructure.ai_detection.catboost_component import load_cb_model
from app.infrastructure.ai_detection.rubric_component import load_rubric_regressor
from ml_pipelines.data.features import augment_data
from ml_pipelines.models.catboost.cb_model import make_pool
from ml_pipelines.models.rubric_regressor.rubric import AXES_DEFINITIONS
from ml_pipelines.models.rubric_regressor.score import score_text_batch


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_REPORT_DIR = PACKAGE_DIR / "reports"
DEFAULT_DATASET = "shahxeebhassan/human_vs_ai_sentences"
DEFAULT_WEIGHTS = dict(ai_detection_config.weights)


@dataclass(frozen=True)
class MetricRow:
    model: str
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float
    brier: float
    log_loss: float
    ece: float
    latency_seconds: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate individual AI detectors and weighted ensembles."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--include-rubric",
        action="store_true",
        help=(
            "Call the configured Groq rubric judge, then evaluate the full "
            "runtime ensemble. This incurs API usage."
        ),
    )
    parser.add_argument(
        "--rubric-batch-size",
        type=int,
        default=8,
        help="Texts per rubric LLM call when --include-rubric is enabled.",
    )
    parser.add_argument(
        "--rubric-request-delay",
        type=float,
        default=1.0,
        help="Minimum pause between rubric requests to reduce rate-limit pressure.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    _configure_plotting(args.report_dir)
    _seed_everything(args.seed)

    texts, labels, dataset_meta = load_external_sample(
        args.dataset,
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    probabilities: dict[str, np.ndarray] = {}
    latencies: dict[str, float] = {}

    predictors: list[tuple[str, Callable[[], np.ndarray]]] = [
        ("catboost", lambda: predict_catboost(texts)),
        ("distilbert", lambda: predict_distilbert(texts, args.batch_size)),
        ("tfidf_svm", lambda: predict_tfidf(texts)),
    ]
    for name, predictor in predictors:
        started = monotonic()
        probabilities[name] = _validate_probabilities(name, predictor(), len(texts))
        latencies[name] = monotonic() - started

    local_components = ["catboost", "distilbert", "tfidf_svm"]
    probabilities["local_weighted_ensemble"] = weighted_ensemble(
        probabilities,
        local_components,
        DEFAULT_WEIGHTS,
    )
    latencies["local_weighted_ensemble"] = sum(latencies.values())

    rubric_status = "not_requested"
    if args.include_rubric:
        started = monotonic()
        try:
            probabilities["rubric_regressor"] = predict_rubric(
                texts,
                batch_size=args.rubric_batch_size,
                cache_path=args.report_dir / "rubric_scores.json",
                request_delay=args.rubric_request_delay,
            )
        except Exception as error:
            rubric_status = f"failed: {type(error).__name__}: {error}"
        else:
            rubric_status = "completed"
            latencies["rubric_regressor"] = monotonic() - started
            probabilities["full_runtime_ensemble"] = weighted_ensemble(
                probabilities,
                [*local_components, "rubric_regressor"],
                DEFAULT_WEIGHTS,
            )
            latencies["full_runtime_ensemble"] = sum(
                latencies[name]
                for name in [*local_components, "rubric_regressor"]
            )

    labels_array = np.asarray(labels, dtype=np.int64)
    metric_rows = [
        calculate_metrics(
            name,
            labels_array,
            scores,
            threshold=args.threshold,
            latency_seconds=latencies.get(name, 0.0),
        )
        for name, scores in probabilities.items()
    ]
    metric_rows.sort(key=lambda row: row.f1, reverse=True)

    comparison = compare_ensemble_to_best_component(
        labels_array,
        probabilities,
        threshold=args.threshold,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        ensemble_name=(
            "full_runtime_ensemble"
            if "full_runtime_ensemble" in probabilities
            else "local_weighted_ensemble"
        ),
    )
    ablations = calculate_ablations(
        labels_array,
        probabilities,
        threshold=args.threshold,
        include_rubric="rubric_regressor" in probabilities,
    )

    write_predictions(args.report_dir, texts, labels_array, probabilities)
    write_metrics(args.report_dir, metric_rows)
    write_json(
        args.report_dir / "assessment.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "dataset": dataset_meta,
            "configuration": {
                "threshold": args.threshold,
                "weights": DEFAULT_WEIGHTS,
                "bootstrap_samples": args.bootstrap_samples,
                "include_rubric": args.include_rubric,
                "rubric_status": rubric_status,
            },
            "metrics": [asdict(row) for row in metric_rows],
            "ensemble_comparison": comparison,
            "ablations": ablations,
            "limitations": limitations(args.include_rubric, rubric_status),
        },
    )
    create_plots(
        args.report_dir,
        labels_array,
        probabilities,
        metric_rows,
        threshold=args.threshold,
    )
    write_markdown_report(
        args.report_dir,
        dataset_meta,
        metric_rows,
        comparison,
        ablations,
        rubric_status,
    )

    print(json.dumps({
        "report": str((args.report_dir / "REPORT.md").resolve()),
        "comparison": comparison,
        "rubric_status": rubric_status,
    }, indent=2))
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.sample_size < 2:
        raise ValueError("sample-size must be at least 2")
    if args.batch_size < 1 or args.rubric_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.rubric_request_delay < 0:
        raise ValueError("rubric-request-delay cannot be negative")
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be positive")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    try:
        args.report_dir.resolve().relative_to(PACKAGE_DIR)
    except ValueError as error:
        raise ValueError(
            f"report-dir must stay beneath {PACKAGE_DIR}"
        ) from error


def _configure_plotting(report_dir: Path) -> None:
    cache_dir = report_dir / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "figure.dpi": 140,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_external_sample(
    dataset_name: str,
    *,
    split: str,
    sample_size: int,
    seed: int,
) -> tuple[list[str], list[int], dict[str, object]]:
    dataset = load_dataset(
        dataset_name,
        split=split,
        cache_dir=str(PACKAGE_DIR / ".cache" / "huggingface"),
    )
    columns = set(dataset.column_names)
    text_column = next(
        (name for name in ("text", "sentence", "content") if name in columns),
        None,
    )
    label_column = next(
        (name for name in ("label", "generated", "class") if name in columns),
        None,
    )
    if text_column is None or label_column is None:
        raise ValueError(
            f"Could not infer text/label columns from {sorted(columns)}"
        )

    labels = np.asarray([normalize_label(value) for value in dataset[label_column]])
    eligible = np.flatnonzero(np.asarray([
        bool(str(value).strip()) for value in dataset[text_column]
    ]))
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    per_class = sample_size // 2
    for label in (0, 1):
        candidates = eligible[labels[eligible] == label]
        if not len(candidates):
            raise ValueError(f"Dataset has no examples for label {label}")
        take = min(per_class, len(candidates))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    remainder = min(sample_size - len(selected), len(eligible) - len(selected))
    if remainder:
        remaining = np.setdiff1d(eligible, np.asarray(selected), assume_unique=False)
        selected.extend(rng.choice(remaining, size=remainder, replace=False).tolist())
    rng.shuffle(selected)

    texts = [str(dataset[index][text_column]).strip() for index in selected]
    sampled_labels = [int(labels[index]) for index in selected]
    counts = {str(label): sampled_labels.count(label) for label in (0, 1)}
    return texts, sampled_labels, {
        "name": dataset_name,
        "split": split,
        "source_rows": len(dataset),
        "sample_rows": len(texts),
        "text_column": text_column,
        "label_column": label_column,
        "label_mapping": {"0": "human", "1": "AI"},
        "class_counts": counts,
        "seed": seed,
        "external_evaluation_only": True,
    }


def normalize_label(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    rendered = str(value).strip().casefold()
    mapping = {
        "0": 0,
        "human": 0,
        "human-written": 0,
        "human_written": 0,
        "1": 1,
        "ai": 1,
        "ai-generated": 1,
        "ai_generated": 1,
        "machine": 1,
    }
    if rendered not in mapping:
        raise ValueError(f"Unsupported label value: {value!r}")
    return mapping[rendered]


def predict_catboost(texts: list[str]) -> np.ndarray:
    model: CatBoostClassifier = load_cb_model()
    frame = pl.DataFrame({"text": texts})
    features = augment_data(frame, "text").drop("text")
    return np.asarray(model.predict_proba(make_pool(features))[:, 1], dtype=float)


def predict_tfidf(texts: list[str]) -> np.ndarray:
    model = joblib.load(PROJECT_ROOT / "models" / "tfidf_svm.joblib")
    return np.asarray(model.predict_proba(texts)[:, 1], dtype=float)


def predict_distilbert(texts: list[str], batch_size: int) -> np.ndarray:
    model_path = PROJECT_ROOT / "models" / "distilbert_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    outputs: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits
            outputs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(outputs).astype(float)


def predict_rubric(
    texts: list[str],
    batch_size: int,
    *,
    cache_path: Path,
    request_delay: float,
) -> np.ndarray:
    regression_model, feature_columns = load_rubric_regressor()
    if set(feature_columns) != set(AXES_DEFINITIONS):
        raise ValueError(
            "Rubric artifact features do not match the current rubric axes: "
            f"artifact={feature_columns}, current={list(AXES_DEFINITIONS)}"
        )
    cache = _load_rubric_cache(cache_path)
    scores_by_row: dict[int, dict[str, int]] = {}
    for index, text in enumerate(texts):
        cached = cache.get(_text_key(text))
        if cached is not None:
            scores_by_row[index] = cached

    last_request_at: float | None = None
    for start in range(0, len(texts), batch_size):
        batch_indexes = [
            index
            for index in range(start, min(start + batch_size, len(texts)))
            if index not in scores_by_row
        ]
        if not batch_indexes:
            continue
        last_request_at = _pace_request(last_request_at, request_delay)
        scored = score_text_batch([texts[index] for index in batch_indexes])
        for local_index, scores in scored.items():
            row_index = batch_indexes[local_index]
            scores_by_row[row_index] = scores
            cache[_text_key(texts[row_index])] = scores
        _write_rubric_cache(cache_path, cache)

        # Structured batch output can legally omit an unscorable row. Recover
        # only those rows individually instead of discarding the completed work.
        missing = [index for index in batch_indexes if index not in scores_by_row]
        for row_index in missing:
            last_request_at = _pace_request(last_request_at, request_delay)
            single = score_text_batch([texts[row_index]])
            if 0 not in single:
                raise RuntimeError(
                    f"Rubric judge omitted row {row_index} even when scored alone"
                )
            scores_by_row[row_index] = single[0]
            cache[_text_key(texts[row_index])] = single[0]
            _write_rubric_cache(cache_path, cache)

    if set(scores_by_row) != set(range(len(texts))):
        missing = sorted(set(range(len(texts))) - set(scores_by_row))
        raise RuntimeError(f"Rubric scoring incomplete; missing rows: {missing}")
    features = np.asarray([
        [scores_by_row[index][axis] for axis in feature_columns]
        for index in range(len(texts))
    ], dtype=float)
    return np.asarray(regression_model.predict_proba(features)[:, 1], dtype=float)


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_rubric_cache(path: Path) -> dict[str, dict[str, int]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Rubric cache must be a JSON object")
    return {
        str(key): {str(axis): int(score) for axis, score in scores.items()}
        for key, scores in payload.items()
    }


def _write_rubric_cache(path: Path, cache: dict[str, dict[str, int]]) -> None:
    path.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")


def _pace_request(last_request_at: float | None, delay: float) -> float:
    if last_request_at is not None:
        sleep(max(0.0, delay - (monotonic() - last_request_at)))
    return monotonic()


def _validate_probabilities(name: str, values: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (size,):
        raise ValueError(f"{name} returned shape {values.shape}; expected {(size,)}")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"{name} returned invalid probabilities")
    return values


def weighted_ensemble(
    probabilities: dict[str, np.ndarray],
    components: list[str],
    weights: dict[str, float],
) -> np.ndarray:
    active_weights = np.asarray([weights[name] for name in components], dtype=float)
    if active_weights.sum() <= 0:
        raise ValueError("Ensemble weights must have positive total weight")
    matrix = np.vstack([probabilities[name] for name in components])
    return np.average(matrix, axis=0, weights=active_weights)


def calculate_metrics(
    name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    latency_seconds: float,
) -> MetricRow:
    predictions = (scores >= threshold).astype(int)
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    return MetricRow(
        model=name,
        accuracy=float(accuracy_score(labels, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(labels, scores)),
        average_precision=float(average_precision_score(labels, scores)),
        brier=float(brier_score_loss(labels, scores)),
        log_loss=float(log_loss(labels, clipped, labels=[0, 1])),
        ece=float(expected_calibration_error(labels, scores)),
        latency_seconds=float(latency_seconds),
    )


def expected_calibration_error(
    labels: np.ndarray,
    scores: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0, 1, bins + 1)
    bucket = np.minimum(np.digitize(scores, edges[1:-1]), bins - 1)
    total = len(labels)
    error = 0.0
    for index in range(bins):
        mask = bucket == index
        if not mask.any():
            continue
        error += mask.sum() / total * abs(labels[mask].mean() - scores[mask].mean())
    return float(error)


def compare_ensemble_to_best_component(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    *,
    threshold: float,
    bootstrap_samples: int,
    seed: int,
    ensemble_name: str,
) -> dict[str, object]:
    component_names = [
        name for name in ("catboost", "distilbert", "tfidf_svm", "rubric_regressor")
        if name in probabilities
    ]
    component_f1 = {
        name: f1_score(labels, probabilities[name] >= threshold, zero_division=0)
        for name in component_names
    }
    best_name = max(component_f1, key=component_f1.get)
    observed = float(
        f1_score(labels, probabilities[ensemble_name] >= threshold, zero_division=0)
        - component_f1[best_name]
    )
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    class_indexes = [np.flatnonzero(labels == value) for value in (0, 1)]
    for _ in range(bootstrap_samples):
        indexes = np.concatenate([
            rng.choice(values, size=len(values), replace=True)
            for values in class_indexes
        ])
        ensemble_f1 = f1_score(
            labels[indexes],
            probabilities[ensemble_name][indexes] >= threshold,
            zero_division=0,
        )
        component_score = f1_score(
            labels[indexes],
            probabilities[best_name][indexes] >= threshold,
            zero_division=0,
        )
        deltas.append(float(ensemble_f1 - component_score))
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "ensemble": ensemble_name,
        "strongest_component": best_name,
        "ensemble_f1": float(f1_score(
            labels,
            probabilities[ensemble_name] >= threshold,
            zero_division=0,
        )),
        "component_f1": float(component_f1[best_name]),
        "f1_delta": observed,
        "paired_bootstrap_95pct_ci": [float(low), float(high)],
        "outperformed": bool(low > 0),
        "interpretation": (
            "Ensemble F1 is higher with a paired bootstrap interval above zero."
            if low > 0 else
            "Ensemble F1 is lower with a paired bootstrap interval below zero."
            if high < 0 else
            "The paired bootstrap interval crosses zero; superiority is not established."
        ),
    }


def calculate_ablations(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    *,
    threshold: float,
    include_rubric: bool,
) -> list[dict[str, object]]:
    components = ["catboost", "distilbert", "tfidf_svm"]
    if include_rubric:
        components.append("rubric_regressor")
    rows = []
    for removed in [None, *components]:
        retained = [name for name in components if name != removed]
        if not retained:
            continue
        scores = weighted_ensemble(probabilities, retained, DEFAULT_WEIGHTS)
        rows.append({
            "configuration": "all" if removed is None else f"without_{removed}",
            "components": retained,
            "f1": float(f1_score(labels, scores >= threshold, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "brier": float(brier_score_loss(labels, scores)),
        })
    return rows


def write_predictions(
    report_dir: Path,
    texts: list[str],
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> None:
    frame = pl.DataFrame({
        "row": np.arange(len(texts)),
        "label": labels,
        "text": texts,
        **{f"probability_{name}": values for name, values in probabilities.items()},
    })
    frame.write_csv(report_dir / "predictions.csv")


def write_metrics(report_dir: Path, rows: list[MetricRow]) -> None:
    pl.DataFrame([asdict(row) for row in rows]).write_csv(report_dir / "metrics.csv")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def create_plots(
    report_dir: Path,
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    rows: list[MetricRow],
    *,
    threshold: float,
) -> None:
    model_names = [row.model for row in rows]
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    figure, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(model_names))
    width = 0.26
    for offset, metric in enumerate(("f1", "roc_auc", "average_precision")):
        values = [getattr(row, metric) for row in rows]
        axis.bar(positions + (offset - 1) * width, values, width, label=metric)
    axis.set_xticks(positions, [name.replace("_", "\n") for name in model_names])
    axis.set_ylim(0, 1)
    axis.set_ylabel("score")
    axis.set_title("Model discrimination metrics")
    axis.legend()
    figure.savefig(report_dir / "metric_comparison.png")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, color in zip(model_names, colors, strict=True):
        scores = probabilities[name]
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        axes[0].plot(fpr, tpr, label=name, color=color)
        axes[1].plot(recall, precision, label=name, color=color)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[0].set(title="ROC curves", xlabel="false-positive rate", ylabel="true-positive rate")
    axes[1].set(title="Precision-recall curves", xlabel="recall", ylabel="precision")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    figure.savefig(report_dir / "discrimination_curves.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 11)
    for name, color in zip(model_names, colors, strict=True):
        scores = probabilities[name]
        indexes = np.minimum(np.digitize(scores, bins[1:-1]), 9)
        observed, predicted = [], []
        for index in range(10):
            mask = indexes == index
            if mask.any():
                observed.append(labels[mask].mean())
                predicted.append(scores[mask].mean())
        axis.plot(predicted, observed, marker="o", label=name, color=color)
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="ideal")
    axis.set(title="Calibration", xlabel="mean predicted probability", ylabel="observed AI rate")
    axis.legend(fontsize=8)
    figure.savefig(report_dir / "calibration.png")
    plt.close(figure)

    figure, axes = plt.subplots(1, len(model_names), figsize=(4 * len(model_names), 4))
    axes = np.atleast_1d(axes)
    for axis, name in zip(axes, model_names, strict=True):
        matrix = confusion_matrix(labels, probabilities[name] >= threshold)
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set(title=name, xlabel="predicted", ylabel="actual", xticks=[0, 1], yticks=[0, 1])
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.savefig(report_dir / "confusion_matrices.png")
    plt.close(figure)

    component_names = [name for name in ("catboost", "distilbert", "tfidf_svm", "rubric_regressor") if name in probabilities]
    if len(component_names) > 1:
        matrix = np.corrcoef(np.vstack([probabilities[name] for name in component_names]))
        figure, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(component_names)), component_names, rotation=30, ha="right")
        axis.set_yticks(range(len(component_names)), component_names)
        for row in range(len(component_names)):
            for column in range(len(component_names)):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
        axis.set_title("Component probability correlation")
        figure.colorbar(image, ax=axis)
        figure.savefig(report_dir / "component_correlation.png")
        plt.close(figure)


def write_markdown_report(
    report_dir: Path,
    dataset_meta: dict[str, object],
    rows: list[MetricRow],
    comparison: dict[str, object],
    ablations: list[dict[str, object]],
    rubric_status: str,
) -> None:
    header = (
        "| model | F1 | ROC AUC | PR AUC | precision | recall | Brier | ECE |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    metric_lines = [header, *[
        f"| {row.model} | {row.f1:.4f} | {row.roc_auc:.4f} | "
        f"{row.average_precision:.4f} | {row.precision:.4f} | "
        f"{row.recall:.4f} | {row.brier:.4f} | {row.ece:.4f} |"
        for row in rows
    ]]
    ablation_lines = [
        f"| {row['configuration']} | {row['f1']:.4f} | "
        f"{row['roc_auc']:.4f} | {row['brier']:.4f} |"
        for row in ablations
    ]
    report = f"""# AI-detection ensemble assessment

Generated: {datetime.now(UTC).isoformat()}

## Conclusion

{comparison['interpretation']}

- Evaluated ensemble: `{comparison['ensemble']}`
- Strongest individual component: `{comparison['strongest_component']}`
- Ensemble F1: {comparison['ensemble_f1']:.4f}
- Component F1: {comparison['component_f1']:.4f}
- Paired F1 delta: {comparison['f1_delta']:+.4f}
- Paired bootstrap 95% interval: [{comparison['paired_bootstrap_95pct_ci'][0]:+.4f}, {comparison['paired_bootstrap_95pct_ci'][1]:+.4f}]
- Statistically supported outperformance under this protocol: **{comparison['outperformed']}**

## Evaluation data

- Dataset: `{dataset_meta['name']}`
- Source split: `{dataset_meta['split']}` (used only for external evaluation)
- Sample: {dataset_meta['sample_rows']} of {dataset_meta['source_rows']} rows
- Classes: {dataset_meta['class_counts']}
- Rubric component: `{rubric_status}`

This dataset contains general English sentences, not resumes. Results measure
out-of-domain sentence detection and must not be interpreted as in-domain resume
performance. The source exposes one split, so this study samples it without
training or tuning any model.

## Model metrics

{chr(10).join(metric_lines)}

## Leave-one-component-out ablation

| configuration | F1 | ROC AUC | Brier |
|---|---:|---:|---:|
{chr(10).join(ablation_lines)}

## Visualizations

![Metric comparison](metric_comparison.png)

![Discrimination curves](discrimination_curves.png)

![Calibration](calibration.png)

![Confusion matrices](confusion_matrices.png)

![Component correlation](component_correlation.png)

## Interpretation guardrails

- The threshold remains the runtime default of 0.5; it was not optimized on this sample.
- F1 superiority is assessed with a paired, class-stratified bootstrap interval.
- The external dataset may overlap conceptually or literally with original training data; no dataset lineage proves otherwise.
- Sentence-level results do not establish performance on resume bullets or full resume sections.
- If the rubric component was not completed, `local_weighted_ensemble` is a renormalized three-artifact ensemble, not the full production ensemble.
- Model artifacts were evaluated as found; no retraining occurred.
"""
    (report_dir / "REPORT.md").write_text(report, encoding="utf-8")


def limitations(include_rubric: bool, rubric_status: str) -> list[str]:
    values = [
        "External data is general sentence prose rather than resume text.",
        "The dataset offers one source split; the study does not tune on it.",
        "Training-data overlap cannot be ruled out from available lineage.",
        "The default decision threshold is evaluated without optimization.",
    ]
    if not include_rubric:
        values.append(
            "The rubric component was omitted because it requires paid/networked LLM scoring."
        )
    elif rubric_status != "completed":
        values.append(f"The rubric component did not complete: {rubric_status}")
    return values


if __name__ == "__main__":
    raise SystemExit(main())
