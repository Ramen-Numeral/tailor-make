import json
from types import SimpleNamespace

import pytest

from ml_pipelines.models.rubric_regressor import score
from ml_pipelines.models.rubric_regressor.rubric import AXES_DEFINITIONS


@pytest.mark.parametrize(
    "payload,error",
    [
        ("not json", ValueError),
        (json.dumps([]), TypeError),
        (json.dumps({"unknown": 3}), ValueError),
        (json.dumps({"global_coherence": 6}), ValueError),
        (json.dumps({"global_coherence": True}), ValueError),
    ],
)
def test_parse_rubric_response_rejects_invalid_payloads(
    payload,
    error,
) -> None:
    with pytest.raises(error):
        score.parse_rubric_response(payload)


def test_parse_rubric_response_accepts_partial_scores() -> None:
    assert score.parse_rubric_response('{"global_coherence": 4}') == {
        "global_coherence": 4
    }


def test_score_text_merges_partial_retries(monkeypatch) -> None:
    axes = list(AXES_DEFINITIONS)

    class Client:
        responses = iter(
            [
                json.dumps({axis: 3 for axis in axes[:5]}),
                json.dumps({axis: 4 for axis in axes[5:]}),
            ]
        )

        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, prompt):
            return next(self.responses)

    monkeypatch.setattr(score, "LLMRoutedClient", Client)
    monkeypatch.setattr(
        score,
        "get_settings",
        lambda: SimpleNamespace(
            runtime=SimpleNamespace(require_groq_api_key=lambda: "test")
        ),
    )

    result = score.score_text("sample", max_attempts=2)

    assert set(result) == set(AXES_DEFINITIONS)


def test_score_text_batch_returns_empty_mapping_for_no_texts() -> None:
    assert score.score_text_batch([]) == {}
