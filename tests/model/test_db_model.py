import torch

from app.infrastructure.ai_detection.distilbert_component import (
    predict_distilbert,
)


class FakeEncoding(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return FakeEncoding(
            input_ids=torch.zeros((1, 4), dtype=torch.long)
        )


class FakeModel:
    def __init__(self, logits):
        self.logits = logits

    def to(self, device):
        return self

    def eval(self):
        return None

    def __call__(self, **kwargs):
        return type("Output", (), {"logits": self.logits})()


def test_distilbert_reports_ai_class_probability() -> None:
    result = predict_distilbert(
        FakeModel(torch.tensor([[10.0, 0.0]])),
        FakeTokenizer(),
        "human text",
    )

    assert result.model_name == "distilbert"
    assert result.ai_probability < 0.01
