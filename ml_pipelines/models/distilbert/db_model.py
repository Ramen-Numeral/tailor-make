from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.settings import get_settings


def save_model(
    model: AutoModelForSequenceClassification,
    tokeniser: AutoTokenizer,
    path: str | Path | None = None,
):
    path = Path(path or get_settings().io.distilbert_model_path)
    path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(path)
    tokeniser.save_pretrained(path)


def load_dbert_model(
    path: str | Path | None = None,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    path = Path(path or get_settings().io.distilbert_model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model directory not found: {path}")

    tokeniser = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return model, tokeniser
