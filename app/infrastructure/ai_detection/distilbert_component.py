"""DistilBERT model loading and inference."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from app.features.ai_detection.config import ai_detection_config
from app.features.ai_detection.schema import (
    ComponentScore,
    TokenAttribution,
)
from config.settings import get_settings
from ml_pipelines.models.config import DistilBertConfig
from ml_pipelines.models.distilbert.db_model import load_dbert_model

distilbert_config = DistilBertConfig()


def _tokenize(
    texts: list[str],
    tokeniser: AutoTokenizer,
) -> BatchEncoding:
    """Tokenize texts for DistilBERT."""
    encoded = tokeniser(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=distilbert_config.max_length,
        stride=distilbert_config.stride,
        return_overflowing_tokens=True,
    )
    encoded.pop("overflow_to_sample_mapping", None)
    return encoded.to(get_settings().runtime.device)


def predict_distilbert(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
) -> ComponentScore:
    """Return probability and Integrated Gradients token evidence."""
    model.to(get_settings().runtime.device)
    model.eval()

    probability = _probability(model, tokenizer, text)
    token_attributions: list[TokenAttribution] = []
    explanation_error: str | None = None
    try:
        token_attributions = _integrated_gradients(model, tokenizer, text)
    except Exception as error:
        explanation_error = f"Integrated Gradients unavailable: {error}"

    return ComponentScore(
        model_name="distilbert",
        ai_probability=probability,
        token_attributions=token_attributions,
        explanation_note=explanation_error,
    )


def _probability(model, tokenizer, text: str) -> float:
    with torch.no_grad():
        outputs = model(**_tokenize([text], tokenizer))
        return float(
            torch.softmax(outputs.logits, dim=-1).mean(dim=0)[1]
        )


def _integrated_gradients(model, tokenizer, text: str) -> list[TokenAttribution]:
    device = get_settings().runtime.device
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=distilbert_config.max_length,
        return_special_tokens_mask=True,
    )
    special_mask = encoded.pop("special_tokens_mask")[0].to(device)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    embeddings = model.get_input_embeddings()
    actual = embeddings(input_ids).detach()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    baseline_ids = torch.full_like(input_ids, pad_id)
    baseline = embeddings(baseline_ids).detach()
    baseline = torch.where(
        special_mask.view(1, -1, 1).bool(),
        actual,
        baseline,
    )
    difference = actual - baseline
    gradients = []
    for alpha in torch.linspace(
        0.0,
        1.0,
        ai_detection_config.integrated_gradients_steps,
        device=device,
    ):
        interpolated = (baseline + alpha * difference).requires_grad_(True)
        output = model(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
        )
        target = torch.softmax(output.logits, dim=-1)[0, 1]
        gradients.append(
            torch.autograd.grad(target, interpolated)[0].detach()
        )
    attribution = (
        difference * torch.stack(gradients).mean(dim=0)
    ).sum(dim=-1)[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return [
        TokenAttribution(
            token=str(token),
            attribution=float(value),
            direction="machine_like" if value >= 0 else "human_like",
        )
        for token, value, special, attended in zip(
            tokens,
            attribution,
            special_mask,
            attention_mask[0],
            strict=True,
        )
        if not bool(special) and bool(attended)
    ]
