import numpy as np
import polars as pl
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from config.train_data import TrainDataConfig
from config.settings import get_settings
from ml_pipelines.models.config import DistilBertConfig
from ml_pipelines.models.distilbert.db_model import save_model
from ml_pipelines.models.util import split_Xy


data_cfg = TrainDataConfig()
distilb_cfg = DistilBertConfig()


def make_dataset(
    X: pl.DataFrame | pl.Series | list[str],
    y: pl.Series | list[int],
) -> Dataset:
    if isinstance(X, pl.DataFrame):
        texts = X.get_column(data_cfg.text_column).cast(pl.Utf8).fill_null("").to_list()
    elif isinstance(X, pl.Series):
        texts = X.cast(pl.Utf8).fill_null("").to_list()
    else:
        texts = ["" if text is None else str(text) for text in X]

    if isinstance(y, pl.Series):
        labels = y.cast(pl.Int64).to_list()
    else:
        labels = [int(label) for label in y]

    return Dataset.from_dict({
        "text": texts,
        "labels": labels,
    })


def tokenize_dataset(
    dataset: Dataset,
    tokeniser,
) -> Dataset:
    def _tokenize(batch):
        return tokeniser(
            batch["text"],
            truncation=True,
            max_length=distilb_cfg.max_length,
        )

    return dataset.map(_tokenize, batched=True)


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    y_pred = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        y_pred,
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(labels, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def fit_distilbert_model(
    tr_X: pl.DataFrame | pl.Series | list[str],
    tr_y: pl.Series | list[int],
    val_X: pl.DataFrame | pl.Series | list[str],
    val_y: pl.Series | list[int],
):
    cuda_available = torch.cuda.is_available()
    tokeniser = distilb_cfg.make_tokenizer()
    model = distilb_cfg.make_model()

    train_ds = tokenize_dataset(make_dataset(tr_X, tr_y), tokeniser)
    val_ds = tokenize_dataset(make_dataset(val_X, val_y), tokeniser)

    args = TrainingArguments(
        output_dir=str(get_settings().io.distilbert_model_path),
        learning_rate=distilb_cfg.learning_rate,
        per_device_train_batch_size=distilb_cfg.train_batch_size,
        per_device_eval_batch_size=distilb_cfg.eval_batch_size,
        num_train_epochs=distilb_cfg.epochs,
        weight_decay=distilb_cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        use_cpu=not cuda_available,
        fp16=cuda_available,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokeniser,
        data_collator=DataCollatorWithPadding(tokenizer=tokeniser),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return model, tokeniser, trainer


def finetune_distilbert(
    tr_df: pl.DataFrame,
    val_df: pl.DataFrame,
    drop_cols: list[str] | None = None,
):
    tr_X, tr_y = split_Xy(
        tr_df,
        target_label=data_cfg.target_column,
        drop_cols=drop_cols,
    )

    val_X, val_y = split_Xy(
        val_df,
        target_label=data_cfg.target_column,
        drop_cols=drop_cols,
    )

    model, tokeniser, trainer = fit_distilbert_model(
        tr_X=tr_X,
        tr_y=tr_y,
        val_X=val_X,
        val_y=val_y,
    )

    save_model(
        model=model,
        tokeniser=tokeniser,
        path=get_settings().io.distilbert_model_path,
    )

    return model, tokeniser, trainer
