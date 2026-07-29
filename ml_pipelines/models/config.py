"""Training configuration for AI-detection ensemble models."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)


class CatBoostConfig(ModelConfig):
    drop_columns: tuple[str, ...] = (
        "text",
        "abstract",
        "title",
        "newline_count",
    )
    target_column: str = "label"

    learning_rate: float = Field(default=0.05, gt=0)
    depth: int = Field(default=9, gt=0)
    iterations: int = Field(default=1000, gt=0)

    loss_function: str = "Logloss"
    eval_metric: str = "AUC"
    l2_leaf_reg: float = Field(default=3.0, ge=0)

    random_seed: int = Field(default=42, ge=0)
    verbose: int = Field(default=100, ge=0)
    early_stopping_rounds: int = Field(default=100, ge=1)

    sweep_learning_rates: tuple[float, ...] = (
        0.05,
        0.10,
        0.15,
        0.20,
    )
    sweep_depths: tuple[int, ...] = (
        1,
        4,
        7,
        10,
        13,
    )

    def make_model(
        self,
        *,
        depth: int | None = None,
        lr: float | None = None,
        iter: int | None = None,
    ):
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            depth=self.depth if depth is None else depth,
            learning_rate=self.learning_rate if lr is None else lr,
            iterations=self.iterations if iter is None else iter,
            loss_function=self.loss_function,
            eval_metric=self.eval_metric,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            verbose=self.verbose,
            early_stopping_rounds=self.early_stopping_rounds,
        )


class DistilBertConfig(ModelConfig):
    model_name: str = "distilbert-base-uncased"
    drop_columns: tuple[str, ...] = ()
    target_column: str = "label"

    max_length: int = Field(default=512, gt=0)
    stride: int = Field(default=128, ge=0)

    learning_rate: float = Field(default=2e-5, gt=0)
    train_batch_size: int = Field(default=8, gt=0)
    eval_batch_size: int = Field(default=8, gt=0)
    epochs: int = Field(default=3, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)

    @model_validator(mode="after")
    def validate_stride(self) -> "DistilBertConfig":
        if self.stride >= self.max_length:
            raise ValueError("stride must be smaller than max_length")
        return self

    def make_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.model_name)

    def make_model(self):
        from transformers import AutoModelForSequenceClassification

        return AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            id2label={0: "HUMAN_WRITTEN", 1: "AI_WRITTEN"},
            label2id={"HUMAN_WRITTEN": 0, "AI_WRITTEN": 1},
        )


class TfidfConfig(ModelConfig):
    target_column: str = "label"

    # Phrase-level features are less prone to treating an isolated resume word
    # as a meaningful authorship signal.
    ngram_range: tuple[int, int] = (2, 4)
    max_features: int = Field(default=50_000, gt=0)
    min_df: int = Field(default=2, ge=1)
    max_df: float = Field(default=0.95, gt=0, le=1)


class EnsembleModelConfig(ModelConfig):
    catboost: CatBoostConfig = Field(default_factory=CatBoostConfig)
    distilbert: DistilBertConfig = Field(default_factory=DistilBertConfig)
    tfidf: TfidfConfig = Field(default_factory=TfidfConfig)
