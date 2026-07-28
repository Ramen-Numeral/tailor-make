"""Training-data configuration."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainDataConfig(BaseModel):
    """Data source, splitting, and preprocessing settings."""

    model_config = ConfigDict(frozen=True)

    data_address: str = "Ateeqq/AI-and-Human-Generated-Text"

    train_pct: float = Field(default=0.8, ge=0, le=1)
    validation_pct: float = Field(default=0.1, ge=0, le=1)
    test_pct: float = Field(default=0.1, ge=0, le=1)

    rubric_candidate_sample_size: int = Field(default=10_000, gt=0)
    rubric_training_sample_size: int = Field(default=200, gt=0)
    rubric_max_text_chars: int = Field(default=1500, gt=0)
    distilbert_training_sample_size: int = Field(default=10_000, gt=0)

    text_column: str = "text"
    target_column: str = "label"

    augmentation_precision: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def validate_split_total(self) -> "TrainDataConfig":
        total = self.train_pct + self.validation_pct + self.test_pct

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "train_pct, validation_pct, and test_pct must sum to 1"
            )

        return self
