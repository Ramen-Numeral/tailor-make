"""Application settings assembled from the active configuration modules."""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from config.io import IOConfig
from config.runtime import RuntimeConfig
from config.train_data import TrainDataConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_io_config() -> IOConfig:
    return IOConfig(
        data_dir=PROJECT_ROOT / "dataset",
        model_dir=PROJECT_ROOT / "models",
        log_dir=PROJECT_ROOT / "log",
        env_file_path=PROJECT_ROOT / ".env",
        template_dir=PROJECT_ROOT / "app/features/renderer/templates",
        resume_output_dir=PROJECT_ROOT / "output/resumes",
        pipeline_output_path=PROJECT_ROOT / "output/pipeline_smoke_test",
        homebrew_library_dir=Path("/opt/homebrew/lib"),
        train_filename="train_set.csv",
        validation_filename="val_set.csv",
        test_filename="test_set.csv",
        catboost_model_filename="cb_model",
        distilbert_model_filename="distilbert_model",
        tfidf_model_filename="tfidf_svm.joblib",
        rubric_regressor_filename="rubric_regression_head.joblib",
        resume_template_filename="resume.html",
        llm_calls_log_filename="llm_calls.log",
        llm_errors_log_filename="llm_errors.log",
    )


def build_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(_env_file=build_io_config().env_file_path)


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime: RuntimeConfig = Field(default_factory=build_runtime_config)
    train_data: TrainDataConfig = Field(default_factory=TrainDataConfig)
    io: IOConfig = Field(default_factory=build_io_config)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
