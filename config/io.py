"""Application directories and derived file paths."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class IOConfig(BaseModel):
    """Filesystem locations used by the application."""

    model_config = ConfigDict(frozen=True)

    data_dir: Path
    model_dir: Path
    log_dir: Path
    env_file_path: Path
    template_dir: Path
    resume_output_dir: Path
    pipeline_output_path: Path
    homebrew_library_dir: Path

    train_filename: str
    validation_filename: str
    test_filename: str

    catboost_model_filename: str
    distilbert_model_filename: str
    tfidf_model_filename: str
    rubric_regressor_filename: str

    resume_template_filename: str
    llm_calls_log_filename: str
    llm_errors_log_filename: str

    @property
    def train_path(self) -> Path:
        return self.data_dir / self.train_filename

    @property
    def validation_path(self) -> Path:
        return self.data_dir / self.validation_filename

    @property
    def test_path(self) -> Path:
        return self.data_dir / self.test_filename

    @property
    def catboost_model_path(self) -> Path:
        return self.model_dir / self.catboost_model_filename

    @property
    def distilbert_model_path(self) -> Path:
        return self.model_dir / self.distilbert_model_filename

    @property
    def tfidf_model_path(self) -> Path:
        return self.model_dir / self.tfidf_model_filename

    @property
    def rubric_regressor_path(self) -> Path:
        return self.model_dir / self.rubric_regressor_filename

    @property
    def resume_template_path(self) -> Path:
        return self.template_dir / self.resume_template_filename

    @property
    def default_resume_output_path(self) -> Path:
        return self.resume_output_dir / "resume"

    @property
    def llm_calls_log_path(self) -> Path:
        return self.log_dir / self.llm_calls_log_filename

    @property
    def llm_errors_log_path(self) -> Path:
        return self.log_dir / self.llm_errors_log_filename
