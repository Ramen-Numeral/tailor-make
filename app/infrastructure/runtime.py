from config.io import IOConfig


def prepare_runtime_directories(config: IOConfig) -> None:
    """Create directories required by the application."""
    for directory in (
        config.data_dir,
        config.model_dir,
        config.log_dir,
        config.resume_output_dir,
        config.pipeline_output_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
