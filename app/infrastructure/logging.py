"""Logging settings."""

import logging
from pathlib import Path

from config.settings import get_settings

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_file_logger(
    name: str,
    path: Path,
    level: int,
) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    logger.setLevel(level)
    logger.propagate = False
    logger.addHandler(handler)

    return logger


def get_llm_calls_logger() -> logging.Logger:
    config = get_settings().io
    return configure_file_logger(
        "llm.calls",
        config.llm_calls_log_path,
        logging.INFO,
    )


def get_llm_errors_logger() -> logging.Logger:
    config = get_settings().io
    return configure_file_logger(
        "llm.errors",
        config.llm_errors_log_path,
        logging.WARNING,
    )
