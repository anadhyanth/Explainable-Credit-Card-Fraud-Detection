"""Shared utility helpers: logging setup and artifact IO."""

import json
import logging
import sys
from pathlib import Path

import joblib

import config


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to stdout with a consistent format."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid duplicate handlers on re-import

    logger.setLevel(config.LOG_LEVEL)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def save_artifact(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: Path):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Artifact not found at {path}. Did you run training first?"
        )
    return joblib.load(path)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)
