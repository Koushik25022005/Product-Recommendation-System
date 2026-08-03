"""Load model hyperparameter config."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _expand_env(raw: str) -> str:
    if "${MLFLOW_TRACKING_URI:-file:./mlruns}" in raw:
        default = "file:./mlruns"
        raw = raw.replace(
            "${MLFLOW_TRACKING_URI:-file:./mlruns}",
            os.getenv("MLFLOW_TRACKING_URI", default),
        )
    return raw


def load_model_config() -> dict:
    path = PROJECT_ROOT / "config" / "model.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(_expand_env(f.read()))
