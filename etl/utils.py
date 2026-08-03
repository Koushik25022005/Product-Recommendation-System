"""Shared utilities for ETL and validation."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_services_config() -> dict:
    config_path = PROJECT_ROOT / "config" / "services.yaml"
    with open(config_path, encoding="utf-8") as f:
        raw = f.read()
    # Expand simple ${VAR:-default} placeholders
    for key in ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"):
        default = {
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "recommendations",
            "POSTGRES_USER": "recuser",
            "POSTGRES_PASSWORD": "recpass",
        }[key]
        raw = raw.replace(f"${{{key}:-{default}}}", os.getenv(key, default))
    return yaml.safe_load(raw)


def get_db_url() -> str:
    cfg = load_services_config()["database"]
    return (
        f"postgresql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['name']}"
    )


def get_engine() -> Engine:
    return create_engine(get_db_url(), pool_pre_ping=True)


def init_schema(engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    schema_path = PROJECT_ROOT / "sql" / "schema.sql"
    with open(schema_path, encoding="utf-8") as f:
        ddl = f.read()
    with engine.begin() as conn:
        conn.execute(text(ddl))


def log_etl_counts(
    engine: Engine,
    run_id: uuid.UUID,
    dataset: str,
    table_name: str,
    stage: str,
    row_count: int,
    null_flags: dict | None = None,
    notes: str | None = None,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO etl_runs (run_id, dataset, table_name, stage, row_count, null_flags, notes)
                VALUES (:run_id, :dataset, :table_name, :stage, :row_count, CAST(:null_flags AS jsonb), :notes)
                """
            ),
            {
                "run_id": str(run_id),
                "dataset": dataset,
                "table_name": table_name,
                "stage": stage,
                "row_count": row_count,
                "null_flags": __import__("json").dumps(null_flags or {}),
                "notes": notes,
            },
        )
