"""Data validation against PostgreSQL using pandera."""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema
from pandera.errors import SchemaErrors
from sqlalchemy import text

from etl.utils import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

users_schema = DataFrameSchema(
    {
        "id": Column(int, Check.ge(1)),
        "external_id": Column(int, Check.ge(1), nullable=False),
        "age": Column(float, nullable=True),
        "gender": Column(str, nullable=True),
        "occupation": Column(str, nullable=True),
        "zip_code": Column(str, nullable=True),
        "created_at": Column("datetime64[ns, UTC]", nullable=False),
    },
    strict=False,
    coerce=True,
)

items_schema = DataFrameSchema(
    {
        "id": Column(int, Check.ge(1)),
        "external_id": Column(int, Check.ge(1), nullable=False),
        "title": Column(str, Check.str_length(min_value=1), nullable=False),
        "release_date": Column(object, nullable=True),
        "video_release": Column(object, nullable=True),
        "imdb_url": Column(str, nullable=True),
        "created_at": Column("datetime64[ns, UTC]", nullable=False),
    },
    strict=False,
    coerce=True,
)

interactions_schema = DataFrameSchema(
    {
        "id": Column(int, Check.ge(1)),
        "user_id": Column(int, Check.ge(1), nullable=False),
        "item_id": Column(int, Check.ge(1), nullable=False),
        "rating": Column(float, Check.in_range(0, 5), nullable=True),
        "signal_type": Column(str, Check.isin(["explicit", "implicit"]), nullable=False),
        "interaction_type": Column(
            str,
            Check.isin(["rating", "click", "view", "purchase"]),
            nullable=False,
        ),
        "interacted_at": Column("datetime64[ns, UTC]", nullable=False),
        "created_at": Column("datetime64[ns, UTC]", nullable=False),
    },
    strict=False,
    coerce=True,
)


def fetch_table(table: str) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(text(f"SELECT * FROM {table}"), engine)


def check_no_nulls_in_required_columns(df: pd.DataFrame, required: list[str], table: str) -> list[str]:
    errors = []
    for col in required:
        if col not in df.columns:
            errors.append(f"{table}: missing column '{col}'")
            continue
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            errors.append(f"{table}.{col}: {null_count} null values in required column")
    return errors


def check_orphaned_interactions(engine) -> list[str]:
    errors = []
    with engine.connect() as conn:
        orphan_users = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM interactions i
                LEFT JOIN users u ON i.user_id = u.id
                WHERE u.id IS NULL
                """
            )
        ).scalar()
        orphan_items = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM interactions i
                LEFT JOIN items it ON i.item_id = it.id
                WHERE it.id IS NULL
                """
            )
        ).scalar()

    if orphan_users:
        errors.append(f"interactions: {orphan_users} rows with orphaned user_id FK")
    if orphan_items:
        errors.append(f"interactions: {orphan_items} rows with orphaned item_id FK")
    return errors


def check_duplicate_external_ids(df: pd.DataFrame, table: str) -> list[str]:
    dupes = df["external_id"].duplicated().sum()
    if dupes:
        return [f"{table}: {dupes} duplicate external_id values"]
    return []


def validate_database() -> bool:
    engine = get_engine()
    all_errors: list[str] = []

    logger.info("Fetching tables from PostgreSQL ...")
    users = fetch_table("users")
    items = fetch_table("items")
    interactions = fetch_table("interactions")

    if users.empty or items.empty or interactions.empty:
        all_errors.append("One or more core tables are empty")
        for err in all_errors:
            logger.error("FAIL: %s", err)
        return False

    logger.info("Row counts — users: %d, items: %d, interactions: %d", len(users), len(items), len(interactions))

    # Pandera schema validation
    for name, df, schema in [
        ("users", users, users_schema),
        ("items", items, items_schema),
        ("interactions", interactions, interactions_schema),
    ]:
        try:
            schema.validate(df, lazy=True)
            logger.info("Pandera schema OK: %s", name)
        except SchemaErrors as exc:
            all_errors.append(f"{name} schema validation failed:\n{exc}")

    # Required null checks
    all_errors.extend(check_no_nulls_in_required_columns(users, ["external_id"], "users"))
    all_errors.extend(check_no_nulls_in_required_columns(items, ["external_id", "title"], "items"))
    all_errors.extend(
        check_no_nulls_in_required_columns(
            interactions,
            ["user_id", "item_id", "signal_type", "interaction_type", "interacted_at"],
            "interactions",
        )
    )

    # FK integrity
    all_errors.extend(check_orphaned_interactions(engine))

    # Uniqueness
    all_errors.extend(check_duplicate_external_ids(users, "users"))
    all_errors.extend(check_duplicate_external_ids(items, "items"))

    # Interaction FK membership
    valid_user_ids = set(users["id"].tolist())
    valid_item_ids = set(items["id"].tolist())
    bad_user = ~interactions["user_id"].isin(valid_user_ids)
    bad_item = ~interactions["item_id"].isin(valid_item_ids)
    if bad_user.any():
        all_errors.append(f"interactions: {bad_user.sum()} user_id values not in users.id")
    if bad_item.any():
        all_errors.append(f"interactions: {bad_item.sum()} item_id values not in items.id")

    if all_errors:
        logger.error("Validation FAILED with %d issue(s):", len(all_errors))
        for err in all_errors:
            logger.error("  - %s", err.replace("\n", " | "))
        return False

    logger.info("Validation PASSED — database is clean.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PostgreSQL data integrity")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on failure")
    args = parser.parse_args()
    ok = validate_database()
    if args.strict and not ok:
        sys.exit(1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
