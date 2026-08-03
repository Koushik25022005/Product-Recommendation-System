"""ETL pipeline: MovieLens 100K -> PostgreSQL."""

from __future__ import annotations

import argparse
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from etl.download_data import download_movielens_100k
from etl.utils import get_engine, init_schema, load_services_config, log_etl_counts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

GENDER_MAP = {"M": "M", "F": "F", "O": "O"}


def _count_nulls(df: pd.DataFrame) -> dict[str, int]:
    return {col: int(df[col].isna().sum()) for col in df.columns}


def _parse_movie_date(value: str) -> pd.Timestamp | pd.NaT:
    value = str(value).strip()
    if not value or value == "NULL":
        return pd.NaT
    for fmt in ("%d-%b-%Y", "%Y"):
        try:
            return pd.to_datetime(value, format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(value, errors="coerce")


def load_raw_users(data_dir: Path) -> pd.DataFrame:
    cols = ["external_id", "age", "gender", "occupation", "zip_code"]
    df = pd.read_csv(
        data_dir / "u.user",
        sep="|",
        names=cols,
        encoding="latin-1",
    )
    df["external_id"] = pd.to_numeric(df["external_id"], errors="coerce").astype("Int64")
    df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("Int64")
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    df["occupation"] = df["occupation"].astype(str).str.strip()
    df["zip_code"] = df["zip_code"].astype(str).str.strip()
    return df


def load_raw_items(data_dir: Path) -> pd.DataFrame:
    base_cols = [
        "external_id",
        "title",
        "release_date",
        "video_release",
        "imdb_url",
    ]
    genre_cols = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]
    df = pd.read_csv(
        data_dir / "u.item",
        sep="|",
        names=base_cols + genre_cols,
        encoding="latin-1",
    )
    genre_names = genre_cols

    def row_genres(row: pd.Series) -> list[str]:
        active = []
        for g in genre_names:
            if int(row.get(g, 0)) == 1:
                active.append(g)
        return active

    df["genres"] = df.apply(row_genres, axis=1)
    df["release_date"] = df["release_date"].apply(_parse_movie_date)
    df["video_release"] = df["video_release"].apply(_parse_movie_date)
    df["title"] = df["title"].astype(str).str.strip()
    df["imdb_url"] = df["imdb_url"].astype(str).str.strip()
    return df[
        ["external_id", "title", "release_date", "video_release", "imdb_url", "genres"]
    ]


def load_raw_interactions(data_dir: Path) -> pd.DataFrame:
    cols = ["user_external_id", "item_external_id", "rating", "timestamp"]
    df = pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        names=cols,
    )
    df["user_external_id"] = pd.to_numeric(df["user_external_id"], errors="coerce").astype("Int64")
    df["item_external_id"] = pd.to_numeric(df["item_external_id"], errors="coerce").astype("Int64")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("Int64")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["interacted_at"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def clean_users(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    null_flags = _count_nulls(df)
    before = len(df)

    df = df.dropna(subset=["external_id"])
    df = df.drop_duplicates(subset=["external_id"], keep="first")
    df["gender"] = df["gender"].where(df["gender"].isin(GENDER_MAP.keys()))
    df.loc[~df["gender"].isin(GENDER_MAP.keys()), "gender"] = None
    df["occupation"] = df["occupation"].replace({"nan": None, "": None})
    df["zip_code"] = df["zip_code"].replace({"nan": None, "": None})

    stats = {
        "before": before,
        "after": len(df),
        "removed": before - len(df),
        "null_flags": null_flags,
    }
    return df, stats


def clean_items(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    null_flags = _count_nulls(df)
    before = len(df)

    df = df.dropna(subset=["external_id", "title"])
    df = df[df["title"].str.len() > 0]
    df = df.drop_duplicates(subset=["external_id"], keep="first")

    stats = {
        "before": before,
        "after": len(df),
        "removed": before - len(df),
        "null_flags": null_flags,
    }
    return df, stats


def clean_interactions(
    df: pd.DataFrame,
    valid_user_ids: set[int],
    valid_item_ids: set[int],
) -> tuple[pd.DataFrame, dict]:
    null_flags = _count_nulls(df)
    before = len(df)

    df = df.dropna(subset=["user_external_id", "item_external_id", "rating", "interacted_at"])
    df = df.drop_duplicates(
        subset=["user_external_id", "item_external_id", "interacted_at"],
        keep="first",
    )

    orphan_users = ~df["user_external_id"].isin(valid_user_ids)
    orphan_items = ~df["item_external_id"].isin(valid_item_ids)
    orphan_count = int((orphan_users | orphan_items).sum())
    df = df[~orphan_users & ~orphan_items]

    df["signal_type"] = "explicit"
    df["interaction_type"] = "rating"
    df["rating"] = df["rating"].clip(0, 5)

    stats = {
        "before": before,
        "after": len(df),
        "removed": before - len(df),
        "orphaned_fk_rows": orphan_count,
        "null_flags": null_flags,
    }
    return df, stats


def load_to_postgres(
    users: pd.DataFrame,
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    truncate: bool = False,
) -> None:
    engine = get_engine()

    if truncate:
        logger.info("Truncating existing tables ...")
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE interactions, items, users RESTART IDENTITY CASCADE"))

    users_load = users.rename(columns={"external_id": "external_id"})[
        ["external_id", "age", "gender", "occupation", "zip_code"]
    ].copy()
    users_load["age"] = users_load["age"].astype(object).where(users_load["age"].notna(), None)

    items_load = items.copy()
    items_load["release_date"] = pd.to_datetime(items_load["release_date"]).dt.date
    items_load["video_release"] = pd.to_datetime(items_load["video_release"]).dt.date
    items_load["genres"] = items_load["genres"].apply(lambda g: g if isinstance(g, list) else [])

    logger.info("Loading %d users ...", len(users_load))
    users_load.to_sql("users", engine, if_exists="append", index=False, method="multi", chunksize=1000)

    logger.info("Loading %d items ...", len(items_load))
    raw_conn = engine.raw_connection()
    try:
        from psycopg2.extras import execute_values

        cur = raw_conn.cursor()
        item_rows = [
            (
                int(row.external_id),
                row.title,
                row.release_date,
                row.video_release,
                row.imdb_url,
                row.genres,
            )
            for row in items_load.itertuples(index=False)
        ]
        execute_values(
            cur,
            """
            INSERT INTO items (external_id, title, release_date, video_release, imdb_url, genres)
            VALUES %s
            """,
            item_rows,
            template="(%s, %s, %s, %s, %s, %s::text[])",
            page_size=1000,
        )
        raw_conn.commit()
    finally:
        raw_conn.close()

    with engine.connect() as conn:
        user_map = pd.read_sql(
            text("SELECT id, external_id FROM users"),
            conn,
        )
        item_map = pd.read_sql(
            text("SELECT id, external_id FROM items"),
            conn,
        )

    interactions_load = interactions.merge(
        user_map,
        left_on="user_external_id",
        right_on="external_id",
        how="inner",
    ).merge(
        item_map,
        left_on="item_external_id",
        right_on="external_id",
        suffixes=("_user", "_item"),
    )
    interactions_load = interactions_load.rename(columns={"id_user": "user_id", "id_item": "item_id"})
    interactions_load = interactions_load[
        ["user_id", "item_id", "rating", "signal_type", "interaction_type", "interacted_at"]
    ]

    logger.info("Loading %d interactions ...", len(interactions_load))
    interactions_load.to_sql(
        "interactions",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=5000,
    )


def run_etl(truncate: bool = False, skip_download: bool = False) -> uuid.UUID:
    run_id = uuid.uuid4()
    cfg = load_services_config()
    dataset = cfg["etl"]["dataset"]

    logger.info("ETL run_id=%s", run_id)

    if not skip_download:
        data_dir = download_movielens_100k()
    else:
        data_dir = Path(__file__).resolve().parent.parent / cfg["etl"]["raw_data_dir"] / "ml-100k"

    engine = get_engine()
    init_schema(engine)

    # --- Extract ---
    users_raw = load_raw_users(data_dir)
    items_raw = load_raw_items(data_dir)
    interactions_raw = load_raw_interactions(data_dir)

    log_etl_counts(engine, run_id, dataset, "users", "raw", len(users_raw), _count_nulls(users_raw))
    log_etl_counts(engine, run_id, dataset, "items", "raw", len(items_raw), _count_nulls(items_raw))
    log_etl_counts(
        engine, run_id, dataset, "interactions", "raw", len(interactions_raw), _count_nulls(interactions_raw)
    )

    logger.info(
        "Raw counts — users: %d, items: %d, interactions: %d",
        len(users_raw),
        len(items_raw),
        len(interactions_raw),
    )

    # --- Transform ---
    users_clean, users_stats = clean_users(users_raw)
    items_clean, items_stats = clean_items(items_raw)
    valid_user_ids = set(users_clean["external_id"].astype(int).tolist())
    valid_item_ids = set(items_clean["external_id"].astype(int).tolist())
    interactions_clean, interactions_stats = clean_interactions(
        interactions_raw, valid_user_ids, valid_item_ids
    )

    log_etl_counts(
        engine, run_id, dataset, "users", "cleaned", len(users_clean), users_stats["null_flags"],
        notes=f"removed={users_stats['removed']}",
    )
    log_etl_counts(
        engine, run_id, dataset, "items", "cleaned", len(items_clean), items_stats["null_flags"],
        notes=f"removed={items_stats['removed']}",
    )
    log_etl_counts(
        engine, run_id, dataset, "interactions", "cleaned", len(interactions_clean),
        interactions_stats["null_flags"],
        notes=(
            f"removed={interactions_stats['removed']}, "
            f"orphaned_fk={interactions_stats['orphaned_fk_rows']}"
        ),
    )

    logger.info(
        "Cleaned counts — users: %d (removed %d), items: %d (removed %d), "
        "interactions: %d (removed %d, orphaned FK %d)",
        users_stats["after"],
        users_stats["removed"],
        items_stats["after"],
        items_stats["removed"],
        interactions_stats["after"],
        interactions_stats["removed"],
        interactions_stats["orphaned_fk_rows"],
    )

    # --- Load ---
    load_to_postgres(users_clean, items_clean, interactions_clean, truncate=truncate)

    with engine.connect() as conn:
        loaded_users = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
        loaded_items = conn.execute(text("SELECT COUNT(*) FROM items")).scalar()
        loaded_interactions = conn.execute(text("SELECT COUNT(*) FROM interactions")).scalar()

    log_etl_counts(engine, run_id, dataset, "users", "loaded", loaded_users)
    log_etl_counts(engine, run_id, dataset, "items", "loaded", loaded_items)
    log_etl_counts(engine, run_id, dataset, "interactions", "loaded", loaded_interactions)

    logger.info(
        "Loaded counts — users: %d, items: %d, interactions: %d",
        loaded_users,
        loaded_items,
        loaded_interactions,
    )
    logger.info("ETL complete at %s", datetime.now(timezone.utc).isoformat())
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MovieLens ETL into PostgreSQL")
    parser.add_argument("--truncate", action="store_true", help="Truncate tables before load")
    parser.add_argument("--skip-download", action="store_true", help="Use existing raw data")
    args = parser.parse_args()
    run_etl(truncate=args.truncate, skip_download=args.skip_download)


if __name__ == "__main__":
    main()
