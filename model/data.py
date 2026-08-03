"""Data loading, temporal/user splits, and negative sampling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sqlalchemy import text
from torch.utils.data import Dataset

from etl.utils import PROJECT_ROOT, get_engine, load_services_config
from model.config import load_model_config

logger = logging.getLogger(__name__)


@dataclass
class InteractionData:
    interactions: pd.DataFrame  # columns: user_idx, item_idx, rating, interacted_at, label
    num_users: int
    num_items: int
    user_id_map: dict[int, int]  # external/db id -> idx
    item_id_map: dict[int, int]
    train_items_by_user: dict[int, set[int]]


def _load_from_postgres() -> pd.DataFrame:
    engine = get_engine()
    query = text(
        """
        SELECT u.id AS user_id, i.id AS item_id, x.rating, x.interacted_at
        FROM interactions x
        JOIN users u ON x.user_id = u.id
        JOIN items i ON x.item_id = i.id
        ORDER BY x.interacted_at
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    if df.empty:
        raise ValueError("PostgreSQL interactions table is empty")
    logger.info("Loaded %d interactions from PostgreSQL", len(df))
    return df


def _load_from_raw() -> pd.DataFrame:
    cfg = load_services_config()
    data_path = PROJECT_ROOT / cfg["etl"]["raw_data_dir"] / "ml-100k" / "u.data"
    if not data_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {data_path}")

    df = pd.read_csv(
        data_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    df["interacted_at"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    logger.info("Loaded %d interactions from raw MovieLens files", len(df))
    return df


def load_interactions(source: str = "auto") -> pd.DataFrame:
    if source == "postgres":
        return _load_from_postgres()
    if source == "raw":
        return _load_from_raw()

    try:
        return _load_from_postgres()
    except Exception as exc:
        logger.warning("PostgreSQL unavailable (%s), falling back to raw files", exc)
        return _load_from_raw()


def _build_index_maps(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int], dict[int, int], int, int]:
    user_ids = sorted(df["user_id"].unique())
    item_ids = sorted(df["item_id"].unique())
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {iid: idx for idx, iid in enumerate(item_ids)}

    out = df.copy()
    out["user_idx"] = out["user_id"].map(user_map)
    out["item_idx"] = out["item_id"].map(item_map)
    return out, user_map, item_map, len(user_ids), len(item_ids)


def _apply_implicit_label(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    out = df.copy()
    out["label"] = (out["rating"] >= threshold).astype(np.int8)
    return out


def split_by_time(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("interacted_at").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def split_by_user(
    df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []

    for _, group in df.groupby("user_idx"):
        group = group.sort_values("interacted_at")
        n = len(group)
        if n < 3:
            train_parts.append(group)
            continue
        indices = np.arange(n)
        rng.shuffle(indices)
        train_end = max(1, int(n * train_ratio))
        val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
        val_end = min(val_end, n - 1)
        ordered = group.iloc[sorted(indices)]
        train_parts.append(ordered.iloc[:train_end])
        val_parts.append(ordered.iloc[train_end:val_end])
        test_parts.append(ordered.iloc[val_end:])

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=df.columns)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)
    return train, val, test


def _safe_int(value: object) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def build_train_items_by_user(train_df: pd.DataFrame) -> dict[int, set[int]]:
    grouped: dict[int, set[int]] = {}
    for row in train_df.itertuples(index=False):
        grouped.setdefault(_safe_int(row.user_idx), set()).add(_safe_int(row.item_idx))
    return grouped


def prepare_data(source: str = "auto") -> tuple[InteractionData, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = load_model_config()["training"]
    raw = load_interactions(source=source)
    indexed, user_map, item_map, num_users, num_items = _build_index_maps(raw)
    labeled = _apply_implicit_label(indexed, cfg["implicit_threshold"])

    if cfg["split_strategy"] == "user":
        train_df, val_df, test_df = split_by_user(
            labeled, cfg["train_ratio"], cfg["val_ratio"], cfg["seed"]
        )
    else:
        train_df, val_df, test_df = split_by_time(labeled, cfg["train_ratio"], cfg["val_ratio"])

    train_positives = train_df[train_df["label"] == 1][["user_idx", "item_idx", "label"]].copy()
    train_items_by_user = build_train_items_by_user(train_df)

    data = InteractionData(
        interactions=train_positives,
        num_users=num_users,
        num_items=num_items,
        user_id_map=user_map,
        item_id_map=item_map,
        train_items_by_user=train_items_by_user,
    )
    logger.info(
        "Split (%s) — train: %d, val: %d, test: %d | users: %d, items: %d, positives: %d",
        cfg["split_strategy"],
        len(train_df),
        len(val_df),
        len(test_df),
        num_users,
        num_items,
        len(train_positives),
    )
    return data, train_df, val_df, test_df


class ImplicitBCELossDataset(Dataset):
    """Positive interactions with on-the-fly negative sampling."""

    def __init__(
        self,
        positives: pd.DataFrame,
        num_items: int,
        train_items_by_user: dict[int, set[int]],
        num_negatives: int = 4,
        seed: int = 42,
    ) -> None:
        self.positives = positives.reset_index(drop=True)
        self.num_items = num_items
        self.train_items_by_user = train_items_by_user
        self.num_negatives = num_negatives
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.positives) * (1 + self.num_negatives)

    def _sample_negative(self, user_idx: int) -> int:
        seen = self.train_items_by_user.get(user_idx, set())
        for _ in range(50):
            candidate = int(self.rng.integers(0, self.num_items))
            if candidate not in seen:
                return candidate
        # fallback: random item (may collide for dense users)
        return int(self.rng.integers(0, self.num_items))

    def __getitem__(self, index: int) -> tuple[int, int, float]:
        pos_index = index // (1 + self.num_negatives)
        slot = index % (1 + self.num_negatives)
        row = self.positives.iloc[pos_index]
        user_idx = _safe_int(row.user_idx)

        if slot == 0:
            return user_idx, _safe_int(row.item_idx), 1.0
        return user_idx, self._sample_negative(user_idx), 0.0


def make_dataloader(
    positives: pd.DataFrame,
    num_items: int,
    train_items_by_user: dict[int, set[int]],
    batch_size: int,
    num_negatives: int,
    seed: int,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    dataset = ImplicitBCELossDataset(
        positives=positives,
        num_items=num_items,
        train_items_by_user=train_items_by_user,
        num_negatives=num_negatives,
        seed=seed,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def build_eval_dict(df: pd.DataFrame) -> dict[int, set[int]]:
    """User -> relevant item indices (positives only) for a split."""
    eval_dict: dict[int, set[int]] = {}
    positives = df[df["label"] == 1]
    for row in positives.itertuples(index=False):
        eval_dict.setdefault(_safe_int(row.user_idx), set()).add(_safe_int(row.item_idx))
    return eval_dict
