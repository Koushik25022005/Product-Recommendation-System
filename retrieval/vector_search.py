"""Vector retrieval functions for similarity search in Qdrant."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from qdrant_client.http import models as qmodels

from etl.utils import PROJECT_ROOT
from model.ncf import NCF
from retrieval.qdrant_indexer import get_qdrant_client

logger = logging.getLogger(__name__)

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


class VectorRecommender:
    """Loaded model + Qdrant client similarity retriever."""

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        ckpt = checkpoint_path or (ARTIFACT_DIR / "latest_ncf.pt")
        maps_file = ARTIFACT_DIR / "latest_id_maps.json"

        if not ckpt.exists() or not maps_file.exists():
            raise FileNotFoundError("Model checkpoint or ID maps not found. Run model training and vector indexing first.")

        payload = torch.load(ckpt, map_location="cpu")
        self.num_users = payload["num_users"]
        self.num_items = payload["num_items"]
        self.embedding_dim = payload["embedding_dim"]
        mlp_layers = payload["mlp_layers"]

        self.model = NCF(self.num_users, self.num_items, self.embedding_dim, mlp_layers)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

        self.user_embeddings = self.model.get_user_embeddings().numpy()
        self.item_embeddings = self.model.get_item_embeddings().numpy()

        maps = json.loads(maps_file.read_text(encoding="utf-8"))
        self.user_id_map: dict[str, int] = maps["user_id_map"]  # external -> idx
        self.item_id_map: dict[str, int] = maps["item_id_map"]

        self.idx_to_user = {v: int(k) for k, v in self.user_id_map.items()}
        self.idx_to_item = {v: int(k) for k, v in self.item_id_map.items()}

        self.qdrant_client, self.collection_name = get_qdrant_client()

    def get_user_vector(self, external_user_id: int) -> tuple[int, list[float]] | None:
        user_idx = self.user_id_map.get(str(external_user_id))
        if user_idx is None or user_idx >= self.num_users:
            return None
        vec = self.user_embeddings[user_idx].tolist()
        return user_idx, vec

    def search_by_vector(self, vector: list[float], top_k: int = 10) -> list[dict[str, Any]]:
        """Query Qdrant collection using query vector."""
        # Use query_points API which works across all qdrant-client versions
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
        ).points

        items = []
        for hit in results:
            payload = hit.payload or {}
            items.append({
                "item_idx": payload.get("item_idx"),
                "external_id": payload.get("external_id"),
                "title": payload.get("title", "Unknown"),
                "genres": payload.get("genres", []),
                "score": round(float(hit.score), 4),
            })
        return items

    def recommend_for_user(self, external_user_id: int, top_k: int = 10) -> list[dict[str, Any]] | None:
        """Get vector recommendations for a known user."""
        user_info = self.get_user_vector(external_user_id)
        if user_info is None:
            return None
        _, user_vec = user_info
        return self.search_by_vector(user_vec, top_k=top_k)


_RECOMMENDER: VectorRecommender | None = None


def get_recommender() -> VectorRecommender:
    global _RECOMMENDER
    if _RECOMMENDER is None:
        _RECOMMENDER = VectorRecommender()
    return _RECOMMENDER
