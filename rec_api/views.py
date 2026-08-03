"""Django REST API views for recommendations, interactions, and items."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from etl.utils import PROJECT_ROOT, load_services_config, get_engine

logger = logging.getLogger(__name__)

# Global dataset cache for popularity fallbacks and item lookups
_ITEM_META_CACHE: pd.DataFrame | None = None
_POPULAR_ITEMS_CACHE: list[dict[str, Any]] | None = None
_NEW_INTERACTIONS: list[dict[str, Any]] = []


def _get_item_metadata() -> pd.DataFrame:
    global _ITEM_META_CACHE
    if _ITEM_META_CACHE is not None:
        return _ITEM_META_CACHE

    cfg = load_services_config()
    raw_dir = PROJECT_ROOT / cfg["etl"]["raw_data_dir"] / "ml-100k"
    item_file = raw_dir / "u.item"

    if item_file.exists():
        cols = [
            "item_id", "title", "release_date", "video_release_date", "imdb_url",
            "unknown", "action", "adventure", "animation", "children", "comedy",
            "crime", "documentary", "drama", "fantasy", "film_noir", "horror",
            "musical", "mystery", "romance", "sci_fi", "thriller", "war", "western",
        ]
        df = pd.read_csv(item_file, sep="|", names=cols, encoding="latin-1")
        genre_cols = cols[5:]
        df["genres"] = df[genre_cols].apply(
            lambda row: [col.replace("_", " ").title() for col in genre_cols if row[col] == 1],
            axis=1
        )
        _ITEM_META_CACHE = df[["item_id", "title", "genres"]].set_index("item_id")
        return _ITEM_META_CACHE
    
    _ITEM_META_CACHE = pd.DataFrame(columns=["title", "genres"])
    return _ITEM_META_CACHE


def _get_popular_items(top_k: int = 10) -> list[dict[str, Any]]:
    global _POPULAR_ITEMS_CACHE
    if _POPULAR_ITEMS_CACHE is not None:
        return _POPULAR_ITEMS_CACHE[:top_k]

    cfg = load_services_config()
    raw_dir = PROJECT_ROOT / cfg["etl"]["raw_data_dir"] / "ml-100k"
    data_file = raw_dir / "u.data"
    meta = _get_item_metadata()

    if data_file.exists():
        data = pd.read_csv(data_file, sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
        stats = data.groupby("item_id").agg(
            rating_count=("rating", "count"),
            avg_rating=("rating", "mean")
        ).reset_index()

        # Popularity score combines rating count and average rating
        stats["popularity_score"] = stats["rating_count"] * stats["avg_rating"]
        stats = stats.sort_values("popularity_score", ascending=False)

        items = []
        for row in stats.head(50).itertuples():
            iid = int(row.item_id)
            title = "Unknown"
            genres = []
            if iid in meta.index:
                title = str(meta.loc[iid, "title"])
                genres = list(meta.loc[iid, "genres"])

            items.append({
                "item_id": iid,
                "external_id": iid,
                "title": title,
                "genres": genres,
                "rating_count": int(row.rating_count),
                "avg_rating": round(float(row.avg_rating), 2),
                "score": round(float(row.popularity_score), 1),
            })
        _POPULAR_ITEMS_CACHE = items
        return _POPULAR_ITEMS_CACHE[:top_k]

    return []


def index_page(request: HttpRequest):
    """Render single-page frontend application."""
    return render(request, "index.html")


@api_view(["GET"])
def get_recommendations(request: HttpRequest, user_id: int):
    """GET /recommendations/<user_id> — Vector retrieval with cold-start fallback."""
    top_k = int(request.GET.get("top_k", 10))

    try:
        from retrieval.vector_search import get_recommender
        recommender = get_recommender()
        recs = recommender.recommend_for_user(external_user_id=user_id, top_k=top_k)

        if recs is not None and len(recs) > 0:
            return Response({
                "user_id": user_id,
                "is_cold_start": False,
                "recommendation_type": "vector_ncf_qdrant",
                "count": len(recs),
                "recommendations": recs,
            })
    except Exception as exc:
        logger.warning("Vector retrieval unvailable for user %s (%s). Falling back to popularity.", user_id, exc)

    # Cold start fallback for unknown users or unindexed users
    popular = _get_popular_items(top_k=top_k)
    return Response({
        "user_id": user_id,
        "is_cold_start": True,
        "recommendation_type": "popularity_fallback",
        "count": len(popular),
        "recommendations": popular,
        "message": f"User #{user_id} has no existing interactions. Serving top popular items.",
    })


@api_view(["POST"])
def log_interaction(request: HttpRequest):
    """POST /interactions — Log new click or rating signal."""
    data = request.data
    user_id = data.get("user_id")
    item_id = data.get("item_id")
    rating = data.get("rating", 5.0)
    signal_type = data.get("signal_type", "explicit")

    if user_id is None or item_id is None:
        return Response(
            {"error": "Both user_id and item_id are required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    record = {
        "user_id": int(user_id),
        "item_id": int(item_id),
        "rating": float(rating),
        "signal_type": signal_type,
        "interacted_at": pd.Timestamp.now().isoformat(),
    }
    _NEW_INTERACTIONS.append(record)

    # Optionally log to Postgres if engine is alive
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(
                "INSERT INTO interactions (user_id, item_id, rating, signal_type) VALUES (:user_id, :item_id, :rating, :signal_type)",
                {"user_id": int(user_id), "item_id": int(item_id), "rating": float(rating), "signal_type": signal_type}
            )
            conn.commit()
    except Exception:
        pass  # Memory logging succeeded fallback

    return Response({
        "status": "success",
        "message": f"Interaction logged for user {user_id} on item {item_id}",
        "record": record,
        "total_new_interactions": len(_NEW_INTERACTIONS),
    })


@api_view(["GET"])
def get_item_detail(request: HttpRequest, item_id: int):
    """GET /items/<id> — Get detailed metadata for an item."""
    meta = _get_item_metadata()
    if item_id in meta.index:
        row = meta.loc[item_id]
        return Response({
            "item_id": item_id,
            "title": str(row["title"]),
            "genres": list(row["genres"]),
        })

    return Response(
        {"error": f"Item #{item_id} not found"},
        status=status.HTTP_404_NOT_FOUND,
    )


@api_view(["GET"])
def list_items(request: HttpRequest):
    """GET /items — List or search items."""
    query = request.GET.get("search", "").lower()
    limit = int(request.GET.get("limit", 20))
    meta = _get_item_metadata()

    items = []
    for iid, row in meta.iterrows():
        title = str(row["title"])
        genres = list(row["genres"])
        if not query or query in title.lower() or any(query in g.lower() for g in genres):
            items.append({
                "item_id": int(iid),
                "title": title,
                "genres": genres,
            })
        if len(items) >= limit:
            break

    return Response({
        "count": len(items),
        "items": items,
    })
