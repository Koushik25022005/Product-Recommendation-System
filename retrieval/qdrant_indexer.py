"""Extract item embeddings from PyTorch NCF and populate Qdrant vector index."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from etl.utils import PROJECT_ROOT, load_services_config
from model.ncf import NCF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def load_item_metadata() -> pd.DataFrame:
    """Load item metadata (title, genres) from raw MovieLens files or DB."""
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
        return df[["item_id", "title", "genres"]]
    
    raise FileNotFoundError("u.item not found for loading metadata")


def get_qdrant_client() -> tuple[QdrantClient, str]:
    """Get QdrantClient instance (attempts server connection, falls back to local storage)."""
    cfg = load_services_config()
    qcfg = cfg.get("qdrant", {})
    host = str(qcfg.get("host", "localhost"))
    raw_port = qcfg.get("port", 6333)
    if "${" in host:
        host = "localhost"
    try:
        port = int(raw_port)
    except (ValueError, TypeError):
        port = 6333
    collection_name = str(qcfg.get("collection_name", "movie_embeddings"))

    try:
        client = QdrantClient(host=host, port=port, timeout=3)
        client.get_collections()
        logger.info("Connected to Qdrant server at %s:%s", host, port)
        return client, collection_name
    except Exception as exc:
        logger.warning("Could not connect to Qdrant server (%s). Falling back to local storage.", exc)
        storage_path = PROJECT_ROOT / "qdrant_storage"
        storage_path.mkdir(exist_ok=True)
        client = QdrantClient(path=str(storage_path))
        return client, collection_name


def index_item_embeddings(checkpoint_path: Path | None = None) -> int:
    """Load model embeddings and upsert into Qdrant collection."""
    ckpt = checkpoint_path or (ARTIFACT_DIR / "latest_ncf.pt")
    maps_file = ARTIFACT_DIR / "latest_id_maps.json"

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt}. Please run model training first.")
    if not maps_file.exists():
        raise FileNotFoundError(f"ID maps not found at {maps_file}.")

    payload = torch.load(ckpt, map_location="cpu")
    num_users = payload["num_users"]
    num_items = payload["num_items"]
    embedding_dim = payload["embedding_dim"]
    mlp_layers = payload["mlp_layers"]

    model = NCF(num_users, num_items, embedding_dim, mlp_layers)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    item_vectors = model.get_item_embeddings().numpy()

    maps = json.loads(maps_file.read_text(encoding="utf-8"))
    item_id_map: dict[str, int] = maps["item_id_map"]  # external_id (str) -> item_idx (int)
    idx_to_external = {v: int(k) for k, v in item_id_map.items()}

    meta_df = load_item_metadata().set_index("item_id")

    client, collection_name = get_qdrant_client()

    # Recreate collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=embedding_dim,
            distance=qmodels.Distance.COSINE,
        ),
    )

    points: list[qmodels.PointStruct] = []
    for item_idx in range(num_items):
        ext_id = idx_to_external.get(item_idx)
        vector = item_vectors[item_idx].tolist()

        title = "Unknown"
        genres = []
        if ext_id and ext_id in meta_df.index:
            row = meta_df.loc[ext_id]
            title = str(row["title"])
            genres = list(row["genres"])

        point_payload = {
            "item_idx": item_idx,
            "external_id": ext_id,
            "title": title,
            "genres": genres,
        }

        points.append(
            qmodels.PointStruct(
                id=item_idx,
                vector=vector,
                payload=point_payload,
            )
        )

    # Upsert in batches of 250
    batch_size = 250
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection_name,
            points=points[i : i + batch_size],
        )

    info = client.get_collection(collection_name)
    logger.info("Successfully indexed %d items into Qdrant collection '%s'", len(points), collection_name)
    return len(points)


def main() -> None:
    index_item_embeddings()


if __name__ == "__main__":
    main()
