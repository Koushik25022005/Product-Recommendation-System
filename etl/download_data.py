"""Download MovieLens 100K dataset."""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests

from etl.utils import PROJECT_ROOT, load_services_config

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download_movielens_100k(force: bool = False) -> Path:
    cfg = load_services_config()
    raw_dir = PROJECT_ROOT / cfg["etl"]["raw_data_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    extract_dir = raw_dir / "ml-100k"
    marker = extract_dir / ".download_complete"

    if marker.exists() and not force:
        print(f"Dataset already present at {extract_dir}")
        return extract_dir

    zip_path = raw_dir / "ml-100k.zip"
    print(f"Downloading MovieLens 100K from {MOVIELENS_100K_URL} ...")
    response = requests.get(MOVIELENS_100K_URL, timeout=120)
    response.raise_for_status()
    zip_path.write_bytes(response.content)

    print(f"Extracting to {raw_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    marker.touch()
    print(f"Dataset ready at {extract_dir}")
    return extract_dir


if __name__ == "__main__":
    download_movielens_100k()
