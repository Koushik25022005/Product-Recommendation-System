"""Train NCF with negative sampling and log runs to MLflow."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import mlflow
from mlflow import pytorch as mlflow_pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.config import load_model_config
from model.data import (
    build_eval_dict,
    make_dataloader,
    prepare_data,
)
from model.metrics import evaluate_model
from model.ncf import NCF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preference: str) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: NCF,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for users, items, labels in loader:
        users = users.to(device)
        items = items.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(users, items)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(total_batches, 1)


def save_model_artifact(model: NCF, data_meta: dict, mlp_layers: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "num_users": model.user_embedding.num_embeddings,
        "num_items": model.item_embedding.num_embeddings,
        "embedding_dim": model.user_embedding.embedding_dim,
        "mlp_layers": mlp_layers,
        "metadata": data_meta,
    }
    torch.save(payload, path)


def run_training(source: str = "auto", epochs_override: int | None = None) -> str:
    cfg = load_model_config()
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    mlflow_cfg = cfg["mlflow"]

    set_seed(train_cfg["seed"])
    device = resolve_device(train_cfg["device"])
    logger.info("Using device: %s", device)

    data, train_df, val_df, test_df = prepare_data(source=source)
    train_loader = make_dataloader(
        positives=data.interactions,
        num_items=data.num_items,
        train_items_by_user=data.train_items_by_user,
        batch_size=train_cfg["batch_size"],
        num_negatives=train_cfg["num_negatives"],
        seed=train_cfg["seed"],
        shuffle=True,
    )

    val_eval = build_eval_dict(val_df)
    test_eval = build_eval_dict(test_df)

    model = NCF(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=model_cfg["embedding_dim"],
        mlp_layers=model_cfg["mlp_layers"],
        dropout=model_cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()
    epochs = epochs_override or train_cfg["epochs"]
    k = train_cfg["eval_k"]

    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    data_meta = {
        "num_users": data.num_users,
        "num_items": data.num_items,
        "split_strategy": train_cfg["split_strategy"],
        "implicit_threshold": train_cfg["implicit_threshold"],
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "data_source": source,
    }

    with mlflow.start_run() as run:
        # Log hyperparameters (MLflow params must be strings)
        mlflow.log_params(
            {
                "model_name": model_cfg["name"],
                "model_embedding_dim": str(model_cfg["embedding_dim"]),
                "model_mlp_layers": json.dumps(model_cfg["mlp_layers"]),
                "model_dropout": str(model_cfg["dropout"]),
                "train_batch_size": str(train_cfg["batch_size"]),
                "train_epochs": str(epochs),
                "train_learning_rate": str(train_cfg["learning_rate"]),
                "train_weight_decay": str(train_cfg["weight_decay"]),
                "train_num_negatives": str(train_cfg["num_negatives"]),
                "train_implicit_threshold": str(train_cfg["implicit_threshold"]),
                "train_split_strategy": str(train_cfg["split_strategy"]),
                "train_eval_k": str(k),
                "train_seed": str(train_cfg["seed"]),
                **{f"data_{k}": str(v) for k, v in data_meta.items()},
            }
        )

        best_val_ndcg = -1.0
        best_state = None
        artifact_path = ARTIFACT_DIR / f"ncf_{run.info.run_id}.pt"

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            logger.info("Epoch %d/%d — train_loss=%.4f", epoch, epochs, train_loss)

            if epoch % train_cfg["eval_every"] == 0:
                val_metrics = evaluate_model(
                    model,
                    val_eval,
                    data.train_items_by_user,
                    data.num_items,
                    device,
                    k=k,
                )
                for name, value in val_metrics.as_dict(prefix="val_").items():
                    mlflow.log_metric(name, value, step=epoch)

                logger.info(
                    "Val @%d — NDCG=%.4f P=%.4f R=%.4f MAP=%.4f F1=%.4f (users=%d)",
                    k,
                    val_metrics.ndcg,
                    val_metrics.precision,
                    val_metrics.recall,
                    val_metrics.map_score,
                    val_metrics.f1,
                    val_metrics.num_users,
                )

                if val_metrics.ndcg > best_val_ndcg:
                    best_val_ndcg = val_metrics.ndcg
                    best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        test_metrics = evaluate_model(
            model,
            test_eval,
            data.train_items_by_user,
            data.num_items,
            device,
            k=k,
        )
        for name, value in test_metrics.as_dict(prefix="test_").items():
            mlflow.log_metric(name, value)

        logger.info(
            "Test @%d — NDCG=%.4f P=%.4f R=%.4f MAP=%.4f F1=%.4f (users=%d)",
            k,
            test_metrics.ndcg,
            test_metrics.precision,
            test_metrics.recall,
            test_metrics.map_score,
            test_metrics.f1,
            test_metrics.num_users,
        )
        logger.info(
            "Note: F1@k is computed on top-k recommendations; implicit feedback is "
            "highly imbalanced (~%d items, sparse positives per user).",
            data.num_items,
        )

        save_model_artifact(model, data_meta, model_cfg["mlp_layers"], artifact_path)
        save_model_artifact(model, data_meta, model_cfg["mlp_layers"], ARTIFACT_DIR / "latest_ncf.pt")
        mlflow.log_artifact(str(artifact_path), artifact_path="model")
        mlflow_pytorch.log_model(model, name="pytorch_model", serialization_format="pickle")

        # Save id maps for serving in later phases
        maps_path = ARTIFACT_DIR / f"id_maps_{run.info.run_id}.json"
        maps_path.parent.mkdir(parents=True, exist_ok=True)
        maps_data = json.dumps(
            {
                "user_id_map": {str(k): v for k, v in data.user_id_map.items()},
                "item_id_map": {str(k): v for k, v in data.item_id_map.items()},
            },
            indent=2,
        )
        maps_path.write_text(maps_data, encoding="utf-8")
        (ARTIFACT_DIR / "latest_id_maps.json").write_text(maps_data, encoding="utf-8")
        mlflow.log_artifact(str(maps_path), artifact_path="metadata")

        mlflow.log_metric("best_val_ndcg", best_val_ndcg)
        logger.info("MLflow run_id=%s", run.info.run_id)
        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NCF recommender with MLflow logging")
    parser.add_argument(
        "--source",
        choices=["auto", "postgres", "raw"],
        default="auto",
        help="Interaction data source",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    args = parser.parse_args()
    run_training(source=args.source, epochs_override=args.epochs)


if __name__ == "__main__":
    main()
