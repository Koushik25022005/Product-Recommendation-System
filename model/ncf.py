"""Neural Collaborative Filtering (NCF) model."""

from __future__ import annotations

import torch
import torch.nn as nn


class NCF(nn.Module):
    """User/item embedding towers fused via MLP for interaction prediction."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        mlp_layers: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        mlp_layers = mlp_layers or [128, 64, 32]

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        layers: list[nn.Module] = []
        in_dim = embedding_dim * 2
        for hidden in mlp_layers:
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        x = torch.cat([user_vec, item_vec], dim=-1)
        return self.mlp(x).squeeze(-1)

    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(user_ids, item_ids))

    def get_user_embeddings(self) -> torch.Tensor:
        return self.user_embedding.weight.detach()

    def get_item_embeddings(self) -> torch.Tensor:
        return self.item_embedding.weight.detach()
