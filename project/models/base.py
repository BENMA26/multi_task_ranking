from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEmbedding(nn.Module):
    """
    为每个特征字段维护独立的 Embedding Table。
    输入：x，shape [B, F]，每列对应一个字段的离散 ID
    输出：所有字段 Embedding 拼接后的向量，shape [B, F * embedding_dim]
    """

    def __init__(self, vocab_sizes: List[int], embedding_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim)
            for vocab_size in vocab_sizes
        ])

        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeds = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(embeds, dim=-1)


class MLP(nn.Module):
    """
    标准多层感知机。
    参数：input_dim, hidden_dims (list), output_dim, dropout, activation='relu'
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        if activation != "relu":
            raise ValueError("Only relu activation is supported in this implementation")

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h

        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim))
            prev_dim = output_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0.5
    if valid.any():
        return F.binary_cross_entropy_with_logits(logits[valid], target[valid])
    return torch.zeros((), device=logits.device, dtype=logits.dtype)
