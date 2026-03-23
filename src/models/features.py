"""
特征工程
支持 Sparse Feature, Dense Feature, Sequence Feature
"""
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class SparseFeature:
    """稀疏特征（类别特征）"""
    name: str
    vocab_size: int
    embed_dim: int
    shared_embed: Optional[str] = None  # 共享嵌入的特征名

@dataclass
class DenseFeature:
    """稠密特征（数值特征）"""
    name: str
    dim: int = 1  # 特征维度

@dataclass
class SequenceFeature:
    """序列特征（用户行为序列）"""
    name: str
    vocab_size: int
    embed_dim: int
    max_len: int
    pooling: str = 'mean'  # 'mean', 'sum', 'max', 'attention'
    shared_embed: Optional[str] = None


class FeatureEmbedding(nn.Module):
    """特征嵌入层"""

    def __init__(
        self,
        sparse_features: List[SparseFeature],
        dense_features: List[DenseFeature],
        sequence_features: List[SequenceFeature],
    ):
        super().__init__()
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.sequence_features = sequence_features

        # 稀疏特征嵌入
        self.sparse_embeds = nn.ModuleDict()
        shared_embeds = {}  # 共享嵌入字典

        for feat in sparse_features:
            if feat.shared_embed and feat.shared_embed in shared_embeds:
                # 使用共享嵌入
                self.sparse_embeds[feat.name] = shared_embeds[feat.shared_embed]
            else:
                embed = nn.Embedding(feat.vocab_size, feat.embed_dim)
                self.sparse_embeds[feat.name] = embed
                if feat.shared_embed:
                    shared_embeds[feat.shared_embed] = embed

        # 序列特征嵌入
        self.sequence_embeds = nn.ModuleDict()
        for feat in sequence_features:
            if feat.shared_embed and feat.shared_embed in shared_embeds:
                self.sequence_embeds[feat.name] = shared_embeds[feat.shared_embed]
            else:
                embed = nn.Embedding(feat.vocab_size, feat.embed_dim)
                self.sequence_embeds[feat.name] = embed
                if feat.shared_embed:
                    shared_embeds[feat.shared_embed] = embed

        # 稠密特征归一化
        self.dense_norms = nn.ModuleDict({
            feat.name: nn.BatchNorm1d(feat.dim)
            for feat in dense_features
        })

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入特征字典
        Returns:
            嵌入后的特征字典
        """
        embeddings = {}

        # 稀疏特征嵌入
        for feat in self.sparse_features:
            embeddings[feat.name] = self.sparse_embeds[feat.name](x[feat.name].long())

        # 稠密特征归一化
        for feat in self.dense_features:
            feat_value = x[feat.name]
            if feat_value.dim() == 1:
                feat_value = feat_value.unsqueeze(1)
            embeddings[feat.name] = self.dense_norms[feat.name](feat_value)

        # 序列特征嵌入和池化
        for feat in self.sequence_features:
            seq_embed = self.sequence_embeds[feat.name](x[feat.name])  # (batch, seq_len, embed_dim)

            # 处理 padding mask
            mask_key = f"{feat.name}_mask"
            if mask_key in x:
                mask = x[mask_key].unsqueeze(-1)  # (batch, seq_len, 1)
                seq_embed = seq_embed * mask

            # 池化
            if feat.pooling == 'mean':
                if mask_key in x:
                    seq_len = x[mask_key].sum(dim=1, keepdim=True).clamp(min=1)
                    pooled = seq_embed.sum(dim=1) / seq_len
                else:
                    pooled = seq_embed.mean(dim=1)
            elif feat.pooling == 'sum':
                pooled = seq_embed.sum(dim=1)
            elif feat.pooling == 'max':
                pooled = seq_embed.max(dim=1)[0]
            else:
                pooled = seq_embed.mean(dim=1)

            embeddings[feat.name] = pooled

        return embeddings

    def get_embedding_dim(self) -> int:
        """获取总嵌入维度"""
        total_dim = 0
        for feat in self.sparse_features:
            total_dim += feat.embed_dim
        for feat in self.dense_features:
            total_dim += feat.dim
        for feat in self.sequence_features:
            total_dim += feat.embed_dim
        return total_dim


class FeatureEncoder(nn.Module):
    """特征编码器"""

    def __init__(
        self,
        sparse_features: List[SparseFeature],
        dense_features: List[DenseFeature],
        sequence_features: List[SequenceFeature],
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_embedding = FeatureEmbedding(
            sparse_features, dense_features, sequence_features
        )

        # MLP
        input_dim = self.feature_embedding.get_embedding_dim()
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入特征字典
        Returns:
            编码后的向量 (batch_size, hidden_dim)
        """
        # 特征嵌入
        embeddings = self.feature_embedding(x)

        # 拼接所有特征
        feat_list = []
        for feat_name, feat_embed in embeddings.items():
            if feat_embed.dim() == 2:
                feat_list.append(feat_embed)
            else:
                feat_list.append(feat_embed.flatten(1))

        x = torch.cat(feat_list, dim=1)

        # MLP 编码
        x = self.mlp(x)

        return x
