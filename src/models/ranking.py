"""
排序模型
包含 Share Bottom, MOE, MMOE, PLE
用于 CTR 和 CVR 多任务学习

特征输入统一为：
    sparse_feature_names : List[str]       — 稀疏特征名称列表（走 Embedding）
    sparse_feature_dims  : Dict[str, int]  — 稀疏特征词表大小字典
    dense_feature_names  : List[str]       — 稠密特征名称列表（直接拼接）

TorchJD 梯度聚合通过以下两个参数统一控制：
    use_torchjd       : bool — 是否启用 TorchJD 梯度聚合
    aggregation_method: str  — 聚合算法，可选 upgrad / mgda / pcgrad / graddrop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List
from torchjd.aggregation import UPGrad, MGDA, PCGrad, GradDrop
from torchjd.autojac import backward
from torchmetrics import AUROC

def _create_aggregator(method: str):
    """根据名称创建 TorchJD 梯度聚合器"""
    if method == "upgrad":
        return UPGrad()
    elif method == "mgda":
        return MGDA()
    elif method == "pcgrad":
        return PCGrad()
    elif method == "graddrop":
        return GradDrop(leak=0.0)
    else:
        raise ValueError(f"未知的聚合方法: '{method}'，可选: upgrad / mgda / pcgrad / graddrop")

class DCNv2CrossLayer(nn.Module):
    """
    DCN-v2 交叉层（Cross Layer），支持全秩和低秩两种模式

    论文：DCN V2: Improved Deep & Cross Network and Practical Lessons for
          Web-scale Learning to Rank Systems (Wang et al., 2021)

    全秩：
        x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l      参数量 O(d^2)
    低秩：
        x_{l+1} = x_0 ⊙ (U_l(V_l · x_l) + b_l) + x_l 参数量 O(d * r)

    Args:
        input_dim : 输入/输出维度（交叉层不改变维度）
        num_layers: 交叉层数量
        dropout   : 每层后的 Dropout
        rank      : 低秩分解维度，0 表示全秩
    """

    def __init__(self, input_dim: int, num_layers: int = 2, dropout: float = 0.0, rank: int = 0):
        super().__init__()
        self.num_layers = num_layers
        self.rank = rank

        if rank > 0:
            self.U = nn.ParameterList([
                nn.Parameter(torch.empty(input_dim, rank))
                for _ in range(num_layers)
            ])
            self.V = nn.ParameterList([
                nn.Parameter(torch.empty(rank, input_dim))
                for _ in range(num_layers)
            ])
            for u, v in zip(self.U, self.V):
                nn.init.xavier_normal_(u)
                nn.init.xavier_normal_(v)
        else:
            self.cross_weights = nn.ParameterList([
                nn.Parameter(torch.empty(input_dim, input_dim))
                for _ in range(num_layers)
            ])
            for w in self.cross_weights:
                nn.init.xavier_normal_(w)

        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(input_dim)
            for _ in range(num_layers)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim]
        Returns:
            out: [B, input_dim]  — 与输入同维度
        """
        x0 = x
        xl = x
        for i in range(self.num_layers):
            if self.rank > 0:
                xl = x0 * (xl @ self.V[i].T @ self.U[i].T + self.cross_biases[i]) + xl
            else:
                xl = x0 * (xl @ self.cross_weights[i] + self.cross_biases[i]) + xl
            xl = self.layer_norms[i](xl)
            xl = self.dropouts[i](xl)
        return xl


class Expert(nn.Module):
    """
    专家网络：支持标准 MLP 或 DCN-v2 并行结构

    并行结构（use_dcn=True）：
        x -> CrossNet -> cross_out  \\
        x -> MLP      -> mlp_out    +-> concat -> proj -> output
    """

    def __init__(
        self,
        input_dim  : int,
        hidden_dims: List[int],
        dropout    : float = 0.2,
        use_dcn    : bool  = False,
        dcn_num_layers      : int       = 2,
        dcn_dropout         : float     = 0.0,
        dcn_rank      : int   = 0,
    ):
        super().__init__()
        self.use_dcn = use_dcn
        self.mlp = self._build_mlp(input_dim, hidden_dims, dropout)

        if use_dcn:
            self.dcn = DCNv2CrossLayer(
                input_dim,
                num_layers=dcn_num_layers,
                dropout=dcn_dropout,
                rank=dcn_rank,
            )
            self.proj = nn.Sequential(
                nn.Linear(input_dim + hidden_dims[-1], hidden_dims[-1]),
                nn.BatchNorm1d(hidden_dims[-1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

    def _build_mlp(self, input_dim: int, hidden_dims: List[int], dropout: float):
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.use_dcn:
            cross_out = self.dcn(x)
            mlp_out = self.mlp(x)
            combined = torch.cat([cross_out, mlp_out], dim=-1)
            return self.proj(combined)
        return self.mlp(x)

class Gate(nn.Module):
    """
    门控网络

    支持计算门控权重的 entropy，用于正则化防止极化问题。
    极化问题：门控网络倾向于只选择少数几个专家，导致其他专家未被充分利用。
    Entropy 正则化：鼓励门控权重分布更均匀，充分利用所有专家。
    """

    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1),
        )

    def forward(self, x, return_entropy: bool = False):
        """
        Args:
            x: [B, input_dim]
            return_entropy: 是否返回 entropy（用于正则化）
        Returns:
            weights: [B, num_experts] 门控权重
            entropy: [B] 每个样本的 entropy（可选）
        """
        weights = self.gate(x)  # [B, num_experts]

        if return_entropy:
            # 计算 entropy: H = -Σ p_i * log(p_i)
            # 添加 eps 防止 log(0)
            eps = 1e-8
            entropy = -(weights * torch.log(weights + eps)).sum(dim=1)  # [B]
            return weights, entropy

        return weights

class Tower(nn.Module):
    """任务塔"""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(dims[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class _MultiTaskMixin:
    """
    为四个模型提供统一的：
      - 稀疏 + 稠密特征编码（_build_feature_layers / _encode_features）
      - TorchJD 初始化
      - training_step 分发（标准加权 / TorchJD）
      - validation / test AUC 计算
      - configure_optimizers

    子类须在 __init__ 中：
      1. 调用 _build_feature_layers(...) 注册 embeddings，并获取 input_dim
      2. 完成各自网络构建
      3. 调用 _init_torchjd(...)
      4. 调用 _init_metrics()

    子类须实现 _forward_logits(x) -> (ctr_logit, cvr_logit)
        其中 x 已是编码后的张量 [B, input_dim]，无需再处理特征。
    """

    # ------------------------------------------------------------------ #
    # 特征层构建与编码                                                      #
    # ------------------------------------------------------------------ #

    def _build_feature_layers(
        self,
        sparse_feature_names: List[str],
        sparse_feature_dims : Dict[str, int],
        dense_feature_names : List[str],
        embedding_dim       : int,
    ) -> int:
        """
        注册 Embedding 层并返回最终 input_dim。
        必须在 __init__ 中、网络构建之前调用。
        """
        self.sparse_feature_names = sparse_feature_names
        self.dense_feature_names  = dense_feature_names

        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(sparse_feature_dims[feat], embedding_dim)
            for feat in sparse_feature_names
        })

        raw_dim = len(sparse_feature_names) * embedding_dim + len(dense_feature_names)
        return raw_dim

    def _encode_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """将原始 inputs 编码为单一张量 [B, input_dim]"""
        parts = []

        # 稀疏特征 → Embedding
        for feat in self.sparse_feature_names:
            parts.append(self.embeddings[feat](inputs[feat]))  # [B, emb_dim]

        # 稠密特征 → 直接使用，统一 shape 为 [B, 1] 或 [B, d]
        for feat in self.dense_feature_names:
            t = inputs[feat]
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            parts.append(t)

        return torch.cat(parts, dim=-1)  # [B, raw_dim]

    # ------------------------------------------------------------------ #
    # TorchJD & 指标初始化                                                  #
    # ------------------------------------------------------------------ #

    def _init_torchjd(
        self,
        use_torchjd       : bool,
        aggregation_method: str,
        ctr_weight        : float,
        cvr_weight        : float,
        esmm              : bool = False,
        sigmoid           : int = 1,
        use_entropy_reg   : bool = False,
        lambda_entropy    : float = 0.01,
        use_ema           : bool = False,
        ema_decay         : float = 0.999,
    ):
        """
        初始化 TorchJD 梯度聚合和 Entropy 正则化参数

        Args:
            use_entropy_reg: 是否使用 Entropy 正则化防止门控极化
            lambda_entropy: Entropy 正则化权重
            use_ema: 是否启用 EMA 参数更新
            ema_decay: EMA 衰减系数（越接近 1 越平滑）
        """
        self.use_torchjd = use_torchjd
        self.ctr_weight  = ctr_weight
        self.cvr_weight  = cvr_weight
        self.esmm        = esmm
        self.sigmoid     = sigmoid
        self.use_entropy_reg = use_entropy_reg
        self.lambda_entropy  = lambda_entropy
        self.use_ema    = use_ema
        self.ema_decay  = ema_decay
        if use_torchjd:
            self.automatic_optimization = False
            self.aggregator = _create_aggregator(aggregation_method)

    def _init_metrics(self):
        self.val_ctr_auc  = AUROC(task="binary")
        self.val_cvr_auc  = AUROC(task="binary")
        self.test_ctr_auc = AUROC(task="binary")
        self.test_cvr_auc = AUROC(task="binary")
        # EMA 专用指标
        self.val_ema_ctr_auc  = AUROC(task="binary")
        self.val_ema_cvr_auc  = AUROC(task="binary")
        self.test_ema_ctr_auc = AUROC(task="binary")
        self.test_ema_cvr_auc = AUROC(task="binary")

    # ------------------------------------------------------------------ #
    # EMA                                                                 #
    # ------------------------------------------------------------------ #

    def on_fit_start(self):
        """训练开始时初始化 EMA 参数副本"""
        if self.use_ema:
            self._ema_params = {
                name: param.data.clone()
                for name, param in self.named_parameters()
            }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """每个训练 batch 结束后更新 EMA 参数"""
        if not self.use_ema:
            return
        decay = self.ema_decay
        for name, param in self.named_parameters():
            self._ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    def _forward_with_ema(self, inputs: Dict[str, torch.Tensor], return_gate_entropy: bool = False):
        """用 EMA 参数做前向推理（临时替换参数，推理后还原）"""
        original = {name: param.data for name, param in self.named_parameters()}
        for name, param in self.named_parameters():
            param.data = self._ema_params[name].to(param.device)
        try:
            result = self.forward(inputs, return_gate_entropy=return_gate_entropy)
        finally:
            for name, param in self.named_parameters():
                param.data = original[name]
        return result

    # ------------------------------------------------------------------ #
    # forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, inputs: Dict[str, torch.Tensor], return_gate_entropy: bool = False):
        x = self._encode_features(inputs)
        result = self._forward_logits(x, return_gate_entropy=return_gate_entropy)
        if return_gate_entropy:
            ctr_logit, cvr_logit, gate_entropy = result
            return ctr_logit.squeeze(-1), cvr_logit.squeeze(-1), gate_entropy
        else:
            ctr_logit, cvr_logit = result
            return ctr_logit.squeeze(-1), cvr_logit.squeeze(-1)

    # ------------------------------------------------------------------ #
    # training_step 分发                                                   #
    # ------------------------------------------------------------------ #

    def training_step(self, batch, batch_idx):
        if self.use_torchjd:
            return self._training_step_torchjd(batch)
        return self._training_step_standard(batch)

    def _compute_losses(self, batch):
        inputs, labels = batch
        ctr_labels = labels["click"]
        cvr_labels = labels["purchase"]

        # 如果启用 entropy 正则化，需要获取 gate entropy
        if self.use_entropy_reg:
            ctr_logits, cvr_logits, gate_entropy = self(inputs, return_gate_entropy=True)
        else:
            ctr_logits, cvr_logits = self(inputs)
            gate_entropy = None

        pCTR = torch.sigmoid(ctr_logits)
        if self.sigmoid == 1:
            pCVR = torch.sigmoid(cvr_logits)
        else:
            pCVR = torch.sigmoid(cvr_logits) * torch.sigmoid(cvr_logits)

        ctr_loss = nn.functional.binary_cross_entropy_with_logits(
            ctr_logits, ctr_labels.float()
        )

        if self.esmm:
            # ESMM：用 pCTR × pCVR 在全量曝光空间监督购买标签
            # CVR Tower 通过 pCTCVR 的梯度间接学习，无显式 L_cvr
            pCTCVR   = pCTR * pCVR
            cvr_loss = nn.functional.binary_cross_entropy(
                pCTCVR, cvr_labels.float()
            )
        else:
            # 标准：直接在点击样本上监督 CVR
            cvr_loss = nn.functional.binary_cross_entropy_with_logits(
                cvr_logits, cvr_labels.float()
            )

        # Entropy 正则化 loss
        # 目标：最大化 entropy，即最小化 -entropy
        # 因此 entropy_loss = -mean(entropy)
        if self.use_entropy_reg and gate_entropy is not None:
            entropy_loss = -gate_entropy.mean()  # 负号：最大化 entropy = 最小化 -entropy
        else:
            entropy_loss = torch.tensor(0.0, device=ctr_logits.device)

        return ctr_logits, cvr_logits, ctr_labels, cvr_labels, ctr_loss, cvr_loss, entropy_loss

    def _training_step_standard(self, batch):
        _, _, _, _, ctr_loss, cvr_loss, entropy_loss = self._compute_losses(batch)
        loss = self.ctr_weight * ctr_loss + self.cvr_weight * cvr_loss + self.lambda_entropy * entropy_loss
        self.log("train_loss",         loss,         prog_bar=True)
        self.log("train_ctr_loss",     ctr_loss)
        self.log("train_cvr_loss",     cvr_loss)
        self.log("train_entropy_loss", entropy_loss)
        return loss

    def _training_step_torchjd(self, batch):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        _, _, _, _, ctr_loss, cvr_loss, entropy_loss = self._compute_losses(batch)
        backward([ctr_loss, cvr_loss], aggregator=self.aggregator)
        optimizer.step()
        total_loss = ctr_loss + cvr_loss + self.lambda_entropy * entropy_loss
        self.log("train_loss",         total_loss,   prog_bar=True)
        self.log("train_ctr_loss",     ctr_loss)
        self.log("train_cvr_loss",     cvr_loss)
        self.log("train_entropy_loss", entropy_loss)
        return total_loss

    # ------------------------------------------------------------------ #
    # validation / test                                                   #
    # ------------------------------------------------------------------ #

    def validation_step(self, batch, batch_idx):
        ctr_logits, cvr_logits, ctr_labels, cvr_labels, ctr_loss, cvr_loss = \
            self._compute_losses(batch)
        loss = self.ctr_weight * ctr_loss + self.cvr_weight * cvr_loss
        self.val_ctr_auc.update(torch.sigmoid(ctr_logits), ctr_labels.long())
        self.val_cvr_auc.update(torch.sigmoid(cvr_logits), cvr_labels.long())
        self.log("val_loss",     loss,     prog_bar=True, sync_dist=True)
        self.log("val_ctr_loss", ctr_loss, sync_dist=True)
        self.log("val_cvr_loss", cvr_loss, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        ctr_auc = self.val_ctr_auc.compute()
        cvr_auc = self.val_cvr_auc.compute()
        self.log("val_ctr_auc",      ctr_auc,           prog_bar=True, sync_dist=True)
        self.log("val_cvr_auc",      cvr_auc,           prog_bar=True, sync_dist=True)
        self.log("val_combined_auc", ctr_auc * cvr_auc, prog_bar=True, sync_dist=True)
        self.val_ctr_auc.reset()
        self.val_cvr_auc.reset()

    def test_step(self, batch, batch_idx):
        ctr_logits, cvr_logits, ctr_labels, cvr_labels, ctr_loss, cvr_loss = \
            self._compute_losses(batch)
        loss = self.ctr_weight * ctr_loss + self.cvr_weight * cvr_loss
        self.test_ctr_auc.update(torch.sigmoid(ctr_logits), ctr_labels.long())
        self.test_cvr_auc.update(torch.sigmoid(cvr_logits), cvr_labels.long())
        self.log("test_loss",     loss,     sync_dist=True)
        self.log("test_ctr_loss", ctr_loss, sync_dist=True)
        self.log("test_cvr_loss", cvr_loss, sync_dist=True)
        return loss

    def on_test_epoch_end(self):
        test_ctr_auc = self.test_ctr_auc.compute()
        test_cvr_auc = self.test_cvr_auc.compute()
        self.log("test_ctr_auc",      test_ctr_auc,                    prog_bar=True, sync_dist=True)
        self.log("test_cvr_auc",      test_cvr_auc,                    prog_bar=True, sync_dist=True)
        self.log("test_combined_auc", test_ctr_auc * test_cvr_auc,     prog_bar=True, sync_dist=True)
        self.test_ctr_auc.reset()
        self.test_cvr_auc.reset()

    # ------------------------------------------------------------------ #
    # optimizer                                                           #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

class _MultiTaskMixin:
    """
    为四个模型提供统一的：
      - 稀疏 + 稠密特征编码（_build_feature_layers / _encode_features）
      - TorchJD 初始化
      - training_step 分发（标准加权 / TorchJD）
      - validation / test AUC 计算
      - configure_optimizers

    子类须在 __init__ 中：
      1. 调用 _build_feature_layers(...) 注册 embeddings，并获取 input_dim
      2. 完成各自网络构建
      3. 调用 _init_torchjd(...)
      4. 调用 _init_metrics()

    子类须实现 _forward_logits(x) -> (ctr_logit, cvr_logit)
        其中 x 已是编码后的张量 [B, input_dim]，无需再处理特征。
    """

    # ------------------------------------------------------------------ #
    # 特征层构建与编码                                                      #
    # ------------------------------------------------------------------ #

    def _build_feature_layers(
        self,
        sparse_feature_names: List[str],
        sparse_feature_dims : Dict[str, int],
        dense_feature_names : List[str],
        embedding_dim       : int,
    ) -> int:
        """
        注册 Embedding 层并返回最终 input_dim。
        必须在 __init__ 中、网络构建之前调用。
        """
        self.sparse_feature_names = sparse_feature_names
        self.dense_feature_names  = dense_feature_names

        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(sparse_feature_dims[feat], embedding_dim)
            for feat in sparse_feature_names
        })

        raw_dim = len(sparse_feature_names) * embedding_dim + len(dense_feature_names)
        return raw_dim

    def _encode_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """将原始 inputs 编码为单一张量 [B, input_dim]"""
        parts = []

        # 稀疏特征 → Embedding
        for feat in self.sparse_feature_names:
            parts.append(self.embeddings[feat](inputs[feat]))  # [B, emb_dim]

        # 稠密特征 → 直接使用，统一 shape 为 [B, 1] 或 [B, d]
        for feat in self.dense_feature_names:
            t = inputs[feat]
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            parts.append(t)

        return torch.cat(parts, dim=-1)  # [B, raw_dim]

    # ------------------------------------------------------------------ #
    # TorchJD & 指标初始化                                                  #
    # ------------------------------------------------------------------ #

    def _init_torchjd(
        self,
        use_torchjd       : bool,
        aggregation_method: str,
        ctr_weight        : float,
        cvr_weight        : float,
        esmm              : bool = False,
        sigmoid           : int = 1,
        use_entropy_reg   : bool = False,
        lambda_entropy    : float = 0.01,
        use_ema           : bool = False,
        ema_decay         : float = 0.999,
    ):
        """
        初始化 TorchJD 梯度聚合和 Entropy 正则化参数

        Args:
            use_entropy_reg: 是否使用 Entropy 正则化防止门控极化
            lambda_entropy: Entropy 正则化权重
            use_ema: 是否启用 EMA 参数更新
            ema_decay: EMA 衰减系数（越接近 1 越平滑）
        """
        self.use_torchjd = use_torchjd
        self.ctr_weight  = ctr_weight
        self.cvr_weight  = cvr_weight
        self.esmm        = esmm
        self.sigmoid     = sigmoid
        self.use_entropy_reg = use_entropy_reg
        self.lambda_entropy  = lambda_entropy
        self.use_ema    = use_ema
        self.ema_decay  = ema_decay
        if use_torchjd:
            self.automatic_optimization = False
            self.aggregator = _create_aggregator(aggregation_method)

    def _init_metrics(self):
        self.val_ctr_auc  = AUROC(task="binary")
        self.val_cvr_auc  = AUROC(task="binary")
        self.test_ctr_auc = AUROC(task="binary")
        self.test_cvr_auc = AUROC(task="binary")
        # EMA 专用指标
        self.val_ema_ctr_auc  = AUROC(task="binary")
        self.val_ema_cvr_auc  = AUROC(task="binary")
        self.test_ema_ctr_auc = AUROC(task="binary")
        self.test_ema_cvr_auc = AUROC(task="binary")

    # ------------------------------------------------------------------ #
    # EMA                                                                 #
    # ------------------------------------------------------------------ #

    def on_fit_start(self):
        """训练开始时初始化 EMA 参数副本"""
        if self.use_ema:
            self._ema_params = {
                name: param.data.clone()
                for name, param in self.named_parameters()
            }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """每个训练 batch 结束后更新 EMA 参数"""
        if not self.use_ema:
            return
        decay = self.ema_decay
        for name, param in self.named_parameters():
            self._ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    def _forward_with_ema(self, inputs: Dict[str, torch.Tensor], return_gate_entropy: bool = False):
        """用 EMA 参数做前向推理（临时替换参数，推理后还原）"""
        original = {name: param.data for name, param in self.named_parameters()}
        for name, param in self.named_parameters():
            param.data = self._ema_params[name].to(param.device)
        try:
            result = self.forward(inputs, return_gate_entropy=return_gate_entropy)
        finally:
            for name, param in self.named_parameters():
                param.data = original[name]
        return result

    # ------------------------------------------------------------------ #
    # forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, inputs: Dict[str, torch.Tensor], return_gate_entropy: bool = False):
        x = self._encode_features(inputs)
        result = self._forward_logits(x, return_gate_entropy=return_gate_entropy)
        if return_gate_entropy:
            ctr_logit, cvr_logit, gate_entropy = result
            return ctr_logit.squeeze(-1), cvr_logit.squeeze(-1), gate_entropy
        else:
            ctr_logit, cvr_logit = result
            return ctr_logit.squeeze(-1), cvr_logit.squeeze(-1)

    # ------------------------------------------------------------------ #
    # training_step 分发                                                   #
    # ------------------------------------------------------------------ #

    def training_step(self, batch, batch_idx):
        if self.use_torchjd:
            return self._training_step_torchjd(batch)
        return self._training_step_standard(batch)

    def _compute_losses(self, batch):
        inputs, labels = batch
        ctr_labels = labels["click"]
        cvr_labels = labels["purchase"]

        # 如果启用 entropy 正则化，需要获取 gate entropy
        if self.use_entropy_reg:
            ctr_logits, cvr_logits, gate_entropy = self(inputs, return_gate_entropy=True)
        else:
            ctr_logits, cvr_logits = self(inputs)
            gate_entropy = None

        pCTR = torch.sigmoid(ctr_logits)
        if self.sigmoid == 1:
            pCVR = torch.sigmoid(cvr_logits)
        else:
            pCVR = torch.sigmoid(cvr_logits) * torch.sigmoid(cvr_logits)

        ctr_loss = nn.functional.binary_cross_entropy_with_logits(
            ctr_logits, ctr_labels.float()
        )

        if self.esmm:
            # ESMM：用 pCTR × pCVR 在全量曝光空间监督购买标签
            # CVR Tower 通过 pCTCVR 的梯度间接学习，无显式 L_cvr
            pCTCVR   = pCTR * pCVR
            cvr_loss = nn.functional.binary_cross_entropy(
                pCTCVR, cvr_labels.float()
            )
        else:
            # 标准：直接在点击样本上监督 CVR
            cvr_loss = nn.functional.binary_cross_entropy_with_logits(
                cvr_logits, cvr_labels.float()
            )

        # Entropy 正则化 loss
        # 目标：最大化 entropy，即最小化 -entropy
        # 因此 entropy_loss = -mean(entropy)
        if self.use_entropy_reg and gate_entropy is not None:
            entropy_loss = -gate_entropy.mean()  # 负号：最大化 entropy = 最小化 -entropy
        else:
            entropy_loss = torch.tensor(0.0, device=ctr_logits.device)

        return ctr_logits, cvr_logits, ctr_labels, cvr_labels, ctr_loss, cvr_loss, entropy_loss

    def _training_step_standard(self, batch):
        _, _, _, _, ctr_loss, cvr_loss, entropy_loss = self._compute_losses(batch)
        loss = self.ctr_weight * ctr_loss + self.cvr_weight * cvr_loss + self.lambda_entropy * entropy_loss
        self.log("train_loss",         loss,         prog_bar=True)
        self.log("train_ctr_loss",     ctr_loss)
        self.log("train_cvr_loss",     cvr_loss)
        self.log("train_entropy_loss", entropy_loss)
        return loss

    def _training_step_torchjd(self, batch):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        _, _, _, _, ctr_loss, cvr_loss, entropy_loss = self._compute_losses(batch)
        backward([ctr_loss, cvr_loss], aggregator=self.aggregator)
        optimizer.step()
        total_loss = ctr_loss + cvr_loss + self.lambda_entropy * entropy_loss
        self.log("train_loss",         total_loss,   prog_bar=True)
        self.log("train_ctr_loss",     ctr_loss)
        self.log("train_cvr_loss",     cvr_loss)
        self.log("train_entropy_loss", entropy_loss)
        return total_loss

    # ------------------------------------------------------------------ #
    # validation / test                                                   #
    # ------------------------------------------------------------------ #

    def _update_auc_metrics(self, ctr_logits, cvr_logits, ctr_labels, cvr_labels,
                            ctr_metric, cvr_metric):
        """更新 AUC 指标（原模型或 EMA 模型通用）"""
        ctr_metric.update(torch.sigmoid(ctr_logits), ctr_labels.long())
        click_mask = ctr_labels.bool()
        if click_mask.any():
            cvr_metric.update(
                torch.sigmoid(cvr_logits[click_mask]),
                cvr_labels[click_mask].long()
            )

    def validation_step(self, batch, batch_idx):
        ctr_logits, cvr_logits, ctr_labels, cvr_labels, ctr_loss, cvr_loss, entropy_loss = \
            self._compute_losses(batch)
        loss = self.ctr_weight * ctr_loss + self.cvr_weight * cvr_loss + self.lambda_entropy * entropy_loss

        self._update_auc_metrics(ctr_logits, cvr_logits, ctr_labels, cvr_labels,
                                 self.val_ctr_auc, self.val_cvr_auc)

        # EMA 评估
        if self.use_ema and hasattr(self, '_ema_params'):
            inputs, _ = batch
            ema_ctr_logits, ema_cvr_logits = self._forward_with_ema(inputs)
            self._update_auc_metrics(ema_ctr_logits, ema_cvr_logits, ctr_labels, cvr_labels,
                                     self.val_ema_ctr_auc, self.val_ema_cvr_auc)

        self.log("val_loss",         loss,         prog_bar=True, sync_dist=True)
        self.log("val_ctr_loss",     ctr_loss,     sync_dist=True)
        self.log("val_cvr_loss",     cvr_loss,     sync_dist=True)
        self.log("val_entropy_loss", entropy_loss, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        ctr_auc = self.val_ctr_auc.compute()
        cvr_auc = self.val_cvr_auc.compute()
        self.log("val_ctr_auc",      ctr_auc,           prog_bar=True, sync_dist=True)
        self.log("val_cvr_auc",      cvr_auc,           prog_bar=True, sync_dist=True)
        self.log("val_combined_auc", ctr_auc * cvr_auc, prog_bar=True, sync_dist=True)
        self.val_ctr_auc.reset()
        self.val_cvr_auc.reset()

        if self.use_ema:
            ema_ctr_auc = self.val_ema_ctr_auc.compute()
            ema_cvr_auc = self.val_ema_cvr_auc.compute()
            self.log("val_ema_ctr_auc",      ema_ctr_auc,                 prog_bar=True, sync_dist=True)
            self.log("val_ema_cvr_auc",      ema_cvr_auc,                 prog_bar=True, sync_dist=True)
            self.log("val_ema_combined_auc", ema_ctr_auc * ema_cvr_auc,   prog_bar=True, sync_dist=True)
            self.val_ema_ctr_auc.reset()
            self.val_ema_cvr_auc.reset()

    def test_step(self, batch, batch_idx):
        ctr_logits, cvr_logits, ctr_labels, cvr_labels, ctr_loss, cvr_loss, entropy_loss = \
            self._compute_losses(batch)
        loss = self.ctr_weight * ctr_loss + self.cvr_weight * cvr_loss + self.lambda_entropy * entropy_loss

        self._update_auc_metrics(ctr_logits, cvr_logits, ctr_labels, cvr_labels,
                                 self.test_ctr_auc, self.test_cvr_auc)

        # EMA 评估
        if self.use_ema and hasattr(self, '_ema_params'):
            inputs, _ = batch
            ema_ctr_logits, ema_cvr_logits = self._forward_with_ema(inputs)
            self._update_auc_metrics(ema_ctr_logits, ema_cvr_logits, ctr_labels, cvr_labels,
                                     self.test_ema_ctr_auc, self.test_ema_cvr_auc)

        self.log("test_loss",         loss,         sync_dist=True)
        self.log("test_ctr_loss",     ctr_loss,     sync_dist=True)
        self.log("test_cvr_loss",     cvr_loss,     sync_dist=True)
        self.log("test_entropy_loss", entropy_loss, sync_dist=True)
        return loss

    def on_test_epoch_end(self):
        test_ctr_auc = self.test_ctr_auc.compute()
        test_cvr_auc = self.test_cvr_auc.compute()
        self.log("test_ctr_auc",      test_ctr_auc,                prog_bar=True, sync_dist=True)
        self.log("test_cvr_auc",      test_cvr_auc,                prog_bar=True, sync_dist=True)
        self.log("test_combined_auc", test_ctr_auc * test_cvr_auc, prog_bar=True, sync_dist=True)
        self.test_ctr_auc.reset()
        self.test_cvr_auc.reset()

        if self.use_ema:
            ema_ctr_auc = self.test_ema_ctr_auc.compute()
            ema_cvr_auc = self.test_ema_cvr_auc.compute()
            self.log("test_ema_ctr_auc",      ema_ctr_auc,                prog_bar=True, sync_dist=True)
            self.log("test_ema_cvr_auc",      ema_cvr_auc,                prog_bar=True, sync_dist=True)
            self.log("test_ema_combined_auc", ema_ctr_auc * ema_cvr_auc,  prog_bar=True, sync_dist=True)
            self.test_ema_ctr_auc.reset()
            self.test_ema_cvr_auc.reset()

    # ------------------------------------------------------------------ #
    # optimizer                                                           #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
class ShareBottomModel(_MultiTaskMixin, pl.LightningModule):
    """
    Share Bottom 多任务学习模型

    所有任务共享同一个底层网络，各自拥有独立的任务塔。

    Args:
        sparse_feature_names : 稀疏特征名称列表（经 Embedding 编码）
        sparse_feature_dims  : 稀疏特征词表大小字典 {feat: vocab_size}
        dense_feature_names  : 稠密特征名称列表（直接拼接，每个特征为标量或向量）
        embedding_dim        : Embedding 维度
        shared_hidden_dims   : 共享底层网络各隐层维度
        tower_hidden_dims    : 任务塔各隐层维度
        dropout              : Dropout 比例
        learning_rate        : Adam 学习率
        ctr_weight           : CTR loss 权重（标准模式）
        cvr_weight           : CVR loss 权重（标准模式）
        use_torchjd          : 是否使用 TorchJD 梯度聚合（默认 False）
        aggregation_method   : TorchJD 聚合算法（use_torchjd=True 时生效）
                               可选 upgrad / mgda / pcgrad / graddrop
    """

    def __init__(
        self,
        sparse_feature_names: List[str],
        sparse_feature_dims : Dict[str, int],
        dense_feature_names : List[str],
        embedding_dim       : int       = 32,
        shared_hidden_dims  : List[int] = [256, 128],
        tower_hidden_dims   : List[int] = [64],
        dropout             : float     = 0.2,
        learning_rate       : float     = 1e-3,
        ctr_weight          : float     = 0.5,
        cvr_weight          : float     = 0.5,
        use_torchjd         : bool      = False,
        aggregation_method  : str       = "upgrad",
        esmm                : bool      = False,
        sigmoid             : int       = 1,
        use_ema             : bool      = False,
        ema_decay           : float     = 0.999,
        use_dcn             : bool      = False,
        dcn_num_layers      : int       = 2,
        dcn_dropout         : float     = 0.0,
        dcn_rank            : int       = 0,
    ):
        pl.LightningModule.__init__(self)
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        input_dim = self._build_feature_layers(
            sparse_feature_names, sparse_feature_dims, dense_feature_names, embedding_dim,
        )

        # 共享底层
        self.shared_bottom = Expert(input_dim, shared_hidden_dims, dropout,
                                    use_dcn=use_dcn, dcn_num_layers=dcn_num_layers, dcn_dropout=dcn_dropout, dcn_rank=dcn_rank)

        # 任务塔
        self.ctr_tower = Tower(shared_hidden_dims[-1], tower_hidden_dims, dropout)
        self.cvr_tower = Tower(shared_hidden_dims[-1], tower_hidden_dims, dropout)

        self._init_torchjd(use_torchjd, aggregation_method, ctr_weight, cvr_weight, esmm, sigmoid, use_ema=use_ema, ema_decay=ema_decay)
        self._init_metrics()

    def _forward_logits(self, x: torch.Tensor, return_gate_entropy: bool = False):
        shared_out = self.shared_bottom(x)
        return self.ctr_tower(shared_out), self.cvr_tower(shared_out)

class MOEModel(_MultiTaskMixin, pl.LightningModule):
    """
    MOE (Mixture-of-Experts) 多任务学习模型 — 共享单门控

    所有任务共用同一组专家网络和同一个门控网络。
    与 MMOE 的区别：MMOE 为每个任务独立设置门控网络。

    Args:
        sparse_feature_names : 稀疏特征名称列表（经 Embedding 编码）
        sparse_feature_dims  : 稀疏特征词表大小字典 {feat: vocab_size}
        dense_feature_names  : 稠密特征名称列表（直接拼接）
        num_experts          : 专家网络数量
        expert_hidden_dims   : 每个专家网络各隐层维度
        tower_hidden_dims    : 任务塔各隐层维度
        use_torchjd          : 是否使用 TorchJD 梯度聚合（默认 False）
        aggregation_method   : TorchJD 聚合算法（use_torchjd=True 时生效）
    """

    def __init__(
        self,
        sparse_feature_names: List[str],
        sparse_feature_dims : Dict[str, int],
        dense_feature_names : List[str],
        embedding_dim       : int       = 32,
        num_experts         : int       = 3,
        expert_hidden_dims  : List[int] = [256, 128],
        tower_hidden_dims   : List[int] = [64],
        dropout             : float     = 0.2,
        learning_rate       : float     = 1e-3,
        ctr_weight          : float     = 0.5,
        cvr_weight          : float     = 0.5,
        use_torchjd         : bool      = False,
        aggregation_method  : str       = "upgrad",
        esmm                : bool      = False,
        sigmoid             : int       = 1,
        use_entropy_reg     : bool      = False,
        lambda_entropy      : float     = 0.01,
        use_ema             : bool      = False,
        ema_decay           : float     = 0.999,
    ):
        pl.LightningModule.__init__(self)
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        input_dim = self._build_feature_layers(
            sparse_feature_names, sparse_feature_dims, dense_feature_names, embedding_dim
        )

        # 专家网络 & 共享门控（MOE 与 MMOE 的核心区别）
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dims, dropout)
            for _ in range(num_experts)
        ])
        self.gate = Gate(input_dim, num_experts)

        # 任务塔
        self.ctr_tower = Tower(expert_hidden_dims[-1], tower_hidden_dims, dropout)
        self.cvr_tower = Tower(expert_hidden_dims[-1], tower_hidden_dims, dropout)

        self._init_torchjd(use_torchjd, aggregation_method, ctr_weight, cvr_weight, esmm, sigmoid, use_entropy_reg, lambda_entropy, use_ema=use_ema, ema_decay=ema_decay)
        self._init_metrics()

    def _forward_logits(self, x: torch.Tensor, return_gate_entropy: bool = False):
        """
        Args:
            x: [B, input_dim]
            return_gate_entropy: 是否返回 gate entropy（用于 entropy 正则化）
        Returns:
            ctr_logit: [B, 1]
            cvr_logit: [B, 1]
            gate_entropy: [B] gate entropy（可选）
        """
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # [B, E, d]

        if return_gate_entropy:
            gate_weights, gate_entropy = self.gate(x, return_entropy=True)  # [B, E], [B]
            gate_weights = gate_weights.unsqueeze(-1)  # [B, E, 1]
            expert_out = torch.sum(expert_outputs * gate_weights, dim=1)  # [B, d]
            return self.ctr_tower(expert_out), self.cvr_tower(expert_out), gate_entropy
        else:
            gate_weights = self.gate(x).unsqueeze(-1)  # [B, E, 1]
            expert_out = torch.sum(expert_outputs * gate_weights, dim=1)  # [B, d]
            return self.ctr_tower(expert_out), self.cvr_tower(expert_out)

class MMOEModel(_MultiTaskMixin, pl.LightningModule):
    """
    MMOE (Multi-gate Mixture-of-Experts) 多任务学习模型

    专家网络共享，每个任务拥有独立的门控网络，
    可学习到不同的专家组合偏好，减少任务间负迁移。

    Args:
        sparse_feature_names : 稀疏特征名称列表（经 Embedding 编码）
        sparse_feature_dims  : 稀疏特征词表大小字典 {feat: vocab_size}
        dense_feature_names  : 稠密特征名称列表（直接拼接）
        num_experts          : 专家网络数量
        expert_hidden_dims   : 每个专家网络各隐层维度
        tower_hidden_dims    : 任务塔各隐层维度
        use_torchjd          : 是否使用 TorchJD 梯度聚合（默认 False）
        aggregation_method   : TorchJD 聚合算法（use_torchjd=True 时生效）
    """

    def __init__(
        self,
        sparse_feature_names: List[str],
        sparse_feature_dims : Dict[str, int],
        dense_feature_names : List[str],
        embedding_dim       : int       = 32,
        num_experts         : int       = 3,
        expert_hidden_dims  : List[int] = [256, 128],
        tower_hidden_dims   : List[int] = [64],
        dropout             : float     = 0.2,
        learning_rate       : float     = 1e-3,
        ctr_weight          : float     = 0.5,
        cvr_weight          : float     = 0.5,
        use_torchjd         : bool      = False,
        aggregation_method  : str       = "upgrad",
        esmm                : bool      = False,
        sigmoid             : int       = 1,
        use_entropy_reg     : bool      = False,
        lambda_entropy      : float     = 0.01,
        use_ema             : bool      = False,
        ema_decay           : float     = 0.999,
        use_dcn             : bool      = False,
        dcn_num_layers      : int       = 2,
        dcn_dropout         : float     = 0.0,
        dcn_rank            : int       = 0,
    ):
        pl.LightningModule.__init__(self)
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        input_dim = self._build_feature_layers(
            sparse_feature_names, sparse_feature_dims, dense_feature_names, embedding_dim,
        )

        # 特征交叉前置：x -> cross(x) -> concat([cross_x, x]) -> MMOE
        self.use_dcn = use_dcn
        if use_dcn:
            self.feature_cross = DCNv2CrossLayer(
                input_dim,
                num_layers=dcn_num_layers,
                dropout=dcn_dropout,
                rank=dcn_rank,
            )
            mmoe_input_dim = input_dim * 2
        else:
            self.feature_cross = None
            mmoe_input_dim = input_dim

        # 共享专家网络（DCN 已前置，Expert 保持纯 MLP）
        self.experts = nn.ModuleList([
            Expert(mmoe_input_dim, expert_hidden_dims, dropout, use_dcn=False)
            for _ in range(num_experts)
        ])

        # 每个任务独立门控（MMOE 核心）
        self.ctr_gate = Gate(mmoe_input_dim, num_experts)
        self.cvr_gate = Gate(mmoe_input_dim, num_experts)

        # 任务塔
        self.ctr_tower = Tower(expert_hidden_dims[-1], tower_hidden_dims, dropout)
        self.cvr_tower = Tower(expert_hidden_dims[-1], tower_hidden_dims, dropout)

        self._init_torchjd(use_torchjd, aggregation_method, ctr_weight, cvr_weight, esmm, sigmoid, use_entropy_reg, lambda_entropy, use_ema=use_ema, ema_decay=ema_decay)
        self._init_metrics()

    def _forward_logits(self, x: torch.Tensor, return_gate_entropy: bool = False):
        """
        Args:
            x: [B, input_dim]
            return_gate_entropy: 是否返回 gate entropy（用于 entropy 正则化）
        Returns:
            ctr_logit: [B, 1]
            cvr_logit: [B, 1]
            gate_entropy: [B] 平均 gate entropy（可选）
        """
        if self.use_dcn:
            cross_x = self.feature_cross(x)
            x = torch.cat([cross_x, x], dim=-1)

        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # [B, E, d]

        if return_gate_entropy:
            ctr_weights, ctr_entropy = self.ctr_gate(x, return_entropy=True)  # [B, E], [B]
            cvr_weights, cvr_entropy = self.cvr_gate(x, return_entropy=True)  # [B, E], [B]
            ctr_weights = ctr_weights.unsqueeze(-1)  # [B, E, 1]
            cvr_weights = cvr_weights.unsqueeze(-1)  # [B, E, 1]

            ctr_out = torch.sum(expert_outputs * ctr_weights, dim=1)
            cvr_out = torch.sum(expert_outputs * cvr_weights, dim=1)

            # 返回两个 gate 的平均 entropy
            avg_entropy = (ctr_entropy + cvr_entropy) / 2.0  # [B]
            return self.ctr_tower(ctr_out), self.cvr_tower(cvr_out), avg_entropy
        else:
            ctr_weights = self.ctr_gate(x).unsqueeze(-1)  # [B, E, 1]
            ctr_out = torch.sum(expert_outputs * ctr_weights, dim=1)

            cvr_weights = self.cvr_gate(x).unsqueeze(-1)
            cvr_out = torch.sum(expert_outputs * cvr_weights, dim=1)

            return self.ctr_tower(ctr_out), self.cvr_tower(cvr_out)

class PLEModel(_MultiTaskMixin, pl.LightningModule):
    """
    PLE (Progressive Layered Extraction) 多任务学习模型
    论文: Tang et al., RecSys 2020

    相比 MMOE 的改进：
      1. 引入任务专属专家（task-specific experts），避免不同任务的负迁移
      2. 支持多层渐进式提取（multi-level CGC），逐层精炼各任务表征
      3. 共享专家在每层汇聚全部任务的信息后，流入下一层

    每层 CGC 结构示意：
      ┌──────────────────────────────────────────────────────────────┐
      │  Layer l                                                     │
      │                                                              │
      │  ctr_specific_experts[l]  ─┐                                │
      │  shared_experts[l]        ─┴─► ctr_gate[l] ─► ctr_out      │
      │                                                              │
      │  cvr_specific_experts[l]  ─┐                                │
      │  shared_experts[l]        ─┴─► cvr_gate[l] ─► cvr_out      │
      │                                                              │
      │  ctr_specific + cvr_specific + shared ─► shared_gate[l]     │
      │    （仅非最后一层，最后一层无需更新共享路径）                   │
      └──────────────────────────────────────────────────────────────┘

    Args:
        sparse_feature_names : 稀疏特征名称列表（经 Embedding 编码）
        sparse_feature_dims  : 稀疏特征词表大小字典 {feat: vocab_size}
        dense_feature_names  : 稠密特征名称列表（直接拼接）
        num_specific_experts : 每个任务专属的专家数量
        num_shared_experts   : 共享专家数量
        expert_hidden_dims   : 每个专家网络各隐层维度
        num_levels           : CGC 层数（≥1，num_levels=1 即单层 PLE）
        tower_hidden_dims    : 任务塔各隐层维度
        use_torchjd          : 是否使用 TorchJD 梯度聚合（默认 False）
        aggregation_method   : TorchJD 聚合算法（use_torchjd=True 时生效）
    """

    _NUM_TASKS = 2  # CTR + CVR

    def __init__(
        self,
        sparse_feature_names : List[str],
        sparse_feature_dims  : Dict[str, int],
        dense_feature_names  : List[str],
        embedding_dim        : int       = 32,
        num_specific_experts : int       = 2,
        num_shared_experts   : int       = 2,
        expert_hidden_dims   : List[int] = [256, 128],
        num_levels           : int       = 2,
        tower_hidden_dims    : List[int] = [64],
        dropout              : float     = 0.2,
        learning_rate        : float     = 1e-3,
        ctr_weight           : float     = 0.5,
        cvr_weight           : float     = 0.5,
        use_torchjd          : bool      = False,
        aggregation_method   : str       = "upgrad",
        esmm                 : bool      = False,
        sigmoid              : int       = 1,
        use_ema              : bool      = False,
        ema_decay            : float     = 0.999,
    ):
        pl.LightningModule.__init__(self)
        self.save_hyperparameters()

        self.learning_rate        = learning_rate
        self.num_levels           = num_levels
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts   = num_shared_experts

        input_dim = self._build_feature_layers(
            sparse_feature_names, sparse_feature_dims, dense_feature_names, embedding_dim
        )

        # ---- 逐层构建专家网络与门控 ----
        # 第 0 层输入维度 = input_dim；后续层 = expert_hidden_dims[-1]
        self.ctr_experts    = nn.ModuleList()  # [level] → ModuleList[Expert]
        self.cvr_experts    = nn.ModuleList()
        self.shared_experts = nn.ModuleList()
        self.ctr_gates      = nn.ModuleList()  # [level] → Gate
        self.cvr_gates      = nn.ModuleList()
        self.shared_gates   = nn.ModuleList()  # 仅 num_levels-1 个

        for level in range(num_levels):
            lvl_in = input_dim if level == 0 else expert_hidden_dims[-1]

            self.ctr_experts.append(nn.ModuleList([
                Expert(lvl_in, expert_hidden_dims, dropout)
                for _ in range(num_specific_experts)
            ]))
            self.cvr_experts.append(nn.ModuleList([
                Expert(lvl_in, expert_hidden_dims, dropout)
                for _ in range(num_specific_experts)
            ]))
            self.shared_experts.append(nn.ModuleList([
                Expert(lvl_in, expert_hidden_dims, dropout)
                for _ in range(num_shared_experts)
            ]))

            # 任务门控：从 task_specific + shared 中选择
            self.ctr_gates.append(Gate(lvl_in, num_specific_experts + num_shared_experts))
            self.cvr_gates.append(Gate(lvl_in, num_specific_experts + num_shared_experts))

            # 共享门控：从 ctr_specific + cvr_specific + shared 中选择（非最后层）
            if level < num_levels - 1:
                self.shared_gates.append(
                    Gate(lvl_in, self._NUM_TASKS * num_specific_experts + num_shared_experts)
                )

        # 任务塔
        self.ctr_tower = Tower(expert_hidden_dims[-1], tower_hidden_dims, dropout)
        self.cvr_tower = Tower(expert_hidden_dims[-1], tower_hidden_dims, dropout)

        self._init_torchjd(use_torchjd, aggregation_method, ctr_weight, cvr_weight, esmm, sigmoid, use_ema=use_ema, ema_decay=ema_decay)
        self._init_metrics()

    def _forward_logits(self, x: torch.Tensor, return_gate_entropy: bool = False):
        # 各路径的当前输入（第 0 层统一为原始编码 x）
        ctr_in    = x
        cvr_in    = x
        shared_in = x

        for level in range(self.num_levels):
            # 各类专家输出
            ctr_specific_out = torch.stack(                                    # [B, ns, d]
                [e(ctr_in)    for e in self.ctr_experts[level]],    dim=1)
            cvr_specific_out = torch.stack(                                    # [B, ns, d]
                [e(cvr_in)    for e in self.cvr_experts[level]],    dim=1)
            shared_out       = torch.stack(                                    # [B, nsh, d]
                [e(shared_in) for e in self.shared_experts[level]], dim=1)

            # CTR 提取：task-specific + shared
            ctr_candidates = torch.cat([ctr_specific_out, shared_out], dim=1) # [B, ns+nsh, d]
            ctr_weights    = self.ctr_gates[level](ctr_in).unsqueeze(-1)      # [B, ns+nsh, 1]
            new_ctr        = torch.sum(ctr_candidates * ctr_weights,  dim=1)  # [B, d]

            # CVR 提取：task-specific + shared
            cvr_candidates = torch.cat([cvr_specific_out, shared_out], dim=1)
            cvr_weights    = self.cvr_gates[level](cvr_in).unsqueeze(-1)
            new_cvr        = torch.sum(cvr_candidates * cvr_weights,  dim=1)

            # Shared 提取：ctr_specific + cvr_specific + shared（非最后层）
            if level < self.num_levels - 1:
                shared_candidates = torch.cat(
                    [ctr_specific_out, cvr_specific_out, shared_out], dim=1)  # [B, 2ns+nsh, d]
                shared_weights    = self.shared_gates[level](shared_in).unsqueeze(-1)
                shared_in         = torch.sum(shared_candidates * shared_weights, dim=1)

            ctr_in = new_ctr
            cvr_in = new_cvr

        return self.ctr_tower(ctr_in), self.cvr_tower(cvr_in)


class AdaFTRModel(_MultiTaskMixin, pl.LightningModule):
    """
    AdaFTR 风格的样本内对比学习多任务模型（MMOE backbone）。

    数据流：
      input -> (optional DCN cross + concat) -> MMOE -> h_ctr/h_cvr
            -> ctr/cvr heads
            -> contrastive loss + relatedness network (dynamic temperature)
    """

    def __init__(
        self,
        sparse_feature_names: List[str],
        sparse_feature_dims : Dict[str, int],
        dense_feature_names : List[str],
        embedding_dim       : int       = 32,
        num_experts         : int       = 8,
        expert_hidden_dims  : List[int] = [256, 128],
        tower_hidden_dims   : List[int] = [64],
        dropout             : float     = 0.2,
        learning_rate       : float     = 1e-3,
        ctr_weight          : float     = 0.5,
        cvr_weight          : float     = 0.5,
        use_torchjd         : bool      = False,
        aggregation_method  : str       = "upgrad",
        esmm                : bool      = False,
        sigmoid             : int       = 1,
        use_entropy_reg     : bool      = False,
        lambda_entropy      : float     = 0.01,
        use_ema             : bool      = False,
        ema_decay           : float     = 0.999,
        use_dcn             : bool      = False,
        dcn_num_layers      : int       = 2,
        dcn_dropout         : float     = 0.0,
        dcn_rank            : int       = 0,
        alpha_contrastive   : float     = 0.1,
        tau_min             : float     = 0.05,
        tau_max             : float     = 0.50,
        relatedness_hidden_dim: int     = 64,
        lambda_rel          : float     = 0.1,
        use_hard_sample     : bool      = False,
        hard_sample_ratio   : float     = 0.2,
        hard_sample_weight  : float     = 2.0,
        hard_sample_warmup_epochs: int  = 1,
    ):
        pl.LightningModule.__init__(self)
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        input_dim = self._build_feature_layers(
            sparse_feature_names, sparse_feature_dims, dense_feature_names, embedding_dim,
        )

        # AdaFTR 论文配置：MMOE backbone expert 输出维度固定为 256
        backbone_dim = expert_hidden_dims[0] if len(expert_hidden_dims) > 0 else 256

        # 特征交叉前置（可选）：x -> cross(x) -> concat([cross_x, x]) -> MMOE
        self.use_dcn = use_dcn
        if use_dcn:
            self.feature_cross = DCNv2CrossLayer(
                input_dim,
                num_layers=dcn_num_layers,
                dropout=dcn_dropout,
                rank=dcn_rank,
            )
            mmoe_input_dim = input_dim * 2
        else:
            self.feature_cross = None
            mmoe_input_dim = input_dim

        # MMOE backbone
        self.experts = nn.ModuleList([
            Expert(mmoe_input_dim, [backbone_dim], dropout)
            for _ in range(num_experts)
        ])
        self.ctr_gate = Gate(mmoe_input_dim, num_experts)
        self.cvr_gate = Gate(mmoe_input_dim, num_experts)

        # AdaFTR task-specific layers: [256, 128, 64]
        if len(tower_hidden_dims) < 3:
            tower_hidden_dims = [256, 128, 64]
        self.rep_ctr = self._build_rep_layer(backbone_dim, tower_hidden_dims, dropout)
        self.rep_cvr = self._build_rep_layer(backbone_dim, tower_hidden_dims, dropout)
        rep_dim = tower_hidden_dims[-1]

        # CTR/CVR heads
        self.ctr_head = nn.Linear(rep_dim, 1)
        self.cvr_head = nn.Linear(rep_dim, 1)

        # Relatedness network: input = [abs(h_ctr-h_cvr), h_ctr*h_cvr]
        self.relatedness_net = nn.Sequential(
            nn.Linear(rep_dim * 2, relatedness_hidden_dim),
            nn.ReLU(),
            nn.Linear(relatedness_hidden_dim, 1),
        )

        # AdaFTR hyperparameters
        self.alpha_contrastive = alpha_contrastive
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.lambda_rel = lambda_rel
        self.use_hard_sample = use_hard_sample
        self.hard_sample_ratio = hard_sample_ratio
        self.hard_sample_weight = hard_sample_weight
        self.hard_sample_warmup_epochs = hard_sample_warmup_epochs

        # 与其它模型保持一致的训练基础能力（TorchJD / EMA / metrics）
        self._init_torchjd(
            use_torchjd,
            aggregation_method,
            ctr_weight,
            cvr_weight,
            esmm,
            sigmoid,
            use_entropy_reg=False,
            lambda_entropy=0.0,
            use_ema=use_ema,
            ema_decay=ema_decay,
        )
        self._init_metrics()

    def _build_rep_layer(self, input_dim: int, hidden_dims: List[int], dropout: float):
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        return nn.Sequential(*layers)

    def _forward_adaftr(self, x: torch.Tensor):
        if self.use_dcn:
            cross_x = self.feature_cross(x)
            x = torch.cat([cross_x, x], dim=-1)

        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # [B, E, d]

        ctr_weights = self.ctr_gate(x).unsqueeze(-1)
        cvr_weights = self.cvr_gate(x).unsqueeze(-1)
        ctr_backbone = torch.sum(expert_outputs * ctr_weights, dim=1)  # [B, d]
        cvr_backbone = torch.sum(expert_outputs * cvr_weights, dim=1)  # [B, d]

        h_ctr = self.rep_ctr(ctr_backbone)  # [B, 64]
        h_cvr = self.rep_cvr(cvr_backbone)  # [B, 64]

        ctr_logit = self.ctr_head(h_ctr).squeeze(-1)  # [B]
        cvr_logit = self.cvr_head(h_cvr).squeeze(-1)  # [B]

        # 关联性网络输入必须截断梯度（AdaFTR 要点）
        diff = torch.abs(h_ctr.detach() - h_cvr.detach())
        prod = h_ctr.detach() * h_cvr.detach()
        fusion = torch.cat([diff, prod], dim=-1)  # [B, 2D]

        relatedness_logit = self.relatedness_net(fusion).squeeze(-1)  # [B]
        temperature = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(relatedness_logit)

        return ctr_logit, cvr_logit, h_ctr, h_cvr, relatedness_logit, temperature

    def _forward_logits(self, x: torch.Tensor, return_gate_entropy: bool = False):
        ctr_logit, cvr_logit, _, _, _, _ = self._forward_adaftr(x)
        if return_gate_entropy:
            dummy_entropy = torch.zeros_like(ctr_logit)
            return ctr_logit.unsqueeze(-1), cvr_logit.unsqueeze(-1), dummy_entropy
        return ctr_logit.unsqueeze(-1), cvr_logit.unsqueeze(-1)

    def _contrastive_loss(self, h_ctr: torch.Tensor, h_cvr: torch.Tensor, temperature: torch.Tensor):
        h_ctr = F.normalize(h_ctr, dim=-1, eps=1e-8)
        h_cvr = F.normalize(h_cvr, dim=-1, eps=1e-8)

        sim = torch.matmul(h_ctr, h_cvr.T)  # [B, B]
        sim = sim / temperature.unsqueeze(1)

        labels = torch.arange(sim.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)

    def _discover_hard_mask(self, score: torch.Tensor, valid_mask: torch.Tensor):
        """
        在有效样本上按 score 做 top-ratio 挖掘，返回 hard mask 与实际 hard ratio。
        """
        hard_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        valid_idx = torch.where(valid_mask)[0]
        valid_n = int(valid_idx.numel())
        if valid_n == 0:
            return hard_mask, 0.0

        ratio = min(max(self.hard_sample_ratio, 0.0), 1.0)
        if ratio <= 0.0:
            return hard_mask, 0.0

        k = max(1, int(valid_n * ratio))
        k = min(k, valid_n)

        valid_score = score[valid_idx]
        top_idx = torch.topk(valid_score, k=k, largest=True).indices
        hard_idx = valid_idx[top_idx]
        hard_mask[hard_idx] = True
        return hard_mask, float(k) / float(valid_n)

    def _reduce_with_hard_weights(
        self,
        loss_vec: torch.Tensor,
        valid_mask: torch.Tensor,
        hard_mask: torch.Tensor,
        for_training: bool,
    ):
        weights = torch.ones_like(loss_vec)
        if (
            for_training
            and self.use_hard_sample
            and self.current_epoch >= self.hard_sample_warmup_epochs
        ):
            hard_w = torch.full_like(weights, self.hard_sample_weight)
            weights = torch.where(hard_mask, hard_w, weights)

        weights = weights * valid_mask.float()
        denom = weights.sum().clamp_min(1.0)
        return (loss_vec * weights).sum() / denom

    def _compute_losses(self, batch, for_training: bool = False):
        inputs, labels = batch
        ctr_labels = labels["click"].float()
        cvr_labels = labels["purchase"].float()

        x = self._encode_features(inputs)
        ctr_logits, cvr_logits, h_ctr, h_cvr, rel_logits, temperature = self._forward_adaftr(x)

        pCTR = torch.sigmoid(ctr_logits)
        if self.sigmoid == 1:
            pCVR = torch.sigmoid(cvr_logits)
        else:
            pCVR = torch.sigmoid(cvr_logits) * torch.sigmoid(cvr_logits)

        # --- CTR loss（逐样本） ---
        ctr_loss_vec = nn.functional.binary_cross_entropy_with_logits(
            ctr_logits, ctr_labels, reduction="none"
        )
        ctr_valid_mask = torch.ones_like(ctr_labels, dtype=torch.bool)
        ctr_hard_mask, hard_ctr_ratio = self._discover_hard_mask(
            ctr_loss_vec.detach(), ctr_valid_mask
        )
        ctr_loss = self._reduce_with_hard_weights(
            ctr_loss_vec, ctr_valid_mask, ctr_hard_mask, for_training
        )

        # --- CVR loss（逐样本） ---
        if self.esmm:
            pCTCVR = pCTR * pCVR
            cvr_loss_vec = nn.functional.binary_cross_entropy(
                pCTCVR, cvr_labels, reduction="none"
            )
            cvr_valid_mask = torch.ones_like(cvr_labels, dtype=torch.bool)
        else:
            cvr_loss_vec = nn.functional.binary_cross_entropy_with_logits(
                cvr_logits, cvr_labels, reduction="none"
            )
            cvr_valid_mask = ctr_labels > 0.5

        cvr_hard_mask, hard_cvr_ratio = self._discover_hard_mask(
            cvr_loss_vec.detach(), cvr_valid_mask
        )
        cvr_loss = self._reduce_with_hard_weights(
            cvr_loss_vec, cvr_valid_mask, cvr_hard_mask, for_training
        )

        contrastive_loss = self._contrastive_loss(h_ctr, h_cvr, temperature)

        # y_rel = 1 if y_ctr == y_cvr else 0
        rel_target = (ctr_labels == cvr_labels).float()
        rel_loss = nn.functional.binary_cross_entropy_with_logits(rel_logits, rel_target)

        return (
            ctr_logits,
            cvr_logits,
            ctr_labels,
            cvr_labels,
            ctr_loss,
            cvr_loss,
            contrastive_loss,
            rel_loss,
            hard_ctr_ratio,
            hard_cvr_ratio,
        )

    def _training_step_standard(self, batch):
        (
            _,
            _,
            _,
            _,
            ctr_loss,
            cvr_loss,
            contrastive_loss,
            rel_loss,
            hard_ctr_ratio,
            hard_cvr_ratio,
        ) = self._compute_losses(batch, for_training=True)

        loss = (
            self.ctr_weight * ctr_loss
            + self.cvr_weight * cvr_loss
            + self.alpha_contrastive * contrastive_loss
            + self.lambda_rel * rel_loss
        )
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_ctr_loss", ctr_loss)
        self.log("train_cvr_loss", cvr_loss)
        self.log("train_contrastive_loss", contrastive_loss)
        self.log("train_relatedness_loss", rel_loss)
        self.log("train_hard_ctr_ratio", hard_ctr_ratio)
        self.log("train_hard_cvr_ratio", hard_cvr_ratio)
        return loss

    def _training_step_torchjd(self, batch):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        (
            _,
            _,
            _,
            _,
            ctr_loss,
            cvr_loss,
            contrastive_loss,
            rel_loss,
            hard_ctr_ratio,
            hard_cvr_ratio,
        ) = self._compute_losses(batch, for_training=True)

        backward([ctr_loss, cvr_loss], aggregator=self.aggregator)
        aux_loss = self.alpha_contrastive * contrastive_loss + self.lambda_rel * rel_loss
        aux_loss.backward()
        optimizer.step()

        total_loss = ctr_loss + cvr_loss + aux_loss
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_ctr_loss", ctr_loss)
        self.log("train_cvr_loss", cvr_loss)
        self.log("train_contrastive_loss", contrastive_loss)
        self.log("train_relatedness_loss", rel_loss)
        self.log("train_hard_ctr_ratio", hard_ctr_ratio)
        self.log("train_hard_cvr_ratio", hard_cvr_ratio)
        return total_loss

    def validation_step(self, batch, batch_idx):
        (
            ctr_logits,
            cvr_logits,
            ctr_labels,
            cvr_labels,
            ctr_loss,
            cvr_loss,
            contrastive_loss,
            rel_loss,
            hard_ctr_ratio,
            hard_cvr_ratio,
        ) = self._compute_losses(batch, for_training=False)

        loss = (
            self.ctr_weight * ctr_loss
            + self.cvr_weight * cvr_loss
            + self.alpha_contrastive * contrastive_loss
            + self.lambda_rel * rel_loss
        )

        self._update_auc_metrics(
            ctr_logits, cvr_logits, ctr_labels, cvr_labels, self.val_ctr_auc, self.val_cvr_auc
        )

        if self.use_ema and hasattr(self, "_ema_params"):
            inputs, _ = batch
            ema_ctr_logits, ema_cvr_logits = self._forward_with_ema(inputs)
            self._update_auc_metrics(
                ema_ctr_logits, ema_cvr_logits, ctr_labels, cvr_labels,
                self.val_ema_ctr_auc, self.val_ema_cvr_auc
            )

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_ctr_loss", ctr_loss, sync_dist=True)
        self.log("val_cvr_loss", cvr_loss, sync_dist=True)
        self.log("val_contrastive_loss", contrastive_loss, sync_dist=True)
        self.log("val_relatedness_loss", rel_loss, sync_dist=True)
        self.log("val_hard_ctr_ratio", hard_ctr_ratio, sync_dist=True)
        self.log("val_hard_cvr_ratio", hard_cvr_ratio, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        (
            ctr_logits,
            cvr_logits,
            ctr_labels,
            cvr_labels,
            ctr_loss,
            cvr_loss,
            contrastive_loss,
            rel_loss,
            hard_ctr_ratio,
            hard_cvr_ratio,
        ) = self._compute_losses(batch, for_training=False)

        loss = (
            self.ctr_weight * ctr_loss
            + self.cvr_weight * cvr_loss
            + self.alpha_contrastive * contrastive_loss
            + self.lambda_rel * rel_loss
        )

        self._update_auc_metrics(
            ctr_logits, cvr_logits, ctr_labels, cvr_labels, self.test_ctr_auc, self.test_cvr_auc
        )

        if self.use_ema and hasattr(self, "_ema_params"):
            inputs, _ = batch
            ema_ctr_logits, ema_cvr_logits = self._forward_with_ema(inputs)
            self._update_auc_metrics(
                ema_ctr_logits, ema_cvr_logits, ctr_labels, cvr_labels,
                self.test_ema_ctr_auc, self.test_ema_cvr_auc
            )

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_ctr_loss", ctr_loss, sync_dist=True)
        self.log("test_cvr_loss", cvr_loss, sync_dist=True)
        self.log("test_contrastive_loss", contrastive_loss, sync_dist=True)
        self.log("test_relatedness_loss", rel_loss, sync_dist=True)
        self.log("test_hard_ctr_ratio", hard_ctr_ratio, sync_dist=True)
        self.log("test_hard_cvr_ratio", hard_cvr_ratio, sync_dist=True)
        return loss
