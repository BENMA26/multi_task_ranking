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
from torchjd.autogram import Engine
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
        use_asym_proj     : bool = False,
        asym_proj_lambda  : float = 1.0,
        asym_proj_tau     : float = 0.0,
        asym_proj_only_10 : bool = False,
        asym_proj_restore_norm: bool = False,
        asym_proj_restore_max_scale: float = 5.0,
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
        self.use_asym_proj = use_asym_proj
        self.asym_proj_lambda = asym_proj_lambda
        self.asym_proj_tau = asym_proj_tau
        self.asym_proj_only_10 = asym_proj_only_10
        self.asym_proj_restore_norm = asym_proj_restore_norm
        self.asym_proj_restore_max_scale = asym_proj_restore_max_scale

        if use_torchjd and use_asym_proj:
            raise ValueError("use_torchjd 与 use_asym_proj 不能同时启用")

        if use_torchjd or use_asym_proj:
            self.automatic_optimization = False

        if use_torchjd:
            self.aggregator = _create_aggregator(aggregation_method)

        # 梯度冲突监控（默认关闭，可由 train_rank.py 动态开启）
        self.configure_grad_conflict_monitor(enabled=False, interval=100)

    def configure_grad_conflict_monitor(self, enabled: bool = False, interval: int = 100):
        self.monitor_grad_conflict = bool(enabled)
        self.grad_conflict_interval = max(1, int(interval))
        self._grad_conflict_engine = None
        self._grad_conflict_monitor_disabled = False
        self._grad_conflict_use_autograd_fallback = False

    @staticmethod
    def _collect_leaf_param_modules(root_module: nn.Module):
        modules = []
        for module in root_module.modules():
            if len(list(module.children())) != 0:
                continue
            if any(p.requires_grad for p in module.parameters(recurse=False)):
                modules.append(module)
        return modules

    def _get_grad_conflict_monitor_modules(self):
        # 默认监控整个模型的叶子参数模块；子类可按共享层覆写
        return self._collect_leaf_param_modules(self)

    def _ensure_grad_conflict_engine(self):
        if self._grad_conflict_engine is not None:
            return True
        modules = self._get_grad_conflict_monitor_modules()
        if len(modules) == 0:
            self._grad_conflict_monitor_disabled = True
            return False
        self._grad_conflict_engine = Engine(*modules, batch_dim=None)
        return True

    def _get_grad_conflict_monitor_params(self):
        params = []
        seen = set()
        for module in self._get_grad_conflict_monitor_modules():
            for p in module.parameters(recurse=False):
                if not p.requires_grad:
                    continue
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                params.append(p)
        return params

    def _compute_conflict_stats(self, loss_a: torch.Tensor, loss_b: torch.Tensor):
        used_fallback = 0.0
        dot = None
        norm_a_sq = None
        norm_b_sq = None

        if not self._grad_conflict_use_autograd_fallback:
            if self._grad_conflict_engine is None:
                if not self._ensure_grad_conflict_engine():
                    return None
                # 本步 forward 已结束，首次初始化后从下一步开始监控
                return None
            try:
                gram = self._grad_conflict_engine.compute_gramian(torch.stack([loss_a, loss_b]))
                dot = gram[0, 1]
                norm_a_sq = gram[0, 0].clamp_min(0.0)
                norm_b_sq = gram[1, 1].clamp_min(0.0)
            except Exception:
                self._grad_conflict_use_autograd_fallback = True
                self._grad_conflict_engine = None
                used_fallback = 1.0

        if self._grad_conflict_use_autograd_fallback:
            try:
                params = self._get_grad_conflict_monitor_params()
                if len(params) == 0:
                    self._grad_conflict_monitor_disabled = True
                    return None
                grads_a_raw = torch.autograd.grad(
                    loss_a, params, retain_graph=True, allow_unused=True
                )
                grads_b_raw = torch.autograd.grad(
                    loss_b, params, retain_graph=True, allow_unused=True
                )
                dot = torch.zeros((), device=loss_a.device)
                norm_a_sq = torch.zeros((), device=loss_a.device)
                norm_b_sq = torch.zeros((), device=loss_a.device)
                for p, g_a, g_b in zip(params, grads_a_raw, grads_b_raw):
                    g_a = torch.zeros_like(p) if g_a is None else g_a
                    g_b = torch.zeros_like(p) if g_b is None else g_b
                    dot = dot + (g_a * g_b).sum()
                    norm_a_sq = norm_a_sq + (g_a * g_a).sum()
                    norm_b_sq = norm_b_sq + (g_b * g_b).sum()
            except Exception:
                self._grad_conflict_monitor_disabled = True
                return None

        if dot is None or norm_a_sq is None or norm_b_sq is None:
            self._grad_conflict_monitor_disabled = True
            return None

        cos = dot / (torch.sqrt(norm_a_sq * norm_b_sq) + 1e-8)
        conflict = (dot < 0).float()
        return dot.detach(), cos.detach(), conflict.detach(), used_fallback

    def _monitor_label_group_conflicts(
        self,
        ctr_logits: torch.Tensor,
        cvr_logits: torch.Tensor,
        ctr_labels: torch.Tensor,
        cvr_labels: torch.Tensor,
    ):
        ctr_labels_f = ctr_labels.float()
        cvr_labels_f = cvr_labels.float()

        ctr_loss_vec = nn.functional.binary_cross_entropy_with_logits(
            ctr_logits, ctr_labels_f, reduction="none"
        )

        pCTR = torch.sigmoid(ctr_logits)
        if self.sigmoid == 1:
            pCVR = torch.sigmoid(cvr_logits)
        else:
            pCVR = torch.sigmoid(cvr_logits) * torch.sigmoid(cvr_logits)

        if self.esmm:
            pCTCVR = pCTR * pCVR
            task2_loss_vec = nn.functional.binary_cross_entropy(
                pCTCVR, cvr_labels_f, reduction="none"
            )
        else:
            task2_loss_vec = nn.functional.binary_cross_entropy_with_logits(
                cvr_logits, cvr_labels_f, reduction="none"
            )

        group_masks = {
            "00": (ctr_labels_f < 0.5) & (cvr_labels_f < 0.5),
            "10": (ctr_labels_f > 0.5) & (cvr_labels_f < 0.5),
            "11": (ctr_labels_f > 0.5) & (cvr_labels_f > 0.5),
        }

        batch_size = max(1, int(ctr_labels_f.numel()))
        for suffix, mask in group_masks.items():
            count = mask.float().sum()
            ratio = count / float(batch_size)
            valid = (count > 0).float()

            self.log(f"train_grad_conflict_count_{suffix}", count, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"train_grad_conflict_ratio_{suffix}", ratio, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"train_grad_conflict_valid_{suffix}", valid, on_step=True, on_epoch=True, sync_dist=True)

            if count.item() == 0:
                self.log(f"train_grad_conflict_dot_{suffix}", torch.zeros((), device=ctr_logits.device), on_step=True, on_epoch=True, sync_dist=True)
                self.log(f"train_grad_conflict_cos_{suffix}", torch.zeros((), device=ctr_logits.device), on_step=True, on_epoch=True, sync_dist=True)
                self.log(f"train_grad_conflict_rate_{suffix}", torch.zeros((), device=ctr_logits.device), on_step=True, on_epoch=True, sync_dist=True)
                continue

            ctr_group_loss = (ctr_loss_vec * mask.float()).sum() / count
            task2_group_loss = (task2_loss_vec * mask.float()).sum() / count
            stats = self._compute_conflict_stats(ctr_group_loss, task2_group_loss)
            if stats is None:
                return
            dot_g, cos_g, conflict_g, _ = stats

            self.log(f"train_grad_conflict_dot_{suffix}", dot_g, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"train_grad_conflict_cos_{suffix}", cos_g, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"train_grad_conflict_rate_{suffix}", conflict_g, on_step=True, on_epoch=True, sync_dist=True)

    def _monitor_task_gradient_conflict(
        self,
        loss_a: torch.Tensor,
        loss_b: torch.Tensor,
        ctr_logits: torch.Tensor | None = None,
        cvr_logits: torch.Tensor | None = None,
        ctr_labels: torch.Tensor | None = None,
        cvr_labels: torch.Tensor | None = None,
    ):
        if not self.monitor_grad_conflict or self._grad_conflict_monitor_disabled:
            return

        if int(self.global_step) % self.grad_conflict_interval != 0:
            return

        stats = self._compute_conflict_stats(loss_a, loss_b)
        if stats is None:
            return
        dot, cos, conflict, used_fallback = stats

        self.log("train_grad_conflict_dot", dot, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_grad_conflict_cos", cos, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_grad_conflict_rate", conflict, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_grad_conflict_fallback", used_fallback, on_step=True, on_epoch=True, sync_dist=True)

        if (
            ctr_logits is not None
            and cvr_logits is not None
            and ctr_labels is not None
            and cvr_labels is not None
        ):
            self._monitor_label_group_conflicts(ctr_logits, cvr_logits, ctr_labels, cvr_labels)

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
        if self.monitor_grad_conflict:
            # Engine 需要在 forward 前初始化，才能捕获模块级导数信息
            self._ensure_grad_conflict_engine()
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
        if self.use_asym_proj:
            return self._training_step_asym_proj(batch)
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

    def _compute_cvr_loss_vector(self, ctr_logits, cvr_logits, cvr_labels):
        """逐样本 CVR/CTCVR loss，供按样本类型做梯度操作。"""
        if self.esmm:
            pCTR = torch.sigmoid(ctr_logits)
            if self.sigmoid == 1:
                pCVR = torch.sigmoid(cvr_logits)
            else:
                pCVR = torch.sigmoid(cvr_logits) * torch.sigmoid(cvr_logits)
            pCTCVR = pCTR * pCVR
            return nn.functional.binary_cross_entropy(
                pCTCVR, cvr_labels.float(), reduction="none"
            )
        return nn.functional.binary_cross_entropy_with_logits(
            cvr_logits, cvr_labels.float(), reduction="none"
        )

    def _training_step_standard(self, batch):
        ctr_logits, cvr_logits, ctr_labels, cvr_labels, ctr_loss, cvr_loss, entropy_loss = self._compute_losses(batch)
        self._monitor_task_gradient_conflict(
            ctr_loss, cvr_loss, ctr_logits, cvr_logits, ctr_labels, cvr_labels
        )
        loss = self.ctr_weight * ctr_loss + self.cvr_weight * cvr_loss + self.lambda_entropy * entropy_loss
        self.log("train_loss",         loss,         prog_bar=True)
        self.log("train_ctr_loss",     ctr_loss)
        self.log("train_cvr_loss",     cvr_loss)
        self.log("train_entropy_loss", entropy_loss)
        return loss

    def _training_step_torchjd(self, batch):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        ctr_logits, cvr_logits, ctr_labels, cvr_labels, ctr_loss, cvr_loss, entropy_loss = self._compute_losses(batch)
        self._monitor_task_gradient_conflict(
            ctr_loss, cvr_loss, ctr_logits, cvr_logits, ctr_labels, cvr_labels
        )
        backward([ctr_loss, cvr_loss], aggregator=self.aggregator)
        optimizer.step()
        total_loss = ctr_loss + cvr_loss + self.lambda_entropy * entropy_loss
        self.log("train_loss",         total_loss,   prog_bar=True)
        self.log("train_ctr_loss",     ctr_loss)
        self.log("train_cvr_loss",     cvr_loss)
        self.log("train_entropy_loss", entropy_loss)
        return total_loss

    def _get_asym_proj_params(self):
        """返回进行非对称投影的参数集合。子类可重写。"""
        return []

    def _training_step_asym_proj(self, batch):
        proj_params = [p for p in self._get_asym_proj_params() if p.requires_grad]
        if len(proj_params) == 0:
            raise RuntimeError("use_asym_proj=True 但未提供可投影参数，请重写 _get_asym_proj_params")

        optimizer = self.optimizers()
        optimizer.zero_grad()

        ctr_logits, cvr_logits, ctr_labels, cvr_labels, ctr_loss, cvr_loss, entropy_loss = self._compute_losses(batch)
        self._monitor_task_gradient_conflict(
            ctr_loss, cvr_loss, ctr_logits, cvr_logits, ctr_labels, cvr_labels
        )

        ctr_obj = self.ctr_weight * ctr_loss
        cvr_obj = self.cvr_weight * cvr_loss
        entropy_obj = self.lambda_entropy * entropy_loss

        use_only_10 = bool(self.asym_proj_only_10)
        if use_only_10:
            cvr_loss_vec = self._compute_cvr_loss_vector(ctr_logits, cvr_logits, cvr_labels)
            mask_10 = (ctr_labels > 0.5) & (cvr_labels < 0.5)
            inv_batch = 1.0 / max(1, cvr_loss_vec.numel())
            cvr_loss_10 = cvr_loss_vec[mask_10].sum() * inv_batch
            cvr_loss_other = cvr_loss_vec[~mask_10].sum() * inv_batch
            cvr_obj_proj = self.cvr_weight * cvr_loss_10
            cvr_obj_other = self.cvr_weight * cvr_loss_other
        else:
            cvr_obj_proj = cvr_obj
            cvr_obj_other = torch.zeros((), device=ctr_loss.device)
            mask_10 = None

        ctr_grads_raw = torch.autograd.grad(
            ctr_obj, proj_params, retain_graph=True, allow_unused=True
        )
        cvr_proj_grads_raw = torch.autograd.grad(
            cvr_obj_proj, proj_params, retain_graph=True, allow_unused=True
        )
        if use_only_10:
            cvr_other_grads_raw = torch.autograd.grad(
                cvr_obj_other, proj_params, retain_graph=True, allow_unused=True
            )
        else:
            cvr_other_grads_raw = [None] * len(proj_params)

        ctr_grads = []
        cvr_proj_grads = []
        cvr_other_grads = []
        dot = torch.zeros((), device=ctr_loss.device)
        ctr_norm_sq = torch.zeros((), device=ctr_loss.device)
        cvr_proj_norm_sq = torch.zeros((), device=ctr_loss.device)
        for p, g_ctr, g_cvr_proj, g_cvr_other in zip(
            proj_params, ctr_grads_raw, cvr_proj_grads_raw, cvr_other_grads_raw
        ):
            g_ctr = torch.zeros_like(p) if g_ctr is None else g_ctr
            g_cvr_proj = torch.zeros_like(p) if g_cvr_proj is None else g_cvr_proj
            g_cvr_other = torch.zeros_like(p) if g_cvr_other is None else g_cvr_other
            ctr_grads.append(g_ctr)
            cvr_proj_grads.append(g_cvr_proj)
            cvr_other_grads.append(g_cvr_other)
            dot = dot + (g_cvr_proj * g_ctr).sum()
            ctr_norm_sq = ctr_norm_sq + (g_ctr * g_ctr).sum()
            cvr_proj_norm_sq = cvr_proj_norm_sq + (g_cvr_proj * g_cvr_proj).sum()

        cos = dot / (torch.sqrt(ctr_norm_sq * cvr_proj_norm_sq) + 1e-8)
        apply_proj = (
            (dot.item() < 0.0)
            and (ctr_norm_sq.item() > 0.0)
            and (cvr_proj_norm_sq.item() > 0.0)
            and (cos.item() < -self.asym_proj_tau)
        )

        restore_scale = torch.ones((), device=ctr_loss.device)
        if apply_proj:
            coeff = dot / (ctr_norm_sq + 1e-8)
            pre_proj_norm_sq = cvr_proj_norm_sq
            cvr_proj_grads = [
                g_cvr - self.asym_proj_lambda * coeff * g_ctr
                for g_ctr, g_cvr in zip(ctr_grads, cvr_proj_grads)
            ]
            if self.asym_proj_restore_norm:
                post_proj_norm_sq = torch.zeros((), device=ctr_loss.device)
                for g_cvr in cvr_proj_grads:
                    post_proj_norm_sq = post_proj_norm_sq + (g_cvr * g_cvr).sum()
                if pre_proj_norm_sq.item() > 0.0 and post_proj_norm_sq.item() > 0.0:
                    restore_scale = torch.sqrt(pre_proj_norm_sq / (post_proj_norm_sq + 1e-8))
                    max_scale = max(1.0, float(self.asym_proj_restore_max_scale))
                    restore_scale = torch.clamp(restore_scale, max=max_scale)
                    cvr_proj_grads = [g_cvr * restore_scale for g_cvr in cvr_proj_grads]

        cvr_grads = [
            g_other + g_proj
            for g_other, g_proj in zip(cvr_other_grads, cvr_proj_grads)
        ]

        total_loss = ctr_obj + cvr_obj + entropy_obj
        self.manual_backward(total_loss)

        # 仅替换投影参数上的梯度，其余参数保持常规反传结果
        for p, g_ctr, g_cvr in zip(proj_params, ctr_grads, cvr_grads):
            p.grad = (g_ctr + g_cvr).detach()

        optimizer.step()

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_ctr_loss", ctr_loss)
        self.log("train_cvr_loss", cvr_loss)
        self.log("train_entropy_loss", entropy_loss)
        self.log("train_asym_grad_dot", dot.detach(), on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_asym_grad_cos", cos.detach(), on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_asym_proj_applied", float(apply_proj), on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_asym_proj_restore_scale", restore_scale.detach(), on_step=False, on_epoch=True, sync_dist=True)
        if use_only_10 and mask_10 is not None:
            self.log("train_asym_proj_10_count", mask_10.float().sum(), on_step=False, on_epoch=True, sync_dist=True)
            self.log("train_asym_proj_10_ratio", mask_10.float().mean(), on_step=False, on_epoch=True, sync_dist=True)
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
        use_asym_proj     : bool = False,
        asym_proj_lambda  : float = 1.0,
        asym_proj_tau     : float = 0.0,
        asym_proj_only_10 : bool = False,
        asym_proj_restore_norm: bool = False,
        asym_proj_restore_max_scale: float = 5.0,
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
        self.use_asym_proj = use_asym_proj
        self.asym_proj_lambda = asym_proj_lambda
        self.asym_proj_tau = asym_proj_tau
        self.asym_proj_only_10 = asym_proj_only_10
        self.asym_proj_restore_norm = asym_proj_restore_norm
        self.asym_proj_restore_max_scale = asym_proj_restore_max_scale

        if use_torchjd and use_asym_proj:
            raise ValueError("use_torchjd 与 use_asym_proj 不能同时启用")

        if use_torchjd or use_asym_proj:
            self.automatic_optimization = False

        if use_torchjd:
            self.aggregator = _create_aggregator(aggregation_method)

        # 梯度冲突监控（默认关闭，可由 train_rank.py 动态开启）
        self.configure_grad_conflict_monitor(enabled=False, interval=100)

    def configure_grad_conflict_monitor(self, enabled: bool = False, interval: int = 100):
        self.monitor_grad_conflict = bool(enabled)
        self.grad_conflict_interval = max(1, int(interval))
        self._grad_conflict_engine = None
        self._grad_conflict_monitor_disabled = False
        self._grad_conflict_use_autograd_fallback = False

    @staticmethod
    def _collect_leaf_param_modules(root_module: nn.Module):
        modules = []
        for module in root_module.modules():
            if len(list(module.children())) != 0:
                continue
            if any(p.requires_grad for p in module.parameters(recurse=False)):
                modules.append(module)
        return modules

    def _get_grad_conflict_monitor_modules(self):
        # 默认监控整个模型的叶子参数模块；子类可按共享层覆写
        return self._collect_leaf_param_modules(self)

    def _ensure_grad_conflict_engine(self):
        if self._grad_conflict_engine is not None:
            return True
        modules = self._get_grad_conflict_monitor_modules()
        if len(modules) == 0:
            self._grad_conflict_monitor_disabled = True
            return False
        self._grad_conflict_engine = Engine(*modules, batch_dim=None)
        return True

    def _get_grad_conflict_monitor_params(self):
        params = []
        seen = set()
        for module in self._get_grad_conflict_monitor_modules():
            for p in module.parameters(recurse=False):
                if not p.requires_grad:
                    continue
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                params.append(p)
        return params

    def _compute_conflict_stats(self, loss_a: torch.Tensor, loss_b: torch.Tensor):
        used_fallback = 0.0
        dot = None
        norm_a_sq = None
        norm_b_sq = None

        if not self._grad_conflict_use_autograd_fallback:
            if self._grad_conflict_engine is None:
                if not self._ensure_grad_conflict_engine():
                    return None
                # 本步 forward 已结束，首次初始化后从下一步开始监控
                return None
            try:
                gram = self._grad_conflict_engine.compute_gramian(torch.stack([loss_a, loss_b]))
                dot = gram[0, 1]
                norm_a_sq = gram[0, 0].clamp_min(0.0)
                norm_b_sq = gram[1, 1].clamp_min(0.0)
            except Exception:
                self._grad_conflict_use_autograd_fallback = True
                self._grad_conflict_engine = None
                used_fallback = 1.0

        if self._grad_conflict_use_autograd_fallback:
            try:
                params = self._get_grad_conflict_monitor_params()
                if len(params) == 0:
                    self._grad_conflict_monitor_disabled = True
                    return None
                grads_a_raw = torch.autograd.grad(
                    loss_a, params, retain_graph=True, allow_unused=True
                )
                grads_b_raw = torch.autograd.grad(
                    loss_b, params, retain_graph=True, allow_unused=True
                )
                dot = torch.zeros((), device=loss_a.device)
                norm_a_sq = torch.zeros((), device=loss_a.device)
                norm_b_sq = torch.zeros((), device=loss_a.device)
                for p, g_a, g_b in zip(params, grads_a_raw, grads_b_raw):
                    g_a = torch.zeros_like(p) if g_a is None else g_a
                    g_b = torch.zeros_like(p) if g_b is None else g_b
                    dot = dot + (g_a * g_b).sum()
                    norm_a_sq = norm_a_sq + (g_a * g_a).sum()
                    norm_b_sq = norm_b_sq + (g_b * g_b).sum()
            except Exception:
                self._grad_conflict_monitor_disabled = True
                return None

        if dot is None or norm_a_sq is None or norm_b_sq is None:
            self._grad_conflict_monitor_disabled = True
            return None

        cos = dot / (torch.sqrt(norm_a_sq * norm_b_sq) + 1e-8)
        conflict = (dot < 0).float()
        return dot.detach(), cos.detach(), conflict.detach(), used_fallback

    def _monitor_label_group_conflicts(
        self,
        ctr_logits: torch.Tensor,
        cvr_logits: torch.Tensor,
        ctr_labels: torch.Tensor,
        cvr_labels: torch.Tensor,
    ):
        ctr_labels_f = ctr_labels.float()
        cvr_labels_f = cvr_labels.float()

        ctr_loss_vec = nn.functional.binary_cross_entropy_with_logits(
            ctr_logits, ctr_labels_f, reduction="none"
        )

        pCTR = torch.sigmoid(ctr_logits)
        if self.sigmoid == 1:
            pCVR = torch.sigmoid(cvr_logits)
        else:
            pCVR = torch.sigmoid(cvr_logits) * torch.sigmoid(cvr_logits)

        if self.esmm:
            pCTCVR = pCTR * pCVR
            task2_loss_vec = nn.functional.binary_cross_entropy(
                pCTCVR, cvr_labels_f, reduction="none"
            )
        else:
            task2_loss_vec = nn.functional.binary_cross_entropy_with_logits(
                cvr_logits, cvr_labels_f, reduction="none"
            )

        group_masks = {
            "00": (ctr_labels_f < 0.5) & (cvr_labels_f < 0.5),
            "10": (ctr_labels_f > 0.5) & (cvr_labels_f < 0.5),
            "11": (ctr_labels_f > 0.5) & (cvr_labels_f > 0.5),
        }

        batch_size = max(1, int(ctr_labels_f.numel()))
        for suffix, mask in group_masks.items():
            count = mask.float().sum()
            ratio = count / float(batch_size)
            valid = (count > 0).float()

            self.log(f"train_grad_conflict_count_{suffix}", count, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"train_grad_conflict_ratio_{suffix}", ratio, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"train_grad_conflict_valid_{suffix}", valid, on_step=True, on_epoch=True, sync_dist=True)

            if count.item() == 0:
                self.log(f"train_grad_conflict_dot_{suffix}", torch.zeros((), device=ctr_logits.device), on_step=True, on_epoch=True, sync_dist=True)
                self.log(f"train_grad_conflict_cos_{suffix}", torch.zeros((), device=ctr_logits.device), on_step=True, on_epoch=True, sync_dist=True)
                self.log(f"train_grad_conflict_rate_{suffix}", torch.zeros((), device=ctr_logits.device), on_step=True, on_epoch=True, sync_dist=True)
                continue

            ctr_group_loss = (ctr_loss_vec * mask.float()).sum() / count
            task2_group_loss = (task2_loss_vec * mask.float()).sum() / count
            stats = self._compute_conflict_stats(ctr_group_loss, task2_group_loss)
            if stats is None:
                return
            dot_g, cos_g, conflict_g, _ = stats

            self.log(f"train_grad_conflict_dot_{suffix}", dot_g, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"train_grad_conflict_cos_{suffix}", cos_g, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"train_grad_conflict_rate_{suffix}", conflict_g, on_step=True, on_epoch=True, sync_dist=True)

    def _monitor_task_gradient_conflict(
        self,
        loss_a: torch.Tensor,
        loss_b: torch.Tensor,
        ctr_logits: torch.Tensor | None = None,
        cvr_logits: torch.Tensor | None = None,
        ctr_labels: torch.Tensor | None = None,
        cvr_labels: torch.Tensor | None = None,
    ):
        if not self.monitor_grad_conflict or self._grad_conflict_monitor_disabled:
            return

        if int(self.global_step) % self.grad_conflict_interval != 0:
            return

        stats = self._compute_conflict_stats(loss_a, loss_b)
        if stats is None:
            return
        dot, cos, conflict, used_fallback = stats

        self.log("train_grad_conflict_dot", dot, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_grad_conflict_cos", cos, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_grad_conflict_rate", conflict, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_grad_conflict_fallback", used_fallback, on_step=True, on_epoch=True, sync_dist=True)

        if (
            ctr_logits is not None
            and cvr_logits is not None
            and ctr_labels is not None
            and cvr_labels is not None
        ):
            self._monitor_label_group_conflicts(ctr_logits, cvr_logits, ctr_labels, cvr_labels)

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
        if self.monitor_grad_conflict:
            # Engine 需要在 forward 前初始化，才能捕获模块级导数信息
            self._ensure_grad_conflict_engine()
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
        if self.use_asym_proj:
            return self._training_step_asym_proj(batch)
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
        ctr_logits, cvr_logits, ctr_labels, cvr_labels, ctr_loss, cvr_loss, entropy_loss = self._compute_losses(batch)
        self._monitor_task_gradient_conflict(
            ctr_loss, cvr_loss, ctr_logits, cvr_logits, ctr_labels, cvr_labels
        )
        loss = self.ctr_weight * ctr_loss + self.cvr_weight * cvr_loss + self.lambda_entropy * entropy_loss
        self.log("train_loss",         loss,         prog_bar=True)
        self.log("train_ctr_loss",     ctr_loss)
        self.log("train_cvr_loss",     cvr_loss)
        self.log("train_entropy_loss", entropy_loss)
        return loss

    def _training_step_torchjd(self, batch):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        ctr_logits, cvr_logits, ctr_labels, cvr_labels, ctr_loss, cvr_loss, entropy_loss = self._compute_losses(batch)
        self._monitor_task_gradient_conflict(
            ctr_loss, cvr_loss, ctr_logits, cvr_logits, ctr_labels, cvr_labels
        )
        backward([ctr_loss, cvr_loss], aggregator=self.aggregator)
        optimizer.step()
        total_loss = ctr_loss + cvr_loss + self.lambda_entropy * entropy_loss
        self.log("train_loss",         total_loss,   prog_bar=True)
        self.log("train_ctr_loss",     ctr_loss)
        self.log("train_cvr_loss",     cvr_loss)
        self.log("train_entropy_loss", entropy_loss)
        return total_loss

    def _get_asym_proj_params(self):
        """返回进行非对称投影的参数集合。子类可重写。"""
        return []

    def _training_step_asym_proj(self, batch):
        proj_params = [p for p in self._get_asym_proj_params() if p.requires_grad]
        if len(proj_params) == 0:
            raise RuntimeError("use_asym_proj=True 但未提供可投影参数，请重写 _get_asym_proj_params")

        optimizer = self.optimizers()
        optimizer.zero_grad()

        ctr_logits, cvr_logits, ctr_labels, cvr_labels, ctr_loss, cvr_loss, entropy_loss = self._compute_losses(batch)
        self._monitor_task_gradient_conflict(
            ctr_loss, cvr_loss, ctr_logits, cvr_logits, ctr_labels, cvr_labels
        )

        ctr_obj = self.ctr_weight * ctr_loss
        cvr_obj = self.cvr_weight * cvr_loss
        entropy_obj = self.lambda_entropy * entropy_loss

        use_only_10 = bool(self.asym_proj_only_10)
        if use_only_10:
            cvr_loss_vec = self._compute_cvr_loss_vector(ctr_logits, cvr_logits, cvr_labels)
            mask_10 = (ctr_labels > 0.5) & (cvr_labels < 0.5)
            inv_batch = 1.0 / max(1, cvr_loss_vec.numel())
            cvr_loss_10 = cvr_loss_vec[mask_10].sum() * inv_batch
            cvr_loss_other = cvr_loss_vec[~mask_10].sum() * inv_batch
            cvr_obj_proj = self.cvr_weight * cvr_loss_10
            cvr_obj_other = self.cvr_weight * cvr_loss_other
        else:
            cvr_obj_proj = cvr_obj
            cvr_obj_other = torch.zeros((), device=ctr_loss.device)
            mask_10 = None

        ctr_grads_raw = torch.autograd.grad(
            ctr_obj, proj_params, retain_graph=True, allow_unused=True
        )
        cvr_proj_grads_raw = torch.autograd.grad(
            cvr_obj_proj, proj_params, retain_graph=True, allow_unused=True
        )
        if use_only_10:
            cvr_other_grads_raw = torch.autograd.grad(
                cvr_obj_other, proj_params, retain_graph=True, allow_unused=True
            )
        else:
            cvr_other_grads_raw = [None] * len(proj_params)

        ctr_grads = []
        cvr_proj_grads = []
        cvr_other_grads = []
        dot = torch.zeros((), device=ctr_loss.device)
        ctr_norm_sq = torch.zeros((), device=ctr_loss.device)
        cvr_proj_norm_sq = torch.zeros((), device=ctr_loss.device)
        for p, g_ctr, g_cvr_proj, g_cvr_other in zip(
            proj_params, ctr_grads_raw, cvr_proj_grads_raw, cvr_other_grads_raw
        ):
            g_ctr = torch.zeros_like(p) if g_ctr is None else g_ctr
            g_cvr_proj = torch.zeros_like(p) if g_cvr_proj is None else g_cvr_proj
            g_cvr_other = torch.zeros_like(p) if g_cvr_other is None else g_cvr_other
            ctr_grads.append(g_ctr)
            cvr_proj_grads.append(g_cvr_proj)
            cvr_other_grads.append(g_cvr_other)
            dot = dot + (g_cvr_proj * g_ctr).sum()
            ctr_norm_sq = ctr_norm_sq + (g_ctr * g_ctr).sum()
            cvr_proj_norm_sq = cvr_proj_norm_sq + (g_cvr_proj * g_cvr_proj).sum()

        cos = dot / (torch.sqrt(ctr_norm_sq * cvr_proj_norm_sq) + 1e-8)
        apply_proj = (
            (dot.item() < 0.0)
            and (ctr_norm_sq.item() > 0.0)
            and (cvr_proj_norm_sq.item() > 0.0)
            and (cos.item() < -self.asym_proj_tau)
        )

        restore_scale = torch.ones((), device=ctr_loss.device)
        if apply_proj:
            coeff = dot / (ctr_norm_sq + 1e-8)
            pre_proj_norm_sq = cvr_proj_norm_sq
            cvr_proj_grads = [
                g_cvr - self.asym_proj_lambda * coeff * g_ctr
                for g_ctr, g_cvr in zip(ctr_grads, cvr_proj_grads)
            ]
            if self.asym_proj_restore_norm:
                post_proj_norm_sq = torch.zeros((), device=ctr_loss.device)
                for g_cvr in cvr_proj_grads:
                    post_proj_norm_sq = post_proj_norm_sq + (g_cvr * g_cvr).sum()
                if pre_proj_norm_sq.item() > 0.0 and post_proj_norm_sq.item() > 0.0:
                    restore_scale = torch.sqrt(pre_proj_norm_sq / (post_proj_norm_sq + 1e-8))
                    max_scale = max(1.0, float(self.asym_proj_restore_max_scale))
                    restore_scale = torch.clamp(restore_scale, max=max_scale)
                    cvr_proj_grads = [g_cvr * restore_scale for g_cvr in cvr_proj_grads]

        cvr_grads = [
            g_other + g_proj
            for g_other, g_proj in zip(cvr_other_grads, cvr_proj_grads)
        ]

        total_loss = ctr_obj + cvr_obj + entropy_obj
        self.manual_backward(total_loss)

        # 仅替换投影参数上的梯度，其余参数保持常规反传结果
        for p, g_ctr, g_cvr in zip(proj_params, ctr_grads, cvr_grads):
            p.grad = (g_ctr + g_cvr).detach()

        optimizer.step()

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_ctr_loss", ctr_loss)
        self.log("train_cvr_loss", cvr_loss)
        self.log("train_entropy_loss", entropy_loss)
        self.log("train_asym_grad_dot", dot.detach(), on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_asym_grad_cos", cos.detach(), on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_asym_proj_applied", float(apply_proj), on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_asym_proj_restore_scale", restore_scale.detach(), on_step=False, on_epoch=True, sync_dist=True)
        if use_only_10 and mask_10 is not None:
            self.log("train_asym_proj_10_count", mask_10.float().sum(), on_step=False, on_epoch=True, sync_dist=True)
            self.log("train_asym_proj_10_ratio", mask_10.float().mean(), on_step=False, on_epoch=True, sync_dist=True)
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
        use_entropy_reg     : bool      = False,
        lambda_entropy      : float     = 0.01,
        use_ema             : bool      = False,
        ema_decay           : float     = 0.999,
        use_dcn             : bool      = False,
        dcn_num_layers      : int       = 2,
        dcn_dropout         : float     = 0.0,
        dcn_rank            : int       = 0,
        use_asym_proj       : bool      = False,
        asym_proj_lambda    : float     = 1.0,
        asym_proj_tau       : float     = 0.0,
        asym_proj_only_10   : bool      = False,
        asym_proj_restore_norm: bool    = False,
        asym_proj_restore_max_scale: float = 5.0,
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

        self._init_torchjd(
            use_torchjd,
            aggregation_method,
            ctr_weight,
            cvr_weight,
            esmm,
            sigmoid,
            use_entropy_reg=use_entropy_reg,
            lambda_entropy=lambda_entropy,
            use_ema=use_ema,
            ema_decay=ema_decay,
            use_asym_proj=use_asym_proj,
            asym_proj_lambda=asym_proj_lambda,
            asym_proj_tau=asym_proj_tau,
            asym_proj_only_10=asym_proj_only_10,
            asym_proj_restore_norm=asym_proj_restore_norm,
            asym_proj_restore_max_scale=asym_proj_restore_max_scale,
        )
        self._init_metrics()

    def _forward_logits(self, x: torch.Tensor, return_gate_entropy: bool = False):
        shared_out = self.shared_bottom(x)
        return self.ctr_tower(shared_out), self.cvr_tower(shared_out)

    def _get_grad_conflict_monitor_modules(self):
        # ShareBottom 仅监控共享底层参数上的任务冲突
        return self._collect_leaf_param_modules(self.shared_bottom)

    def _get_asym_proj_params(self):
        # ShareBottom 的梯度冲突主要发生在共享底层上
        return list(self.shared_bottom.parameters())

    def _compute_cvr_loss_vector(self, ctr_logits, cvr_logits, cvr_labels):
        if self.esmm:
            pCTR = torch.sigmoid(ctr_logits)
            if self.sigmoid == 1:
                pCVR = torch.sigmoid(cvr_logits)
            else:
                pCVR = torch.sigmoid(cvr_logits) * torch.sigmoid(cvr_logits)
            pCTCVR = pCTR * pCVR
            return nn.functional.binary_cross_entropy(
                pCTCVR, cvr_labels.float(), reduction="none"
            )
        return nn.functional.binary_cross_entropy_with_logits(
            cvr_logits, cvr_labels.float(), reduction="none"
        )

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
        use_asym_proj       : bool      = False,
        asym_proj_lambda    : float     = 1.0,
        asym_proj_tau       : float     = 0.0,
        asym_proj_only_10   : bool      = False,
        asym_proj_restore_norm: bool    = False,
        asym_proj_restore_max_scale: float = 5.0,
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

        self._init_torchjd(
            use_torchjd,
            aggregation_method,
            ctr_weight,
            cvr_weight,
            esmm,
            sigmoid,
            use_entropy_reg,
            lambda_entropy,
            use_ema=use_ema,
            ema_decay=ema_decay,
            use_asym_proj=use_asym_proj,
            asym_proj_lambda=asym_proj_lambda,
            asym_proj_tau=asym_proj_tau,
            asym_proj_only_10=asym_proj_only_10,
            asym_proj_restore_norm=asym_proj_restore_norm,
            asym_proj_restore_max_scale=asym_proj_restore_max_scale,
        )
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

    def _get_asym_proj_params(self):
        # MOE 的共享表示层 = 专家网络 + 共享门控
        return list(self.experts.parameters()) + list(self.gate.parameters())

    def _compute_cvr_loss_vector(self, ctr_logits, cvr_logits, cvr_labels):
        if self.esmm:
            pCTR = torch.sigmoid(ctr_logits)
            if self.sigmoid == 1:
                pCVR = torch.sigmoid(cvr_logits)
            else:
                pCVR = torch.sigmoid(cvr_logits) * torch.sigmoid(cvr_logits)
            pCTCVR = pCTR * pCVR
            return nn.functional.binary_cross_entropy(
                pCTCVR, cvr_labels.float(), reduction="none"
            )
        return nn.functional.binary_cross_entropy_with_logits(
            cvr_logits, cvr_labels.float(), reduction="none"
        )

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
        use_asym_proj       : bool      = False,
        asym_proj_lambda    : float     = 1.0,
        asym_proj_tau       : float     = 0.0,
        asym_proj_only_10   : bool      = False,
        asym_proj_restore_norm: bool    = False,
        asym_proj_restore_max_scale: float = 5.0,
        use_hard_sample     : bool      = False,
        hard_sample_ratio   : float     = 0.2,
        hard_sample_weight  : float     = 2.0,
        hard_sample_warmup_epochs: int  = 1,
    ):
        pl.LightningModule.__init__(self)
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.use_hard_sample = use_hard_sample
        self.hard_sample_ratio = hard_sample_ratio
        self.hard_sample_weight = hard_sample_weight
        self.hard_sample_warmup_epochs = hard_sample_warmup_epochs

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

        self._init_torchjd(
            use_torchjd,
            aggregation_method,
            ctr_weight,
            cvr_weight,
            esmm,
            sigmoid,
            use_entropy_reg,
            lambda_entropy,
            use_ema=use_ema,
            ema_decay=ema_decay,
            use_asym_proj=use_asym_proj,
            asym_proj_lambda=asym_proj_lambda,
            asym_proj_tau=asym_proj_tau,
            asym_proj_only_10=asym_proj_only_10,
            asym_proj_restore_norm=asym_proj_restore_norm,
            asym_proj_restore_max_scale=asym_proj_restore_max_scale,
        )
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

    def _get_asym_proj_params(self):
        # MMOE 仅在共享表示层（DCN 交叉层 + 共享专家）上做投影
        params = []
        if self.feature_cross is not None:
            params.extend(list(self.feature_cross.parameters()))
        params.extend(list(self.experts.parameters()))
        return params

    def _build_hard_sample_weights(self, difficulty: torch.Tensor):
        """
        基于 batch 内难度分数（越大越难）做 top-ratio 样本挖掘并返回权重。
        """
        n = difficulty.numel()
        if (
            (not self.use_hard_sample)
            or (self.current_epoch < self.hard_sample_warmup_epochs)
            or n == 0
        ):
            return torch.ones_like(difficulty), 0.0

        ratio = float(min(max(self.hard_sample_ratio, 0.0), 1.0))
        if ratio <= 0.0:
            return torch.ones_like(difficulty), 0.0

        k = max(1, int(n * ratio))
        k = min(k, n)
        top_idx = torch.topk(difficulty, k=k, largest=True).indices

        weights = torch.ones_like(difficulty)
        hard_w = max(float(self.hard_sample_weight), 1.0)
        weights[top_idx] = hard_w
        return weights, float(k) / float(n)

    def _compute_losses(self, batch):
        inputs, labels = batch
        ctr_labels = labels["click"]
        cvr_labels = labels["purchase"]

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

        # 逐样本 loss，便于 hard sample 挖掘和重加权
        ctr_loss_vec = nn.functional.binary_cross_entropy_with_logits(
            ctr_logits, ctr_labels.float(), reduction="none"
        )

        if self.esmm:
            pCTCVR = pCTR * pCVR
            cvr_loss_vec = nn.functional.binary_cross_entropy(
                pCTCVR, cvr_labels.float(), reduction="none"
            )
        else:
            cvr_loss_vec = nn.functional.binary_cross_entropy_with_logits(
                cvr_logits, cvr_labels.float(), reduction="none"
            )

        difficulty = (self.ctr_weight * ctr_loss_vec + self.cvr_weight * cvr_loss_vec).detach()
        hard_weights, hard_ratio = self._build_hard_sample_weights(difficulty)
        denom = hard_weights.sum().clamp_min(1.0)

        ctr_loss = (ctr_loss_vec * hard_weights).sum() / denom
        cvr_loss = (cvr_loss_vec * hard_weights).sum() / denom

        if self.use_entropy_reg and gate_entropy is not None:
            entropy_loss = -gate_entropy.mean()
        else:
            entropy_loss = torch.tensor(0.0, device=ctr_logits.device)

        if self.training and self.use_hard_sample and (self._trainer is not None):
            self.log("train_hard_ratio", hard_ratio, on_step=False, on_epoch=True)

        return ctr_logits, cvr_logits, ctr_labels, cvr_labels, ctr_loss, cvr_loss, entropy_loss

    def _compute_cvr_loss_vector(self, ctr_logits, cvr_logits, cvr_labels):
        if self.esmm:
            pCTR = torch.sigmoid(ctr_logits)
            if self.sigmoid == 1:
                pCVR = torch.sigmoid(cvr_logits)
            else:
                pCVR = torch.sigmoid(cvr_logits) * torch.sigmoid(cvr_logits)
            return nn.functional.binary_cross_entropy(
                pCTR * pCVR, cvr_labels.float(), reduction="none"
            )
        return nn.functional.binary_cross_entropy_with_logits(
            cvr_logits, cvr_labels.float(), reduction="none"
        )

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
        use_entropy_reg      : bool      = False,
        lambda_entropy       : float     = 0.01,
        use_ema              : bool      = False,
        ema_decay            : float     = 0.999,
        use_dcn              : bool      = False,
        dcn_num_layers       : int       = 2,
        dcn_dropout          : float     = 0.0,
        dcn_rank             : int       = 0,
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
                Expert(
                    lvl_in,
                    expert_hidden_dims,
                    dropout,
                    use_dcn=use_dcn,
                    dcn_num_layers=dcn_num_layers,
                    dcn_dropout=dcn_dropout,
                    dcn_rank=dcn_rank,
                )
                for _ in range(num_specific_experts)
            ]))
            self.cvr_experts.append(nn.ModuleList([
                Expert(
                    lvl_in,
                    expert_hidden_dims,
                    dropout,
                    use_dcn=use_dcn,
                    dcn_num_layers=dcn_num_layers,
                    dcn_dropout=dcn_dropout,
                    dcn_rank=dcn_rank,
                )
                for _ in range(num_specific_experts)
            ]))
            self.shared_experts.append(nn.ModuleList([
                Expert(
                    lvl_in,
                    expert_hidden_dims,
                    dropout,
                    use_dcn=use_dcn,
                    dcn_num_layers=dcn_num_layers,
                    dcn_dropout=dcn_dropout,
                    dcn_rank=dcn_rank,
                )
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

        self._init_torchjd(
            use_torchjd,
            aggregation_method,
            ctr_weight,
            cvr_weight,
            esmm,
            sigmoid,
            use_entropy_reg=use_entropy_reg,
            lambda_entropy=lambda_entropy,
            use_ema=use_ema,
            ema_decay=ema_decay,
        )
        self._init_metrics()

    def _forward_logits(self, x: torch.Tensor, return_gate_entropy: bool = False):
        # 各路径的当前输入（第 0 层统一为原始编码 x）
        ctr_in    = x
        cvr_in    = x
        shared_in = x
        gate_entropies = []

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
            if return_gate_entropy:
                ctr_weights_raw, ctr_entropy = self.ctr_gates[level](ctr_in, return_entropy=True)
                gate_entropies.append(ctr_entropy)
                ctr_weights = ctr_weights_raw.unsqueeze(-1)                   # [B, ns+nsh, 1]
            else:
                ctr_weights = self.ctr_gates[level](ctr_in).unsqueeze(-1)     # [B, ns+nsh, 1]
            new_ctr        = torch.sum(ctr_candidates * ctr_weights,  dim=1)  # [B, d]

            # CVR 提取：task-specific + shared
            cvr_candidates = torch.cat([cvr_specific_out, shared_out], dim=1)
            if return_gate_entropy:
                cvr_weights_raw, cvr_entropy = self.cvr_gates[level](cvr_in, return_entropy=True)
                gate_entropies.append(cvr_entropy)
                cvr_weights = cvr_weights_raw.unsqueeze(-1)
            else:
                cvr_weights = self.cvr_gates[level](cvr_in).unsqueeze(-1)
            new_cvr        = torch.sum(cvr_candidates * cvr_weights,  dim=1)

            # Shared 提取：ctr_specific + cvr_specific + shared（非最后层）
            if level < self.num_levels - 1:
                shared_candidates = torch.cat(
                    [ctr_specific_out, cvr_specific_out, shared_out], dim=1)  # [B, 2ns+nsh, d]
                if return_gate_entropy:
                    shared_weights_raw, shared_entropy = self.shared_gates[level](shared_in, return_entropy=True)
                    gate_entropies.append(shared_entropy)
                    shared_weights = shared_weights_raw.unsqueeze(-1)
                else:
                    shared_weights = self.shared_gates[level](shared_in).unsqueeze(-1)
                shared_in         = torch.sum(shared_candidates * shared_weights, dim=1)

            ctr_in = new_ctr
            cvr_in = new_cvr

        ctr_logit = self.ctr_tower(ctr_in)
        cvr_logit = self.cvr_tower(cvr_in)
        if return_gate_entropy:
            if len(gate_entropies) == 0:
                avg_entropy = torch.zeros(ctr_logit.size(0), device=ctr_logit.device)
            else:
                avg_entropy = torch.stack(gate_entropies, dim=0).mean(dim=0)
            return ctr_logit, cvr_logit, avg_entropy
        return ctr_logit, cvr_logit


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
        ) = self._compute_losses(batch, for_training=True)

        self._monitor_task_gradient_conflict(
            ctr_loss, cvr_loss, ctr_logits, cvr_logits, ctr_labels, cvr_labels
        )
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
        ) = self._compute_losses(batch, for_training=True)

        self._monitor_task_gradient_conflict(
            ctr_loss, cvr_loss, ctr_logits, cvr_logits, ctr_labels, cvr_labels
        )
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
