"""
Tenrec: ESMM + MMOE

任务:
    click, like, collect

概率关系:
    p(ct_like)   = p(click) * p(like|click)
    p(ct_collect)= p(click) * p(collect|click)
"""
from __future__ import annotations

from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import AUROC


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Gate(nn.Module):
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, num_experts), nn.Softmax(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Tower(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TenrecMMOEESMMModel(pl.LightningModule):
    def __init__(
        self,
        sparse_feature_names: List[str],
        sparse_feature_dims: Dict[str, int],
        dense_feature_names: List[str],
        embedding_dim: int = 32,
        num_experts: int = 6,
        expert_hidden_dims: List[int] = [256, 128],
        tower_hidden_dims: List[int] = [64],
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        w_click: float = 1.0,
        w_like: float = 1.0,
        w_collect: float = 1.0,
        use_asym_proj: bool = False,
        asym_proj_lambda: float = 1.0,
        asym_proj_tau: float = 0.0,
        asym_proj_restore_norm: bool = False,
        asym_proj_restore_max_scale: float = 5.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.sparse_feature_names = sparse_feature_names
        self.dense_feature_names = dense_feature_names
        self.w_click = w_click
        self.w_like = w_like
        self.w_collect = w_collect
        self.use_asym_proj = use_asym_proj
        self.asym_proj_lambda = asym_proj_lambda
        self.asym_proj_tau = asym_proj_tau
        self.asym_proj_restore_norm = asym_proj_restore_norm
        self.asym_proj_restore_max_scale = asym_proj_restore_max_scale

        if self.use_asym_proj:
            self.automatic_optimization = False

        self.embeddings = nn.ModuleDict(
            {
                f: nn.Embedding(sparse_feature_dims[f], embedding_dim)
                for f in sparse_feature_names
            }
        )
        input_dim = len(sparse_feature_names) * embedding_dim + len(dense_feature_names)

        self.experts = nn.ModuleList(
            [_MLP(input_dim, expert_hidden_dims, dropout) for _ in range(num_experts)]
        )
        expert_out_dim = expert_hidden_dims[-1]

        self.click_gate = _Gate(input_dim, num_experts)
        self.like_gate = _Gate(input_dim, num_experts)
        self.collect_gate = _Gate(input_dim, num_experts)

        self.click_tower = _Tower(expert_out_dim, tower_hidden_dims, dropout)
        self.like_tower = _Tower(expert_out_dim, tower_hidden_dims, dropout)
        self.collect_tower = _Tower(expert_out_dim, tower_hidden_dims, dropout)

        self.val_click_auc = AUROC(task="binary")
        self.val_like_auc = AUROC(task="binary")
        self.val_collect_auc = AUROC(task="binary")

        self.test_click_auc = AUROC(task="binary")
        self.test_like_auc = AUROC(task="binary")
        self.test_collect_auc = AUROC(task="binary")

    def _encode_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for f in self.sparse_feature_names:
            parts.append(self.embeddings[f](inputs[f].long()))
        for f in self.dense_feature_names:
            x = inputs[f]
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            parts.append(x)
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _mix_experts(expert_outs: torch.Tensor, gate_weight: torch.Tensor) -> torch.Tensor:
        # expert_outs: [B, E, D], gate_weight: [B, E]
        return torch.sum(expert_outs * gate_weight.unsqueeze(-1), dim=1)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        x = self._encode_features(inputs)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)

        click_repr = self._mix_experts(expert_outs, self.click_gate(x))
        like_repr = self._mix_experts(expert_outs, self.like_gate(x))
        collect_repr = self._mix_experts(expert_outs, self.collect_gate(x))

        click_logit = self.click_tower(click_repr)
        like_logit = self.like_tower(like_repr)      # p(like|click) 的 logit
        collect_logit = self.collect_tower(collect_repr)  # p(collect|click) 的 logit
        return click_logit, like_logit, collect_logit

    def _compute_losses(self, batch):
        inputs, labels = batch
        click = labels["click"].float()
        like = labels["like"].float()
        collect = labels["collect"].float()

        click_logit, like_logit, collect_logit = self(inputs)
        p_click = torch.sigmoid(click_logit)
        p_like_cond = torch.sigmoid(like_logit)
        p_collect_cond = torch.sigmoid(collect_logit)

        p_ct_like = p_click * p_like_cond
        p_ct_collect = p_click * p_collect_cond

        click_loss = nn.functional.binary_cross_entropy_with_logits(click_logit, click)
        like_loss = nn.functional.binary_cross_entropy(p_ct_like, like)
        collect_loss = nn.functional.binary_cross_entropy(p_ct_collect, collect)

        total = (
            self.w_click * click_loss
            + self.w_like * like_loss
            + self.w_collect * collect_loss
        )
        return (
            total,
            click_loss,
            like_loss,
            collect_loss,
            p_click,
            p_ct_like,
            p_ct_collect,
            click,
            like,
            collect,
        )

    def training_step(self, batch, batch_idx):
        if self.use_asym_proj:
            return self._training_step_asym_proj(batch)
        total, click_loss, like_loss, collect_loss, *_ = self._compute_losses(batch)
        self.log("train_loss", total, prog_bar=True)
        self.log("train_click_loss", click_loss)
        self.log("train_like_loss", like_loss)
        self.log("train_collect_loss", collect_loss)
        return total

    def _get_shared_params(self):
        # asymmetric projection 只作用在共享表示层参数上
        return list(self.embeddings.parameters()) + list(self.experts.parameters())

    @staticmethod
    def _normalize_grads(params, grads_raw):
        grads = []
        for p, g in zip(params, grads_raw):
            grads.append(torch.zeros_like(p) if g is None else g)
        return grads

    @staticmethod
    def _masked_mean(loss_vec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.float()
        return (loss_vec * weight).sum() / torch.clamp(weight.sum(), min=1.0)

    def _project_task_grad(self, g_task, g_click, device):
        dot = torch.zeros((), device=device)
        task_norm_sq = torch.zeros((), device=device)
        click_norm_sq = torch.zeros((), device=device)
        for gt, gc in zip(g_task, g_click):
            dot = dot + (gt * gc).sum()
            task_norm_sq = task_norm_sq + (gt * gt).sum()
            click_norm_sq = click_norm_sq + (gc * gc).sum()

        cos = dot / (torch.sqrt(task_norm_sq * click_norm_sq) + 1e-8)
        apply = (
            (dot.item() < 0.0)
            and (task_norm_sq.item() > 0.0)
            and (click_norm_sq.item() > 0.0)
            and (cos.item() < -self.asym_proj_tau)
        )
        restore_scale = torch.ones((), device=device)
        if not apply:
            return g_task, dot.detach(), cos.detach(), 0.0, restore_scale

        coeff = dot / (click_norm_sq + 1e-8)
        pre_proj_norm_sq = task_norm_sq
        proj = [
            gt - self.asym_proj_lambda * coeff * gc
            for gt, gc in zip(g_task, g_click)
        ]
        if self.asym_proj_restore_norm:
            post_proj_norm_sq = torch.zeros((), device=device)
            for g in proj:
                post_proj_norm_sq = post_proj_norm_sq + (g * g).sum()
            if pre_proj_norm_sq.item() > 0.0 and post_proj_norm_sq.item() > 0.0:
                restore_scale = torch.sqrt(pre_proj_norm_sq / (post_proj_norm_sq + 1e-8))
                max_scale = max(1.0, float(self.asym_proj_restore_max_scale))
                restore_scale = torch.clamp(restore_scale, max=max_scale)
                proj = [g * restore_scale for g in proj]
        return proj, dot.detach(), cos.detach(), 1.0, restore_scale

    def _training_step_asym_proj(self, batch):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        inputs, labels = batch
        click = labels["click"].float()
        like = labels["like"].float()
        collect = labels["collect"].float()

        click_logit, like_logit, collect_logit = self(inputs)
        p_click = torch.sigmoid(click_logit)
        p_like_cond = torch.sigmoid(like_logit)
        p_collect_cond = torch.sigmoid(collect_logit)

        p_ct_like = p_click * p_like_cond
        p_ct_collect = p_click * p_collect_cond

        click_loss_vec = nn.functional.binary_cross_entropy_with_logits(
            click_logit,
            click,
            reduction="none",
        )
        like_loss_vec = nn.functional.binary_cross_entropy(
            p_ct_like,
            like,
            reduction="none",
        )
        collect_loss_vec = nn.functional.binary_cross_entropy(
            p_ct_collect,
            collect,
            reduction="none",
        )

        click_loss = click_loss_vec.mean()
        like_loss = like_loss_vec.mean()
        collect_loss = collect_loss_vec.mean()

        l_click = self.w_click * click_loss
        l_like = self.w_like * like_loss
        l_collect = self.w_collect * collect_loss

        # 独立冲突定义：
        # 1) click=1, like=0 -> like 任务梯度相对 click 做投影
        # 2) click=1, collect=0 -> collect 任务梯度相对 click 做投影
        like_conflict_mask = ((click > 0.5) & (like < 0.5)).float()
        collect_conflict_mask = ((click > 0.5) & (collect < 0.5)).float()
        like_non_conflict_mask = 1.0 - like_conflict_mask
        collect_non_conflict_mask = 1.0 - collect_conflict_mask
        union_conflict_mask = torch.clamp(like_conflict_mask + collect_conflict_mask, max=1.0)

        click_like_conf = self._masked_mean(click_loss_vec, like_conflict_mask)
        click_collect_conf = self._masked_mean(click_loss_vec, collect_conflict_mask)
        like_conf = self._masked_mean(like_loss_vec, like_conflict_mask)
        collect_conf = self._masked_mean(collect_loss_vec, collect_conflict_mask)

        like_non_conf = self._masked_mean(like_loss_vec, like_non_conflict_mask)
        collect_non_conf = self._masked_mean(collect_loss_vec, collect_non_conflict_mask)

        shared_params = [p for p in self._get_shared_params() if p.requires_grad]
        g_click_all = self._normalize_grads(
            shared_params,
            torch.autograd.grad(l_click, shared_params, retain_graph=True, allow_unused=True),
        )

        g_click_like_conf = self._normalize_grads(
            shared_params,
            torch.autograd.grad(
                self.w_click * click_like_conf,
                shared_params,
                retain_graph=True,
                allow_unused=True,
            ),
        )
        g_click_collect_conf = self._normalize_grads(
            shared_params,
            torch.autograd.grad(
                self.w_click * click_collect_conf,
                shared_params,
                retain_graph=True,
                allow_unused=True,
            ),
        )
        g_like_conf = self._normalize_grads(
            shared_params,
            torch.autograd.grad(
                self.w_like * like_conf,
                shared_params,
                retain_graph=True,
                allow_unused=True,
            ),
        )
        g_collect_conf = self._normalize_grads(
            shared_params,
            torch.autograd.grad(
                self.w_collect * collect_conf,
                shared_params,
                retain_graph=True,
                allow_unused=True,
            ),
        )
        g_like_non_conf = self._normalize_grads(
            shared_params,
            torch.autograd.grad(
                self.w_like * like_non_conf,
                shared_params,
                retain_graph=True,
                allow_unused=True,
            ),
        )
        g_collect_non_conf = self._normalize_grads(
            shared_params,
            torch.autograd.grad(
                self.w_collect * collect_non_conf,
                shared_params,
                retain_graph=True,
                allow_unused=True,
            ),
        )

        device = click_loss.device
        g_like_conf_p, dot_like, cos_like, app_like, like_restore_scale = self._project_task_grad(
            g_like_conf, g_click_like_conf, device
        )
        g_collect_conf_p, dot_collect, cos_collect, app_collect, collect_restore_scale = self._project_task_grad(
            g_collect_conf, g_click_collect_conf, device
        )

        total = l_click + l_like + l_collect
        self.manual_backward(total)

        # 仅替换共享参数梯度，其他任务专属层保持常规梯度。
        # 非冲突样本保持原梯度；冲突样本使用投影后梯度。
        for p, gc_all, gl_nc, gc_nc, gl_c_p, gc_c_p in zip(
            shared_params,
            g_click_all,
            g_like_non_conf,
            g_collect_non_conf,
            g_like_conf_p,
            g_collect_conf_p,
        ):
            p.grad = (gc_all + gl_nc + gc_nc + gl_c_p + gc_c_p).detach()

        optimizer.step()

        self.log("train_loss", total, prog_bar=True)
        self.log("train_click_loss", click_loss)
        self.log("train_like_loss", like_loss)
        self.log("train_collect_loss", collect_loss)
        self.log("train_conflict_ratio", union_conflict_mask.mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_like_conflict_ratio", like_conflict_mask.mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_collect_conflict_ratio", collect_conflict_mask.mean(), on_step=False, on_epoch=True, sync_dist=True)

        self.log("train_asym_like_dot", dot_like, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_asym_collect_dot", dot_collect, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_asym_like_cos", cos_like, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_asym_collect_cos", cos_collect, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_asym_like_applied", app_like, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_asym_collect_applied", app_collect, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_asym_like_restore_scale", like_restore_scale.detach(), on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_asym_collect_restore_scale", collect_restore_scale.detach(), on_step=False, on_epoch=True, sync_dist=True)
        return total

    def validation_step(self, batch, batch_idx):
        (
            total,
            click_loss,
            like_loss,
            collect_loss,
            p_click,
            p_ct_like,
            p_ct_collect,
            click,
            like,
            collect,
        ) = self._compute_losses(batch)
        self.log("val_loss", total, prog_bar=True, sync_dist=True)
        self.log("val_click_loss", click_loss, sync_dist=True)
        self.log("val_like_loss", like_loss, sync_dist=True)
        self.log("val_collect_loss", collect_loss, sync_dist=True)

        self.val_click_auc.update(p_click, click.long())
        self.val_like_auc.update(p_ct_like, like.long())
        self.val_collect_auc.update(p_ct_collect, collect.long())

    def on_validation_epoch_end(self):
        click_auc = self.val_click_auc.compute()
        like_auc = self.val_like_auc.compute()
        collect_auc = self.val_collect_auc.compute()
        combined_auc = (click_auc + like_auc + collect_auc) / 3.0

        self.log("val_click_auc", click_auc, prog_bar=True, sync_dist=True)
        self.log("val_like_auc", like_auc, sync_dist=True)
        self.log("val_collect_auc", collect_auc, sync_dist=True)
        self.log("val_combined_auc", combined_auc, prog_bar=True, sync_dist=True)

        self.val_click_auc.reset()
        self.val_like_auc.reset()
        self.val_collect_auc.reset()

    def test_step(self, batch, batch_idx):
        (
            total,
            _click_loss,
            _like_loss,
            _collect_loss,
            p_click,
            p_ct_like,
            p_ct_collect,
            click,
            like,
            collect,
        ) = self._compute_losses(batch)
        self.log("test_loss", total, sync_dist=True)

        self.test_click_auc.update(p_click, click.long())
        self.test_like_auc.update(p_ct_like, like.long())
        self.test_collect_auc.update(p_ct_collect, collect.long())

    def on_test_epoch_end(self):
        click_auc = self.test_click_auc.compute()
        like_auc = self.test_like_auc.compute()
        collect_auc = self.test_collect_auc.compute()
        combined_auc = (click_auc + like_auc + collect_auc) / 3.0

        self.log("test_click_auc", click_auc, prog_bar=True, sync_dist=True)
        self.log("test_like_auc", like_auc, sync_dist=True)
        self.log("test_collect_auc", collect_auc, sync_dist=True)
        self.log("test_combined_auc", combined_auc, prog_bar=True, sync_dist=True)

        self.test_click_auc.reset()
        self.test_like_auc.reset()
        self.test_collect_auc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
