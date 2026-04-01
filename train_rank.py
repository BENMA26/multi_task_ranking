"""
排序模型训练脚本（层次化 CLI）
仅保留 ShareBottom / MMOE / PLE。

层次化接口：
    train -> model -> regularization -> gradient surgery -> scope

示例：
    python train_rank.py train --model mmoe --regularization entropy \
        --grad_surgery upgrad --grad_scope on10
"""
import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.dataset import RankDataModule
from src.models.ranking import MMOEModel, PLEModel, ShareBottomModel
from src.utils.constants import (
    CONTEXT_SPARSE,
    CROSS_DENSE,
    CROSS_SPARSE,
    ID_FEATURES,
    ITEM_DENSE,
    ITEM_SPARSE,
    USER_DENSE,
    USER_SPARSE,
    vocabulary_size,
)

MODEL_MAP = {
    "share_bottom": ShareBottomModel,
    "mmoe": MMOEModel,
    "ple": PLEModel,
}

ENTROPY_MODELS = {"mmoe", "ple"}
ASYM_PROJ_MODELS = {"share_bottom", "mmoe", "ple"}
GRAD_SURGERY_CHOICES = ["normal", "pcgrad", "upgrad", "asymmetric_projection"]
GRAD_SCOPE_CHOICES = ["all", "on10"]
REG_CHOICES = ["none", "entropy"]


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    # 数据路径
    parser.add_argument("--train_path", type=str, default="/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_train.parquet")
    parser.add_argument("--val_path", type=str, default="/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_val.parquet")
    parser.add_argument("--test_path", type=str, default="/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_test.parquet")

    # 模型与任务
    parser.add_argument("--model", type=str, default="mmoe", choices=list(MODEL_MAP.keys()), help="模型类型")
    parser.add_argument("--esmm", action="store_true", default=False, help="是否使用 ESMM 损失（pCTR × pCVR 监督购买标签）")
    parser.add_argument("--sigmoid", type=int, default=1)

    # 层次化方法接口
    parser.add_argument("--regularization", type=str, default="none", choices=REG_CHOICES, help="正则化策略")
    parser.add_argument("--lambda_entropy", type=float, default=0.01, help="Entropy 正则化权重（regularization=entropy 时生效）")
    parser.add_argument("--grad_surgery", type=str, default="normal", choices=GRAD_SURGERY_CHOICES, help="梯度手术方法")
    parser.add_argument("--grad_scope", type=str, default="all", choices=GRAD_SCOPE_CHOICES, help="梯度手术作用范围：全样本或仅(1,0)")

    # 手术细节参数
    parser.add_argument("--asym_proj_lambda", type=float, default=1.0, help="非对称投影强度 λ，1.0 为完整投影")
    parser.add_argument("--asym_proj_tau", type=float, default=0.0, help="仅当 cos(g_ctr,g_ctcvr) < -tau 时触发投影")
    parser.add_argument("--asym_proj_restore_norm", action="store_true", default=False, help="投影后将被投影分量的模长恢复到投影前")
    parser.add_argument("--asym_proj_restore_max_scale", type=float, default=5.0, help="模长恢复时的最大放大倍数上限")

    # 通用超参数
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--tower_hidden_dims", type=int, nargs="+", default=[64], help="任务塔各隐层维度")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "swish"], help="隐藏层激活函数")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--ctr_weight", type=float, default=0.5, help="CTR loss 权重")
    parser.add_argument("--cvr_weight", type=float, default=0.5, help="CVR loss 权重")

    # 监控
    parser.add_argument("--monitor_grad_conflict", action="store_true", default=False, help="是否动态监控任务梯度冲突")
    parser.add_argument("--grad_conflict_interval", type=int, default=100, help="梯度冲突监控间隔（每 N 个 step）")

    # 模型专用
    parser.add_argument("--shared_hidden_dims", type=int, nargs="+", default=[256, 128], help="共享底层各隐层维度（仅 share_bottom）")
    parser.add_argument("--num_experts", type=int, default=6, help="专家数量（仅 mmoe）")
    parser.add_argument("--expert_hidden_dims", type=int, nargs="+", default=[256, 128], help="每个专家网络隐层维度（mmoe/ple）")
    parser.add_argument("--num_specific_experts", type=int, default=1, help="每个任务专属专家数量（仅 ple）")
    parser.add_argument("--num_shared_experts", type=int, default=2, help="共享专家数量（仅 ple）")
    parser.add_argument("--num_levels", type=int, default=2, help="CGC 层数（仅 ple，>=1）")

    # 数据加载
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_subset_frac", type=float, default=1.0, help="训练集子采样比例")
    parser.add_argument("--train_subset_seed", type=int, default=42, help="训练集子采样随机种子")
    parser.add_argument("--train_subset_stratify", action="store_true", default=False, help="训练集子采样时是否按 (click,purchase) 分层抽样")

    # 训练配置
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--strategy", type=str, default="ddp_find_unused_parameters_false")
    parser.add_argument("--gradient_clip_val", type=float, default=0.0, help="梯度裁剪阈值，<=0 表示关闭")
    parser.add_argument("--gradient_clip_algorithm", type=str, default="norm", choices=["norm", "value"], help="梯度裁剪方式")
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42, help="训练随机种子")

    # 输出
    parser.add_argument("--exp_dir", type=str, default="experiments/rank", help="实验输出根目录")



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="排序模型训练脚本（层次化 CLI）")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="训练 CTR/CVR 多任务模型")
    _add_train_args(p_train)

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    # 兼容旧调用：未显式写子命令时，自动补成 train
    if not raw_argv or raw_argv[0] not in {"train"}:
        raw_argv = ["train", *raw_argv]

    return parser.parse_args(raw_argv)


def _derive_runtime_flags(args) -> None:
    args.use_entropy_reg = args.regularization == "entropy"

    args.use_torchjd = args.grad_surgery in {"pcgrad", "upgrad"}
    args.aggregation_method = args.grad_surgery if args.use_torchjd else "upgrad"
    args.torchjd_only_10 = args.use_torchjd and args.grad_scope == "on10"

    args.use_asym_proj = args.grad_surgery == "asymmetric_projection"
    args.asym_proj_only_10 = args.use_asym_proj and args.grad_scope == "on10"



def _validate_args(args) -> None:
    if args.use_asym_proj and args.model not in ASYM_PROJ_MODELS:
        raise ValueError(f"asymmetric_projection 当前仅支持 model in {sorted(ASYM_PROJ_MODELS)}")
    if args.use_entropy_reg and args.model not in ENTROPY_MODELS:
        raise ValueError(f"entropy regularization 当前仅支持 model in {sorted(ENTROPY_MODELS)}")

    if args.lambda_entropy < 0.0:
        raise ValueError("lambda_entropy 需 >= 0.0")
    if args.gradient_clip_val < 0.0:
        raise ValueError("gradient_clip_val 需 >= 0.0")
    if args.asym_proj_restore_max_scale < 1.0:
        raise ValueError("asym_proj_restore_max_scale 需 >= 1.0")
    if not (0.0 < args.train_subset_frac <= 1.0):
        raise ValueError("train_subset_frac 需在 (0, 1] 区间")

    if args.asym_proj_restore_norm and not args.use_asym_proj:
        raise ValueError("asym_proj_restore_norm 仅在 grad_surgery=asymmetric_projection 时生效")



def build_model(args):
    model_cls = MODEL_MAP[args.model]

    sparse_feature_names = USER_SPARSE + ITEM_SPARSE + ID_FEATURES + CONTEXT_SPARSE + CROSS_SPARSE
    dense_feature_names = USER_DENSE + ITEM_DENSE + CROSS_DENSE

    common = dict(
        sparse_feature_names=sparse_feature_names,
        sparse_feature_dims=vocabulary_size,
        dense_feature_names=dense_feature_names,
        embedding_dim=args.embedding_dim,
        tower_hidden_dims=args.tower_hidden_dims,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        ctr_weight=args.ctr_weight,
        cvr_weight=args.cvr_weight,
        use_torchjd=args.use_torchjd,
        aggregation_method=args.aggregation_method,
        esmm=args.esmm,
        sigmoid=args.sigmoid,
    )

    if args.model == "share_bottom":
        return model_cls(
            **common,
            shared_hidden_dims=args.shared_hidden_dims,
            activation=args.activation,
            use_entropy_reg=False,
            lambda_entropy=0.0,
            use_asym_proj=args.use_asym_proj,
            asym_proj_lambda=args.asym_proj_lambda,
            asym_proj_tau=args.asym_proj_tau,
            asym_proj_only_10=args.asym_proj_only_10,
            asym_proj_restore_norm=args.asym_proj_restore_norm,
            asym_proj_restore_max_scale=args.asym_proj_restore_max_scale,
            torchjd_only_10=args.torchjd_only_10,
        )

    if args.model == "mmoe":
        return model_cls(
            **common,
            num_experts=args.num_experts,
            expert_hidden_dims=args.expert_hidden_dims,
            activation=args.activation,
            use_entropy_reg=args.use_entropy_reg,
            lambda_entropy=args.lambda_entropy,
            use_asym_proj=args.use_asym_proj,
            asym_proj_lambda=args.asym_proj_lambda,
            asym_proj_tau=args.asym_proj_tau,
            asym_proj_only_10=args.asym_proj_only_10,
            asym_proj_restore_norm=args.asym_proj_restore_norm,
            asym_proj_restore_max_scale=args.asym_proj_restore_max_scale,
            torchjd_only_10=args.torchjd_only_10,
        )

    # ple
    return model_cls(
        **common,
        expert_hidden_dims=args.expert_hidden_dims,
        num_specific_experts=args.num_specific_experts,
        num_shared_experts=args.num_shared_experts,
        num_levels=args.num_levels,
        use_entropy_reg=args.use_entropy_reg,
        lambda_entropy=args.lambda_entropy,
        use_asym_proj=args.use_asym_proj,
        asym_proj_lambda=args.asym_proj_lambda,
        asym_proj_tau=args.asym_proj_tau,
        asym_proj_only_10=args.asym_proj_only_10,
        asym_proj_restore_norm=args.asym_proj_restore_norm,
        asym_proj_restore_max_scale=args.asym_proj_restore_max_scale,
        torchjd_only_10=args.torchjd_only_10,
    )


def train_rank(args):
    _derive_runtime_flags(args)
    _validate_args(args)

    pl.seed_everything(args.seed, workers=True)

    sparse_feature_names = USER_SPARSE + ITEM_SPARSE + ID_FEATURES + CONTEXT_SPARSE + CROSS_SPARSE
    dense_feature_names = USER_DENSE + ITEM_DENSE + CROSS_DENSE

    data_module = RankDataModule(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        sparse_feature_names=sparse_feature_names,
        dense_feature_names=dense_feature_names,
        label_cols=["click", "purchase"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_subset_frac=args.train_subset_frac,
        train_subset_seed=args.train_subset_seed,
        train_subset_stratify=args.train_subset_stratify,
    )

    model = build_model(args)
    if hasattr(model, "configure_grad_conflict_monitor"):
        model.configure_grad_conflict_monitor(
            enabled=args.monitor_grad_conflict,
            interval=args.grad_conflict_interval,
        )

    act_suffix = f"_act_{args.activation}"
    reg_suffix = f"_reg_{args.regularization}"
    if args.use_entropy_reg:
        lam = str(args.lambda_entropy).replace(".", "p")
        reg_suffix += f"_l{lam}"

    surgery_suffix = f"_gs_{args.grad_surgery}_{args.grad_scope}"
    asym_suffix = ""
    if args.use_asym_proj:
        lam = str(args.asym_proj_lambda).replace(".", "p")
        tau = str(args.asym_proj_tau).replace(".", "p")
        asym_suffix = f"_l{lam}_t{tau}"
        if args.asym_proj_restore_norm:
            rmax = str(args.asym_proj_restore_max_scale).replace(".", "p")
            asym_suffix += f"_rn_m{rmax}"

    conflict_suffix = f"_gconf_i{args.grad_conflict_interval}" if args.monitor_grad_conflict else ""

    clip_suffix = ""
    if args.gradient_clip_val > 0.0:
        clip_val = str(args.gradient_clip_val).replace(".", "p")
        clip_suffix = f"_gclip_{args.gradient_clip_algorithm}_{clip_val}"

    subset_suffix = ""
    if args.train_subset_frac < 1.0:
        frac_tag = str(args.train_subset_frac).replace(".", "p")
        strat_tag = "_strat" if args.train_subset_stratify else ""
        subset_suffix = f"_sub{frac_tag}{strat_tag}"

    exp_name = f"{args.model}{act_suffix}{reg_suffix}{surgery_suffix}{asym_suffix}{conflict_suffix}{clip_suffix}{subset_suffix}"
    exp_dir = Path(args.exp_dir) / exp_name

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(exp_dir / "checkpoints"),
        filename="best_val_combined_auc",
        monitor="val_combined_auc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_combined_auc",
        patience=args.early_stop_patience,
        mode="max",
        verbose=True,
    )
    logger = TensorBoardLogger(save_dir=str(exp_dir), name="logs")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        strategy=args.strategy,
        profiler="simple",
        devices=args.devices,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 0.0 else None,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        fast_dev_run=False,
    )

    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)

    print(f"训练完成！最佳模型保存在: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    args = parse_args()
    if args.command == "train":
        train_rank(args)
