"""
Tenrec 训练脚本（MMOE + ESMM）

对比实验:
    1) MMOE + ESMM
    2) MMOE + ESMM + asymmetric projection
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.tenrec_dataset import TenrecDataModule
from src.models.tenrec_mmoe_esmm import TenrecMMOEESMMModel


def parse_args():
    parser = argparse.ArgumentParser(description="Tenrec ESMM+MMOE 训练")

    # 数据路径
    parser.add_argument("--input_path", type=str, default=None, help="Tenrec 单文件路径（自动按顺序 8:1:1 切分）")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)

    # 切分
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    # 标签列映射（内部统一为 click/like/collect）
    parser.add_argument("--click_label", type=str, default="click")
    parser.add_argument("--like_label", type=str, default="like")
    parser.add_argument("--collect_label", type=str, default="favorite")
    parser.add_argument("--collect_label_fallback", type=str, default="follow")
    parser.add_argument(
        "--filter_click0_with_other_behavior",
        action="store_true",
        default=False,
        help="过滤 click=0 且 like/collect 为正的异常样本（读入时过滤）",
    )

    # 模型超参
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--num_experts", type=int, default=6)
    parser.add_argument("--expert_hidden_dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--tower_hidden_dims", type=int, nargs="+", default=[64])
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning_rate", type=float, default=3e-4)

    # 任务权重
    parser.add_argument("--w_click", type=float, default=1.0)
    parser.add_argument("--w_like", type=float, default=1.0)
    parser.add_argument("--w_collect", type=float, default=1.0)

    # 训练
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--strategy", type=str, default="ddp_find_unused_parameters_false")
    parser.add_argument("--log_every_n_steps", type=int, default=100)

    # 梯度裁剪策略（对比开关）
    parser.add_argument("--gradient_clip_val", type=float, default=0.0, help="<=0 表示关闭")
    parser.add_argument("--gradient_clip_algorithm", type=str, default="norm", choices=["norm", "value"])
    parser.add_argument("--use_asym_proj", action="store_true", default=False, help="是否启用 asymmetric gradient projection")
    parser.add_argument("--asym_proj_lambda", type=float, default=1.0, help="asym projection 强度 lambda")
    parser.add_argument("--asym_proj_tau", type=float, default=0.0, help="仅当 cos < -tau 时触发投影")
    parser.add_argument("--asym_proj_restore_norm", action="store_true", default=False, help="是否对投影后任务梯度做模长还原")
    parser.add_argument("--asym_proj_restore_max_scale", type=float, default=5.0, help="梯度模长还原的最大缩放倍数")

    # 输出
    parser.add_argument("--exp_dir", type=str, default="experiments/tenrec/outputs")

    args = parser.parse_args()
    if args.input_path is None and (args.train_path is None or args.val_path is None or args.test_path is None):
        raise ValueError("请提供 --input_path 或 train/val/test 三个路径")
    if args.gradient_clip_val < 0.0:
        raise ValueError("gradient_clip_val 必须 >= 0")
    if args.asym_proj_lambda < 0.0:
        raise ValueError("asym_proj_lambda 必须 >= 0")
    if args.asym_proj_tau < 0.0:
        raise ValueError("asym_proj_tau 必须 >= 0")
    if args.asym_proj_restore_max_scale <= 0.0:
        raise ValueError("asym_proj_restore_max_scale 必须 > 0")
    return args


def train_tenrec(args):
    data_module = TenrecDataModule(
        input_path=args.input_path,
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        click_label=args.click_label,
        like_label=args.like_label,
        collect_label=args.collect_label,
        collect_label_fallback=args.collect_label_fallback,
        filter_click0_with_other_behavior=args.filter_click0_with_other_behavior,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_module.prepare_datasets()

    model = TenrecMMOEESMMModel(
        sparse_feature_names=data_module.sparse_feature_names,
        sparse_feature_dims=data_module.sparse_feature_dims,
        dense_feature_names=data_module.dense_feature_names,
        embedding_dim=args.embedding_dim,
        num_experts=args.num_experts,
        expert_hidden_dims=args.expert_hidden_dims,
        tower_hidden_dims=args.tower_hidden_dims,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        w_click=args.w_click,
        w_like=args.w_like,
        w_collect=args.w_collect,
        use_asym_proj=args.use_asym_proj,
        asym_proj_lambda=args.asym_proj_lambda,
        asym_proj_tau=args.asym_proj_tau,
        asym_proj_restore_norm=args.asym_proj_restore_norm,
        asym_proj_restore_max_scale=args.asym_proj_restore_max_scale,
    )

    clip_suffix = ""
    if args.gradient_clip_val > 0:
        clip_tag = str(args.gradient_clip_val).replace(".", "p")
        clip_suffix = f"_gclip_{args.gradient_clip_algorithm}_{clip_tag}"
    asym_suffix = ""
    if args.use_asym_proj:
        lam = str(args.asym_proj_lambda).replace(".", "p")
        tau = str(args.asym_proj_tau).replace(".", "p")
        asym_suffix = f"_asymproj_l{lam}_t{tau}"
        if args.asym_proj_restore_norm:
            max_scale = str(args.asym_proj_restore_max_scale).replace(".", "p")
            asym_suffix = f"{asym_suffix}_rs{max_scale}"

    exp_name = (
        f"tenrec_mmoe_esmm"
        f"_e{args.num_experts}"
        f"_lr{str(args.learning_rate).replace('.', 'p')}"
        f"_do{str(args.dropout).replace('.', 'p')}"
        f"{clip_suffix}{asym_suffix}"
    )
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
        devices=args.devices,
        strategy=args.strategy,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 0.0 else None,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)
    print(f"训练完成！最佳模型: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    train_tenrec(parse_args())
