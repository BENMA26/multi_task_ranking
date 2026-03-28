"""
排序模型训练脚本
支持 ShareBottom / MOE / MMOE / PLE，通过 --use_torchjd 和 --aggregation_method 控制梯度聚合
"""
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
from pathlib import Path

from src.models.ranking import ShareBottomModel, MOEModel, MMOEModel, PLEModel
from src.data.dataset import RankDataModule
from src.utils.constants import (
    USER_SPARSE, USER_DENSE,
    ITEM_SPARSE, ITEM_DENSE,
    ID_FEATURES, CONTEXT_SPARSE,
    CROSS_SPARSE, CROSS_DENSE,
    vocabulary_size,
)

MODEL_MAP = {
    'share_bottom': ShareBottomModel,
    'moe'         : MOEModel,
    'mmoe'        : MMOEModel,
    'ple'         : PLEModel,
}

# 仅 MOE / MMOE / PLE 有专家网络相关参数
EXPERT_MODELS = {'moe', 'mmoe', 'ple'}

# 仅 PLE 有的参数
PLE_MODELS = {'ple'}


def parse_args():
    parser = argparse.ArgumentParser(description='排序模型训练脚本')

    # ── 数据路径 ────────────────────────────────────────────────────────────
    parser.add_argument('--train_path', type=str, default='/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_train.parquet')
    parser.add_argument('--val_path', type=str, default='/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_val.parquet')
    parser.add_argument('--test_path', type=str, default='/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_test.parquet')

    # ── 模型选择 ────────────────────────────────────────────────────────────
    parser.add_argument('--model', type=str, default='mmoe', choices=list(MODEL_MAP.keys()), help='模型类型')
    parser.add_argument('--esmm', action='store_true', default=False, help='是否使用 ESMM 损失（pCTR × pCVR 监督购买标签）')
    parser.add_argument('--sigmoid', type=int, default=1)

    # ── 通用超参数 ──────────────────────────────────────────────────────────
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--tower_hidden_dims', type=int, nargs='+', default=[64], help='任务塔各隐层维度')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--ctr_weight', type=float, default=0.5, help='CTR loss 权重（use_torchjd=False 时生效）')
    parser.add_argument('--cvr_weight', type=float, default=0.5, help='CVR loss 权重（use_torchjd=False 时生效）')

    # ── TorchJD 梯度聚合 ────────────────────────────────────────────────────
    parser.add_argument('--use_torchjd', action='store_true', default=False, help='是否启用 TorchJD 梯度聚合')
    parser.add_argument('--aggregation_method', type=str, default='upgrad', choices=['upgrad', 'mgda', 'pcgrad', 'graddrop'], help='TorchJD 聚合算法（use_torchjd=True 时生效）')

    # ── Entropy 正则化（防止门控极化）────────────────────────────────────────
    parser.add_argument('--use_entropy_reg', action='store_true', default=False, help='是否启用 Entropy 正则化防止门控极化')
    parser.add_argument('--lambda_entropy', type=float, default=0.01, help='Entropy 正则化权重')

    # ── EMA 参数更新 ────────────────────────────────────────────────────────
    parser.add_argument('--use_ema', action='store_true', default=False, help='是否启用 EMA 参数更新')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA 衰减系数')

    # ── DCN-v2 特征交叉 ─────────────────────────────────────────────────────
    parser.add_argument('--use_dcn', action='store_true', default=False, help='是否在特征编码后加 DCN-v2 交叉层（ShareBottom/MMOE 生效）')
    parser.add_argument('--dcn_num_layers', type=int, default=2, help='DCN-v2 交叉层数量')
    parser.add_argument('--dcn_dropout', type=float, default=0.0, help='DCN-v2 交叉层 Dropout')

    # ── ShareBottom 专用 ────────────────────────────────────────────────────
    parser.add_argument('--shared_hidden_dims', type=int, nargs='+', default=[256, 128], help='共享底层各隐层维度（仅 share_bottom 有效）')

    # ── MOE / MMOE / PLE 专用 ──────────────────────────────────────────────
    parser.add_argument('--num_experts', type=int, default=6, help='专家数量（仅 moe / mmoe 有效）')
    parser.add_argument('--expert_hidden_dims', type=int, nargs='+', default=[256, 128], help='每个专家网络各隐层维度（moe / mmoe / ple 有效）')

    # ── PLE 专用 ────────────────────────────────────────────────────────────
    parser.add_argument('--num_specific_experts', type=int, default=1, help='每个任务专属专家数量（仅 ple 有效）')
    parser.add_argument('--num_shared_experts', type=int, default=2, help='共享专家数量（仅 ple 有效）')
    parser.add_argument('--num_levels', type=int, default=2, help='CGC 层数（仅 ple 有效，>=1）')

    # ── 数据加载 ────────────────────────────────────────────────────────────
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)

    # ── 训练配置 ────────────────────────────────────────────────────────────
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=4)
    parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_false')
    parser.add_argument('--log_every_n_steps', type=int, default=100)
    parser.add_argument('--early_stop_patience', type=int, default=2)

    # ── 输出路径 ────────────────────────────────────────────────────────────
    parser.add_argument('--exp_dir', type=str, default='experiments/rank',
        help='实验输出根目录')

    return parser.parse_args()


def build_model(args):
    """根据 args 构造对应模型实例"""
    model_cls = MODEL_MAP[args.model]

    sparse_feature_names = USER_SPARSE + ITEM_SPARSE + ID_FEATURES + CONTEXT_SPARSE + CROSS_SPARSE
    dense_feature_names  = USER_DENSE + ITEM_DENSE + CROSS_DENSE

    # 所有模型共有的参数
    common = dict(
        sparse_feature_names = sparse_feature_names,
        sparse_feature_dims  = vocabulary_size,
        dense_feature_names  = dense_feature_names,
        embedding_dim        = args.embedding_dim,
        tower_hidden_dims    = args.tower_hidden_dims,
        dropout              = args.dropout,
        learning_rate        = args.learning_rate,
        ctr_weight           = args.ctr_weight,
        cvr_weight           = args.cvr_weight,
        use_torchjd          = args.use_torchjd,
        aggregation_method   = args.aggregation_method,
        esmm                 = args.esmm,
        sigmoid              = args.sigmoid,
        use_entropy_reg      = args.use_entropy_reg,
        lambda_entropy       = args.lambda_entropy,
        use_ema              = args.use_ema,
        ema_decay            = args.ema_decay,
        use_dcn              = args.use_dcn,
        dcn_num_layers       = args.dcn_num_layers,
        dcn_dropout          = args.dcn_dropout,
    )

    # ShareBottom 专用参数
    if args.model == 'share_bottom':
        return model_cls(**common, shared_hidden_dims=args.shared_hidden_dims)

    # PLE 专用参数
    if args.model in PLE_MODELS:
        return model_cls(
            **common,
            expert_hidden_dims   = args.expert_hidden_dims,
            num_specific_experts = args.num_specific_experts,
            num_shared_experts   = args.num_shared_experts,
            num_levels           = args.num_levels,
        )

    # MOE / MMOE 专用参数
    return model_cls(
        **common,
        num_experts        = args.num_experts,
        expert_hidden_dims = args.expert_hidden_dims,
    )


def train_rank(args):

    sparse_feature_names = USER_SPARSE + ITEM_SPARSE + ID_FEATURES + CONTEXT_SPARSE + CROSS_SPARSE
    dense_feature_names  = USER_DENSE + ITEM_DENSE + CROSS_DENSE

    data_module = RankDataModule(
        train_path           = args.train_path,
        val_path             = args.val_path,
        test_path            = args.test_path,
        sparse_feature_names = sparse_feature_names,
        dense_feature_names  = dense_feature_names,
        label_cols           = ['click', 'purchase'],
        batch_size           = args.batch_size,
        num_workers          = args.num_workers,
    )

    model = build_model(args)

    # 实验名：模型名 + 是否使用 torchjd（若是则加聚合方法）
    jd_suffix = f"_jd_{args.aggregation_method}" if args.use_torchjd else ""
    exp_name  = f"{args.model}{jd_suffix}"
    exp_dir   = Path(args.exp_dir) / exp_name

    checkpoint_callback = ModelCheckpoint(
        dirpath  = str(exp_dir / 'checkpoints'),
        filename = 'best_val_combined_auc',
        monitor  = 'val_combined_auc',
        mode     = 'max',
        save_top_k = 1,
        save_last  = True,
    )
    early_stop_callback = EarlyStopping(
        monitor = 'val_combined_auc',
        patience = args.early_stop_patience,
        mode    = 'max',
        verbose = True,
    )
    logger = TensorBoardLogger(save_dir=str(exp_dir), name='logs')

    trainer = pl.Trainer(
        max_epochs          = args.max_epochs,
        accelerator         = args.accelerator,
        strategy            = args.strategy,
        profiler            = 'simple',
        devices             = args.devices,
        callbacks           = [checkpoint_callback, early_stop_callback],
        logger              = logger,
        log_every_n_steps   = args.log_every_n_steps,
        fast_dev_run        = False,
    )

    trainer.fit(model, data_module)

    trainer.test(model, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)

    print(f"训练完成！最佳模型保存在: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    train_rank(parse_args())
