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

from src.models.ranking import ShareBottomModel, MOEModel, MMOEModel, PLEModel, AdaFTRModel
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
    'adaftr'      : AdaFTRModel,
}

# 仅 MOE / MMOE / PLE 有专家网络相关参数
EXPERT_MODELS = {'moe', 'mmoe', 'ple', 'adaftr'}

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
    parser.add_argument('--use_asym_proj', action='store_true', default=False, help='是否启用非对称梯度投影（当前支持 ShareBottom / MMOE）')
    parser.add_argument('--asym_proj_lambda', type=float, default=1.0, help='非对称投影强度 λ，1.0 为完整投影')
    parser.add_argument('--asym_proj_tau', type=float, default=0.0, help='仅当 cos(g_ctr,g_ctcvr) < -tau 时触发投影')
    parser.add_argument('--asym_proj_only_10', action='store_true', default=False, help='仅对 (click=1,purchase=0) 样本的 cvr/ctcvr 梯度做投影，其它样本梯度不投影')
    parser.add_argument('--asym_proj_restore_norm', action='store_true', default=False, help='投影后将被投影分量的梯度模长恢复到投影前（仅作用于被投影分量）')
    parser.add_argument('--asym_proj_restore_max_scale', type=float, default=5.0, help='模长恢复时的最大放大倍数上限')
    parser.add_argument('--monitor_grad_conflict', action='store_true', default=False, help='是否动态监控任务梯度冲突（TorchJD autogram API）')
    parser.add_argument('--grad_conflict_interval', type=int, default=100, help='梯度冲突监控间隔（每 N 个 step 监控一次）')

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
    parser.add_argument('--dcn_rank', type=int, default=0, help='DCN-v2 低秩分解维度，0 表示全秩')

    # ── ShareBottom 专用 ────────────────────────────────────────────────────
    parser.add_argument('--shared_hidden_dims', type=int, nargs='+', default=[256, 128], help='共享底层各隐层维度（仅 share_bottom 有效）')

    # ── MOE / MMOE / PLE 专用 ──────────────────────────────────────────────
    parser.add_argument('--num_experts', type=int, default=6, help='专家数量（仅 moe / mmoe 有效）')
    parser.add_argument('--expert_hidden_dims', type=int, nargs='+', default=[256, 128], help='每个专家网络各隐层维度（moe / mmoe / ple 有效）')

    # ── PLE 专用 ────────────────────────────────────────────────────────────
    parser.add_argument('--num_specific_experts', type=int, default=1, help='每个任务专属专家数量（仅 ple 有效）')
    parser.add_argument('--num_shared_experts', type=int, default=2, help='共享专家数量（仅 ple 有效）')
    parser.add_argument('--num_levels', type=int, default=2, help='CGC 层数（仅 ple 有效，>=1）')

    # ── AdaFTR 专用 ─────────────────────────────────────────────────────────
    parser.add_argument('--adaftr_tower_hidden_dims', type=int, nargs='+', default=[256, 128, 64], help='AdaFTR 任务特异层维度')
    parser.add_argument('--alpha_contrastive', type=float, default=0.1, help='AdaFTR 对比损失权重 alpha')
    parser.add_argument('--tau_min', type=float, default=0.05, help='AdaFTR 动态温度下界')
    parser.add_argument('--tau_max', type=float, default=0.50, help='AdaFTR 动态温度上界')
    parser.add_argument('--relatedness_hidden_dim', type=int, default=64, help='AdaFTR 关联性网络隐层维度')
    parser.add_argument('--lambda_rel', type=float, default=0.1, help='AdaFTR 关联性损失权重')
    parser.add_argument('--use_hard_sample', action='store_true', default=False, help='是否启用 hard sample 发现与重加权（MMOE/AdaFTR）')
    parser.add_argument('--hard_sample_ratio', type=float, default=0.2, help='每个 batch 挖掘 hard sample 的比例')
    parser.add_argument('--hard_sample_weight', type=float, default=2.0, help='hard sample 的损失权重放大倍数')
    parser.add_argument('--hard_sample_warmup_epochs', type=int, default=1, help='启用 hard sample 之前的 warmup epoch 数')

    # ── 数据加载 ────────────────────────────────────────────────────────────
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_subset_frac', type=float, default=1.0, help='训练集子采样比例，用于参数选择（默认 1.0=全量）')
    parser.add_argument('--train_subset_seed', type=int, default=42, help='训练集子采样随机种子')
    parser.add_argument('--train_subset_stratify', action='store_true', default=False, help='训练集子采样时是否按 (click,purchase) 分层抽样')

    # ── 训练配置 ────────────────────────────────────────────────────────────
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=4)
    parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_false')
    parser.add_argument('--gradient_clip_val', type=float, default=0.0, help='梯度裁剪阈值，<=0 表示关闭')
    parser.add_argument('--gradient_clip_algorithm', type=str, default='norm', choices=['norm', 'value'], help='梯度裁剪方式')
    parser.add_argument('--log_every_n_steps', type=int, default=100)
    parser.add_argument('--early_stop_patience', type=int, default=2)

    # ── 输出路径 ────────────────────────────────────────────────────────────
    parser.add_argument('--exp_dir', type=str, default='experiments/rank',
        help='实验输出根目录')

    return parser.parse_args()


def build_model(args):
    """根据 args 构造对应模型实例"""
    model_cls = MODEL_MAP[args.model]

    if args.use_torchjd and args.use_asym_proj:
        raise ValueError("use_torchjd 与 use_asym_proj 不能同时启用")
    if args.use_asym_proj and args.model not in {'share_bottom', 'mmoe'}:
        raise ValueError("use_asym_proj 当前仅支持 model in {share_bottom, mmoe}")
    if args.asym_proj_only_10 and not args.use_asym_proj:
        raise ValueError("asym_proj_only_10 仅在 use_asym_proj=True 时生效")
    if args.asym_proj_restore_norm and not args.use_asym_proj:
        raise ValueError("asym_proj_restore_norm 仅在 use_asym_proj=True 时生效")
    if args.asym_proj_restore_max_scale < 1.0:
        raise ValueError("asym_proj_restore_max_scale 需 >= 1.0")
    if args.use_asym_proj and args.model == 'mmoe' and args.use_hard_sample:
        raise ValueError("当前 mmoe + use_asym_proj 暂不支持 use_hard_sample")
    if args.gradient_clip_val < 0.0:
        raise ValueError("gradient_clip_val 需 >= 0.0")
    if not (0.0 < args.train_subset_frac <= 1.0):
        raise ValueError("train_subset_frac 需在 (0, 1] 区间")

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
        dcn_rank             = args.dcn_rank,
    )

    # ShareBottom 专用参数
    if args.model == 'share_bottom':
        return model_cls(
            **common,
            shared_hidden_dims=args.shared_hidden_dims,
            use_asym_proj=args.use_asym_proj,
            asym_proj_lambda=args.asym_proj_lambda,
            asym_proj_tau=args.asym_proj_tau,
            asym_proj_only_10=args.asym_proj_only_10,
            asym_proj_restore_norm=args.asym_proj_restore_norm,
            asym_proj_restore_max_scale=args.asym_proj_restore_max_scale,
        )

    # PLE 专用参数
    if args.model in PLE_MODELS:
        return model_cls(
            **common,
            expert_hidden_dims   = args.expert_hidden_dims,
            num_specific_experts = args.num_specific_experts,
            num_shared_experts   = args.num_shared_experts,
            num_levels           = args.num_levels,
        )

    # AdaFTR 专用参数
    if args.model == 'adaftr':
        common_adaftr = {k: v for k, v in common.items() if k != 'tower_hidden_dims'}
        return model_cls(
            **common_adaftr,
            num_experts           = args.num_experts,
            expert_hidden_dims    = args.expert_hidden_dims,
            tower_hidden_dims     = args.adaftr_tower_hidden_dims,
            alpha_contrastive     = args.alpha_contrastive,
            tau_min               = args.tau_min,
            tau_max               = args.tau_max,
            relatedness_hidden_dim= args.relatedness_hidden_dim,
            lambda_rel            = args.lambda_rel,
            use_hard_sample       = args.use_hard_sample,
            hard_sample_ratio     = args.hard_sample_ratio,
            hard_sample_weight    = args.hard_sample_weight,
            hard_sample_warmup_epochs = args.hard_sample_warmup_epochs,
        )

    # MMOE + hard sample 专用参数
    if args.model == 'mmoe':
        return model_cls(
            **common,
            num_experts        = args.num_experts,
            expert_hidden_dims = args.expert_hidden_dims,
            use_asym_proj      = args.use_asym_proj,
            asym_proj_lambda   = args.asym_proj_lambda,
            asym_proj_tau      = args.asym_proj_tau,
            asym_proj_only_10  = args.asym_proj_only_10,
            asym_proj_restore_norm = args.asym_proj_restore_norm,
            asym_proj_restore_max_scale = args.asym_proj_restore_max_scale,
            use_hard_sample       = args.use_hard_sample,
            hard_sample_ratio     = args.hard_sample_ratio,
            hard_sample_weight    = args.hard_sample_weight,
            hard_sample_warmup_epochs = args.hard_sample_warmup_epochs,
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
        train_subset_frac    = args.train_subset_frac,
        train_subset_seed    = args.train_subset_seed,
        train_subset_stratify= args.train_subset_stratify,
    )

    model = build_model(args)
    if hasattr(model, "configure_grad_conflict_monitor"):
        model.configure_grad_conflict_monitor(
            enabled=args.monitor_grad_conflict,
            interval=args.grad_conflict_interval,
        )

    # 实验名：模型名 + 训练策略后缀
    jd_suffix = f"_jd_{args.aggregation_method}" if args.use_torchjd else ""
    if args.use_asym_proj:
        lam = str(args.asym_proj_lambda).replace('.', 'p')
        tau = str(args.asym_proj_tau).replace('.', 'p')
        scope = "_on10" if args.asym_proj_only_10 else ""
        if args.asym_proj_restore_norm:
            rmax = str(args.asym_proj_restore_max_scale).replace('.', 'p')
            restore = f"_rn_m{rmax}"
        else:
            restore = ""
        asym_suffix = f"_asymproj_l{lam}_t{tau}{scope}{restore}"
    else:
        asym_suffix = ""
    conflict_suffix = f"_gconf_i{args.grad_conflict_interval}" if args.monitor_grad_conflict else ""
    clip_suffix = ""
    if args.gradient_clip_val > 0.0:
        clip_val = str(args.gradient_clip_val).replace('.', 'p')
        clip_suffix = f"_gclip_{args.gradient_clip_algorithm}_{clip_val}"
    subset_suffix = ""
    if args.train_subset_frac < 1.0:
        frac_tag = str(args.train_subset_frac).replace('.', 'p')
        strat_tag = "_strat" if args.train_subset_stratify else ""
        subset_suffix = f"_sub{frac_tag}{strat_tag}"
    exp_name  = f"{args.model}{jd_suffix}{asym_suffix}{conflict_suffix}{clip_suffix}{subset_suffix}"
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
        gradient_clip_val   = args.gradient_clip_val if args.gradient_clip_val > 0.0 else None,
        gradient_clip_algorithm = args.gradient_clip_algorithm,
        fast_dev_run        = False,
    )

    trainer.fit(model, data_module)

    trainer.test(model, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)

    print(f"训练完成！最佳模型保存在: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    train_rank(parse_args())
