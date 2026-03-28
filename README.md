# Multi-Task Ranking

一个面向推荐排序场景的多任务学习实验仓库，主要研究 CTR / CVR 联合建模，以及不同多任务架构、损失设计和训练策略对排序效果与训练稳定性的影响。

当前代码以 Ali-CCP 风格的数据组织为默认输入，训练目标是同时预测：

- `click`
- `purchase`

仓库整体更偏研究与实验验证，而不是面向生产部署的完整系统。

## 项目目标

本项目主要围绕以下问题展开：

- 不同多任务排序模型在点击与转化联合建模中的效果差异
- ESMM 对点击后转化建模的帮助
- 多任务梯度冲突的缓解方法
- 门控网络极化问题的缓解方法
- 训练稳定性的提升方法
- 显式特征交叉在多任务架构中的收益

## 当前支持的模型与训练能力

训练入口位于 `train_rank.py`，当前支持以下模型：

- `share_bottom`
- `moe`
- `mmoe`
- `ple`

核心能力包括：

- 标准 CTR/CVR 多任务训练
- `ESMM` 训练方式，用 `pCTR * pCVR` 监督购买标签
- `TorchJD` 梯度聚合，支持 `upgrad` / `mgda` / `pcgrad` / `graddrop`
- `Entropy Regularization`，缓解 MOE/MMOE 的 gate polarization
- `EMA` 参数滑动平均与额外评估指标
- `DCN-v2` 特征交叉层

## 数据与特征

默认数据路径配置在 `train_rank.py` 中：

- `train_path`: `/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_train.parquet`
- `val_path`: `/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_val.parquet`
- `test_path`: `/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_test.parquet`

当前特征定义位于 `src/utils/constants.py`，按来源分为：

- 用户稀疏特征 `USER_SPARSE`
- 物品稀疏特征 `ITEM_SPARSE`
- 交叉稀疏特征 `CROSS_SPARSE`
- 交叉稠密特征 `CROSS_DENSE`
- ID 特征 `ID_FEATURES`
- 上下文特征 `CONTEXT_SPARSE`

数据读取逻辑位于 `src/data/dataset.py`：

- 仅按需要读取列，减少内存占用
- 支持 `parquet` / `csv` / `tsv`
- 标签列固定为 `click` 和 `purchase`

## 代码结构

```text
multi_task_ranking/
├── train_rank.py
├── src/
│   ├── data/
│   │   └── dataset.py
│   ├── models/
│   │   └── ranking.py
│   └── utils/
│       ├── constants.py
│       └── metrics.py
├── experiments/
│   ├── basic_experiment/
│   ├── entropy_regularization/
│   ├── ema_stability/
│   ├── dcn_v2_feature_cross/
│   └── ...
└── setup.py
```

关键文件说明：

- `train_rank.py`: 训练入口、参数解析、模型构建、Lightning Trainer 配置
- `src/models/ranking.py`: 所有多任务排序模型及训练逻辑
- `src/data/dataset.py`: 数据集与 DataModule
- `src/utils/constants.py`: 特征列表与词表大小
- `experiments/*`: 不同实验版本对应的批量运行脚本

## 安装

建议在独立 Python 环境中安装。

```bash
python -m pip install -e .
```

根据当前代码，至少需要以下依赖：

- `torch`
- `pytorch-lightning`
- `torchmetrics`
- `pandas`
- `pyarrow`
- `torchjd`

如果缺少某些依赖，请根据报错补充安装。

## 使用方式

### 1. 基础训练

```bash
python train_rank.py \
  --model mmoe \
  --batch_size 1024 \
  --num_workers 32 \
  --max_epochs 20 \
  --learning_rate 1e-3 \
  --dropout 0.2 \
  --embedding_dim 32 \
  --expert_hidden_dims 256 128 \
  --num_experts 6 \
  --tower_hidden_dims 64 \
  --accelerator gpu \
  --devices 4 \
  --strategy ddp_find_unused_parameters_false \
  --esmm \
  --exp_dir experiments/rank
```

### 2. 启用 TorchJD

```bash
python train_rank.py \
  --model mmoe \
  --use_torchjd \
  --aggregation_method upgrad \
  --esmm
```

### 3. 启用 Entropy Regularization

```bash
python train_rank.py \
  --model mmoe \
  --use_entropy_reg \
  --lambda_entropy 0.01 \
  --esmm
```

### 4. 启用 EMA

```bash
python train_rank.py \
  --model mmoe \
  --use_ema \
  --ema_decay 0.999 \
  --esmm
```

### 5. 启用 DCN-v2

```bash
python train_rank.py \
  --model mmoe \
  --expert_hidden_dims 256 128 \
  --num_experts 6 \
  --use_dcn \
  --dcn_num_layers 2 \
  --esmm
```

## 训练与评估逻辑

当前训练流程由 PyTorch Lightning 驱动，主要特点如下：

- 使用 `ModelCheckpoint` 保存最佳模型
- 使用 `EarlyStopping` 提前停止
- 默认监控指标是 `val_combined_auc`
- `val_combined_auc = val_ctr_auc * val_cvr_auc`
- 测试阶段会加载最佳 checkpoint

如果开启 `EMA`，验证与测试阶段还会额外记录一组 EMA 参数下的 AUC 指标。

## 实验脚本

`experiments/` 目录下包含多组实验脚本，通常面向集群环境，使用 `sbatch` 提交。当前可以看到的主要实验方向包括：

- `basic_experiment`: 基础模型与梯度聚合对比
- `entropy_regularization`: 门控熵正则实验
- `ema_stability`: 多随机种子下的 EMA 稳定性实验
- `dcn_v2_feature_cross`: ShareBottom / MMOE 的 DCN-v2 对比实验

示例：

```bash
bash experiments/dcn_v2_feature_cross/run_dcn_experiments.sh
```

注意：

- 这些脚本默认使用多卡 GPU 和较大的 batch size
- 输出通常写入 `experiments/*/outputs`
- 日志通常写入 `experiments/*/logs`

## Git 分支与实验版本

这个仓库没有使用 tag 管理正式版本，主要通过 branch 演进不同实验方向。

### 主要分支

- `main`
  - 较早的基础版本，包含多任务排序模型的基本实现

- `inter-sample-contrastive`
  - 早期实验分支
  - 从当前可见历史看，它更像后续多个实验分支的分叉基础

- `intra-sample-contrastive`
  - 引入样本内对比学习
  - 采用 AdaFTR 风格思路，引入 `RelatednessNet`、动态温度和额外相关性损失

- `entropy-regularization`
  - 在 MOE/MMOE 中加入 gate entropy regularization
  - 目标是缓解门控只偏向少数专家的极化问题

- `ema-training`
  - 单独探索过一套基于 callback / shadow parameters 的 EMA 方案
  - 更像早期独立实现

- `dcn-v2-feature-cross`
  - 当前工作分支
  - 在 ShareBottom 和 MMOE 上引入 DCN-v2
  - 最新提交将 DCN-v2 从统一特征编码阶段移入 `Expert` 网络内部

### 分支使用建议

- 运行某个实验脚本前，先确认自己位于对应分支
- 不同分支的参数接口可能并不完全一致
- 对比学习相关脚本请优先以对应分支中的代码为准，不要直接假设它们与当前分支完全兼容

## 当前分支说明

当前分支 `dcn-v2-feature-cross` 的重点是：

- 保留多任务基础架构
- 保留 ESMM / TorchJD / Entropy Regularization / EMA
- 增加 DCN-v2 特征交叉层
- 将 DCN-v2 放到 `Expert` 前向开头，而不是共享编码后的统一位置

这意味着当前分支主要关注的问题是：

- 显式特征交叉是否能增强多任务排序模型
- DCN-v2 放在专家网络内部是否比放在共享输入后更合理

## 已知特点与限制

- 仓库当前更偏实验性质，依赖版本和运行环境需要使用者自己维护
- 默认训练配置偏向多卡集群环境，不一定适合单卡或本地机器
- README 是根据当前代码和 git 历史整理而成，若后续分支继续演进，内容需要同步更新

## 后续可扩展方向

- 增加统一的 `requirements.txt` 或 `pyproject.toml`
- 为不同分支整理更明确的实验配置文档
- 将实验结果表格化，便于横向比较
- 增加单卡 / 本地调试用的最小可运行配置
