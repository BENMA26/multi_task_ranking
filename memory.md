# Multi-Task Ranking Memory

更新时间：2026-03-31 23:54 CST (+0800)

## 项目简介

这是一个 Ali-CCP 多任务排序项目，核心目标是联合建模 CTR / CVR（ESMM 范式），当前主训练入口是：

- `train_rank.py`（层次化 CLI）

当前主要实验维度：

- 模型：`share_bottom`、`mmoe`、`ple`
- 正则：`none`、`entropy`
- 梯度手术：`normal`、`upgrad`、`pcgrad`、`asymmetric_projection`
- 作用范围：`all`、`on10`（即 `(click=1, purchase=0)`）

## 本次会话我们做了什么

1. 通读并恢复了项目记忆
- 阅读了 `README.md`、`train_rank.py`、`src/models/ranking.py`、`src/data/dataset.py` 等核心文件。
- 梳理了现有实验脚本与结果目录结构。

2. 修复并补齐了 asymmetric projection 的 restore-norm 对照实验脚本
- 之前问题：部分脚本只有 `restore_norm` 或只有非 `restore_norm`，缺少成对可比对照。
- 处理结果：为 share_bottom（`all`/`on10`）和 mmoe（`on10+entropy`）都补成了成对实验模式（`off/on`）。

3. 你要求删除 `experiments` 后，我执行了删除并提交
- 提交记录：
  - commit: `fb80b2d`
  - message: `chore: remove experiments directory and snapshot current training code`
- 推送状态：两次 `git push` 因网络失败（GitHub 443 连接问题）未成功。

4. 按你的新需求重新创建实验目录并新增双节点串行脚本
- 新目录：`experiments/eight_settings_dualnode/`
- 新脚本：
  - `run_nodeA_serial.sh`（setting 1-4）
  - `run_nodeB_serial.sh`（setting 5-8）
- 资源配置按你确认：
  - partition=`normal`
  - time=`144:00:00`
  - cpus-per-task=`40`
  - mem=`96G`
  - 每作业 `gpu:4`
- 实验设置共 8 组，默认 seeds 为 `42,2027,3407`，总任务数 24（每节点 12，串行执行）。

5. 新增 smoke test 脚本并完成快速验证
- 脚本：`scripts/smoke_test_rank_settings.sh`
- 覆盖 8 个 setting 的代码路径完整性验证。
- 快速测试结果：`8/8 OK`（小数据 + 1 epoch + CPU smoke 跑通）。

## 当前集群运行状态（检测时）

检测时间：2026-03-31 23:54 CST  
`squeue -u maben` 显示与本项目相关作业正在运行：

- `146854`（nodeA 脚本）
- `146855`（nodeB 脚本）

对应日志目录：

- `experiments/eight_settings_dualnode/slurm/`

## 当前仓库状态（需注意）

当前有未跟踪文件（尚未提交）：

- `.smoke_data/`（smoke 临时数据）
- `experiments/`（新建双节点实验脚本与日志）
- `scripts/smoke_test_rank_settings.sh`

建议后续：

1. 运行完成后先清理 `.smoke_data/` 与不必要日志。
2. 选择保留的脚本与必要结果，整理后再提交。
3. 网络恢复后执行 `git push origin <branch>`。

