# ESMM 梯度流向分析与改进方法研究

*Gradient Flow Analysis and Improvement of Entire Space Multi-Task Model*

---

## 摘要

本文针对阿里巴巴提出的 ESMM（Entire Space Multi-Task Model）框架，系统分析了其在 CTR/CVR 联合预估任务中的梯度流向问题。通过对三类样本——曝光未点击 $(y=0,z=0)$、点击未转化 $(y=1,z=0)$、点击且转化 $(y=1,z=1)$——的梯度方向与大小进行精确推导，揭示了 ESMM 在训练过程中存在的梯度冲突现象、CVR tower 信号稀疏问题以及两个 tower 之间的耦合不稳定性。在此基础上，本文提出基于 stop-gradient 的改进方案，并讨论了与 MMoE/CGC 的结合方式，以及 ESCM² 的反事实风险正则化方法。

---

## 1. 问题定义

### 1.1 CVR 预估的核心挑战

在电商推荐系统中，用户行为遵循序贯模式：**曝光 → 点击 → 转化**。CVR（转化率）定义为用户点击后发生购买的条件概率，是排序系统的核心指标之一。传统 CVR 模型面临两个根本性问题：

**样本选择偏差（Sample Selection Bias, SSB）**：CVR 模型的训练数据仅包含被点击的样本，但推断时需要对全部曝光样本打分。训练空间与推断空间的分布不一致，导致模型泛化能力受损。

**数据稀疏（Data Sparsity, DS）**：转化样本数量远少于点击样本，点击样本又远少于曝光样本，正样本极度稀疏，模型难以充分学习。

### 1.2 符号定义

设 $x$ 为曝光样本的特征向量，$y \in \{0,1\}$ 为点击标签，$z \in \{0,1\}$ 为转化标签（仅在 $y=1$ 时有意义）。三种有效样本类型为：

| 样本类型 | 含义 | 占比 |
|---|---|---|
| $(y=0, z=0)$ | 曝光但未点击 | 最多，约占 96% |
| $(y=1, z=0)$ | 点击但未转化 | 次多 |
| $(y=1, z=1)$ | 点击且转化 | 极少，正样本 |

### 1.3 ESMM 的建模方式

ESMM 利用概率链式法则将 CVR 隐式建模在全曝光空间上：

$$P(y=1, z=1 \mid x) = P(y=1 \mid x) \times P(z=1 \mid y=1, x)$$

$$\text{即：} \hat{p}_{CTCVR} = \hat{p}_{CTR} \times \hat{p}_{CVR}$$

其中 CTR 和 CTCVR 均可在全曝光空间上获得监督信号，CVR 不直接训练，而是通过乘积关系隐式约束。模型联合训练的损失函数为：

$$\mathcal{L} = \mathcal{L}_{CTR}(y, p) + \mathcal{L}_{CTCVR}(y \cdot z,\ p \cdot q)$$

其中 $p = \hat{p}_{CTR}$，$q = \hat{p}_{CVR}$，使用二元交叉熵损失。CVR tower 没有直接的监督 loss，梯度完全来自 CTCVR 项。

---

## 2. 梯度流向推导

### 2.1 损失函数展开

将损失函数按样本类型展开：

$$\mathcal{L}_{CTR} = -\left[y \log p + (1-y)\log(1-p)\right]$$

$$\mathcal{L}_{CTCVR} = -\left[yz \log(pq) + (1-yz)\log(1-pq)\right]$$

对 CTR tower 和 CVR tower 分别求偏导，得到各类样本上的梯度表达式。

### 2.2 Case 1：$(y=0, z=0)$ 曝光未点击

损失函数为：$\mathcal{L}_{CTR} = -\log(1-p)$，$\mathcal{L}_{CTCVR} = -\log(1-pq)$

对 $p$ 的梯度：

$$\frac{\partial \mathcal{L}}{\partial p} = \frac{1}{1-p} + \frac{q}{1-pq} > 0 \quad \text{（压低 } p \text{）}$$

对 $q$ 的梯度：

$$\frac{\partial \mathcal{L}}{\partial q} = \frac{p}{1-pq} > 0 \quad \text{（压低 } q \text{）}$$

**结论**：两个梯度恒为正，方向固定，压低 $p$ 和 $q$，无边界效应。这是正确的监督信号——未点击样本应降低 CTR 和 CVR 预测值。

### 2.3 Case 2：$(y=1, z=1)$ 点击且转化

损失函数为：$\mathcal{L}_{CTR} = -\log p$，$\mathcal{L}_{CTCVR} = -\log(pq)$

对 $p$ 的梯度：

$$\frac{\partial \mathcal{L}}{\partial p} = -\frac{1}{p} - \frac{1}{p} = -\frac{2}{p} < 0 \quad \text{（抬升 } p \text{）}$$

对 $q$ 的梯度：

$$\frac{\partial \mathcal{L}}{\partial q} = -\frac{1}{q} < 0 \quad \text{（抬升 } q \text{）}$$

**结论**：两个 loss 方向完全一致，$p$ 和 $q$ 在全空间被强力抬升，无冲突，无边界效应。

### 2.4 Case 3：$(y=1, z=0)$ 点击未转化（核心分析）

损失函数为：$\mathcal{L}_{CTR} = -\log p$，$\mathcal{L}_{CTCVR} = -\log(1-pq)$

对 $p$ 的梯度：

$$\frac{\partial \mathcal{L}}{\partial p} = \underbrace{-\frac{1}{p}}_{\mathcal{L}_{CTR},\ \text{抬升}} + \underbrace{\frac{q}{1-pq}}_{\mathcal{L}_{CTCVR},\ \text{压低}}$$

对 $q$ 的梯度：

$$\frac{\partial \mathcal{L}}{\partial q} = \frac{p}{1-pq} > 0 \quad \text{（恒为压低，无冲突）}$$

**对 $q$ 的分析**：CVR tower 在 $(y=1,z=0)$ 样本上只收到压低信号，方向恒定，符合真实标签（该用户未转化）。

**对 $p$ 的分析**：CTR tower 存在两项方向相反的力。梯度方向由 $pq$ 的大小决定，令 $\frac{\partial \mathcal{L}}{\partial p} = 0$，解得决策边界：

$$\frac{q}{1-pq} = \frac{1}{p} \implies pq = 0.5$$

| 条件 | CTR 项 | CTCVR 项 | 净方向 |
|---|---|---|---|
| $pq < 0.5$（早期） | $-1/p$（抬升） | $+q/(1-pq)$（压低，较小） | 净抬升（正确） |
| $pq = 0.5$（边界） | $-1/p$（抬升） | $+q/(1-pq)$（压低） | 梯度为 0 |
| $pq > 0.5$（后期） | $-1/p$（抬升） | $+q/(1-pq)$（压低，较大） | 净压低（冲突显现） |

### 2.5 梯度汇总

| 样本类型 | $\partial \mathcal{L}/\partial p$（CTR） | $\partial \mathcal{L}/\partial q$（CVR） | 冲突 |
|---|---|---|---|
| $(0,0)$ 曝光未点击 | ↓↓ 两路压低 | ↓ CTCVR 压低 | 无 |
| $(1,0)$ 点击未转化 | ↑↓ 对抗，边界 $pq=0.5$ | ↓ 恒压低 | $p$ 存在对抗 |
| $(1,1)$ 点击且转化 | ↑↑ 两路抬升 | ↑ CTCVR 抬升 | 无 |

---

## 3. 关键问题分析

### 3.1 CVR Tower 的梯度稀疏问题

CVR tower 没有直接 loss，完全依赖 CTCVR 的代理梯度学习。在三类样本上，CVR tower 的有效抬升信号只来自极少数 $(1,1)$ 样本，其余样本均为压低信号。

以训练早期典型值 $p=0.1$，$q=0.1$ 为例，在 $(y=1,z=0)$ 样本上：

$$\frac{\partial \mathcal{L}}{\partial p} = -\frac{1}{0.1} + \frac{0.1}{1-0.01} \approx -10 + 0.101 \approx -9.9 \quad \text{（正确抬升）}$$

$$\frac{\partial \mathcal{L}}{\partial q} = \frac{0.1}{1-0.01} \approx +0.101 \quad \text{（压低）}$$

CTR tower 早期梯度信号充足且方向正确，CVR tower 收到的抬升信号几乎全部来自稀缺的 $(1,1)$ 样本，是数据稀疏问题在梯度层面的直接体现。

### 3.2 训练耦合的不稳定性

ESMM 存在三方耦合结构：

- CVR 需要收敛 → 才能减少对 CTR 的梯度干扰
- CVR 的收敛 → 依赖 CTCVR 信号质量
- CTCVR 的信号质量 → 受 CTR 准确性影响

三者互相依赖，形成鸡生蛋问题，早期训练处于不稳定状态。具体表现为：不稳定的 $q$ 作为系数出现在 CTR 梯度中（$\partial \mathcal{L}_{CTCVR}/\partial p = q/(1-pq)$），向 CTR tower 引入噪声。

### 3.3 收敛状态分析

$(y=1,z=0)$ 样本的理想收敛状态为 $p \to 1$，$q \to 0$。此时：

$$\frac{\partial \mathcal{L}}{\partial p} = \underbrace{-\frac{1}{1}}_{=-1} + \underbrace{\frac{0}{1-0}}_{=0} = -1$$

CTCVR 对 CTR 的压制力归零，梯度冲突自动消解。然而这一收敛过程本身极难达到，原因正是 CVR 梯度信号的稀疏性——没有足够的 $(1,1)$ 样本将 $q$ 推向 0。

---

## 4. 改进方案

### 4.1 Stop-Gradient 方案

在 $(y=1,z=0)$ 样本的 CTCVR loss 反向传播时，对 $p$ 做梯度截断：

```python
# 仅对 (y=1, z=0) 样本
ctcvr = p.detach() * q
```

截断效果：

| Tower | 操作前 | 操作后 |
|---|---|---|
| CTR $(p)$ | $-1/p + q/(1-pq)$，有冲突 | 只剩 $-1/p$，干净抬升 |
| CVR $(q)$ | $p/(1-pq)$，压低 | 不变，完整保留 |

对 $(0,0)$ 和 $(1,1)$ 样本不做截断——这两类样本上 CTCVR 对 CTR 的梯度方向与 $\mathcal{L}_{CTR}$ 一致，是有效信号。

该方案与 ESCM² 附录中的工程实现完全一致——ESCM² 在 IPS 正则化项中对倾向得分（CTR 输出）同样做了 `.detach()` 处理，防止 CVR loss 污染 CTR 估计。

### 4.2 与 MMoE/CGC 的结合

ESMM 的共享 Embedding 是**硬共享**，梯度冲突在 Embedding 层直接叠加。

**MMoE（软共享）**：让 CTR 和 CVR 通过各自的 gating network 动态选择 Expert 组合，在特征层隔离梯度冲突。局限在于 Expert 坍缩——CVR gating 因信号稀疏，早期退化为均匀分布，软共享退化为硬共享。

**CGC（PLE）**：显式区分 task-specific Expert 和 shared Expert，强制隔离跨任务梯度，不依赖 gating 自发学会分化，从结构上打破耦合。

两种改进互补，作用在不同层面：

| 层面 | 问题 | 解法 |
|---|---|---|
| 梯度流 | $(1,0)$ 样本上的对抗梯度 | Stop-gradient |
| 特征层（软） | 硬共享导致梯度叠加 | MMoE |
| 特征层（硬） | Expert 坍缩，软共享失效 | CGC |

### 4.3 ESCM² 的反事实正则化

ESCM² 引入反事实风险最小化器，直接给 CVR tower 施加全空间监督，解决两个问题：

**IEB（Inherent Estimation Bias）**：ESMM 的 CVR 估计恒大于真实值。由 Jensen 不等式证明：

$$\mathbb{E}_\mathcal{D}[\hat{R}] \geq \frac{\mathbb{E}_\mathcal{D}[\hat{C}]}{\mathbb{E}_\mathcal{D}[\hat{O}]} = \mathbb{E}_\mathcal{O}[R] > \mathbb{E}_\mathcal{D}[R]$$

**PIP（Potential Independence Priority）**：ESMM 隐含 $O \perp R$ 假设，忽略了点击到转化的因果链 $O \to R$。

最终损失函数：

$$\mathcal{L}_{ESCM^2} = \mathcal{L}_{CTR} + \lambda_c \mathcal{L}_{CVR} + \lambda_g \mathcal{L}_{CTCVR}$$

其中 $\mathcal{L}_{CVR}$ 通过 IPS 或 DR 正则化实现无偏估计，实现时对倾向得分做 stop-gradient 防止反向污染 CTR。

---

## 5. 结论

通过对 ESMM 三类样本梯度的系统推导，本文揭示了以下核心问题：

- $(y=1,z=0)$ 样本上 CTR tower 存在梯度对抗，边界由 $pq=0.5$ 决定
- CVR tower 缺乏直接监督，仅依赖稀疏的 $(1,1)$ 样本抬升
- 两个 tower 通过乘法结构形成三方耦合，早期训练不稳定
- CVR 系统性高估源于训练分布与推断分布不一致

基于梯度分析推导的改进方案共同构成一个层次清晰的体系：

| 改进 | 解决的问题 | 作用层面 |
|---|---|---|
| Stop-gradient | $(1,0)$ 样本梯度冲突 | 梯度流 |
| MMoE + CGC | Embedding 层梯度耦合 | 特征共享 |
| ESCM²（IPS/DR） | CVR 偏估计，因果链缺失 | 无偏性 |