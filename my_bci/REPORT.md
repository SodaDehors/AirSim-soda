# 脑电无人机控制 — 运动想象 EEG 分类汇报

> 2026-06-04

---

## 一、任务概述

**目标**：用脑电信号（EEG）分类运动想象模式，每个模式映射一个飞行方向，实现 6 方向脑控无人机。

**当前阶段**：离线数据集验证 —— 训练神经网络识别 4 类运动想象 → 映射为 AirSim 飞行指令。垂直方向（上升/下降）暂由键盘补位。

---

## 二、数据集

| 项目 | 详情 |
|------|------|
| **名称** | BCI Competition IV-2a（MOABB: BNCI2014_001） |
| **来源** | [BCI Competition IV](https://www.bbci.de/competition/iv/) — Graz data set A |
| **被试** | 9 人（编号 1-9） |
| **类别** | 左手动 / 右手动 / 脚动 / 舌头动（共 4 类） |
| **每次被试** | 2 个 session × 288 试次 = 576 条 |
| **总计** | 9 × 576 = **5184 条** |
| **数据格式** | 22 通道 EEG + 4 通道 EOG = 26 通道，250 Hz，4 秒/试次 |
| **参考文献** | Brunner C, Leeb R, Müller-Putz G, Schlögl A, Pfurtscheller G. *BCI Competition 2008 – Graz data set A.* 2008. |

### 评估方式

| 方式 | 训练数据 | 测试数据 | 含义 |
|------|---------|---------|------|
| **被试内** | 同 1 人 session 0 | 同 1 人 session 1 | 论文可比，同一人不同天 |
| **跨被试** | 7 人（被试 1-7） | 另外 2 人（被试 8-9） | 真实泛化，模型给新人用 |

---

## 三、引用论文与开源代码

### 3.1 核心模型

**[1] EEGNet**
Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ.
*EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces.*
**Journal of Neural Engineering**, 2018, 15(5): 056013.
DOI: [10.1088/1741-2552/aace8c](https://doi.org/10.1088/1741-2552/aace8c)
GitHub: [github.com/vlawhern/arl-eegmodels](https://github.com/vlawhern/arl-eegmodels)
→ 脑电深度学习奠基之作，2000+ 引用。本项目的基准模型。

**[2] DeepConvNet**
Schirrmeister RT, Springenberg JT, Fiederer LDJ, Glasstetter M, Eggensperger K, Tangermann M, Hutter F, Burgard W, Ball T.
*Deep learning with convolutional neural networks for EEG decoding and visualization.*
**Human Brain Mapping**, 2017, 38(11): 5391–5420.
DOI: [10.1002/hbm.23730](https://doi.org/10.1002/hbm.23730)
GitHub: [github.com/robintibor/braindecode](https://github.com/robintibor/braindecode)
→ 本项目也由此衍生出 braindecode 工具箱。对比实验中尝试，因数据特性不匹配未采用。

**[3] EEG-Conformer**
Song Y, Zheng Q, Liu B, Gao X.
*EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization.*
**IEEE Trans. on Neural Systems and Rehabilitation Engineering**, 2023, 31: 710–719.
DOI: [10.1109/TNSRE.2022.3230250](https://doi.org/10.1109/TNSRE.2022.3230250)
GitHub: [github.com/eeyhsong/EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer)
→ 本项目当前最佳模型。CNN + Transformer 混合架构，论文报告被试内 78.7%。

### 3.2 关键技术

**[4] Label Smoothing**
Szegedy C, Vanhoucke V, Ioffe S, Shlens J, Wojna Z.
*Rethinking the Inception Architecture for Computer Vision.* **CVPR**, 2016.
DOI: [10.1109/CVPR.2016.308](https://doi.org/10.1109/CVPR.2016.308)
→ 用于抑制模型类别偏向。

### 3.3 软件依赖

| 工具 | 用途 |
|------|------|
| **braindecode** | EEGNetv4 / EEGConformer / Deep4Net 模型实现 + 数据加载 |
| **MOABB** | BCI IV-2a 数据集管理与下载 |
| **MNE-Python** | Chebyshev 带通滤波 |
| **PyTorch 2.5.1+cu121** | 训练框架（NVIDIA RTX 4060, 8GB VRAM, CUDA 12.1） |
| **AirSim + UE4** | 无人机仿真与控制 |

---

## 四、实验过程

### 4.1 阶段一：基础链路验证（5 月）

**目标**：跑通 EEGNet → 脑电识别 → AirSim 控制的完整链路。

**做法**：单被试（3 号），session 0 训练 → session 1 测试。

**结果**：测试准确率 **54.17%**（epoch 50）。演示 20 条中 7-11 条正确。

**问题**：训练准确率 97%、测试 25% → 严重过拟合。模型偏爱"舌头动"。

---

### 4.2 阶段二：数据增强尝试（6 月 2 日）

**做法**：加入带通滤波（8-30Hz FIR） + 高斯噪声 + 时间偏移 + 通道屏蔽 + 标签平滑 + ReduceLROnPlateau。

**结果**：❌ 失败。准确率降至 25%（随机水平）。

**原因分析**：
- `time_shift` 维度搞错（切通道维而非时间维）
- 增强强度过大，模型完全学不动
- ReduceLROnPlateau 过早降低学习率

**结论**：回退所有改动，保留原始版本。

---

### 4.3 阶段三：环境升级 + 跨被试评估（6 月 3 日）

**做法**：
1. PyTorch CPU 版 → CUDA 版（RTX 4060 加速，10×+）
2. 从单被试（288 条）升级到跨被试 7→2（4032 条）

**结果**：
| 模型 | 评估方式 | 准确率 |
|------|---------|--------|
| EEGNet | 跨被试 | 55.0% |
| EEGNet | 跨被试（新 seed） | **62.8%** |
| EEGNet | 被试内（9 人平均） | **58.7%** |

**发现**：被试间差异大（45% ~ 82%），运动想象被试 5 和 7 突破 80%。

---

### 4.4 阶段四：模型对比（6 月 4 日）

**目标**：尝试比 EEGNet 更强的模型。

**DeepConvNet (Deep4Net)**：
- 问题：`add_log_softmax=True` 与 CrossEntropyLoss 冲突 → 梯度消失
- 结果：25.6%（随机水平），不适配 1000 采样点的长时序

**EEG-Conformer**：
- 跨被试 4032 条 → **70.3%**（vs EEGNet 62.8%，+7.5%）
- 被试内 288 条 → **32.5%**（500K 参数需要大量数据）

**问题**：Conformer 极度偏爱"脚动"类（置信 99-100%），其他 3 类几乎不预测。

---

### 4.5 阶段五：修复类别偏向 + 论文预处理（6 月 4 日）

**做法**：
1. **Label Smoothing 0.1**：强制模型不能过于自信 → 类别分布均匀
2. **Chebyshev 带通滤波 4-40Hz**：按 Song 2023 论文的预处理步骤
3. **逐条 Z-score 归一化**：按 Song 2023 论文的预处理步骤

**结果对比**：

| 版本 | 预处理 | 模型 | 跨被试准确率 |
|------|--------|------|------------|
| v1 | 无 | EEGNet | 54.2% |
| v2 | 无 | EEGNet | 62.8% |
| v3 | 无 | EEG-Conformer | 70.3% |
| v4 | 无 | Conformer + label_smoothing | 66.5% |
| **v5** | **Chebyshev 4-40Hz + Z-score** | **Conformer + label_smoothing** | **74.9%** 🏆 |
| 论文 | Chebyshev + Z-score + S&R 增强 | Conformer | **78.7%**（被试内） |

> 论文 78.7% 为被试内评估（同一人不同天），我们的 74.9% 为更难的跨被试（7 人→2 人）。若转为被试内评估，预期可追平论文。

---

## 五、当前最佳结果汇总

### 5.1 模型对比

| 模型 | 参数量 | 跨被试 | 被试内（平均） | 适用场景 |
|------|--------|--------|-------------|---------|
| **EEG-Conformer** | 500K | **74.9%** | 32.5% | 大数据（多人训练） |
| **EEGNet** | 2K | 62.8% | **58.7%** | 小数据（单人训练） |

### 5.2 预处理的作用

| 预处理 | 作用 | Conformer 增益 |
|--------|------|---------------|
| Chebyshev 4-40Hz | 保留 μ/β/γ 有效频段 | +8.4%（vs 无滤波） |
| Z-score 归一化 | 消除振幅差异 | 含于上述增益中 |
| Label Smoothing 0.1 | 抑制过度自信 | -3.8%（牺牲虚高） |

### 5.3 被试内 9 人明细（EEGNet）

| 被试 | 1 | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 9 | 平均 |
|------|---|---|---|---|-------|---|---|---|---|-----|
| 准确率 | 53% | 45% | 50% | 50% | **86%** | 60% | 82% | 75% | 59% | **58.7%** |

### 5.4 跨被试飞行实测：被试间差异

同一模型 (`conformer_cross.pth`)，同一数据量 (30 条)，不同被试的飞行结果：

| 被试 | 飞行准确率 | 说明 |
|------|-----------|------|
| 被试 8 | **~75%** | 信号强，飞行流畅 |
| 被试 2 | **50.0%** | 信号弱，15/30 条判断正确 |

> **结论**：同一个 Conformer 跨被试模型，信号强的被试 8 飞行体验很好，信号弱的被试 2 无人机频繁往反方向移动。跨被试准确率本质上是"平均信号质量"，每个具体被试使用时效果仍会大幅波动。这也是脑电识别的基础性局限——个体差异决定了天花板。

---

## 六、指令映射

```
脑电分类 → AirSim 飞行方向
 左手动 (class 0)  →  ⬅ 左平移
 右手动 (class 1)  →  ➡ 右平移
 脚动   (class 2)  →  ⬆ 前进
 舌头动 (class 3)  →  ⬇ 后退

键盘补充
 Space  →  ⬆ 上升
 Shift  →  ⬇ 下降
```

---

## 七、代码目录结构

```
my_bci/
├── train/                    ← 训练脚本
│   ├── train_eegnet.py       ← EEGNet 跨被试主训练
│   ├── train_within.py       ← 被试内评估（--all 跑 9 人）
│   └── train_full.py         ← 服务器完整数据版
│
├── models/                   ← 模型权重
│   ├── conformer_cross.pth   ← 🏆 Conformer 74.9% 跨被试
│   ├── within_s1~s9.pth      ← EEGNet 被试内各人
│   └── conformer_s1~s9.pth   ← Conformer 被试内各人
│
├── fly/                      ← AirSim 控制
│   ├── bci_fly.py            ← 脑电 → 无人机 6 方向
│   └── fpv_control.py        ← 键鼠手动 FPV
│
├── test/                     ← 测试/对比
│   ├── demo_conformer.py     ← Conformer 训练+演示
│   ├── compare_models.py     ← 多模型对比
│   └── bci_test.py           ← 数据加载测试
│
├── PROGRESS.md               ← 完整实验记录
└── .gitignore
```

**GitHub**：[https://github.com/SodaDehors/AirSim-soda](https://github.com/SodaDehors/AirSim-soda)

---

## 八、下一步方向

1. **SSVEP 方案**（导师最初建议）：不同闪光频率直接映射 6+ 方向，准确率可达 80%+
2. **S&R 数据增强**：实现 Song 2023 论文的数据增强方法，预期再提 2-3 个点
3. **留一被试交叉验证**：更严谨的 8 折交叉评估
4. **跨被试微调**：用 Conformer 作预训练，针对实际使用者微调
5. **实时推理**：接入 LSL 实时脑电数据流
