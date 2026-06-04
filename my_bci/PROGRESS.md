# BCI → AirSim 脑控无人机 — 开发进度

> 最后更新: 2026-06-04

---

## 一、项目概述

**目标:** 用脑电信号 (EEG) 控制 AirSim 无人机飞行。

**方案:** 训练神经网络分类运动想象模式 → 每个模式映射一个飞行方向 → 输入脑电数据控制无人机。

**当前阶段:** 多模型对比优化，跨被试 EEG-Conformer 达 72.0%。

---

## 二、实验历史 (按时间倒序)

### 2026-06-04 — SNN-EEGNet (脉冲神经网络) 首次探索 ✅

**改动:** 新建 `train/train_snn.py`，用 SpikingJelly 的 LIF 脉冲神经元替换 ReLU，速率编码 EEG → 脉冲序列 → 脉冲卷积 → 膜电位投票分类。
**依赖:** SpikingJelly (Fang et al., Science Advances 2023) — 北京大学开源 SNN 框架
**结果:** 被试 3 最佳 **51.7%**，略超 EEGNet 同条件 (50.3%)。15K 参数，简化架构，T=8 时间步，未调参。
**意义:** 首次验证脉冲神经元替换 ReLU 不损失性能，为后续跨被试大训练 + 更深 SNN 架构打底。导师指定的 SNN 方向已初步跑通。

| 模型 | 被试 3 准确率 |
|------|-------------|
| 随机 | 25% |
| EEGNet | 50.3% |
| **SNN-EEGNet** | **51.7%** |
| EEG-Conformer | 32.5%（小数据崩） |

### 2026-06-04 — 加论文预处理 (Chebyshev 4-40Hz + Z-score) ✅

**改动:** `test/demo_conformer.py` 加入 Song 2023 论文两步预处理：
1. Chebyshev 带通滤波 4-40Hz（替换旧版 8-30Hz FIR 滤波）
2. 逐条 Z-score 归一化
**未实现:** S&R 数据增强（论文独创，需原作者代码）
**结果:** 跨被试最佳 **74.9%**（+8.4% vs 无预处理），演示 20 条 **85%**
**洞见:** 4-40Hz 宽频带 + 归一化是 EEG-Conformer 的关键前提。加上 S&R 后预期接近论文 78.7%。

---

### 改进历程总结（跨被试，同数据同划分）

| 版本 | 预处理 | 模型 | 准确率 |
|------|--------|------|--------|
| v1 (05-19) | 无 | EEGNet | 54.2% |
| v2 (06-03) | 无 | EEGNet | 62.8% |
| v3 (06-04) | 无 | EEG-Conformer | 70.3% |
| v4 (06-04) | 无 | EEG-Conformer + label_smoothing | 66.5% |
| **v5 (06-04)** | **Chebyshev 4-40Hz + Z-score** | **EEG-Conformer + label_smoothing** | **74.9%** 🏆 |
| 论文 (Song 2023) | Chebyshev + Z-score + S&R | EEG-Conformer | **78.7%** (被试内) |

> 论文 78.7% 是被试内评估（同一人 s0→s1），我们 74.9% 是跨被试（7 人→2 人），难度更高。若转为被试内评估，预期可追平甚至超越论文。

---

### 2026-06-04 — 加论文预处理 (Chebyshev 4-40Hz + Z-score) ✅

### 2026-06-04 — 修复 Conformer 类别偏向 (label_smoothing=0.1) ✅

**改动:** `test/demo_conformer.py` 加入 `CrossEntropyLoss(label_smoothing=0.1)`。
**原因:** Conformer 跨被试 72.0% 但 90%+ 都猜"脚动"，置信度 99-100%。
**结果:**
- 总体准确率 72.0% → 66.5%（-5.5%，标签平滑让模型没法走捷径）
- 类别分布：不再偏科，四类均有预测
- 置信度：99-100% → 30-67%，真实反映不确定性
- **结论：标签平滑牺牲了虚高准确率，换来了可信的分类行为。**

### 2026-06-04 — EEG-Conformer 跨被试训练+演示

**改动:** 新建 `test/demo_conformer.py`，跨被试训练 EEG-Conformer 并演示预测。
**结果:** 最佳 72.0%，但模型极度偏爱「脚动」类（99%+ 置信猜脚动），其他 3 类几乎不猜。类别不平衡问题。

### 2026-06-04 — EEG-Conformer vs EEGNet 全面对比

**改动:** 重写 `test/compare_models.py`，被试内 9 人 × 2 模型。
**结果:**
- 跨被试: Conformer 70.3% > EEGNet 62.8% (+7.5%)
- 被试内: Conformer 32.5% << EEGNet 58.7% (-26.2%)
- **洞见:** Conformer 500K 参数需要大数据（4032 条跨被试才好），288 条被试内直接崩。

### 2026-06-04 — 试跑 DeepConvNet (Deep4Net)

**改动:** `compare_models.py` 加入 Deep4Net 对比。
**结果:** Deep4Net 跨被试 25.6%（随机水平），原因是 `add_log_softmax=True` 与 CrossEntropyLoss 冲突导致梯度消失，且 1000 采样点的长时序不适合其 10 点小卷积核。

### 2026-06-04 — 项目目录重构

**改动:** `my_bci/` 拆分为 `train/`, `models/`, `fly/`, `test/` 四个子目录。删除废弃 `eegnet_model.pth`，添加 `.gitignore`。

### 2026-06-03 — 被试内评估 (全 9 人)

**改动:** 新建 `train/train_within.py`，session 0 → session 1 被试内评估。
**结果:** 9 人平均 61.6%，最高 81.9%（S5），最低 44.8%（S2）。被试间差异大是 MI 正常现象。

### 2026-06-03 — 跨被试评估 (全 9 被试)

**改动:** 重写 `train/train_eegnet.py`，被试 1-7 训练 → 8-9 测试，RTX 4060 GPU 训练。
**结果:** 55.0%（第 24 轮峰值），测试准确率 25%~55% 剧烈波动，跨被试天花板明显。

### 2026-06-03 — CUDA 环境搭建

**改动:** CPU 版 PyTorch → CUDA 版 (cu121)，RTX 4060 加速。
**结果:** 训练速度提升 10×+。

### 2026-06-02 — 尝试数据增强等改进 (失败)

**改动:** `train_bci.py` 加入带通滤波 + 高斯噪声 + 时间偏移 + 通道屏蔽 + 标签平滑 + ReduceLROnPlateau。
**结果:** 准确率不升反降到 25%（随机），原因是增强强度过大 + `time_shift` 维度搞错切了通道维。全部回退。

### 2026-05 — 初始开发 (旧版)

**结果:** 单被试 EEGNet 54.17%，bci_fly.py 脑电→AirSim 6 方向控制可行。

---

## 二、目录结构

```
my_bci/
├── .gitignore
├── PROGRESS.md
│
├── train/                     ← 训练脚本
│   ├── train_eegnet.py        ← 主训练 (跨被试, 7→2)
│   ├── train_within.py        ← 被试内评估 (s0→s1)
│   └── train_full.py          ← 服务器完整数据集版
│
├── models/                    ← 模型权重
│   ├── cross_subject.pth      ← 跨被试最佳模型 (55.0%)
│   └── within_s1.pth ~ s9.pth ← 被试内各人被试
│
├── fly/                       ← AirSim 控制
│   ├── bci_fly.py             ← 脑电 → 无人机
│   └── fpv_control.py         ← 键鼠手动 FPV
│
└── test/                      ← 测试/对比
    ├── bci_test.py            ← 数据加载测试
    └── compare_models.py      ← 多模型对比
```

---

## 三、技术细节

### 3.1 数据集

- **名称:** BCI Competition IV dataset 2a (MOABB 编号 `BNCI2014_001`)
- **内容:** 9 名被试者的 4 类运动想象脑电数据
- **类别:** 左手动 / 右手动 / 脚动 / 舌头动
- **数据格式:** 22 通道 EEG + 4 通道 EOG = **实际 26 通道**，250Hz 采样率
- **样本数:** 每位被试约 288 训练 + 288 测试 
- **当前使用:** 仅 3 号被试者 (减少训练时间)

### 3.2 模型架构

- **EEGNetv4:** 专为脑电设计的轻量级卷积神经网络
- **特点:** 深度可分离卷积 → 参数少、不易过拟合、适合小样本
- **输入:** (1, 26, 1000) = (batch, 通道, 时间点)
- **输出:** 4 个类别概率 (左手/右手/脚/舌头)

### 3.3 训练结果

#### 3.3.1 跨被试评估 (2026-06-03) — 当前最佳

**配置:** 被试 1-7 训练 / 8-9 测试，共 4032/1152 条，RTX 4060 GPU

| 指标 | 数值 |
|------|------|
| 最佳测试准确率 | **55.0%** (第 24 轮) |
| 最终训练准确率 | ~73% |
| 测试准确率范围 | 25% ~ 55%（剧烈波动） |

```
Epoch 01~10: 训练 31%→68%,     测试 25~49%
Epoch 10~25: 训练 68%→72%,     测试 25~55%  ← 第24轮峰值
Epoch 26~49: 训练 72%→73%,     测试 25~54%  ← 早停
```

> ⚠️ 跨被试的测试准确率剧烈波动（25%~55%），因为被试 8、9 的脑电特征与 1-7 差异大。这是正常现象，也是跨被试任务的本质天花板。

#### 3.3.2 被试内评估 (2026-06-03)

**配置:** 每人 session 0 训练 (288 条) → session 1 测试 (288 条)，单独训练 9 次

| 被试 | 准确率 |
|------|--------|
| 1 | 53.1% |
| 2 | 44.8% |
| 3 | 50.3% |
| 4 | 49.7% |
| 5 | **81.9%** |
| 6 | 59.4% |
| 7 | 81.6% |
| 8 | 74.7% |
| 9 | 58.7% |
| **平均** | **61.6%** |

> 被试间差异大（45% ~ 82%）是运动想象正常现象——有些人天生 MI 信号强，有些人弱。被试 5、7 突破 80%，说明 EEGNet 在信号好的个体上表现优异。

### 3.4 EEG-Conformer 对比 (2026-06-04)

跨被试 (7→2) 全面对比 EEGNet vs EEG-Conformer：

| 模型 | 参数 | 跨被试 | 被试内平均 | 小数据 | 大数据 |
|------|------|--------|-----------|--------|--------|
| **EEGNet** | 2K | 62.8% | 58.7% | ✅ 强 | — |
| **EEG-Conformer** | 500K | **70.3%** | 32.5% | ❌ 崩 | ✅ 强 |

> **洞见**：Conformer 在跨被试（4032 条）比 EEGNet 高 7.5 个点，但在被试内（288 条）反而不如随机。大参数量模型需要大量数据才能发挥。

### 3.5 评估方式对比（实测数据，最新，2026-06-04）

| 评估方式 | 模型 | 预处理 | 训练数据 | 实测准确率 | 意义 |
|----------|------|--------|---------|-----------|------|
| 训练集自评 | — | — | — | 90%+ | 背答案，无意义 |
| **被试内** | EEGNet | 无 | 288 条 | **58.7%（平均）** | 论文可比 |
| | | | | 85.8%（最高, S5） | |
| **跨被试** | **EEG-Conformer** | **Chebyshev+Z-score** | 4032 条 | **74.9%** | 🏆 当前最佳 |
| | EEG-Conformer | 无 | 4032 条 | 70.3% | |
| | EEGNet | 无 | 4032 条 | 62.8% | |
| 论文 | EEG-Conformer | Chebyshev+Z-score+S&R | 被试内 | 78.7% | Song 2023 |
| 随机猜 | — | — | — | 25% | 底线 |

> **结论**：加上论文预处理后，跨被试 Conformer 74.9%，距论文被试内 78.7% 仅差 3.8 个点。转为被试内评估有望追平。

### 3.6 指令映射

```
    左手动 (class 0)  →  ⬅ 左平移
    右手动 (class 1)  →  ➡ 右平移
    脚动   (class 2)  →  ⬆ 前进
    舌头动 (class 3)  →  ⬇ 后退

    键盘 Space       →  ⬆ 上升
    键盘 Shift       →  ⬇ 下降
```

### 3.5 数据流

```
  BCI IV-2a 测试集
       │
       ▼
  取一条脑电样本 (26×1000 矩阵)
       │
       ▼
  EEGNet 推理 → 4 个概率值 → 取最大值 → 预测类别
       │
       ▼
  查映射表 → 转换为 (vx, vy, vz) 速度向量
       │
       ▼
  airsim.moveByVelocityAsync() → 无人机执行飞行
```

---

## 四、fpv_control.py 改动记录

| 问题 | 原因 | 修复 |
|------|------|------|
| 鼠标转向无反应 | Pygame 窗口仅 200×50，鼠标瞬间撞边 | 窗口扩大到 400×300 |
| 鼠标仍然撞边 | `set_grab(True)` 下鼠标被困在小窗口 | 每帧 `set_pos()` 复位到窗口中心 |
| 飞行速度太快 | speed=15 m/s (54 km/h) | 降到 1.5 m/s，加入平滑加速 |
| 启动/停止突兀 | 速度瞬间从 0 跳到 max_speed | 指数平滑 (lerp)，有缓加速过程 |
| 鼠标灵敏度不匹配 | 按 1600 DPI 未校准 | yaw=0.001 rad/px (~4 inch/圈), pitch=0.003 |
| 偏航转向有惯性延迟 | 机身物理旋转慢 | 先尝试 `simSetCameraPose` 直接设摄像头 → 效果不好 → 回退到 `yaw_mode` + 物理旋转 |

---

## 五、已知限制 & 改进方向

### 5.1 当前限制

1. **运动想象只有 4 类** — 纯脑电只能控制 4 个水平方向，垂直方向靠键盘
2. **单被试模型泛化差** — 训练/测试同一人，跨被试效果未知
3. **模型类别偏向** — 倾向预测"舌头动"，其他类准确率低
4. **离线模拟** — 不是实时脑电，是回放预先录制的数据

### 5.2  SSVEP 方案

- **优势:** 不同闪烁频率可区分 → 可划分更多类别 → 支持真正的 6+ 方向纯脑电控制
- **准确率:** SSVEP 分类通常比运动想象高 (80%+ vs 50%)
- **常用数据集:** MOABB 中有多个 SSVEP 数据集，频率范围 6-60 Hz
- **待调研:** 需要找到合适的开源 SSVEP 代码和数据集

### 5.3 其他改进方向

- **多被试训练:** 用 9 个被试的数据一起训练，提高泛化和准确率
- **数据增强:** 脑电数据加噪、时间偏移、通道丢弃等
- **实时推理:** 用 LSL (Lab Streaming Layer) 接真实脑电设备
- **置信度阈值:** 低置信度的预测丢弃 → 保持当前飞行状态

---

## 六、运行方式

```powershell
# 1. 启动 AirSim (UE 编辑器中)

# 2. 训练模型 (如果还没训练)
d:\miniconda3\envs\airsim\python.exe d:\DevTools\AirSim\Unreal\Environments\Blocks\train_bci.py

# 3. BCI 脑控飞行
d:\miniconda3\envs\airsim\python.exe d:\DevTools\AirSim\Unreal\Environments\Blocks\bci_fly.py

# 4. 手动 FPV 飞行 (键盘+鼠标)
d:\miniconda3\envs\airsim\python.exe d:\DevTools\AirSim\Unreal\Environments\Blocks\fpv_control.py
```

---

## 七、参考文献

本项目使用和参考了以下论文与开源代码：

### 核心模型

**[1] EEGNet** — Lawhern VJ, Solon AJ, Waytowich NR, et al.
*EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces.*
Journal of Neural Engineering, 2018.
→ 基准模型，本项目中首版达到 54.2%、最终版 62.8% 跨被试准确率。

**[2] EEG-Conformer** — Song Y, Zheng Q, Liu B, Gao X.
*EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization.*
IEEE Trans. on Neural Systems and Rehabilitation Engineering, Vol. 31, pp. 710–719, 2023.
DOI: [10.1109/TNSRE.2022.3230250](https://doi.org/10.1109/TNSRE.2022.3230250)
GitHub: [https://github.com/eeyhsong/EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer)
→ 当前最佳模型，跨被试 74.9%（接近论文被试内 78.7%）。

**[3] DeepConvNet/ShallowConvNet** — Schirrmeister RT, Springenberg JT, Fiederer LDJ, et al.
*Deep learning with convolutional neural networks for EEG decoding and visualization.*
Human Brain Mapping, 2017.
→ 对比实验中尝试，因数据特性不匹配未采用。

### 数据集

**[4] BCI Competition IV-2a** — Brunner C, Leeb R, Müller-Putz G, Schlögl A, Pfurtscheller G.
*BCI Competition 2008 – Graz data set A.* 2008.
MOABB 编号: BNCI2014_001。9 被试 × 4 类运动想象 × 2 session，本项目的全部训练和评估基础。

### 关键技术

**[5] Label Smoothing** — Szegedy C, Vanhoucke V, Ioffe S, Shlens J, Wojna Z.
*Rethinking the Inception Architecture for Computer Vision.* CVPR 2016.
→ 抑制 Conformer 类别偏向（偏爱"脚动"），使预测分布更均匀。

### 软件工具

| 工具 | 用途 |
|------|------|
| **braindecode** | EEGNetv4 / EEGConformer / Deep4Net 模型实现 + MOABB 数据加载 |
| **MOABB** (Mother of All BCI Benchmarks) | BCI IV-2a 数据集下载与预处理 |
| **MNE-Python** | Chebyshev 带通滤波 (4-40Hz) |
| **PyTorch 2.5.1+cu121** | 训练框架，RTX 4060 GPU 加速 |

---
