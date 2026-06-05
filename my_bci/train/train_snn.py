"""
SNN-EEGNet 脉冲神经网络训练脚本（跨被试 / 单被试）
=====================================================

【什么是脉冲神经网络 (SNN)】
  普通网络（如 EEGNet）用 ReLU 激活，输出连续实数。
  SNN 用 LIF（Leaky Integrate-and-Fire）神经元激活，输出 0/1 脉冲。
  脉冲 = 像生物神经元一样"达到阈值就放电，放完电就重置"。

【数据处理流程】
  EEG(26×1000 矩阵) → 归一化到 [0,1] → 速率编码 → 脉冲序列 → SNN → 膜电位投票 → 分类

【速率编码 (Rate Coding)】
  连续值越大 → 泊松脉冲发放概率越高 → 单位时间内脉冲越多。
  相当于把 EEG 的幅值强度"翻译"成脉冲频率。

【LIF 神经元 (Leaky Integrate-and-Fire)】
  公式: V(t) = decay × V(t-1) + (1-decay) × R × input
  - 膜电位持续积分输入脉冲
  - 达到阈值 v_threshold 就发放一个脉冲，然后重置为 v_reset
  - decay (tau) 控制"记忆衰减速度"，t=2.0 表示膜电位每步衰减到 ≈61%

【代理梯度 (Surrogate Gradient)】
  脉冲"发不发"是不可导的阶梯函数，无法正常反向传播。
  用反正切函数 (ATan) 在反向传播时替代阶梯的导数，骗过自动微分。
  这是 SNN 能训练的根本原因（SpikingJelly 框架的核心贡献）。

【BPTT (Backpropagation Through Time)】
  SNN 在 T 个时间步上展开，每个时间步跑一遍网络。
  膜电位跨时间步累积，梯度也跨时间步回传——所以叫"时间反向传播"。
  functional.reset_net() 必须在每次前向后重置，不然膜电位残留到下一批数据。

【和普通 EEGNet 的关键区别】
  1. 编码:  连续值 → 泊松脉冲序列 (rate_encode)
  2. 激活:  ReLU → LIFNode (脉冲发放)
  3. 重置:  每批数据前 functional.reset_net(net)
  4. 输出:  一次性 logits → T 步膜电位平均 → 分类

【参考】
  SpikingJelly: Fang et al., Science Advances 2023
  EEG-Conformer: Song et al., IEEE TNSRE 2023

【运行方式】
  python train_snn.py                  # 被试 3 快速测试
  python train_snn.py --cross_subject  # 跨被试 7→2
  python train_snn.py --cross_subject --T 16 --epochs 100  # 完整跑
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events

# SpikingJelly: 北大开源脉冲神经网络框架
# neuron:    LIF 神经元
# functional: reset_net 等工具函数
# surrogate: 代理梯度（反正切近似脉冲导数）
from spikingjelly.activation_based import neuron, functional, surrogate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 速率编码 (Rate Coding)
# ============================================================
def rate_encode(x, T=8):
    """
    连续 EEG 值 → 脉冲序列（泊松编码）

    【原理】
      EEG 幅值归一化到 [0,1] 后作为脉冲发放概率。
      比如某点 x[i,j] = 0.7，则每个时间步以 70% 概率发脉冲 (值=1)。
      幅值越大的 EEG 区域 → 单位时间内脉冲越多 → SNN 对其越敏感。

    【输入/输出】
      x: [B, C, L]       B=批, C=通道数(26), L=时间采样点(1000)
      返回: [B, T, 1, C, L]   T=脉冲时间步数，1 是卷积通道维度
    """
    B, C, L = x.shape
    # 每个时间步生成 [0,1) 随机数，小于 x 的地方发脉冲 = 1，否则 = 0
    # x 大的位置 → 更容易小于随机数 → 脉冲多（泊松过程）
    rand = torch.rand(T, B, C, L, device=x.device)
    spikes = (rand < x.unsqueeze(0)).float()  # [T, B, C, L]
    # permute 转成批次优先: [B, T, C, L]
    # unsqueeze(2) 加卷积通道维: [B, T, 1, C, L]
    spikes = spikes.permute(1, 0, 2, 3).unsqueeze(2)
    return spikes


# ============================================================
# SpikingEEGNet: 脉冲版 EEG 分类网络
# ============================================================
class SpikingEEGNet(nn.Module):
    """
    脉冲版简化 EEGNet 架构

    【架构】
      Conv2d(1→16, 核(1,64)) → BN → LIF   ← 时间卷积，提取局部波形模式
      Conv2d(16→32, 核(26,1)) → BN → Pool → LIF  ← 空间卷积，融合 26 通道 + 降采样
      AdaptivePool → FC(256→4)           ← 分类头

    【为什么用 LIF 而非 ReLU】
      ReLU: y = max(0, x)          —— 输入负数就静默，正数就线性通过
      LIF:  V += input; if V>th: 发脉冲, V=0  —— 输入弱则膜电位衰减消失
      有记忆(tau 衰减) + 有阈值(只重要的信号才输出) = 天然稀疏+抗噪

    【关键参数】
      tau=2.0: 膜电位衰减时间常数。tau 越大记忆越久。
      代理梯度 ATan: 阶梯函数的可导近似，反向传播需要。
      T: 输入脉冲序列的时间步数（在 rate_encode 和 forward 中一致）
    """
    def __init__(self, n_chans=26, n_times=1000, n_classes=4):
        super().__init__()
        F1, F2 = 16, 32  # F1=第一层卷积核数, F2=第二层

        # ATan = 反正切代理梯度。正向：脉冲阶梯；反向：用光滑的 arctan 导数替代
        self.surrogate = surrogate.ATan()

        # ===== Block 1: 时间卷积 + LIF =====
        # Conv2d(1, 16, (1,64)): 16 个 1×64 卷积核，沿时间轴滑动
        # 每个核学习一种"波形模板"——比如 mu 节律 10Hz 的正弦模式
        # padding=(0,32): 时间维度两侧各补 32 个零，保证输出长度不变
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32))
        self.bn1 = nn.BatchNorm2d(F1)   # 批归一化：稳定训练
        self.lif1 = neuron.LIFNode(     # LIF 脉冲神经元
            tau=2.0,                     # 膜电位衰减时间常数
            surrogate_function=self.surrogate  # 代理梯度函数
        )

        # ===== Block 2: 空间卷积 + 池化 + LIF =====
        # Conv2d(16, 32, (26,1)): 32 个 26×1 卷积核，跨 26 个通道做空间融合
        # 相当于"把 26 个电极的信息按最优比例混合"
        self.conv2 = nn.Conv2d(F1, F2, (n_chans, 1))
        self.bn2 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))      # 时间维 8 倍降采样，减少计算量
        self.lif2 = neuron.LIFNode(
            tau=2.0,
            surrogate_function=self.surrogate
        )

        # ===== 分类头 =====
        self.pool = nn.AdaptiveAvgPool2d((1, 8))  # 自适应池化到固定大小 1×8
        self.fc = nn.Linear(F2 * 8, n_classes)    # 256 → 4 (四类运动想象)

    def forward(self, x):
        """
        前向传播: 时间维循环展开

        【输入】
          x: [B, T, 1, C, L]   B=批, T=脉冲时间步, C=26, L=1000
        【输出】
          [B, 4]  四类运动想象的分类分数

        【为什么 for t in range(T_sim) 循环？】
          SNN 的状态（膜电位）跨时间步累积。
          每个时间步在同一个网络上跑一遍，但网络内部膜电位在变化。
          就像同一张图给你看 16 遍，你每看一遍理解加深一点。
        """
        B, T_sim, _, _, _ = x.shape

        outputs = []
        for t in range(T_sim):
            xt = x[:, t]                    # 取第 t 步的脉冲: [B, 1, C, L]
            # Block 1: 时间卷积 → BN → LIF 发放脉冲
            xt = self.conv1(xt)
            xt = self.bn1(xt)
            xt = self.lif1(xt)              # 输出是 0/1 脉冲，不再是连续值!
            # Block 2: 空间卷积 → BN → 池化 → LIF 发放脉冲
            xt = self.conv2(xt)
            xt = self.bn2(xt)
            xt = self.pool2(xt)
            xt = self.lif2(xt)
            # 池化 + 展平
            xt = self.pool(xt).view(B, -1)  # [B, 256]
            outputs.append(xt)

        # 关键: T 步膜电位取均值后再 FC
        # 每个时间步的"理解"可能不同，取平均 = 时间维度上的集成投票
        out = torch.stack(outputs, 0).mean(0)  # [B, 256]
        out = self.fc(out)                      # [B, 4]
        return out


# ============================================================
# 数据加载
# ============================================================
def load_data(cross_subject=False):
    """
    加载 BCI IV-2a 数据集

    cross_subject=True:  跨被试 被试 1-7(训练) → 8-9(测试)
    cross_subject=False: 单被试 被试 3 session 0 → session 1
    """
    if cross_subject:
        train_sets, test_sets = [], []
        for sid in range(1, 10):
            ds = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[sid])
            # 截取运动想象事件对应的 4 秒 EEG 片段
            w = create_windows_from_events(ds, 0, 0, preload=True)
            sess = w.split("session")
            keys = sorted(sess.keys())
            # 合并 2 个 session
            all_trials = ConcatDataset([sess[k] for k in keys])
            # 被试 1-7 进训练，8-9 进测试
            (train_sets if sid <= 7 else test_sets).append(all_trials)
        return ConcatDataset(train_sets), ConcatDataset(test_sets)
    else:
        ds = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[3])
        w = create_windows_from_events(ds, 0, 0, preload=True)
        sess = w.split("session")
        keys = sorted(sess.keys())
        return sess[keys[0]], sess[keys[1]]


# ============================================================
# 主训练流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross_subject", action="store_true",
                        help="跨被试 (7→2) 评估")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--T", type=int, default=8,
                        help="脉冲时间步数 (越大越准但越慢)")
    args = parser.parse_args()

    print(f"设备: {DEVICE} | T={args.T}")
    print("=" * 50)
    print("  SNN-EEGNet (LIF 脉冲神经元)")
    print("=" * 50)

    # ===== 1. 加载数据 =====
    mode = "跨被试 7→2" if args.cross_subject else "单被试 3"
    print(f"\n[数据] {mode}")
    train_set, test_set = load_data(args.cross_subject)
    print(f"  训练 {len(train_set)} / 测试 {len(test_set)}")

    # 自动检测数据维度（26 通道 × 1000 采样点）
    sample_x, _, _ = train_set[0]
    n_chans, n_times = sample_x.shape[0], sample_x.shape[1]

    # ===== 2. 构建模型 =====
    model = SpikingEEGNet(n_chans, n_times).to(DEVICE)
    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0.0
    patience = 0  # 早停计数器

    print(f"\n[训练] {args.epochs} 轮")
    print("-" * 50)

    # ===== 3. 训练循环 =====
    for epoch in range(args.epochs):
        # ---- 训练阶段 ----
        model.train()
        train_correct, train_total = 0, 0

        for X, y, _ in train_loader:
            X, y = X.float().to(DEVICE), y.long().to(DEVICE)

            # 步骤 1: min-max 归一化到 [0,1]
            # （每个试次的各个通道分别归一化）
            X_norm = (X - X.min(dim=-1, keepdim=True)[0]) / \
                     (X.max(dim=-1, keepdim=True)[0] - X.min(dim=-1, keepdim=True)[0] + 1e-8)

            # 步骤 2: 速率编码 — 连续值 → 脉冲序列
            spikes = rate_encode(X_norm, args.T)

            # 步骤 3: 重置 SNN 内部状态（膜电位归零）
            # ⚠️ 必须做! 否则上一批数据的膜电位残留会影响当前批次
            functional.reset_net(model)

            # 步骤 4: 标准训练流程
            optimizer.zero_grad()
            outputs = model(spikes)
            loss = criterion(outputs, y)
            loss.backward()        # BPTT: 梯度跨 T 个时间步回传
            optimizer.step()

            train_total += y.size(0)
            train_correct += (outputs.argmax(1) == y).sum().item()

        # ---- 测试阶段 ----
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():  # 测试不记录梯度（省显存加速）
            for X, y, _ in test_loader:
                X, y = X.float().to(DEVICE), y.long().to(DEVICE)
                X_norm = (X - X.min(dim=-1, keepdim=True)[0]) / \
                         (X.max(dim=-1, keepdim=True)[0] - X.min(dim=-1, keepdim=True)[0] + 1e-8)
                spikes = rate_encode(X_norm, args.T)
                functional.reset_net(model)
                outputs = model(spikes)
                _, pred = torch.max(outputs, 1)
                test_total += y.size(0)
                test_correct += (pred == y).sum().item()

        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total
        print(f"Epoch {epoch+1:03d}/{args.epochs} "
              f"| Train {train_acc:.1f}% | Test {test_acc:.1f}%")

        # ---- 保存最佳模型 + 早停 ----
        if test_acc > best_acc:
            best_acc = test_acc
            patience = 0
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "snn_best.pth"))
            print(f"    新最佳! {best_acc:.1f}%")
        else:
            patience += 1
        # 连续 20 轮不涨 → 停止
        if patience >= 20:
            print(f"    早停 @ {epoch+1}")
            break

    print(f"\nSNN 最佳: {best_acc:.1f}%")
    print("基准: EEGNet 62.8% | EEG-Conformer 74.9%")


if __name__ == "__main__":
    main()
