"""
SNN-EEGNet 被试内评估脚本（session 0 → session 1）
=====================================================

每人 288 条训练 → 288 条测试，跑全部 9 个被试，汇总对比 EEGNet。

【和 train_snn.py 的区别】
  - train_snn.py: 跨被试 (7 人→2 人) 或单被试
  - 本脚本: 被试内 (同一人 s0→s1)，跑 9 人出汇总

【被试内 vs 跨被试】
  被试内: 同一个人不同天 → 评估"模型能否适应改天的脑电"
  跨被试: 不同人 → 评估"模型能否泛化到没见过的人"

【运行方式】
  python train_snn_within.py --all       # 全部 9 人
  python train_snn_within.py --subject 3 # 仅被试 3

【参考】
  SpikingJelly: Fang et al., Science Advances 2023
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events

# SpikingJelly SNN 组件
# neuron:  LIF 脉冲神经元
# functional: reset_net — 每次前向后必须重置膜电位
# surrogate: 代理梯度 — 用光滑函数近似脉冲阶梯的导数
from spikingjelly.activation_based import neuron, functional, surrogate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 速率编码: 连续 EEG → 脉冲序列
# ============================================================
def rate_encode(x, T=8):
    """
    泊松速率编码：幅值越大 → 脉冲概率越高

    输入: [B, C, L]       归一化 EEG (值在 [0,1] 之间)
    输出: [B, T, 1, C, L] 脉冲序列 (0 或 1)
          T = 脉冲时间步数 = 同一条 EEG 被"复读"的次数

    原理:
      rand < x → 脉冲=1, 否则=0
      x=0.9 → 90% 概率发脉冲, x=0.1 → 10% 概率
    """
    B, C, L = x.shape
    rand = torch.rand(T, B, C, L, device=x.device)    # [0,1) 随机数
    spikes = (rand < x.unsqueeze(0)).float()           # 比较 → 0/1
    spikes = spikes.permute(1, 0, 2, 3).unsqueeze(2)  # 重整形状
    return spikes  # [B, T, 1, C, L]


# ============================================================
# 脉冲版 EEG 分类网络
# ============================================================
class SpikingEEGNet(nn.Module):
    """
    简化脉冲 EEGNet: Conv1 → LIF → Conv2 → Pool → LIF → FC

    【关键概念】
    - LIF 神经元: 像积分电路，输入脉冲累加到膜电位
      达到阈值 → 发脉冲 + 重置，没达到 → 膜电位自然衰减
    - tau=2.0: 膜电位时间常数，决定"记忆"多长
    - ATan 代理梯度: 脉冲的"发/不发"不可导，用反正切函数近似
    """
    def __init__(self, n_chans=26, n_times=1000, n_classes=4):
        super().__init__()
        F1, F2 = 16, 32
        self.surrogate = surrogate.ATan()

        # Block 1: 时间卷积 (1×64 核) → BN → LIF
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32))
        self.bn1 = nn.BatchNorm2d(F1)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=self.surrogate)

        # Block 2: 空间卷积 (26×1 核) → BN → 时间池化 → LIF
        self.conv2 = nn.Conv2d(F1, F2, (n_chans, 1))
        self.bn2 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=self.surrogate)

        # 分类头: 自适应池化 + 全连接
        self.pool = nn.AdaptiveAvgPool2d((1, 8))
        self.fc = nn.Linear(F2 * 8, n_classes)  # 256 → 4

    def forward(self, x):
        """
        x: [B, T, 1, C, L]  脉冲序列
        返回: [B, 4]  类别分数

        for t in range(T): 同一个网络跑 T 遍，
        但每遍膜电位在变（有记忆），最后 T 步取均值 = 时间集成投票
        """
        B, T_sim, _, _, _ = x.shape
        outputs = []
        for t in range(T_sim):
            xt = x[:, t]
            xt = self.conv1(xt); xt = self.bn1(xt); xt = self.lif1(xt)
            xt = self.conv2(xt); xt = self.bn2(xt); xt = self.pool2(xt); xt = self.lif2(xt)
            xt = self.pool(xt).view(B, -1)
            outputs.append(xt)
        return self.fc(torch.stack(outputs, 0).mean(0))


# ============================================================
# 训练单个被试
# ============================================================
def train_one(sid, args):
    """
    被试内训练: session 0(288条) 训练 → session 1(288条) 测试

    返回: 最佳测试准确率 (%)
    模型保存: models/snn_s{sid}.pth
    """
    # 加载指定被试的数据
    ds = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[sid])
    w = create_windows_from_events(ds, 0, 0, preload=True)
    sess = w.split("session")
    keys = sorted(sess.keys())
    train_set, test_set = sess[keys[0]], sess[keys[1]]
    print(f"  被试 {sid}: train {len(train_set)} → test {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False)

    # 自动检测通道数和采样点数
    sample_x, _, _ = train_set[0]
    model = SpikingEEGNet(sample_x.shape[0], sample_x.shape[1]).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    patience = 0  # 早停计数器

    for epoch in range(args.epochs):
        # ----- 训练 -----
        model.train()
        for X, y, _ in train_loader:
            X, y = X.float().to(DEVICE), y.long().to(DEVICE)

            # min-max 归一化 → 速率编码 → 脉冲序列
            X_norm = (X - X.min(dim=-1, keepdim=True)[0]) / \
                     (X.max(dim=-1, keepdim=True)[0] - X.min(dim=-1, keepdim=True)[0] + 1e-8)
            spikes = rate_encode(X_norm, args.T)

            # ⚠️ 必须重置 SNN 内部状态（膜电位、脉冲记录）
            functional.reset_net(model)
            optimizer.zero_grad()
            loss = criterion(model(spikes), y)
            loss.backward()  # BPTT: 在 T 个时间步上反向传播
            optimizer.step()

        # ----- 测试 (不更新参数) -----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y, _ in test_loader:
                X, y = X.float().to(DEVICE), y.long().to(DEVICE)
                X_norm = (X - X.min(dim=-1, keepdim=True)[0]) / \
                         (X.max(dim=-1, keepdim=True)[0] - X.min(dim=-1, keepdim=True)[0] + 1e-8)
                spikes = rate_encode(X_norm, args.T)
                functional.reset_net(model)
                _, pred = torch.max(model(spikes), 1)
                total += y.size(0)
                correct += (pred == y).sum().item()

        acc = 100 * correct / total

        # 保存最佳
        if acc > best_acc:
            best_acc = acc
            patience = 0
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, f"snn_s{sid}.pth"))
        else:
            patience += 1

        # 还低于 30% 时每 10 轮报一次（早期训练慢）
        if epoch % 10 == 0 and best_acc < 30:
            print(f"    Epoch {epoch+1:02d}: {acc:.1f}%")
        elif patience >= 20:
            break

    return best_acc


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="跑全部 9 个被试")
    parser.add_argument("--subject", type=int, default=3,
                        help="指定被试编号")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--T", type=int, default=16,
                        help="脉冲时间步数（被试内用 16，数据少需要更多步）")
    args = parser.parse_args()

    print(f"设备: {DEVICE} | T={args.T}")
    print("=" * 50)

    if args.all:
        # 逐个被试训练
        results = {}
        for sid in range(1, 10):
            print(f"\n--- 被试 {sid} ---")
            acc = train_one(sid, args)
            results[sid] = acc
            print(f"  → {acc:.1f}%")

        # ----- 汇总 -----
        avg = sum(results.values()) / 9
        print(f"\n{'='*40}")
        print(f"  SNN-EEGNet 被试内汇总")
        print(f"{'='*40}")
        print(f"{'被试':>5} | {'SNN':>7}")
        print("-" * 20)
        for sid in range(1, 10):
            print(f"  {sid:2d}  | {results[sid]:6.1f}%")
        print("-" * 20)
        print(f" 平均 | {avg:6.1f}%")
        print(f" 最高 | {max(results.values()):6.1f}% "
              f"(S{max(results, key=results.get)})")
        print(f" 最低 | {min(results.values()):6.1f}% "
              f"(S{min(results, key=results.get)})")

        # ----- 和 EEGNet 对比 -----
        print(f"\n对比 EEGNet 被试内:")
        eegnet = {1: 53.1, 2: 44.8, 3: 50.3, 4: 49.7,
                  5: 81.9, 6: 59.4, 7: 81.6, 8: 74.7, 9: 58.7}
        for sid in range(1, 10):
            d = results[sid] - eegnet[sid]
            print(f"  被试 {sid}: SNN {results[sid]:.1f}% vs "
                  f"EEGNet {eegnet[sid]:.1f}% ({d:+.1f})")
        print(f"  SNN 平均: {avg:.1f}% vs "
              f"EEGNet 平均: {sum(eegnet.values())/9:.1f}%")
    else:
        acc = train_one(args.subject, args)
        print(f"\n被试 {args.subject}: {acc:.1f}%")
        print(f"基准 EEGNet: 50.3%")


if __name__ == "__main__":
    main()
