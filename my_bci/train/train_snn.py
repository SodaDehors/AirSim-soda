"""
SNN-EEGNet 脉冲神经网络训练脚本
===============================

用 SpikingJelly 的 LIF 脉冲神经元构建 EEG 分类网络。
脑电幅值 → 速率编码为脉冲序列 → 脉冲卷积 + LIF → 膜电位投票分类。

【参考】
  SpikingJelly: Fang et al., Science Advances 2023
  EEG-Conformer: Song et al., IEEE TNSRE 2023 (当前 SOTA 基准)

【运行方式】
  python train_snn.py                  # 被试 3 快速测试
  python train_snn.py --cross_subject  # 跨被试 7→2
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

from spikingjelly.activation_based import neuron, functional, surrogate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rate_encode(x, T=8):
    """
    速率编码: 连续 EEG 值 → 脉冲序列

    x: [B, C, T_orig]    归一化后的 EEG
    → [B, T, 1, C, T_orig]   脉冲 (0/1)
    """
    B, C, L = x.shape
    # 随机数 vs 归一化幅值 = 泊松脉冲
    rand = torch.rand(T, B, C, L, device=x.device)
    spikes = (rand < x.unsqueeze(0)).float()
    spikes = spikes.permute(1, 0, 2, 3)  # [B, T, C, L]
    spikes = spikes.unsqueeze(2)           # [B, T, 1, C, L]
    return spikes


class SpikingEEGNet(nn.Module):
    """
    脉冲版简化 EEGNet:
    Conv2d(1, F1) → LIF → Conv2d(F1, F2) → LIF → Pool → FC
    """
    def __init__(self, n_chans=26, n_times=1000, n_classes=4):
        super().__init__()
        F1, F2 = 16, 32
        self.surrogate = surrogate.ATan()

        # Conv1: 时间卷积 (1, 64)
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32))
        self.bn1 = nn.BatchNorm2d(F1)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=self.surrogate)

        # Conv2: 空间卷积 (n_chans, 1) + 时间池化
        self.conv2 = nn.Conv2d(F1, F2, (n_chans, 1))
        self.bn2 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=self.surrogate)

        # 分类头
        self.pool = nn.AdaptiveAvgPool2d((1, 8))
        self.fc = nn.Linear(F2 * 8, n_classes)

    def forward(self, x):
        # x: [B, T, 1, C, L]
        B, T_sim, _, _, _ = x.shape

        outputs = []
        for t in range(T_sim):
            xt = x[:, t]  # [B, 1, C, L]
            xt = self.conv1(xt)
            xt = self.bn1(xt)
            xt = self.lif1(xt)
            xt = self.conv2(xt)
            xt = self.bn2(xt)
            xt = self.pool2(xt)
            xt = self.lif2(xt)
            xt = self.pool(xt).view(B, -1)
            outputs.append(xt)

        # 所有时间步取均值 → FC
        out = torch.stack(outputs, 0).mean(0)
        out = self.fc(out)
        return out


def load_data(cross_subject=False):
    if cross_subject:
        train_sets, test_sets = [], []
        for sid in range(1, 10):
            ds = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[sid])
            w = create_windows_from_events(ds, 0, 0, preload=True)
            sess = w.split("session")
            keys = sorted(sess.keys())
            all_trials = ConcatDataset([sess[k] for k in keys])
            (train_sets if sid <= 7 else test_sets).append(all_trials)
        return ConcatDataset(train_sets), ConcatDataset(test_sets)
    else:
        ds = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[3])
        w = create_windows_from_events(ds, 0, 0, preload=True)
        sess = w.split("session")
        keys = sorted(sess.keys())
        return sess[keys[0]], sess[keys[1]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross_subject", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--T", type=int, default=8)
    args = parser.parse_args()

    print(f"设备: {DEVICE} | T={args.T}")
    print("=" * 50)
    print("  SNN-EEGNet (LIF 脉冲神经元)")
    print("=" * 50)

    mode = "跨被试 7→2" if args.cross_subject else "单被试 3"
    print(f"\n[数据] {mode}")
    train_set, test_set = load_data(args.cross_subject)
    print(f"  训练 {len(train_set)} / 测试 {len(test_set)}")

    sample_x, _, _ = train_set[0]
    n_chans, n_times = sample_x.shape[0], sample_x.shape[1]

    model = SpikingEEGNet(n_chans, n_times).to(DEVICE)
    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0.0
    patience = 0

    print(f"\n[训练] {args.epochs} 轮")
    print("-" * 50)

    for epoch in range(args.epochs):
        model.train()
        train_correct, train_total = 0, 0

        for X, y, _ in train_loader:
            X, y = X.float().to(DEVICE), y.long().to(DEVICE)

            # 速率编码: 归一化 → 脉冲
            X_norm = (X - X.min(dim=-1, keepdim=True)[0]) / \
                     (X.max(dim=-1, keepdim=True)[0] - X.min(dim=-1, keepdim=True)[0] + 1e-8)
            spikes = rate_encode(X_norm, args.T)

            functional.reset_net(model)
            optimizer.zero_grad()
            outputs = model(spikes)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_total += y.size(0)
            train_correct += (outputs.argmax(1) == y).sum().item()

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
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
        print(f"Epoch {epoch+1:03d}/{args.epochs} | Train {train_acc:.1f}% | Test {test_acc:.1f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            patience = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "snn_best.pth"))
            print(f"    新最佳! {best_acc:.1f}%")
        else:
            patience += 1
        if patience >= 20:
            print(f"    早停 @ {epoch+1}")
            break

    print(f"\nSNN 最佳: {best_acc:.1f}%")
    print("基准: EEGNet 62.8% | EEG-Conformer 74.9%")


if __name__ == "__main__":
    main()
