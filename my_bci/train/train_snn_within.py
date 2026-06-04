"""
SNN-EEGNet 被试内评估脚本
==========================

session 0 训练 → session 1 测试，跑全部 9 被试，汇总对比。

【运行方式】
  python train_snn_within.py --all
  python train_snn_within.py --subject 3  # 单被试
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

from spikingjelly.activation_based import neuron, functional, surrogate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rate_encode(x, T=8):
    B, C, L = x.shape
    rand = torch.rand(T, B, C, L, device=x.device)
    spikes = (rand < x.unsqueeze(0)).float()
    spikes = spikes.permute(1, 0, 2, 3).unsqueeze(2)  # [B,T,1,C,L]
    return spikes


class SpikingEEGNet(nn.Module):
    def __init__(self, n_chans=26, n_times=1000, n_classes=4):
        super().__init__()
        F1, F2 = 16, 32
        self.surrogate = surrogate.ATan()

        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32))
        self.bn1 = nn.BatchNorm2d(F1)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=self.surrogate)

        self.conv2 = nn.Conv2d(F1, F2, (n_chans, 1))
        self.bn2 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=self.surrogate)

        self.pool = nn.AdaptiveAvgPool2d((1, 8))
        self.fc = nn.Linear(F2 * 8, n_classes)

    def forward(self, x):
        B, T_sim, _, _, _ = x.shape
        outputs = []
        for t in range(T_sim):
            xt = x[:, t]
            xt = self.conv1(xt); xt = self.bn1(xt); xt = self.lif1(xt)
            xt = self.conv2(xt); xt = self.bn2(xt); xt = self.pool2(xt); xt = self.lif2(xt)
            xt = self.pool(xt).view(B, -1)
            outputs.append(xt)
        return self.fc(torch.stack(outputs, 0).mean(0))


def train_one(sid, args):
    ds = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[sid])
    w = create_windows_from_events(ds, 0, 0, preload=True)
    sess = w.split("session")
    keys = sorted(sess.keys())
    train_set, test_set = sess[keys[0]], sess[keys[1]]
    print(f"  被试 {sid}: train {len(train_set)} → test {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False)

    sample_x, _, _ = train_set[0]
    model = SpikingEEGNet(sample_x.shape[0], sample_x.shape[1]).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    patience = 0

    for epoch in range(args.epochs):
        model.train()
        for X, y, _ in train_loader:
            X, y = X.float().to(DEVICE), y.long().to(DEVICE)
            X_norm = (X - X.min(dim=-1, keepdim=True)[0]) / \
                     (X.max(dim=-1, keepdim=True)[0] - X.min(dim=-1, keepdim=True)[0] + 1e-8)
            spikes = rate_encode(X_norm, args.T)
            functional.reset_net(model)
            optimizer.zero_grad()
            loss = criterion(model(spikes), y)
            loss.backward()
            optimizer.step()

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
        if acc > best_acc:
            best_acc = acc
            patience = 0
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, f"snn_s{sid}.pth"))
        else:
            patience += 1

        if epoch % 10 == 0 and best_acc < 30:
            print(f"    Epoch {epoch+1:02d}: {acc:.1f}%")
        elif patience >= 20:
            break

    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--T", type=int, default=16)
    args = parser.parse_args()

    print(f"设备: {DEVICE} | T={args.T}")
    print("=" * 50)

    if args.all:
        results = {}
        for sid in range(1, 10):
            print(f"\n--- 被试 {sid} ---")
            acc = train_one(sid, args)
            results[sid] = acc
            print(f"  → {acc:.1f}%")

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
        print(f" 最高 | {max(results.values()):6.1f}% (S{max(results, key=results.get)})")
        print(f" 最低 | {min(results.values()):6.1f}% (S{min(results, key=results.get)})")

        print(f"\n对比 EEGNet 被试内:")
        eegnet = {1:53.1,2:44.8,3:50.3,4:49.7,5:81.9,6:59.4,7:81.6,8:74.7,9:58.7}
        for sid in range(1, 10):
            d = results[sid] - eegnet[sid]
            print(f"  被试 {sid}: SNN {results[sid]:.1f}% vs EEGNet {eegnet[sid]:.1f}% ({d:+.1f})")
        print(f"  SNN 平均: {avg:.1f}% vs EEGNet 平均: {sum(eegnet.values())/9:.1f}%")
    else:
        acc = train_one(args.subject, args)
        print(f"\n被试 {args.subject}: {acc:.1f}%")
        print(f"基准 EEGNet: 50.3%")


if __name__ == "__main__":
    main()
