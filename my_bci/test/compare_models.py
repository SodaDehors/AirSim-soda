"""
DeepConvNet vs EEGNet 对比脚本
===============================

被试内评估，先跑被试 3 快速对比，确认有提升再跑全部。
"""

import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.models import EEGNetv4, Deep4Net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {DEVICE}")


def train_one(model, train_set, test_set, epochs=80, batch=16, lr=0.001):
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False)

    sample_x, _, _ = train_set[0]
    n_chans, n_times = sample_x.shape[0], sample_x.shape[1]

    kwargs = dict(n_chans=n_chans, n_outputs=4, n_times=n_times,
                  final_conv_length="auto")
    if model is Deep4Net:
        kwargs["add_log_softmax"] = False
    model = model(**kwargs).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    patience = 0

    for epoch in range(epochs):
        model.train()
        for X, y, _ in train_loader:
            X, y = X.float().to(DEVICE), y.long().to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y, _ in test_loader:
                X, y = X.float().to(DEVICE), y.long().to(DEVICE)
                _, pred = torch.max(model(X), 1)
                total += y.size(0)
                correct += (pred == y).sum().item()

        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc
            patience = 0
        else:
            patience += 1
        if patience >= 25:
            break
        if epoch % 10 == 0 or patience == 0:
            print(f"    Epoch {epoch+1:02d}: {acc:.1f}%")

    return best_acc


# 跨被试 1-7 训练 / 8-9 测试（数据量够 DeepConvNet 发挥）
print("\n加载跨被试数据 (1-7 → 8,9)...")
train_sets, test_sets = [], []
for sid in range(1, 10):
    ds = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[sid])
    w = create_windows_from_events(ds, 0, 0, preload=True)
    sess = w.split("session")
    keys = sorted(sess.keys())
    all_trials = torch.utils.data.ConcatDataset([sess[k] for k in keys])
    if sid <= 7:
        train_sets.append(all_trials)
    else:
        test_sets.append(all_trials)

train_set = torch.utils.data.ConcatDataset(train_sets)
test_set = torch.utils.data.ConcatDataset(test_sets)
print(f"  训练: {len(train_set)} 条, 测试: {len(test_set)} 条")

print("\n--- EEGNet (跨被试) ---")
eegnet_acc = train_one(EEGNetv4, train_set, test_set, batch=32)

print("\n--- DeepConvNet (跨被试) ---")
deep_acc = train_one(Deep4Net, train_set, test_set, epochs=120, batch=32, lr=0.0005)

print(f"\n======== 跨被试对比 ========")
print(f"  EEGNet:      {eegnet_acc:.1f}%")
print(f"  DeepConvNet: {deep_acc:.1f}%")
print(f"  提升:        {deep_acc - eegnet_acc:+.1f}%")
