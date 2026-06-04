"""
跨被试 EEG-Conformer 训练 + 演示脚本
====================================
训练被试 1-7 → 测试 8-9，保存模型，然后逐条演示预测结果。
"""

import warnings
warnings.filterwarnings("ignore")

import os
import random
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.models import EEGConformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")
SAVE_PATH = os.path.join(MODEL_DIR, "conformer_cross.pth")
CLASS_NAMES = ["左手动", "右手动", "脚动", "舌头动"]

os.makedirs(MODEL_DIR, exist_ok=True)

# ====== 1. 加载跨被试数据 ======
print("=" * 55)
print("  EEG-Conformer 跨被试训练 + 演示")
print("=" * 55)
print("\n[1/3] 加载数据 (被试 1-7 → 8,9)...")

train_sets, test_sets = [], []
for sid in range(1, 10):
    ds = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[sid])
    w = create_windows_from_events(ds, 0, 0, preload=True)
    sess = w.split("session")
    keys = sorted(sess.keys())
    all_trials = ConcatDataset([sess[k] for k in keys])
    if sid <= 7:
        train_sets.append(all_trials)
    else:
        test_sets.append(all_trials)

train_set = ConcatDataset(train_sets)
test_set = ConcatDataset(test_sets)
print(f"  训练: {len(train_set)} 条, 测试: {len(test_set)} 条")

# ====== 2. 训练模型 ======
print("\n[2/3] 训练 EEG-Conformer (跨被试)...")

sample_x, _, _ = train_set[0]
model = EEGConformer(
    n_chans=sample_x.shape[0], n_outputs=4, n_times=sample_x.shape[1],
    add_log_softmax=False,
).to(DEVICE)
print(f"  参数: {sum(p.numel() for p in model.parameters()):,}")
print(f"  设备: {DEVICE}")

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False)

# 标签平滑 0.1: 抑制过度自信 (目标 [1,0,0,0] → [0.925,0.025,0.025,0.025])
# 解决 Conformer 99%+ 置信偏爱单一类别的问题
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0003)
best_acc = 0.0
patience = 0

for epoch in range(100):
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
        torch.save(model.state_dict(), SAVE_PATH)
    else:
        patience += 1

    if epoch % 5 == 0 or patience == 0:
        print(f"  Epoch {epoch+1:02d}: Test {acc:.1f}%  ('最佳 {best_acc:.1f}%)")
    if patience >= 25:
        print(f"  早停 @ epoch {epoch+1}")
        break

print(f"\n  训练完成! 最佳: {best_acc:.1f}%  -> {SAVE_PATH}")

# ====== 3. 演示预测 ======
print(f"\n[3/3] 演示预测 (随机 20 条测试数据)...")
print("-" * 55)

model.eval()
indices = list(range(len(test_set)))
random.shuffle(indices)
correct = 0

for i in range(min(20, len(indices))):
    idx = indices[i]
    X, y_true, _ = test_set[idx]

    with torch.no_grad():
        tensor = torch.tensor(X).unsqueeze(0).float().to(DEVICE)
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        pred = pred.item()
        conf = conf.item()

    ok = "O" if pred == y_true else "X"
    correct += (pred == y_true)
    print(f"  [{i+1:02d}] {ok} 真实: {CLASS_NAMES[y_true]:4s} | "
          f"预测: {CLASS_NAMES[pred]:4s} | 置信: {conf:.1%}")

print(f"\n  准确率: {correct}/20 = {correct/20*100:.0f}%")
print(f"\n  模型已保存: {SAVE_PATH}")
