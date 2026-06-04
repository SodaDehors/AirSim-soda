"""
EEG-Conformer vs EEGNet 全面对比 — 被试内评估 (全部 9 人)
==========================================================

每人 session 0 训练 → session 1 测试，保存最佳模型，汇总对比。
"""

import warnings
warnings.filterwarnings("ignore")

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.models import EEGNetv4, EEGConformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")


def train_one(ModelClass, train_set, test_set, epochs, batch, lr, save_path):
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False)

    sample_x, _, _ = train_set[0]
    n_chans, n_times = sample_x.shape[0], sample_x.shape[1]

    kwargs = dict(n_chans=n_chans, n_outputs=4, n_times=n_times)
    if ModelClass is EEGNetv4:
        kwargs["final_conv_length"] = "auto"
    if ModelClass is EEGConformer:
        kwargs["add_log_softmax"] = False
    model = ModelClass(**kwargs).to(DEVICE)

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
            torch.save(model.state_dict(), save_path)
        else:
            patience += 1
        if patience >= 25:
            break

    return best_acc


os.makedirs(MODEL_DIR, exist_ok=True)

eegnet_results = {}
conf_results = {}

for sid in range(1, 10):
    ds = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[sid])
    w = create_windows_from_events(ds, 0, 0, preload=True)
    sess = w.split("session")
    keys = sorted(sess.keys())
    train_set, test_set = sess[keys[0]], sess[keys[1]]

    print(f"\n{'='*50}")
    print(f"被试 {sid} | 训练 {len(train_set)} 条 → 测试 {len(test_set)} 条")

    print(f"\n  EEGNet...")
    eegnet_acc = train_one(EEGNetv4, train_set, test_set,
                           epochs=80, batch=16, lr=0.001,
                           save_path=os.path.join(MODEL_DIR, f"within_s{sid}.pth"))
    eegnet_results[sid] = eegnet_acc
    print(f"    -> {eegnet_acc:.1f}%")

    print(f"\n  EEG-Conformer...")
    conf_acc = train_one(EEGConformer, train_set, test_set,
                         epochs=100, batch=8, lr=0.0003,
                         save_path=os.path.join(MODEL_DIR, f"conformer_s{sid}.pth"))
    conf_results[sid] = conf_acc
    print(f"    -> {conf_acc:.1f}%")

# 汇总
print(f"\n{'='*50}")
print(f"           被试内评估汇总")
print(f"{'='*50}")
print(f"{'被试':>5} | {'EEGNet':>8} | {'Conformer':>9} | {'提升':>6}")
print("-" * 40)
for sid in range(1, 10):
    delta = conf_results[sid] - eegnet_results[sid]
    print(f"  {sid:2d}  | {eegnet_results[sid]:7.1f}% | {conf_results[sid]:8.1f}% | {delta:+6.1f}%")
print("-" * 40)
avg_e = sum(eegnet_results.values()) / 9
avg_c = sum(conf_results.values()) / 9
print(f" 平均 | {avg_e:7.1f}% | {avg_c:8.1f}% | {avg_c-avg_e:+6.1f}%")
