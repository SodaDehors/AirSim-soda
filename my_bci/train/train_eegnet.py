"""
EEGNet 脑电分类模型训练脚本 (完整数据集版)
===========================================

默认加载 BCI IV-2a 全部 9 个被试，跨被试训练。
如需单被试快速实验，改下面 TRAIN_SUBJECTS 即可。

【运行方式】
  python train_bci.py                  # 全部 9 被试，跨被试
  python train_bci.py --epochs 120     # 自定义轮数
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.models import EEGNetv4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(MODEL_DIR, "..", "models")

# ==========================================
# 配置：改这里切换 单被试 / 全被试
# ==========================================
# 单被试模式（快速实验）: TRAIN_SUBJECTS = [3,4,5,6,7], TEST_SUBJECTS = [8,9]
# 全被试模式（完整训练）: TRAIN_SUBJECTS = [1,2,3,4,5,6,7], TEST_SUBJECTS = [8,9]
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7]
TEST_SUBJECTS = [8, 9]


def load_dataset(subject_ids):
    """加载指定被试的数据，返回 (session_0, session_1) 合并后的数据集"""
    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=subject_ids)
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        preload=True,
    )
    # 合并该被试的所有 session
    sessions = windows_dataset.split("session")
    sets = [sessions[k] for k in sorted(sessions.keys())]
    if len(sets) == 1:
        return sets[0]
    return ConcatDataset(sets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 计算设备: {device}")

    # ==========================================
    # 阶段 1: 加载数据
    # ==========================================
    print(f" 训练被试: {TRAIN_SUBJECTS}")
    print(f" 测试被试: {TEST_SUBJECTS}")

    train_sets = []
    for sid in TRAIN_SUBJECTS:
        ds = load_dataset([sid])
        print(f"   被试 {sid}: {len(ds)} trials")
        train_sets.append(ds)
    train_set = ConcatDataset(train_sets)

    test_sets = []
    for sid in TEST_SUBJECTS:
        ds = load_dataset([sid])
        print(f"   被试 {sid}: {len(ds)} trials")
        test_sets.append(ds)
    test_set = ConcatDataset(test_sets)

    print(f" 合计 — 训练 {len(train_set)} / 测试 {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False)

    # ==========================================
    # 阶段 2: 构建模型
    # ==========================================
    sample_x, _, _ = train_set[0]
    actual_channels = sample_x.shape[0]
    actual_times = sample_x.shape[1]
    print(f" 数据维度: {actual_channels} 通道 x {actual_times} 采样点")

    model = EEGNetv4(
        in_chans=actual_channels,
        n_classes=4,
        input_window_samples=actual_times,
        final_conv_length="auto",
        drop_prob=0.5,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ==========================================
    # 阶段 3: 训练
    # ==========================================
    epochs = args.epochs
    best_test_acc = 0.0
    patience = 0

    print(f"\n 开始训练 ({epochs} 轮, batch={args.batch})")
    print("-" * 55)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X, y, _ in train_loader:
            X, y = X.float().to(device), y.long().to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            train_total += y.size(0)
            train_correct += (pred == y).sum().item()

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for X, y, _ in test_loader:
                X, y = X.float().to(device), y.long().to(device)
                outputs = model(X)
                _, pred = torch.max(outputs, 1)
                test_total += y.size(0)
                test_correct += (pred == y).sum().item()

        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total

        print(f"Epoch [{epoch+1:03d}/{epochs}] "
              f"| Loss {train_loss/len(train_loader):.4f} "
              f"| Train {train_acc:.1f}% "
              f"| Test {test_acc:.1f}%")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "eegnet_best_model.pth"))
            print(f"    新最佳 ({best_test_acc:.1f}%), 已保存")
        else:
            patience += 1

        if patience >= 25:
            print(f"\n 早停: {patience} 轮未创新高")
            break

    print(f"\n 完成 — 最佳: {best_test_acc:.1f}%")
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "eegnet_model.pth"))


if __name__ == "__main__":
    main()
