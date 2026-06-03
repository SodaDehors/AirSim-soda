"""
EEGNet 完整数据集训练脚本 (服务器版)
====================================

加载 BCI IV-2a 全部 9 个被试，跨被试训练+评估。

【运行方式】
  python train_full.py
  python train_full.py --device cuda:0
  python train_full.py --epochs 120 --batch 64

【数据规模】
  9 被试 × 2 session × 288 trials = 5184 条
  训练/测试用被试级别划分 (前 7 被试训练，后 2 被试测试)
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.models import EEGNetv4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")


def load_all_subjects():
    """加载全部 9 个被试，返回按 session 分组后的数据"""
    print(" 加载 BCI IV-2a 全部 9 个被试...")
    dataset = MOABBDataset(dataset_name="BNCI2014_001")
    # 全部被试: 1, 2, 3, 4, 5, 6, 7, 8, 9

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        preload=True,
    )

    # 按被试拆分
    subjects = windows_dataset.split("subject")
    subject_ids = sorted(subjects.keys(), key=int)
    print(f" 被试: {subject_ids}")

    all_sessions = {}
    for sid in subject_ids:
        subj_sessions = subjects[sid].split("session")
        for sk in subj_sessions.keys():
            all_sessions[f"{sid}_{sk}"] = subj_sessions[sk]

    session_keys = sorted(all_sessions.keys(), key=lambda x: (int(x[0]), x[1:]))
    for k in session_keys:
        print(f"   {k}: {len(all_sessions[k])} trials")

    return all_sessions, session_keys


def build_datasets(all_sessions, session_keys):
    """按被试划分训练/测试集 --- 前 7 个被试训练，后 2 个测试"""
    train_sessions = []
    test_sessions = []

    for k in session_keys:
        sid = k[0]  # subject id digit
        if sid in ["8", "9"]:
            test_sessions.append(all_sessions[k])
        else:
            train_sessions.append(all_sessions[k])

    train_set = ConcatDataset(train_sessions)
    test_set = ConcatDataset(test_sessions)
    print(f"\n 训练集: {len(train_set)} trials (被试 1-7)")
    print(f" 测试集: {len(test_set)} trials (被试 8-9)")

    return train_set, test_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f" 计算设备: {device}")

    # ==========================================
    # 阶段 1: 加载全部被试
    # ==========================================
    t0 = time.time()
    all_sessions, session_keys = load_all_subjects()
    train_set, test_set = build_datasets(all_sessions, session_keys)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False)
    print(f" 数据加载耗时 {time.time()-t0:.0f}s")

    # ==========================================
    # 阶段 2: 构建模型
    # ==========================================
    sample_x, _, _ = train_set[0]
    actual_channels = sample_x.shape[0]
    actual_times = sample_x.shape[1]
    print(f"\n 数据维度: {actual_channels} 通道 × {actual_times} 采样点")

    model = EEGNetv4(
        in_chans=actual_channels,
        n_classes=4,
        input_window_samples=actual_times,
        final_conv_length="auto",
        drop_prob=0.5,  # 跨被试任务更难，加大 Dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ==========================================
    # 阶段 3: 训练
    # ==========================================
    epochs = args.epochs
    best_test_acc = 0.0
    patience = 0
    max_patience = 20

    print(f"\n 开始训练 (跨被试: 1-7 => 8-9)")
    print("-" * 60)

    for epoch in range(epochs):
        t_epoch = time.time()

        # --- 训练 ---
        model.train()
        train_correct, train_total = 0, 0
        train_loss = 0.0
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

        # --- 测试 ---
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for X, y, _ in test_loader:
                X, y = X.float().to(device), y.long().to(device)
                outputs = model(X)
                _, pred = torch.max(outputs, 1)
                test_total += y.size(0)
                test_correct += (pred == y).sum().item()

        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total

        print(
            f"Epoch [{epoch+1:03d}/{epochs}] "
            f"| 耗时 {time.time()-t_epoch:.1f}s "
            f"| Train Loss {train_loss/len(train_loader):.4f} "
            f"| Train {train_acc:.1f}% "
            f"| Test {test_acc:.1f}%"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience = 0
            torch.save(
                model.state_dict(),
                os.path.join(MODEL_DIR, "eegnet_full_best.pth"),
            )
            print(f"    新最佳模型 -> 保存 ({best_test_acc:.1f}%)")
        else:
            patience += 1

        if patience >= max_patience:
            print(f"\n 早停: 连续 {max_patience} 轮未创最佳")
            break

    # ==========================================
    # 结果
    # ==========================================
    print(f"\n 训练完成 — 最佳测试准确率: {best_test_acc:.1f}%")
    torch.save(
        model.state_dict(),
        os.path.join(MODEL_DIR, "eegnet_full_final.pth"),
    )
    print(f"  最终模型: eegnet_full_final.pth")
    print(f"  最佳模型: eegnet_full_best.pth")


if __name__ == "__main__":
    main()
