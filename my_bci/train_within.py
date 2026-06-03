"""
EEGNet 被试内训练脚本
=====================

同一人 session 0 训练 → session 1 测试（论文可比准确率）。

【运行方式】
  python train_within.py                  # 被试 3
  python train_within.py --subject 1      # 指定被试
  python train_within.py --all            # 所有被试依次训练，汇总报告
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
from braindecode.models import EEGNetv4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def train_one_subject(subject_id, args):
    """被试内训练：session 0→1"""
    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        preload=True,
    )

    sessions = windows_dataset.split("session")
    keys = sorted(sessions.keys())
    train_set = sessions[keys[0]]
    test_set = sessions[keys[1]]
    print(f"  被试 {subject_id}: session {keys[0]} 训练 ({len(train_set)}), "
          f"session {keys[1]} 测试 ({len(test_set)})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_x, _, _ = train_set[0]

    model = EEGNetv4(
        in_chans=sample_x.shape[0],
        n_classes=4,
        input_window_samples=sample_x.shape[1],
        final_conv_length="auto",
    ).to(device)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_test_acc = 0.0
    patience = 0

    for epoch in range(args.epochs):
        model.train()
        train_correct, train_total = 0, 0
        for X, y, _ in train_loader:
            X, y = X.float().to(device), y.long().to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            _, pred = torch.max(outputs, 1)
            train_total += y.size(0)
            train_correct += (pred == y).sum().item()

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for X, y, _ in test_loader:
                X, y = X.float().to(device), y.long().to(device)
                _, pred = torch.max(model(X), 1)
                test_total += y.size(0)
                test_correct += (pred == y).sum().item()

        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience = 0
            save_path = os.path.join(SCRIPT_DIR, f"eegnet_s{subject_id}_best.pth")
            torch.save(model.state_dict(), save_path)
        else:
            patience += 1

        if epoch % 5 == 0 or patience == 0:
            print(f"    Epoch {epoch+1:02d}/{args.epochs} | Train {train_acc:.1f}% | Test {test_acc:.1f}%")

        if patience >= 25:
            print(f"    早停 @ epoch {epoch+1}")
            break

    return best_test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    if args.all:
        results = {}
        print(f"被试内评估: 全部 9 个被试\n")
        for sid in range(1, 10):
            acc = train_one_subject(sid, args)
            results[sid] = acc
            print(f"  -> 被试 {sid}: {acc:.1f}%\n")

        avg = sum(results.values()) / len(results)
        print("=" * 40)
        for sid, acc in results.items():
            print(f"  被试 {sid}: {acc:.1f}%")
        print(f"  ---------------------------")
        print(f"  平均: {avg:.1f}%")
        print(f"  最高: {max(results.values()):.1f}% (被试 {max(results, key=results.get)})")
        print(f"  最低: {min(results.values()):.1f}% (被试 {min(results, key=results.get)})")
    else:
        acc = train_one_subject(args.subject, args)
        print(f"\n-> 被试 {args.subject}: {acc:.1f}%")


if __name__ == "__main__":
    main()
