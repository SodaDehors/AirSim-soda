"""
EEGNet 脑电分类模型训练脚本
============================

【运行方式】
  d:/miniconda3/envs/airsim/python.exe train_bci.py
"""

import warnings
warnings.filterwarnings('ignore')

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.models import EEGNetv4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # ==========================================
    # 阶段 1: 加载数据
    # ==========================================
    print("⏳ 正在加载 BCI IV-2a 数据 (从本地缓存)...")
    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[3])

    print("✂️ 正在对连续脑电波进行切片 (提取运动想象时刻)...")
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        preload=True
    )

    splitted = windows_dataset.split('session')
    session_keys = list(splitted.keys())
    print(f"🔍 探测到最新版本库的 Session 键名为: {session_keys}")

    train_set = splitted[session_keys[0]]
    test_set = splitted[session_keys[1]]
    print(f"✅ 切片完成！得到 {len(train_set)} 个训练样本，{len(test_set)} 个测试样本。")

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    # ==========================================
    # 阶段 2: 构建模型
    # ==========================================
    print("🧠 正在组装 EEGNet 神经网络...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 当前使用的计算设备: {device}")

    sample_x, _, _ = train_set[0]
    actual_channels = sample_x.shape[0]
    actual_times = sample_x.shape[1]
    print(f"📊 动态探测到真实脑电维度 -> 通道数: {actual_channels}, 采样点数: {actual_times}")

    model = EEGNetv4(
        in_chans=actual_channels,
        n_classes=4,
        input_window_samples=actual_times,
        final_conv_length='auto',
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ==========================================
    # 阶段 3: 训练
    # ==========================================
    epochs = 60
    best_test_acc = 0.0

    print("\n🔥 开始深度训练模型 (引入测试集实时监控)...")

    for epoch in range(epochs):
        # ----- 训练 -----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y, _ in train_loader:
            batch_X, batch_y = batch_X.float().to(device), batch_y.long().to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        train_acc = 100 * train_correct / train_total

        # ----- 测试 -----
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch_X, batch_y, _ in test_loader:
                batch_X, batch_y = batch_X.float().to(device), batch_y.long().to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()

        test_acc = 100 * test_correct / test_total

        print(f"Epoch [{epoch+1:02d}/{epochs}] | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, "eegnet_best_model.pth"))
            print(f"   🌟 发现更强模型！已保存权重 (当前最高 Test Acc: {best_test_acc:.2f}%)")

    print(f"\n🎉 炼丹彻底结束！最终你拿到的最强模型准确率为: {best_test_acc:.2f}%")
    torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, "eegnet_model.pth"))
    print("💾 模型权重已保存至本地: eegnet_model.pth")


if __name__ == "__main__":
    main()
