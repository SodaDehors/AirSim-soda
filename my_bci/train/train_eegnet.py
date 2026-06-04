"""
EEGNet 脑电分类模型训练脚本 (完整数据集版)
===========================================

这是最基础的训练脚本，学习一个轻量级卷积网络 (EEGNet) 识别运动想象模式。
整个项目的起点——后续所有改进 (Conformer、预处理、标签平滑) 都是在此基础上的迭代。

【核心概念】
  - 输入: 一条脑电数据 = (26 通道 × 1000 采样点) 的矩阵
  - 输出: 4 个数字，代表左手/右手/脚/舌头的概率
  - 跨被试: 用 7 个人的数据训练 → 测试另外 2 个没见过的人

【运行方式】
  python train_eegnet.py                  # 全部 9 被试，跨被试
  python train_eegnet.py --epochs 120     # 自定义轮数
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import torch
from torch import nn, optim                        # nn: 神经网络层, optim: Adam 优化器
from torch.utils.data import DataLoader, ConcatDataset
from braindecode.datasets import MOABBDataset       # 自动下载/缓存 BCI IV-2a 数据集
from braindecode.preprocessing import create_windows_from_events  # 截取运动想象时刻的脑电片段
from braindecode.models import EEGNetv4             # 2K 参数轻量脑电专用 CNN

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")  # 模型存到 ../models/

# ==========================================
# 配置：改这里切换 单被试 / 全被试
# ==========================================
# 单被试模式:  TRAIN_SUBJECTS = [3], TEST_SUBJECTS = [3]
# 跨被试模式:  TRAIN_SUBJECTS = [1,2,3,4,5,6,7], TEST_SUBJECTS = [8,9]
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7]
TEST_SUBJECTS = [8, 9]


def load_dataset(subject_ids):
    """
    加载指定被试的脑电数据，返回 训练session + 测试session 合并后的集合。

    BCI IV-2a 每人有两次采集 (session 0 和 1)，
    这里把它们合并，因为跨被试评估不区分 session。

    【数据格式】
      每条试次 = (26 通道, 1000 采样点) 的 ndarray
      标签：0=左手, 1=右手, 2=脚, 3=舌头
    """
    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=subject_ids)
    # 从连续脑电中按事件标记截取出每个 4 秒的试次
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,   # 试次开始偏移=0，从标记点开始
        trial_stop_offset_samples=0,    # 试次结束偏移=0，到标记点结束
        preload=True,                   # 预加载到内存，避免每次读硬盘
    )
    sessions = windows_dataset.split("session")
    sets = [sessions[k] for k in sorted(sessions.keys())]
    if len(sets) == 1:
        return sets[0]
    return ConcatDataset(sets)  # 多个 session 拼成一个大集合


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

    # 逐个被试加载 → 拼成大训练集
    train_sets = []
    for sid in TRAIN_SUBJECTS:
        ds = load_dataset([sid])
        print(f"   被试 {sid}: {len(ds)} trials")
        train_sets.append(ds)
    train_set = ConcatDataset(train_sets)  # 所有训练被试的数据拼一起

    test_sets = []
    for sid in TEST_SUBJECTS:
        ds = load_dataset([sid])
        print(f"   被试 {sid}: {len(ds)} trials")
        test_sets.append(ds)
    test_set = ConcatDataset(test_sets)

    print(f" 合计 — 训练 {len(train_set)} / 测试 {len(test_set)}")

    # DataLoader: 每次取 batch_size 条数据给 GPU 训练
    # shuffle=True: 每轮随机打乱，防止模型记住顺序
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False)

    # ==========================================
    # 阶段 2: 构建模型
    # ==========================================
    sample_x, _, _ = train_set[0]  # 取一条看维度
    actual_channels = sample_x.shape[0]  # 26 通道
    actual_times = sample_x.shape[1]     # 1000 采样点
    print(f" 数据维度: {actual_channels} 通道 x {actual_times} 采样点")

    model = EEGNetv4(
        in_chans=actual_channels,          # 输入通道数 = 26
        n_classes=4,                       # 输出类别 = 4
        input_window_samples=actual_times, # 时间步数 = 1000
        final_conv_length="auto",          # 自动计算最后一层卷积尺寸
        drop_prob=0.5,                     # 训练时随机丢弃 50% 神经元，防过拟合
    ).to(device)  # 搬到 GPU（如果有的话）

    # 损失函数: 衡量预测值和真实标签的差距，越小越好
    criterion = nn.CrossEntropyLoss()
    # Adam 优化器: 根据梯度自动调参数，lr=学习率
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ==========================================
    # 阶段 3: 训练循环
    # ==========================================
    epochs = args.epochs
    best_test_acc = 0.0   # 记录全程最高测试准确率
    patience = 0           # 早停计数器

    print(f"\n 开始训练 ({epochs} 轮, batch={args.batch})")
    print("-" * 55)

    for epoch in range(epochs):
        # ----- 训练阶段 -----
        model.train()  # 开启 Dropout
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X, y, _ in train_loader:
            X, y = X.float().to(device), y.long().to(device)
            optimizer.zero_grad()            # 清空上一轮的梯度
            outputs = model(X)               # 前向传播: 输入 → 4 个分数
            loss = criterion(outputs, y)     # 算损失
            loss.backward()                  # 反向传播: 算梯度
            optimizer.step()                 # 更新参数: 朝最优方向走一步
            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)  # 取 4 个分数中最大的作为预测
            train_total += y.size(0)
            train_correct += (pred == y).sum().item()

        # ----- 验证阶段（不更新参数）-----
        model.eval()  # 关闭 Dropout
        test_correct = 0
        test_total = 0
        with torch.no_grad():  # 不记录梯度，省显存加速
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

        # 只在测试准确率创新高时保存模型（避免过拟合版本覆盖好模型）
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "cross_subject.pth"))
            print(f"    新最佳 ({best_test_acc:.1f}%), 已保存")
        else:
            patience += 1

        # 早停: 连续 N 轮不涨就停止，不用跑完所有轮
        if patience >= 25:
            print(f"\n 早停: {patience} 轮未创新高")
            break

    print(f"\n 完成 — 最佳: {best_test_acc:.1f}%")
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "eegnet_final.pth"))


if __name__ == "__main__":
    main()
