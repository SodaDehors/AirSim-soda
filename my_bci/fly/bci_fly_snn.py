"""
BCI → AirSim SNN 脉冲神经网络版脑控无人机
==========================================

加载 SNN-EEGNet 脉冲模型，速率编码脑电为脉冲序列后推理飞行。

【数据流】
  脑电样本(26×1000) → min-max 归一化 → 速率编码(T 步脉冲) → SNN → 膜电位投票 → 分类

【和 bci_fly.py (Conformer 版) 的核心区别】
  |                  | Conformer 版        | SNN 版 (这个脚本)     |
  |------------------|--------------------|-----------------------|
  | 模型             | EEG-Conformer      | SpikingEEGNet         |
  | 参数             | 500K               | 15K                   |
  | 预处理           | Z-score 归一化     | min-max 归一化         |
  | 编码             | 连续实数           | 泊松脉冲 (0/1)        |
  | 推理             | 一次前向           | T=16 次前向 + 膜电位累积 |
  | 模型文件         | conformer_cross.pth| snn_best.pth           |
  | 跨被试准确率      | 74.9%              | 72.4%                 |

【速率编码过程（每次推理都会执行）】
  1. EEG(26×1000) → min-max 归一化到 [0,1]
  2. 每个值 x → 泊松过程: 以 x 的概率在每个时间步发脉冲
  3. 输出: [1, T, 1, 26, 1000] 的 0/1 脉冲序列

【膜电位投票】
  每个时间步 SNN 跑一遍，LIF 神经元膜电位累积。
  T 步后，取最后一层膜电位的平均值 → FC → 4 分类。
  等价于"看了 T 遍同一条 EEG，每次理解加深，综合做决定"。

【运行方式】
  先启动 AirSim (UE 里点 Play) →
  d:/miniconda3/envs/airsim/python.exe bci_fly_snn.py
"""

import warnings
warnings.filterwarnings("ignore")

import time
import os
import random
import torch
import airsim
import threading
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events, Preprocessor, preprocess

# 从 train_snn.py 导入模型架构和编码函数
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "train"))
from train_snn import SpikingEEGNet, rate_encode
from spikingjelly.activation_based import functional  # reset_net

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False

# ============================================================
# 可调参数
# ============================================================
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "snn_best.pth")
SPEED = 1.0              # 水平飞行速度 (m/s)
VERTICAL_SPEED = 0.5     # 垂直升降速度 (m/s)
CMD_DURATION = 2.0       # 每条脑电指令持续时间 (秒)
HOVER_PAUSE = 1.0        # 指令间悬停间隔 (秒)
T_STEPS = 16             # 脉冲编码时间步数（和训练时一致）

CLASS_NAMES = ["左手动", "右手动", "脚动", "舌头动"]
COMMAND_MAP = {
    #         描述       前进  左右
    0: ("⬅ 左平移",  0.0, -1.0),
    1: ("➡ 右平移",  0.0,  1.0),
    2: ("⬆ 前进",    1.0,  0.0),
    3: ("⬇ 后退",   -1.0,  0.0),
}

# ---- 键盘线程共享变量 ----
kb_vz = 0.0      # 垂直速度: 负=上升, 正=下降
kb_running = True


def keyboard_thread():
    """后台监听键盘: Space↑  Shift↓  Esc→退出"""
    global kb_vz, kb_running
    while kb_running:
        space = keyboard.is_pressed("space")
        shift = keyboard.is_pressed("shift")
        if space and not shift:
            kb_vz = -VERTICAL_SPEED
        elif shift and not space:
            kb_vz = VERTICAL_SPEED
        else:
            kb_vz = 0.0
        if keyboard.is_pressed("esc"):
            kb_running = False
            break
        time.sleep(0.05)


def load_model():
    """
    加载 SNN-EEGNet 脉冲神经网络模型

    输入维度: 26 通道 × 1000 采样点（BCI IV-2a 标准）
    模型文件: models/snn_best.pth（跨被试 72.4%）
    """
    model = SpikingEEGNet(n_chans=26, n_times=1000).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def predict(model, x):
    """
    SNN 推理一条脑电数据

    【步骤】
      1. min-max 归一化到 [0,1]
      2. 速率编码: 连续值 → T 步泊松脉冲序列
      3. reset_net: 清零膜电位（⚠️ 必须做! 否则膜电位残留）
      4. forward: T 步前向，膜电位累积后投票
      5. softmax → 置信度 + 预测类别

    【输入】x: numpy array (26, 1000)
    【输出】(预测类别编号, Softmax 置信度)
    """
    # 步骤 1: min-max 归一化
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)

    # 步骤 2: 速率编码 → 脉冲序列
    # unsqueeze(0): (26,1000) → (1,26,1000) 加 batch 维度
    spikes = rate_encode(
        torch.tensor(x).unsqueeze(0).float().to(DEVICE), T_STEPS
    )

    # 步骤 3+4: 推理
    functional.reset_net(model)  # ⚠️ 清零膜电位
    with torch.no_grad():
        output = model(spikes)
        probs = torch.softmax(output, dim=1)   # → 概率分布
        conf, pred = torch.max(probs, 1)       # 取最大概率

    return pred.item(), conf.item()


def execute_command(client, pred_class, confidence):
    """脑电分类 → AirSim 飞行指令"""
    desc, vx_r, vy_r = COMMAND_MAP[pred_class]
    vx, vy = vx_r * SPEED, vy_r * SPEED
    vz = kb_vz  # 键盘控制垂直
    print(f"   {CLASS_NAMES[pred_class]} "
          f"(置信度 {confidence:.0%}) → {desc}")
    client.moveByVelocityAsync(
        vx, vy, vz, CMD_DURATION,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0),
    )


def main():
    global kb_running

    print("=" * 55)
    print("  SNN → AirSim 脉冲脑控无人机")
    print("=" * 55)

    # ===== 步骤 1/4: 加载模型 =====
    print("\n[1/4] 加载 SNN-EEGNet 脉冲神经网络...")
    model = load_model()
    params = sum(p.numel() for p in model.parameters())
    print(f"   参数: {params:,} | 设备: {DEVICE} | T={T_STEPS}")

    # ===== 步骤 2/4: 加载测试数据 =====
    print(f"\n[2/4] 加载被试 8 测试集 (Chebyshev 4-40Hz 滤波)...")
    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[8])
    # Chebyshev 4-40Hz 带通滤波
    preprocess(dataset, [Preprocessor(fn="filter", l_freq=4, h_freq=40)])
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0, trial_stop_offset_samples=0, preload=True,
    )
    session_keys = list(windows_dataset.split("session").keys())
    test_set = windows_dataset.split("session")[session_keys[1]]
    print(f"   {len(test_set)} 个样本")

    # ===== 步骤 3/4: 连接 AirSim =====
    print("\n[3/4] 连接 AirSim...")
    client = airsim.MultirotorClient(timeout_value=10)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    time.sleep(1)
    print("   起飞完成")

    # ===== 步骤 4/4: 主循环 =====
    print("\n[4/4] 开始 SNN 脑控飞行...")
    print("-" * 55)
    print("  左手动→左平移  右手动→右平移")
    print("  脚动→前进      舌头动→后退")
    if HAS_KEYBOARD:
        print("  Space→上升      Shift→下降")
    print("-" * 55)

    if HAS_KEYBOARD:
        threading.Thread(target=keyboard_thread, daemon=True).start()

    correct, total = 0, 0
    n_samples = min(len(test_set), 30)  # 最多 30 条
    indices = list(range(len(test_set)))
    random.shuffle(indices)  # 随机打乱

    try:
        for i in range(n_samples):
            if not kb_running:
                break

            idx = indices[i]
            X, y_true, _ = test_set[idx]
            pred_class, confidence = predict(model, X)

            correct += (pred_class == y_true)
            total += 1
            status = "✅" if pred_class == y_true else "❌"

            print(f"\n[样本 {i+1:02d}/{n_samples}] "
                  f"真实: {CLASS_NAMES[y_true]} → "
                  f"{status} 预测: {CLASS_NAMES[pred_class]} "
                  f"(置信度 {confidence:.0%})")

            execute_command(client, pred_class, confidence)
            time.sleep(HOVER_PAUSE)

    except KeyboardInterrupt:
        print("\n⏸  中断")
    except Exception as e:
        print(f"\n❌ {e}")
        import traceback
        traceback.print_exc()
    finally:
        kb_running = False
        print("\n 降落...")
        try:
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
        except Exception:
            pass
        if total > 0:
            print(f"\n SNN 准确率: "
                  f"{correct}/{total} = {correct/total*100:.1f}%")
        print(" 结束。")


if __name__ == "__main__":
    main()
