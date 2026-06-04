"""
BCI → AirSim SNN 脉冲神经网络版脑控无人机
==========================================

加载 SNN-EEGNet 脉冲模型，速率编码脑电为脉冲序列后推理飞行。

【和 bci_fly.py 的区别】
  - 模型: SpikingEEGNet (LIF 脉冲神经元) 而非 Conformer
  - 预处理: 速率编码 → 脉冲序列，而非 Z-score 归一化
  - 模型: snn_best.pth (跨被试 72.4%)

【运行方式】
  先启动 AirSim → d:/miniconda3/envs/airsim/python.exe bci_fly_snn.py
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

# 导入 SNN 模型架构
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
from train_snn import SpikingEEGNet, rate_encode
from spikingjelly.activation_based import functional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False

# ====== 参数 ======
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "snn_best.pth")
SPEED = 1.0
VERTICAL_SPEED = 0.5
CMD_DURATION = 2.0
HOVER_PAUSE = 1.0
T_STEPS = 16  # 脉冲时间步数

CLASS_NAMES = ["左手动", "右手动", "脚动", "舌头动"]
COMMAND_MAP = {
    0: ("⬅ 左平移", 0.0, -1.0),
    1: ("➡ 右平移", 0.0, 1.0),
    2: ("⬆ 前进", 1.0, 0.0),
    3: ("⬇ 后退", -1.0, 0.0),
}

kb_vz = 0.0
kb_running = True


def keyboard_thread():
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
    sample_x = torch.zeros(1, 1, 26, 1000)  # dummy 用于推断维度
    model = SpikingEEGNet(n_chans=26, n_times=1000).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def predict(model, x):
    # 归一化 → 速率编码 → 脉冲序列 → 推理
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    spikes = rate_encode(
        torch.tensor(x).unsqueeze(0).float().to(DEVICE), T_STEPS
    )
    functional.reset_net(model)
    with torch.no_grad():
        output = model(spikes)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    return pred.item(), conf.item()


def execute_command(client, pred_class, confidence):
    desc, vx_r, vy_r = COMMAND_MAP[pred_class]
    vx, vy = vx_r * SPEED, vy_r * SPEED
    vz = kb_vz
    print(f"   {CLASS_NAMES[pred_class]} (置信度 {confidence:.0%}) → {desc}")
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

    print("\n[1/4] 加载 SNN-EEGNet 脉冲神经网络...")
    model = load_model()
    params = sum(p.numel() for p in model.parameters())
    print(f"   参数: {params:,} | 设备: {DEVICE} | T={T_STEPS}")

    print(f"\n[2/4] 加载被试 8 测试集...")
    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[8])
    preprocess(dataset, [Preprocessor(fn="filter", l_freq=4, h_freq=40)])
    windows_dataset = create_windows_from_events(
        dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0, preload=True
    )
    session_keys = list(windows_dataset.split("session").keys())
    test_set = windows_dataset.split("session")[session_keys[1]]
    print(f"   {len(test_set)} 个样本")

    print("\n[3/4] 连接 AirSim...")
    client = airsim.MultirotorClient(timeout_value=10)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    time.sleep(1)
    print("   起飞完成")

    print("\n[4/4] 开始 SNN 脑控飞行...")
    print("-" * 55)
    print("  左手动→左平移 右手动→右平移 脚动→前进 舌头动→后退")
    if HAS_KEYBOARD:
        print("  Space→上升 Shift→下降")
    print("-" * 55)

    if HAS_KEYBOARD:
        threading.Thread(target=keyboard_thread, daemon=True).start()

    correct, total = 0, 0
    n_samples = min(len(test_set), 30)
    indices = list(range(len(test_set)))
    random.shuffle(indices)

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
            print(f"\n SNN 准确率: {correct}/{total} = {correct/total*100:.1f}%")
        print(" 结束。")


if __name__ == "__main__":
    main()
