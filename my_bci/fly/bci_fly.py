"""
BCI → AirSim 脑控无人机 (运动想象 4 分类 → 6 方向控制)
=========================================================

【任务背景】
  导师要求：用开源脑电分类代码，训练一个网络识别运动想象/SSVEP 模式，
  每个模式映射到一个飞行方向，输入脑电数据 → 网络识别 → 控制无人机。
  先在数据集上离线模拟，验证整个链路可行。

【本脚本做什么】
  1. 加载 train_bci.py 训练好的 EEGNet 模型 (eegnet_best_model.pth)
  2. 从 BCI IV-2a 测试集中逐条取出脑电样本
  3. 每条样本喂给模型推理 → 得到预测类别 (左手/右手/脚/舌头)
  4. 将类别映射为 AirSim 飞行指令
  5. 配合键盘实现 6 方向控制 (4 类 + 上/下)

【为什么加载 eegnet_best_model.pth 而不是 eegnet_model.pth】
  train_bci.py 在训练过程中会过拟合：训练集准确率 97%+，测试集准确率却掉回 25%。
  所以训练代码只在测试集准确率创新高时才保存 eegnet_best_model.pth。
  eegnet_model.pth 是最后一个 epoch 的模型，大概率已经过拟合退化。

【当前局限性 (导师已知)】
  - 运动想象只有 4 类，纯脑电只能控制 4 个方向。上/下用键盘补位。
  - 单被试训练，测试准确率约 44~54%，模型有类别偏向 (偏爱预测"舌头动")。
  - SSVEP 方案可区分更多频率模式，适合扩展为纯脑电 6+ 方向控制。

【运行方式】
  先启动 AirSim (UE 里点 Play) → d:/miniconda3/envs/airsim/python.exe bci_fly.py

【指令映射】
  左手动 (class 0) → 左平移      右手动 (class 1) → 右平移
  脚动   (class 2) → 前进        舌头动 (class 3) → 后退
  键盘 Space       → 上升        键盘 Shift       → 下降
"""

import warnings
warnings.filterwarnings('ignore')  # 忽略 MOABB/braindecode 底层库的兼容性警告

import time
import os
import random
import torch
import airsim
import threading
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.models import EEGNetv4

# 确保模型文件路径始终相对于脚本所在目录，无论从哪里执行都能找到
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# keyboard 库用于监听全局键盘
try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False
    print(" 未安装 keyboard 库，垂直方向需手动控制。安装: pip install keyboard")

# ==========================================
# 可调参数
# ==========================================
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "cross_subject.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选 GPU/CPU

SPEED = 1.0              # 水平飞行速度 (m/s)，演示用
VERTICAL_SPEED = 0.5     # 垂直升降速度 (m/s)
CMD_DURATION = 2.0       # 每条脑电指令的执行时长 (秒)
HOVER_PAUSE = 1.0        # 两条指令之间的悬停间隔 (秒)，便于看清每条指令效果

# BCI IV-2a 数据集的 4 个类别 (按标签顺序)
CLASS_NAMES = ["左手动", "右手动", "脚动", "舌头动"]

# 分类标签 → 飞行指令映射
# 格式: {标签: (描述文字, 前后方向比例, 左右方向比例)}
# vx: 正值=前进, 负值=后退。vy: 正值=右移, 负值=左移
COMMAND_MAP = {
    0: ("⬅ 左平移",  0.0, -1.0),
    1: ("➡ 右平移",  0.0,  1.0),
    2: ("⬆ 前进",    1.0,  0.0),
    3: ("⬇ 后退",   -1.0,  0.0),
}

# ---- 键盘线程共享变量 ----
# Python 的 global 变量在多线程中读写简单变量是安全的 (GIL 保护)
kb_vz = 0.0        # 当前键盘垂直速度: 负值=上升, 正值=下降, 0=无操作
kb_running = True   # 控制线程退出的标志


def keyboard_thread():
    """
    后台线程：持续监听键盘
    Space → 上升 (AirSim 中 Z 轴向上为负)
    Shift → 下降
    Esc   → 停止程序
    """
    global kb_vz, kb_running
    while kb_running:
        space = keyboard.is_pressed('space')
        shift = keyboard.is_pressed('shift')
        # 避免同时按下 Space+Shift 时的冲突
        if space and not shift:
            kb_vz = -VERTICAL_SPEED  # AirSim 中 Z 负方向 = 上升
        elif shift and not space:
            kb_vz = VERTICAL_SPEED   # AirSim 中 Z 正方向 = 下降
        else:
            kb_vz = 0.0              # 没有按键 = 垂直悬停
        if keyboard.is_pressed('esc'):
            kb_running = False
            break
        time.sleep(0.05)  # 50ms 轮询一次，不占 CPU


def load_model():
    """
    加载训练好的 EEGNet 脑电分类模型
    参数必须和 train_bci.py 训练时完全一致：
      in_chans=26  (22 EEG + 4 EOG = 26 通道)
      n_classes=4  (左手/右手/脚/舌头)
      input_window_samples=1000 (4秒 × 250Hz = 1000个采样点)
    """
    model = EEGNetv4(
        in_chans=26, n_classes=4,
        input_window_samples=1000, final_conv_length='auto'
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()  # 切换到评估模式：关闭 Dropout/BatchNorm 训练行为
    return model


def predict(model, x):
    """
    对一条脑电数据进行推理
    输入: x = numpy array，形状 (26, 1000) → 26 个通道 × 1000 个时间点
    输出: (预测类别编号, Softmax 置信度)
    """
    with torch.no_grad():  # 不计算梯度，省显存 + 加速
        # unsqueeze(0): (26,1000) → (1,26,1000)，增加 batch 维度
        tensor = torch.tensor(x).unsqueeze(0).float().to(DEVICE)
        output = model(tensor)  # 前向传播，得到 4 个原始分数 (logits)
        probs = torch.softmax(output, dim=1)  # logits → 概率分布
        confidence, pred = torch.max(probs, 1)  # 取最大概率的类别
        return pred.item(), confidence.item()


def execute_command(client, pred_class, confidence):
    """
    将模型预测类别 + 键盘垂直状态 合并为无人机飞行指令
    - 水平方向 (前后左右): 由脑电分类决定
    - 垂直方向 (上升下降): 由键盘决定 (Space/Shift)
    """
    desc, vx_r, vy_r = COMMAND_MAP[pred_class]
    vx = vx_r * SPEED      # 前后速度
    vy = vy_r * SPEED      # 左右速度
    vz = kb_vz             # 垂直速度 (来自键盘线程)

    print(f"   {CLASS_NAMES[pred_class]} (置信度 {confidence:.0%}) → {desc}")

    # MaxDegreeOfFreedom: 无人机可以在任意方向移动，不需要机头朝向运动方向
    # is_rate=True, yaw_or_rate=0: 不旋转机身，只平移
    client.moveByVelocityAsync(
        vx, vy, vz, CMD_DURATION,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0)
    )


def main():
    global kb_running

    print("=" * 55)
    print("  BCI → AirSim 脑控无人机")
    print("=" * 55)

    # ==========================================
    # 步骤 1/4: 加载 AI
    # ==========================================
    print("\n[1/4] 加载 EEGNet 模型...")
    model = load_model()
    print(f"   模型加载设备: {DEVICE})")

    # ==========================================
    # 步骤 2/4: 准备测试用的脑电数据
    # ==========================================
    print("\n[2/4]  加载 BCI IV-2a 测试集...")
    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[3])
    windows_dataset = create_windows_from_events(
        dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0, preload=True
    )
    session_keys = list(windows_dataset.split('session').keys())
    test_set = windows_dataset.split('session')[session_keys[1]]
    print(f"   测试集: {len(test_set)} 个运动想象样本")

    # ==========================================
    # 步骤 3/4: 连接 AirSim 无人机
    # ==========================================
    print("\n[3/4]  连接 AirSim 无人机...")
    client = airsim.MultirotorClient(timeout_value=10)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    time.sleep(1)
    print("   起飞完成")

    # ==========================================
    # 步骤 4/4: 主循环 —— 脑电 → 指令 → 飞行
    # ==========================================
    print("\n[4/4]  开始识别...")
    print("-" * 55)
    print("  指令映射:")
    print("    左手动 → 左平移      右手动 → 右平移")
    print("    脚动   → 前进        舌头动 → 后退")
    if HAS_KEYBOARD:
        print("    Space  → 上升        Shift  → 下降")
    print("-" * 55)

    if HAS_KEYBOARD:
        threading.Thread(target=keyboard_thread, daemon=True).start()

    correct = 0
    total = 0
    n_samples = min(len(test_set), 30)

    # 随机打乱样本顺序，每次运行看不同数据
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
            acc = correct / total * 100
            desc, _, _ = COMMAND_MAP[pred_class]

            print(f"\n[样本 {i+1:02d}/{n_samples}] "
                  f"真实: {CLASS_NAMES[y_true]} → "
                  f"{status} 预测: {CLASS_NAMES[pred_class]} "
                  f"(置信度 {confidence:.0%})")
            print(f"         🛸 飞行指令: {desc}")

            execute_command(client, pred_class, confidence)
            time.sleep(HOVER_PAUSE)

            if (i + 1) % 10 == 0:
                try:
                    state = client.getMultirotorState()
                    pos = state.kinematics_estimated.position
                    print(f"  📍 坐标: X={pos.x_val:.1f} Y={pos.y_val:.1f} Z={pos.z_val:.1f}")
                except Exception:
                    pass  # 获取状态失败则跳过，不影响主流程

    except KeyboardInterrupt:
        print("\n⏸  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        kb_running = False
        print("\n 降落中...")
        try:
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
        except Exception:
            pass  # AirSim 已断连则忽略
        if total > 0:
            print(f"\n 最终准确率: {correct}/{total} = {correct/total*100:.1f}%")
        print(" 演示结束。")


if __name__ == "__main__":
    main()
