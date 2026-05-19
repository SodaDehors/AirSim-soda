"""
AirSim 无人机 FPV 手动控制 (键盘 + 鼠标)
==========================================

【功能】
  - W/S → 前进/后退
  - A/D → 左/右平移
  - Space/Shift → 上升/下降
  - 鼠标左右 → 偏航转向 (Yaw)
  - 鼠标上下 → 摄像头俯仰 (Pitch)
  - Esc → 安全降落退出

【核心修复记录】
  1. 窗口从 200×50 扩大到 400×300，否则鼠标瞬间撞边无法转向
  2. 每帧把鼠标重置回窗口中心 (set_pos)，彻底解决鼠标撞边问题
  3. 速度从 15m/s 降到 1.5m/s，加入平滑加减速 (不再突兀启停)
  4. 偏航灵敏度按 1600DPI 校准：0.001 rad/px ≈ 4 英寸转一圈
  5. 俯仰灵敏度 0.003 rad/px，比偏航稍快 (俯仰范围只有 ±81°)

【鼠标灵敏度计算】
  yaw_sens = 0.001 rad/像素
  转一圈 360° = 2π ≈ 6.28 rad → 需要 6283 像素
  1600 DPI 鼠标 → 6283 / 1600 ≈ 3.9 英寸 (约 10cm) 转一圈

【平滑加减速原理】
  不直接设 v = 目标速度，而是每帧做指数平滑 (lerp):
    curr = curr + (target - curr) × factor
  效果: 按 W 时速度从 0 逐渐增长到最大值，松手时平滑回 0
  accel_factor=0.12: 加速系数 (较慢 → 启动平滑)
  decel_factor=0.20: 减速系数 (较快 → 松手响应快)
"""

import airsim
import pygame
import sys
import math
import os


def main():
    # 控制 Pygame 窗口在屏幕上的弹出位置 (左上角坐标)
    os.environ['SDL_VIDEO_WINDOW_POS'] = "10,30"

    # ---- 连接 AirSim + 起飞 ----
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)    # 接管控制权
    client.armDisarm(True)           # 解锁电机

    print("正在自动起飞...")
    client.takeoffAsync().join()     # 等待起飞完成

    # 读取起飞后无人机的偏航角 (用于初始化朝向)
    state = client.getMultirotorState()
    _, _, current_yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)

    # ---- Pygame 窗口 ----
    pygame.init()
    # 💥 修复 1: 窗口扩大到 400×300 (原 200×50)，鼠标有足够移动空间
    screen = pygame.display.set_mode((400, 300))
    screen_center = (screen.get_width() // 2, screen.get_height() // 2)
    pygame.display.set_caption('FPV Control - 点此窗口激活鼠标')

    # ---- 可调飞行参数 ----
    max_speed = 1.5           # 最大水平飞行速度 (m/s)
    max_vertical_speed = 0.7  # 最大垂直升降速度 (m/s)
    yaw_sens = 0.001          # 偏航灵敏度 (rad/像素)，按 1600DPI 校准 ~4inch/圈
    pitch_sens = 0.003        # 俯仰灵敏度 (rad/像素)，比偏航快因为俯仰范围小
    accel_factor = 0.12       # 加速平滑系数: 数值越大加速越快 (每帧逼近比例)
    decel_factor = 0.20       # 减速平滑系数: 比加速快 → 松手即停
    camera_pitch = 0.0        # 摄像头初始俯仰角
    curr_vx, curr_vy, curr_vz = 0.0, 0.0, 0.0  # 当前平滑后的速度
    clock = pygame.time.Clock()

    # 隐藏鼠标光标，锁定到窗口内
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)   # 鼠标不会离开窗口
    running = True

    print("\n 起飞完成！")
    print("👉 点那个小窗口激活鼠标")
    print("-" * 30)
    print("🎮 操作说明：")
    print("  [W/S] : 前进 / 后退")
    print("  [A/D] : 左侧平移 / 右侧平移")
    print("  [空格]: 上升")
    print("  [Shift]: 下降")
    print("  [鼠标左右]: 转向 (Yaw)")
    print("  [鼠标上下]: 抬头 / 低头 (Pitch)")
    print("  [ESC] : 安全降落并退出")
    print("-" * 30 + "\n")

    try:
        while running:
            # 处理窗口事件 (退出、ESC 等)
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or \
                   (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

            # ---- 键盘输入: 设定目标速度 ----
            keys = pygame.key.get_pressed()
            target_vx, target_vy, target_vz = 0.0, 0.0, 0.0

            if keys[pygame.K_w]: target_vx = max_speed
            if keys[pygame.K_s]: target_vx = -max_speed
            if keys[pygame.K_d]: target_vy = max_speed
            if keys[pygame.K_a]: target_vy = -max_speed
            if keys[pygame.K_SPACE]: target_vz = -max_vertical_speed    # Z负=上升
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: target_vz = max_vertical_speed

            # 💥 修复 3: 平滑加减速
            # 公式: new = old + (target - old) × factor
            # 这是指数平滑 (lerp)，效果是逐渐逼近目标值
            # 有按键时用 accel_factor 加速，松手时用 decel_factor 减速
            if abs(target_vx) > 0.01:
                curr_vx += (target_vx - curr_vx) * accel_factor
            else:
                curr_vx += (0 - curr_vx) * decel_factor

            if abs(target_vy) > 0.01:
                curr_vy += (target_vy - curr_vy) * accel_factor
            else:
                curr_vy += (0 - curr_vy) * decel_factor

            if abs(target_vz) > 0.01:
                curr_vz += (target_vz - curr_vz) * accel_factor
            else:
                curr_vz += (0 - curr_vz) * decel_factor

            # 当前帧的最终速度
            vx, vy, vz = curr_vx, curr_vy, curr_vz

            # ---- 鼠标输入: 视角控制 ----
            mouse_dx, mouse_dy = pygame.mouse.get_rel()  # 获取鼠标相对位移

            # 💥 修复 2: 手动把鼠标拉回窗口中心，防止撞边
            # 这是标准 FPS 做法 —— 读完位移立刻复位，鼠标永远不会碰到窗口边缘
            pygame.mouse.set_pos(screen_center)
            pygame.mouse.get_rel()  # 吃掉 warp 产生的虚假位移

            # 偏航 (Yaw): 鼠标左右 → 无人机转向
            if mouse_dx != 0:
                current_yaw += mouse_dx * yaw_sens
                current_yaw = current_yaw % (2 * math.pi)  # 保持在 [0, 2π)

            # 俯仰 (Pitch): 鼠标上下 → 摄像头仰角
            # 限制在 ±81° 内，防止万向节死锁 (gimbal lock)
            if mouse_dy != 0:
                camera_pitch -= mouse_dy * pitch_sens
                camera_pitch = max(-math.pi / 2.2, min(math.pi / 2.2, camera_pitch))

            # ---- 速度方向变换: 机体坐标系 → 世界坐标系 ----
            # 因为 current_yaw 是无人机当前的偏航角，
            # vx(期望前进) 和 vy(期望右移) 需要旋转到世界坐标系
            world_vx = vx * math.cos(current_yaw) - vy * math.sin(current_yaw)
            world_vy = vx * math.sin(current_yaw) + vy * math.cos(current_yaw)

            # ---- 发送飞行指令 ----
            # MaxDegreeOfFreedom: 无人机可以任意方向移动，不依赖机头朝向
            # yaw_mode (绝对角度): 告诉无人机保持指定偏航角
            client.moveByVelocityAsync(
                world_vx, world_vy, vz,
                duration=0.1,  # 100ms 后需要重新发送 (刷新率高 = 控制更实时)
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(current_yaw))
            )

            # ---- 摄像头俯仰 (独立于机身) ----
            # simSetCameraPose: 直接设置摄像头在机身上的相对姿态
            # to_quaternion(pitch, roll, yaw): 欧拉角 → 四元数
            if mouse_dy != 0:
                camera_pose = airsim.Pose(
                    airsim.Vector3r(0, 0, 0),
                    airsim.to_quaternion(camera_pitch, 0, 0)
                )
                client.simSetCameraPose("0", camera_pose)

            # ---- 实时坐标显示 ----
            if vx != 0 or vy != 0 or vz != 0:
                current_state = client.getMultirotorState()
                pos = current_state.kinematics_estimated.position
                print(f"实时坐标 | X: {pos.x_val:.1f} | Y: {pos.y_val:.1f} | Z: {pos.z_val:.1f}", end='\r')

            clock.tick(60)  # 限制帧率 60FPS

    except Exception as e:
        print(f"\n 发生崩溃: {e}")

    finally:
        # ---- 安全退出 ----
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)
        pygame.quit()
        try:
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
        except:
            pass
        print("\n 控制已释放。")


if __name__ == "__main__":
    main()
