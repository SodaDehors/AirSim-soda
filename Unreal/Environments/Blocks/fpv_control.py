import airsim
import pygame
import sys
import math
import os

def main():
    # 把窗口弹出位置
    os.environ['SDL_VIDEO_WINDOW_POS'] = "10,30"

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    print("正在自动起飞...")
    client.takeoffAsync().join()
    
    state = client.getMultirotorState()
    _, _, current_yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)

    pygame.init()
    # 弹出窗口大小（需要足够大，鼠标才不会频繁撞边）
    screen = pygame.display.set_mode((400, 300))
    screen_center = (screen.get_width() // 2, screen.get_height() // 2)
    pygame.display.set_caption('click')


    max_speed = 1.5           # 最大水平速度
    max_vertical_speed = 0.7  # 最大垂直速度
    yaw_sens = 0.001          # 偏航灵敏度rad (1600DPI ~4inch/360°)
    pitch_sens = 0.003        # 俯仰灵敏度rad
    accel_factor = 0.12       # 加速平滑系数（每帧逼近比例）
    decel_factor = 0.20       # 减速平滑系数
    camera_pitch = 0.0        # 摄像头的初始俯仰角，防止鼠标移动时报错
    curr_vx, curr_vy, curr_vz = 0.0, 0.0, 0.0
    clock = pygame.time.Clock()

    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
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
            pygame.event.pump() 
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

            keys = pygame.key.get_pressed()
            target_vx, target_vy, target_vz = 0.0, 0.0, 0.0

            if keys[pygame.K_w]: target_vx = max_speed
            if keys[pygame.K_s]: target_vx = -max_speed
            if keys[pygame.K_d]: target_vy = max_speed
            if keys[pygame.K_a]: target_vy = -max_speed
            if keys[pygame.K_SPACE]: target_vz = -max_vertical_speed
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: target_vz = max_vertical_speed

            # 平滑加减速：有按键时向目标加速，松开时向 0 减速
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

            vx, vy, vz = curr_vx, curr_vy, curr_vz

            mouse_dx, mouse_dy = pygame.mouse.get_rel()
            # 每帧把鼠标重置回窗口中心，防止撞到窗口边缘后丢失位移
            pygame.mouse.set_pos(screen_center)
            pygame.mouse.get_rel()  # 吃掉 warp 产生的合成事件

            if mouse_dx != 0:
                current_yaw += mouse_dx * yaw_sens
                current_yaw = current_yaw % (2 * math.pi)

            if mouse_dy != 0:
                camera_pitch -= mouse_dy * pitch_sens
                camera_pitch = max(-math.pi/2.2, min(math.pi/2.2, camera_pitch))

            world_vx = vx * math.cos(current_yaw) - vy * math.sin(current_yaw)
            world_vy = vx * math.sin(current_yaw) + vy * math.cos(current_yaw)

            client.moveByVelocityAsync(
                world_vx, world_vy, vz,
                duration=0.1,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(current_yaw))
            )

            if mouse_dy != 0:
                camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(camera_pitch, 0, 0))
                client.simSetCameraPose("0", camera_pose)

            # 
            if vx != 0 or vy != 0 or vz != 0:
                current_state = client.getMultirotorState()
                pos = current_state.kinematics_estimated.position
                print(f"实时坐标 | X: {pos.x_val:.1f} | Y: {pos.y_val:.1f} | Z: {pos.z_val:.1f}", end='\r')

            clock.tick(60)

    except Exception as e:
        print(f"\n 发生崩溃: {e}")
        
    finally:
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