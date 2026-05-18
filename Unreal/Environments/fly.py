import airsim
import time
import sys

def main():
    # 1. 初始化客户端并连接虚幻引擎
    client = airsim.MultirotorClient()
    try:
        client.confirmConnection()
    except Exception as e:
        print(f"❌ 无法连接到 AirSim 仿真器！请确保虚幻引擎已经点击 'Play' 运行。")
        print(f"错误细节: {e}")
        sys.exit(1)

    print("🚀 成功连接 AirSim！准备接管无人机...")

    try:
        # 2. 显式申请 API 控制权（必须步骤）
        client.enableApiControl(True)
        print("-> 已成功获取外部 API 控制权。")
        
        # 3. 解锁无人机电机 (Arm)
        client.armDisarm(True)
        print("-> 无人机电机已解锁 (Armed)。")

        # 4. 自主起飞
        print("-> 正在发起自主起飞 (Takeoff)...")
        # .join() 是异步转同步的操作，意味着直到起飞动作彻底完成，才会执行下一行代码
        client.takeoffAsync().join()
        print("✨ 起飞完成！已进入安全悬停状态。")
        time.sleep(1)

        # 5. 执行三维空间轨迹飞行
        # 【核心避坑指南】：AirSim 采用的是 NED（北-东-地）坐标系！
        # X 轴正数 = 向前，Y 轴正数 = 向右，Z 轴负数 = 向上 (因为地面是0，地下是正)
        # 这里的参数含义为：moveToPositionAsync(X, Y, Z, 速度)
        
        print("-> 正在向正前方飞行 5 米...")
        client.moveToPositionAsync(5, 0, -3, 2).join()

        print("-> 正在向正右方飞行 5 米...")
        client.moveToPositionAsync(5, 5, -3, 2).join()

        # 6. 获取无人机当前的实时传感器状态 (Telemetry)
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        print(f"📊 【当前空间坐标】X: {pos.x_val:.2f}, Y: {pos.y_val:.2f}, Z: {pos.z_val:.2f}")

        print("-> 原地悬停保持 3 秒...")
        time.sleep(3)

        # 7. 自动降落
        print("-> 正在执行自动降落 (Landing)...")
        client.landAsync().join()
        print("🏁 降落成功！")

    except KeyboardInterrupt:
        print("\n⚠️ 检测到用户手动中断 (Ctrl+C)，正在紧急终止任务...")
    except Exception as e:
        print(f"💥 运行过程中发生未知错误: {e}")
        
    finally:
        # 8. 无论代码是正常跑完还是中途报错，都必须执行安全退出机制
        # 否则虚幻引擎会一直卡在外部控制模式，导致键盘无法手动接管
        print("-> 正在清理现场，释放控制权并关闭电机...")
        client.armDisarm(False)
        client.enableApiControl(False)
        print("🤝 API 控制权已安全交还给系统，任务圆满结束！")

if __name__ == "__main__":
    main()