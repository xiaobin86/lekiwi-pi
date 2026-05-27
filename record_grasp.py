#!/usr/bin/env python
"""
LeKiwi 抓取纸团录制脚本
使用主臂遥操作控制从臂 + 键盘控制底盘移动
"""

from lerobot.common.control_utils import init_keyboard_listener
from lerobot.datasets import LeRobotDataset
from lerobot.processor import make_default_processors
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# ==================== 配置 ====================
# 录制参数
NUM_EPISODES = 10          # 录制 episode 数量
FPS = 30                   # 帧率
EPISODE_TIME_SEC = 30      # 每个 episode 时长（秒）
RESET_TIME_SEC = 10        # 重置环境时长（秒）
TASK_DESCRIPTION = "Grasp paper ball with follower arm"  # 任务描述

# 数据集配置
HF_REPO_ID = "your_username/lekiwi_grasp_paper_ball"  # Hugging Face 数据集仓库

# 硬件配置
RASPBERRY_PI_IP = "192.168.3.176"  # 树莓派 IP 地址
LEADER_ARM_PORT = "COM5"            # 主臂串口（Windows: COMx, Linux: /dev/ttyACM0）
LEADER_ARM_ID = "L07252802"         # 主臂 ID


def main():
    print("=" * 60)
    print("LeKiwi 抓取纸团录制")
    print("=" * 60)
    print(f"\n任务: {TASK_DESCRIPTION}")
    print(f"树莓派: {RASPBERRY_PI_IP}")
    print(f"主臂端口: {LEADER_ARM_PORT}")
    print(f"Episode 数量: {NUM_EPISODES}")
    print(f"Episode 时长: {EPISODE_TIME_SEC}秒")
    print("=" * 60)

    # 1. 创建配置
    print("\n[1/4] 初始化配置...")
    robot_config = LeKiwiClientConfig(
        remote_ip=RASPBERRY_PI_IP,
        id="lekiwi",
        port_zmq_cmd=5555,
        port_zmq_observations=5556,
    )
    
    # 主臂配置（SO100/SO101）
    leader_arm_config = SO100LeaderConfig(
        port=LEADER_ARM_PORT,
        id=LEADER_ARM_ID,
    )
    
    keyboard_config = KeyboardTeleopConfig()

    # 2. 初始化机器人和遥操作器
    print("[2/4] 连接硬件...")
    robot = LeKiwiClient(robot_config)
    leader_arm = SO100Leader(leader_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    # 3. 配置数据集
    print("[3/4] 配置数据集...")
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # 4. 连接硬件
    print("[4/4] 连接硬件...")
    print("  - 连接树莓派...")
    robot.connect()
    print("  - 连接主臂...")
    leader_arm.connect()
    print("  - 连接键盘...")
    keyboard.connect()

    # 检查连接状态
    if not robot.is_connected:
        raise ValueError("❌ 无法连接到树莓派！请检查：\n"
                        "  1. 树莓派是否已启动 host 程序\n"
                        "  2. IP 地址是否正确\n"
                        "  3. 网络连接是否正常")
    
    if not leader_arm.is_connected:
        raise ValueError("❌ 无法连接到主臂！请检查：\n"
                        "  1. 主臂是否已上电\n"
                        "  2. 串口是否正确\n"
                        "  3. USB 线是否连接")

    print("\n✅ 所有硬件已连接！")

    # 初始化键盘监听器和可视化
    listener, events = init_keyboard_listener()
    init_rerun(session_name="lekiwi_grasp_record")

    # 创建默认处理器
    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    print("\n" + "=" * 60)
    print("录制控制:")
    print("  主臂: 控制从臂抓取纸团")
    print("  键盘: W/A/S/D=移动, Z/X=旋转, R/F=速度, Q=退出")
    print("  空格键: 提前结束当前 episode")
    print("  R键: 重新录制当前 episode")
    print("=" * 60)

    try:
        recorded_episodes = 0
        while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"开始录制 episode {recorded_episodes + 1}/{NUM_EPISODES}")
            print(f"\n🎬 录制 Episode {recorded_episodes + 1}/{NUM_EPISODES}...")

            # 主录制循环
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                dataset=dataset,
                teleop=[leader_arm, keyboard],
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )

            # 检查是否需要重置环境
            if not events["stop_recording"] and (
                (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
            ):
                if events["rerecord_episode"]:
                    log_say("重新录制")
                    print("🔄 重新录制当前 episode...")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue
                else:
                    log_say("重置环境")
                    print("🔄 重置环境...")
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=FPS,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=[leader_arm, keyboard],
                        control_time_s=RESET_TIME_SEC,
                        single_task=TASK_DESCRIPTION,
                        display_data=True,
                    )

            # 保存 episode
            dataset.save_episode()
            recorded_episodes += 1
            print(f"✅ Episode {recorded_episodes} 已保存！")

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    finally:
        print("\n清理资源...")
        log_say("停止录制")
        
        # 断开连接
        if robot.is_connected:
            robot.disconnect()
        if leader_arm.is_connected:
            leader_arm.disconnect()
        if keyboard.is_connected:
            keyboard.disconnect()
        listener.stop()

        # 保存并上传数据集
        print("\n保存数据集...")
        dataset.finalize()
        
        print("上传到 Hugging Face...")
        dataset.push_to_hub()
        
        print(f"\n✅ 录制完成！")
        print(f"   已录制 {recorded_episodes} 个 episodes")
        print(f"   数据集: {HF_REPO_ID}")
        print(f"   本地路径: {dataset.root}")


if __name__ == "__main__":
    main()
