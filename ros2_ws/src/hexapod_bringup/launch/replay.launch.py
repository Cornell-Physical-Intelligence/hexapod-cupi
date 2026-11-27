"""Replay a bag's raw lidar and IMU topics through FAST-LIO and a Nav2 costmap.

Usage: ros2 launch hexapod_bringup replay.launch.py bag:=<input bag> [roll:= pitch:= height:= rviz:=true]
See ros2_ws/README.md, "Replay a bag".
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Raw sensor topics only; the bags also hold the Jetson's FAST-LIO output.
RAW_TOPICS = ["/livox/lidar", "/livox/imu"]


def generate_launch_description():
    bag = LaunchConfiguration("bag")
    share = get_package_share_directory("hexapod_bringup")
    fast_lio_config = os.path.join(get_package_share_directory("fast_lio"), "config", "mid360.yaml")

    fast_lio = Node(
        package="fast_lio",
        executable="fastlio_mapping",
        # Later entries override earlier ones; skip the map file written on exit.
        parameters=[fast_lio_config, {"pcd_save.pcd_save_en": False}],
        output="screen",
    )
    # odom -> camera_init: levels FAST-LIO's start frame and puts odom on the floor.
    level = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--z",
            LaunchConfiguration("height"),
            "--roll",
            LaunchConfiguration("roll"),
            "--pitch",
            LaunchConfiguration("pitch"),
            "--frame-id",
            "odom",
            "--child-frame-id",
            "camera_init",
        ],
    )
    costmap = Node(
        package="nav2_costmap_2d",
        executable="nav2_costmap_2d",
        name="costmap",
        parameters=[os.path.join(share, "config", "costmap.yaml")],
        output="screen",
    )
    # The costmap is a lifecycle node; this configures and activates it.
    # Activation waits for FAST-LIO's first pose, which outlasts the default
    # 4 s heartbeat (bond) check, so the check is off.
    lifecycle = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_costmap",
        parameters=[{"node_names": ["costmap"], "autostart": True, "bond_timeout": 0.0}],
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", os.path.join(share, "config", "replay.rviz")],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )
    player = ExecuteProcess(
        cmd=["ros2", "bag", "play", bag, "--topics", *RAW_TOPICS],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("bag", description="Input bag folder"),
            DeclareLaunchArgument("roll", default_value="0.0", description="Levelling roll, rad"),
            DeclareLaunchArgument("pitch", default_value="0.0", description="Levelling pitch, rad"),
            DeclareLaunchArgument(
                "height", default_value="0.0", description="Lidar height above the floor, m"
            ),
            DeclareLaunchArgument("rviz", default_value="false", description="Open RViz"),
            fast_lio,
            level,
            costmap,
            lifecycle,
            rviz,
            # Let FAST-LIO subscribe before the first scan arrives.
            TimerAction(period=3.0, actions=[player]),
            # Stop everything when playback ends.
            RegisterEventHandler(
                OnProcessExit(
                    target_action=player,
                    on_exit=[EmitEvent(event=Shutdown(reason="bag finished"))],
                )
            ),
        ]
    )
