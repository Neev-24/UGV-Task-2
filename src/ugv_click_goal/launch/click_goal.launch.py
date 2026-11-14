import os

from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node


def generate_launch_description():

    ugv_path = get_package_share_directory("ugv")

    world_file = os.path.join(
        ugv_path,
        "worlds",
        "empty_with_state.world",
    )

    xacro_file = os.path.join(
        ugv_path,
        "urdf",
        "ugv.urdf.xacro",
    )

    gazebo_launch = os.path.join(
        get_package_share_directory("gazebo_ros"),
        "launch",
        "gazebo.launch.py",
    )

    plugin = os.path.join(
        get_package_prefix("ugv_click_goal"),
        "lib",
        "libugv_click_goal_gui.so",
    )

    robot_description = {
        "robot_description": Command(["xacro ", xacro_file])
    }

    return LaunchDescription([

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch),
            launch_arguments={
                "world": world_file,
                "gui": "false",
            }.items(),
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            parameters=[
                robot_description,
                {"use_sim_time": True},
            ],
        ),

        Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            arguments=[
                "-topic", "robot_description",
                "-entity", "ugv",
                "-z", "0.14",
            ],
            output="screen",
        ),

        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster"],
            output="screen",
        ),

        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["diff_drive_controller"],
            output="screen",
        ),

        ExecuteProcess(
            cmd=[
                "gzclient",
                "--gui-client-plugin=" + plugin,
            ],
            output="screen",
        ),
    ])