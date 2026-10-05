import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node


def generate_launch_description():

    package_name = 'ugv'

    package_path = get_package_share_directory(package_name)

    xacro_file = os.path.join(
        package_path,
        'urdf',
        'ugv.urdf.xacro'
    )

    gazebo_launch_file = os.path.join(
        get_package_share_directory('gazebo_ros'),
        'launch',
        'gazebo.launch.py'
    )

    robot_description = {
        'robot_description': Command([
            'xacro ',
            xacro_file
        ])
    }

    return LaunchDescription([

        # Start Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                gazebo_launch_file
            )
        ),

        # Publish robot description and TF
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                robot_description,
                {'use_sim_time': True}
            ]
        ),

        # Spawn UGV into Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-topic',
                'robot_description',
                '-entity',
                'ugv',
                '-z',
                '0.14'
            ],
            output='screen'
        ),

        # Joint state broadcaster
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                'joint_state_broadcaster'
            ],
            output='screen'
        ),

        # Differential drive controller
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                'diff_drive_controller'
            ],
            output='screen'
        ),

    ])