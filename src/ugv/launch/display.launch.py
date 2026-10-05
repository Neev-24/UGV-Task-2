import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node


def generate_launch_description():

    package_name = 'ugv'

    rviz_config = os.path.join(
        get_package_share_directory(package_name),
        'rviz',
        'ugv.rviz'
    )

    xacro_file = os.path.join(
        get_package_share_directory(package_name),
        'urdf',
        'ugv.urdf.xacro'
    )

    robot_description = {
        'robot_description': Command(['xacro ', xacro_file])
    }

    return LaunchDescription([

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[robot_description]
        ),

        Node(
            package='ugv',
            executable='ugv_controller',
            name='ugv_controller'
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config]
        )

    ])