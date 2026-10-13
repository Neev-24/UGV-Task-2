from setuptools import setup

package_name = 'ugv_rl'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/ugv_rl']),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Gymnasium reinforcement learning environment for a ROS 2 UGV',
    license='Apache-2.0',
)
