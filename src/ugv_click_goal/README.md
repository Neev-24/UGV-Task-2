# Click-to-goal testing

This starts the existing Gazebo Classic UGV world with a GUI plugin that publishes
left-clicks as world coordinates. The separate test runner moves the existing goal
marker and optionally runs a trained TD3 policy. Training and the UGV environment
are not modified.

Build from the ROS 2 Humble workspace and launch the new simulation:

```bash
source /opt/ros/humble/setup.bash
cd ~/task2_ws
colcon build --symlink-install --packages-select ugv ugv_rl ugv_click_goal
source install/setup.bash
ros2 launch ugv_click_goal click_goal.launch.py
```

In another terminal in the same container, preview clicks without a model:

```bash
source /opt/ros/humble/setup.bash
source ~/venvs/ugv_rl/bin/activate
cd ~/task2_ws
source install/setup.bash
python src/ugv_rl/ugv_rl/test_click_goal.py --preview
```

Left-click the arena floor; the marker should move. The test starts by resetting
the UGV once. Stop the preview with Ctrl+C. To drive toward clicked goals, run:

```bash
python src/ugv_rl/ugv_rl/test_click_goal.py --model runs/ugv_td3_v1/final.zip
```

The test accepts subsequent clicks while moving and waits for the next click after
success, collision, or timeout. It ignores clicks outside the arena or inside a
pillar. The existing environment still supplies its steering and rewards, and
does not guarantee that a trained model can reach a goal. Use only with Gazebo.
