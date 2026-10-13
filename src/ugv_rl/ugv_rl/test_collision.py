import rclpy
from ugv_env import UGVBridge, UGVEnv

rclpy.init()
bridge = UGVBridge()
env = UGVEnv(bridge)

try:
    # Episode 1: reset should clear the collision flag.
    bridge.collision_detected = True
    env.reset()

    print("Collision flag after reset:", bridge.has_collision())

finally:
    env.close()
    bridge.destroy_node()
    rclpy.shutdown()