import time
import rclpy
from stable_baselines3 import PPO
from ugv_rl.ugv_env import UGVBridge, UGVEnv
import threading


def main():
    rclpy.init()

    bridge = UGVBridge()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(bridge)
    threading.Thread(target=executor.spin, daemon=True).start()
    env = UGVEnv(bridge)

    model = PPO.load("newruns/ugv_ppo_v1/final.zip")

    try:
        obs, info = env.reset()
        total_reward = 0.0

        while True:
            action, _ = model.predict(obs, deterministic=True)

            # print(
            #     f"Linear action: {action[0]:.3f} | "
            #     f"Angular action: {action[1]:.3f} | "
            #     f"Front LiDAR: {obs[10:14]}"
            # )

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(
                f"Reward: {reward:.3f} | "
                f"Total: {total_reward:.3f}"
            )

            if terminated or truncated:
                print("Episode ended:", info)
                break


    except KeyboardInterrupt:
        print("Inference stopped.")

    finally:
        env.close()
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
