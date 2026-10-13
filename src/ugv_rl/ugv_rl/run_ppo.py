import time
import rclpy
from stable_baselines3 import PPO
from ugv_env import UGVBridge, UGVEnv


def main():
    rclpy.init()

    bridge = UGVBridge()
    env = UGVEnv(bridge)

    model = PPO.load("runs/ugv_ppo/final.zip")

    try:
        obs, info = env.reset()
        total_reward = 0.0

        while True:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(
                f"Reward: {reward:.3f} | "
                f"Total: {total_reward:.3f}"
            )

            if terminated or truncated:
                print("Episode ended:", info)
                break

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Inference stopped.")

    finally:
        env.close()
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
