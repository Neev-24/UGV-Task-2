#!/usr/bin/env python3
import argparse
import rclpy
from stable_baselines3 import PPO, SAC
from ugv_rl.ugv_env import UGVBridge, UGVEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    model_path = f"runs/ugv_{args.algo}/final.zip"
    AlgoClass = PPO if args.algo == "ppo" else SAC

    rclpy.init()
    bridge = UGVBridge()
    env = UGVEnv(bridge)
    model = AlgoClass.load(model_path)

    try:
        for ep in range(1, args.episodes + 1):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            print(f"Episode {ep} | Reward: {total_reward:.1f} | Info: {info}")
    finally:
        env.close()
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()