#!/usr/bin/env python3

import argparse
import time

import rclpy
from stable_baselines3 import SAC

from ugv_rl.ugv_env import UGVBridge, UGVEnv


def main():
    parser = argparse.ArgumentParser(description="Test a trained SAC UGV policy")
    parser.add_argument(
        "--model",
        default="runs/ugv_sac_v7/final.zip",
        help="Path to the trained SAC model",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()

    rclpy.init()
    bridge = None
    env = None

    try:
        bridge = UGVBridge()
        env = UGVEnv(bridge)

        model = SAC.load(args.model, device="cpu")
        print(f"\nLoaded model: {args.model}")
        print(f"Testing for {args.episodes} episodes.\n")

        successes = 0
        collisions = 0

        for episode in range(1, args.episodes + 1):
            obs, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"action={action}")

                total_reward += float(reward)
                steps += 1

                if args.delay > 0:
                    time.sleep(args.delay)

            success = bool(info.get("success", False))
            collision = bool(info.get("collision", False))

            successes += int(success)
            collisions += int(collision)

            print(
                f"Episode {episode}/{args.episodes} | "
                f"Steps: {steps} | "
                f"Reward: {total_reward:.2f} | "
                f"Success: {success} | "
                f"Collision: {collision} | "
                f"Info: {info}"
            )

        print("\n========== TEST SUMMARY ==========")
        print(f"Episodes:   {args.episodes}")
        print(f"Successes:  {successes}/{args.episodes}")
        print(f"Collisions: {collisions}/{args.episodes}")
        print(f"Success rate: {100 * successes / args.episodes:.1f}%")

    finally:
        if env is not None:
            env.close()
        if bridge is not None:
            bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()