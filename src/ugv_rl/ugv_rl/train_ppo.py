#!/usr/bin/env python3

import argparse
import signal
from pathlib import Path

import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

from ugv_rl.ugv_env import UGVBridge, UGVEnv


class EpisodeSummaryCallback(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode:
                print(
                    f"Episode steps={episode['l']}, "
                    f"return={episode['r']:.2f}",
                    flush=True,
                )
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("runs/ugv_ppo"))
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")

    rclpy.init()
    bridge = None
    env = None

    try:
        bridge = UGVBridge()
        env = Monitor(UGVEnv(bridge))

        args.out.mkdir(parents=True, exist_ok=True)

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1000,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs={"net_arch": [128, 128]},
            verbose=1,
            device="auto",
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=500,
            save_path=str(args.out / "checkpoints"),
            name_prefix="ugv_ppo",
        )

        summary_callback = EpisodeSummaryCallback()

        print(f"Training for {args.steps} steps.", flush=True)
        print("Press Ctrl+C to stop training.", flush=True)

        model.learn(
            total_timesteps=args.steps,
            callback=[checkpoint_callback, summary_callback],
        )

        model.save(str(args.out / "final"))
        print(f"Model saved to {args.out / 'final.zip'}", flush=True)

    except KeyboardInterrupt:
        print("Training interrupted.", flush=True)

    finally:
        if env is not None:
            env.close()
        if bridge is not None:
            bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()