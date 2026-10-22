
#!/usr/bin/env python3

import argparse
import threading
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
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--out", type=Path, default=Path("runs/ugv_ppo"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint, or final model if no checkpoint exists.",
    )
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")

    checkpoint_dir = args.out / "checkpoints"
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resume_path = None
    if args.resume:
        checkpoints = list(checkpoint_dir.glob("ugv_ppo_*_steps.zip"))
        if checkpoints:
            def checkpoint_step(path):
                try:
                    return int(path.stem.rsplit("_", 2)[1])
                except (ValueError, IndexError):
                    return -1

            resume_path = max(checkpoints, key=checkpoint_step)
        elif (args.out / "final.zip").exists():
            resume_path = args.out / "final.zip"
        else:
            parser.error(
                f"No checkpoint or final.zip found in {args.out}. "
                "Run without --resume first."
            )

    rclpy.init()
    bridge = None
    env = None
    executor = None
    try:
        bridge = UGVBridge()
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(bridge)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        env = Monitor(UGVEnv(bridge))

        if resume_path is not None:
            print(f"Resuming from: {resume_path}", flush=True)
            model = PPO.load(
                str(resume_path),
                env=env,
                device="auto",
            )
        else:


            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=1024,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                policy_kwargs={"net_arch": [128, 128]},
                verbose=1,
                device="auto",
            )

        checkpoint_callback = CheckpointCallback(
            save_freq=500,
            save_path=str(checkpoint_dir),
            name_prefix="ugv_ppo",
        )

        summary_callback = EpisodeSummaryCallback()

        print(f"Training for another {args.steps} timesteps.", flush=True)
        print("Press Ctrl+C to stop training.", flush=True)

        model.learn(
            total_timesteps=args.steps,
            callback=[checkpoint_callback, summary_callback],
            reset_num_timesteps=(resume_path is None),
        )

        model.save(str(args.out / "final"))
        print(f"Model saved to {args.out / 'final.zip'}", flush=True)

    except KeyboardInterrupt:
        print(
            "Training interrupted. Use --resume to continue from a saved checkpoint.",
            flush=True,
        )

    finally:
        if env is not None:
            env.close()
        if executor is not None:
            executor.shutdown()
        if bridge is not None:
            bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
