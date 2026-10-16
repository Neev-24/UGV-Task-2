
#!/usr/bin/env python3

import argparse
import glob
import os
import re

import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from ugv_rl.ugv_env import UGVBridge, UGVEnv


def latest_checkpoint(checkpoint_dir):
    pattern = os.path.join(checkpoint_dir, "sac_*_steps.zip")
    matches = []

    for path in glob.glob(pattern):
        match = re.search(r"sac_(\d+)_steps\.zip$", path)
        if match:
            matches.append((int(match.group(1)), path))

    return max(matches, default=(0, None), key=lambda item: item[0])[1]


def main():
    parser = argparse.ArgumentParser(
        description="Train or resume SAC for UGV navigation"
    )
    parser.add_argument("--steps", type=int, default=10000,
                        help="Additional timesteps to train")
    parser.add_argument("--out", default="runs/ugv_sac_v1/final.zip",
                        help="Output directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")

    output_dir = os.path.abspath(args.out)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    monitor_dir = os.path.join(output_dir, "monitor")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)

    rclpy.init()
    bridge = None
    env = None

    try:
        bridge = UGVBridge()
        base_env = UGVEnv(bridge)
        env = Monitor(
            base_env,
            filename=os.path.join(monitor_dir, "train"),
        )

        final_path = os.path.join(output_dir, "final.zip")
        final_replay = os.path.join(output_dir, "replay_buffer.pkl")

        if args.resume:
            checkpoint = latest_checkpoint(checkpoint_dir)

            if checkpoint is None:
                raise FileNotFoundError(
                    f"No SAC checkpoints found in {checkpoint_dir}"
                )

            print(f"Resuming model from: {checkpoint}")
            model = SAC.load(
                checkpoint,
                env=env,
                device="auto",
            )

            replay_path = checkpoint.replace(
                ".zip", "_replay_buffer.pkl"
            )

            match = re.search(r"sac_(\d+)_steps\.zip$", checkpoint)

            if not match:
                raise ValueError(f"Unexpected checkpoint filename: {checkpoint}")

            replay_path = os.path.join(
                checkpoint_dir,
                f"sac_replay_buffer_{match.group(1)}_steps.pkl",
            )

            if not os.path.exists(replay_path):
                raise FileNotFoundError(
                    f"Matching replay buffer not found:\n{replay_path}"
                )

            model.load_replay_buffer(replay_path)
            print(f"Loaded replay buffer: {replay_path}")

            # Continue the timestep count from the checkpoint.
            model.tensorboard_log = os.path.join(
                output_dir, "tensorboard"
            )

        else:
            if glob.glob(os.path.join(checkpoint_dir, "sac_*_steps.zip")):
                raise RuntimeError(
                    f"Checkpoints already exist in {checkpoint_dir}.\n"
                    "Use --resume to continue, or choose a new --out directory."
                )

            print("Starting a fresh SAC model.")

            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                buffer_size=100_000,
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                ent_coef="auto",
                policy_kwargs={"net_arch": [128, 128]},
                tensorboard_log=os.path.join(
                    output_dir, "tensorboard"
                ),
                verbose=1,
                device="auto",
                seed=42,
            )

        callback = CheckpointCallback(
            save_freq=500,
            save_path=checkpoint_dir,
            name_prefix="sac",
            save_replay_buffer=True,
            save_vecnormalize=False,
        )

        print(f"Training {args.steps} additional timesteps.")

        model.learn(
            total_timesteps=args.steps,
            callback=callback,
            reset_num_timesteps=not args.resume,
            tb_log_name="sac_ugv",
        )

        model.save(final_path)
        model.save_replay_buffer(final_replay)

        print("Training finished.")
        print(f"Final model: {final_path}")
        print(f"Final replay buffer: {final_replay}")

    finally:
        if env is not None:
            env.close()
        if bridge is not None:
            bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
