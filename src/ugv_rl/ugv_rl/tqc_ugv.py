#!/usr/bin/env python3
"""Train or evaluate TQC with the existing ROS 2 UGV environment."""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import rclpy

from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor


# Prefer the ugv_env.py next to this script over an older installed copy.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ugv_rl.ugv_env import UGVBridge, UGVEnv


def latest_pair(folder):
    """Find the newest complete TQC model and replay-buffer pair."""
    pairs = []

    final_model = folder / "final.zip"
    final_replay = folder / "replay_buffer.pkl"

    if final_model.is_file() and final_replay.is_file():
        pairs.append((final_model, final_replay))

    for model_path in (folder / "checkpoints").glob("tqc_*_steps.zip"):
        match = re.fullmatch(
            r"tqc_(\d+)_steps\.zip",
            model_path.name,
        )

        if match:
            replay_path = (
                model_path.parent
                / f"tqc_replay_buffer_{match.group(1)}_steps.pkl"
            )

            if replay_path.is_file():
                pairs.append((model_path, replay_path))

    if not pairs:
        raise FileNotFoundError(
            f"No matching TQC model and replay buffer in {folder}"
        )

    return max(
        pairs,
        key=lambda pair: min(p.stat().st_mtime_ns for p in pair),
    )


def train(env, folder, args, resume_pair):
    checkpoint_dir = folder / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if resume_pair is not None:
        model_path, replay_path = resume_pair

        print(f"Resuming model: {model_path}", flush=True)
        print(f"Replay buffer: {replay_path}", flush=True)

        model = TQC.load(
            str(model_path),
            env=env,
            device="auto",
        )

        model.load_replay_buffer(str(replay_path))

        print(
            f"Previously trained steps: {model.num_timesteps}",
            flush=True,
        )

    else:
        model = TQC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            top_quantiles_to_drop_per_net=2,
            policy_kwargs={
                "net_arch": [128, 128],
                "n_critics": 5,
                "n_quantiles": 25,
            },
            seed=args.seed,
            verbose=1,
            device="auto",
        )

        print(
            "New TQC model: 5 critics, 25 quantiles per critic.",
            flush=True,
        )

    checkpoint = CheckpointCallback(
        save_freq=1_000,
        save_path=str(checkpoint_dir),
        name_prefix="tqc",
        save_replay_buffer=True,
    )

    print(
        f"Starting {args.steps} additional environment steps.",
        flush=True,
    )

    try:
        model.learn(
            total_timesteps=args.steps,
            callback=checkpoint,
            reset_num_timesteps=not args.resume,
            log_interval=1,
        )

    except KeyboardInterrupt:
        print(
            "Interrupted; saving the current model and replay buffer.",
            flush=True,
        )

    finally:
        model.save(str(folder / "final.zip"))
        model.save_replay_buffer(str(folder / "replay_buffer.pkl"))

        print(f"Saved model: {folder / 'final.zip'}", flush=True)

        print(
            f"Saved replay buffer: {folder / 'replay_buffer.pkl'}",
            flush=True,
        )

        print(
            f"Total trained steps: {model.num_timesteps}",
            flush=True,
        )


def evaluate(env, model_path, args):
    model = TQC.load(
        str(model_path),
        env=env,
        device="auto",
    )

    print(f"Loaded TQC model: {model_path}", flush=True)

    print(
        f"Total trained steps: {model.num_timesteps}",
        flush=True,
    )

    successes = 0
    collisions = 0
    timeouts = 0
    returns = []

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)

        ended = False
        total_reward = 0.0
        steps = 0

        while not ended:
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, info = env.step(action)

            ended = terminated or truncated
            total_reward += float(reward)
            steps += 1

        success = bool(info.get("success", False))
        collision = bool(info.get("collision", False))
        timeout = bool(truncated and not terminated)

        successes += int(success)
        collisions += int(collision)
        timeouts += int(timeout)
        returns.append(total_reward)

        print(
            f"Episode {episode + 1}/{args.episodes}: "
            f"steps={steps}, "
            f"reward={total_reward:.1f}, "
            f"success={success}, "
            f"collision={collision}, "
            f"timeout={timeout}",
            flush=True,
        )

    print(
        f"\nSuccesses: {successes}/{args.episodes}; "
        f"collisions: {collisions}/{args.episodes}; "
        f"timeouts: {timeouts}/{args.episodes}; "
        f"mean return: {np.mean(returns):.1f}",
        flush=True,
    )

    print(
        "Success and collision can both be true in the same episode.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train or test TQC on UGVEnv"
    )

    parser.add_argument(
        "--mode",
        choices=("train", "test"),
        default="train",
    )

    parser.add_argument(
        "--out",
        default="runs/ugv_tqc_random_world",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=18_000,
        help="Training steps to add in this run",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--model",
        help="Optional TQC model path for testing",
    )

    args = parser.parse_args()

    if args.steps <= 0 or args.episodes <= 0 or args.max_steps <= 0:
        parser.error(
            "--steps, --episodes and --max-steps must be positive"
        )

    if args.resume and args.mode != "train":
        parser.error("--resume can only be used in training mode")

    if args.model and args.mode != "test":
        parser.error(
            "--model is for testing; use --resume to continue training"
        )

    folder = Path(args.out)
    resume_pair = None

    model_path = (
        Path(args.model)
        if args.model
        else folder / "final.zip"
    )

    if args.mode == "train":
        if args.resume:
            try:
                resume_pair = latest_pair(folder)

            except FileNotFoundError as exc:
                parser.error(str(exc))

        elif folder.exists() and any(folder.iterdir()):
            parser.error(
                "Output folder is not empty. "
                "Use --resume or a different --out folder."
            )

        folder.mkdir(parents=True, exist_ok=True)

    elif not model_path.is_file():
        parser.error(f"TQC model not found: {model_path}")

    rclpy.init()

    bridge = None
    env = None

    try:
        bridge = UGVBridge()

        env = UGVEnv(
            bridge,
            max_steps=args.max_steps,
        )

        print(
            f"Episode step limit: {args.max_steps}",
            flush=True,
        )

        print(
            f"Observation: {env.observation_space}; "
            f"action: {env.action_space}",
            flush=True,
        )

        if args.mode == "train":
            monitor_path = folder / "train.monitor.csv"

            env = Monitor(
                env,
                filename=str(monitor_path),
                info_keywords=("success", "collision"),
                override_existing=not (
                    args.resume and monitor_path.exists()
                ),
            )

            train(env, folder, args, resume_pair)

        else:
            evaluate(env, model_path, args)

    except KeyboardInterrupt:
        print("\nStopping test.", flush=True)

    finally:
        if env is not None:
            env.close()

        if bridge is not None:
            bridge.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()