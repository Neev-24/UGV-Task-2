#!/usr/bin/env python3
"""Train or evaluate TD3 with the existing ROS 2 UGVEnv."""
import argparse
import re
from pathlib import Path

import numpy as np
import rclpy
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from ugv_rl.ugv_env import UGVBridge, UGVEnv


def latest_pair(folder):
    """Find the newest complete model + replay-buffer pair."""
    pairs = []
    final_model = folder / "final.zip"
    final_replay = folder / "replay_buffer.pkl"
    if final_model.is_file() and final_replay.is_file():
        pairs.append((final_model, final_replay))
    for model_path in (folder / "checkpoints").glob("td3_*_steps.zip"):
        match = re.fullmatch(r"td3_(\d+)_steps\.zip", model_path.name)
        if match:
            replay_path = model_path.parent / f"td3_replay_buffer_{match.group(1)}_steps.pkl"
            if replay_path.is_file():
                pairs.append((model_path, replay_path))
    if not pairs:
        raise FileNotFoundError(f"No matching model and replay buffer in {folder}")
    return max(pairs, key=lambda pair: min(p.stat().st_mtime_ns for p in pair))


def train(env, folder, args, resume_pair):
    checkpoint_dir = folder / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if resume_pair:
        model_path, replay_path = resume_pair
        print(f"Resuming model: {model_path}\nReplay buffer: {replay_path}", flush=True)
        model = TD3.load(str(model_path), env=env, device="auto")
        model.load_replay_buffer(str(replay_path))
    else:
        n_actions = env.action_space.shape[0]
        noise = NormalActionNoise(
            mean=np.zeros(n_actions, dtype=np.float32),
            sigma=0.1 * np.ones(n_actions, dtype=np.float32),
        )
        model = TD3(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            action_noise=noise,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            policy_kwargs={"net_arch": [128, 128]},
            seed=args.seed,
            verbose=1,
            device="auto",
        )
    callback = CheckpointCallback(
        save_freq=1_000,
        save_path=str(checkpoint_dir),
        name_prefix="td3",
        save_replay_buffer=True,
    )
    print(f"Starting {args.steps} additional environment steps.", flush=True)
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=callback,
            reset_num_timesteps=not args.resume,
            log_interval=1,
        )
    except KeyboardInterrupt:
        print("Interrupted; saving the current model and replay buffer.", flush=True)
    model.save(str(folder / "final.zip"))
    model.save_replay_buffer(str(folder / "replay_buffer.pkl"))
    print(f"Saved: {folder / 'final.zip'} and {folder / 'replay_buffer.pkl'}", flush=True)


def evaluate(env, folder, args):
    model = None
    if args.policy == "td3":
        path = Path(args.model) if args.model else folder / "final.zip"
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")
        model = TD3.load(str(path), env=env, device="cpu")
        print(f"Loaded TD3 model: {path}", flush=True)
    else:
        print("Zero-action baseline: the environment's steering rules still run.", flush=True)
    successes = collisions = timeouts = 0
    returns = []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        ended = False
        total_reward = 0.0
        steps = 0
        while not ended:
            action = np.zeros(env.action_space.shape, dtype=np.float32) if model is None else model.predict(obs, deterministic=True)[0]
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
        print(f"Episode {episode + 1}/{args.episodes}: steps={steps}, reward={total_reward:.1f}, "
              f"success={success}, collision={collision}, timeout={timeout}", flush=True)
    print(f"\nSuccesses: {successes}/{args.episodes}; collisions: {collisions}/{args.episodes}; "
          f"timeouts: {timeouts}/{args.episodes}; mean return: {np.mean(returns):.1f}", flush=True)
    print("Success and collision can both be true in the same episode.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train or test TD3 on the existing UGVEnv")
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--out", default="runs/ugv_td3_v1")
    parser.add_argument("--steps", type=int, default=2_000, help="Additional training steps")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--policy", choices=("td3", "zero"), default="td3")
    parser.add_argument("--model", help="Optional TD3 model path for testing")
    args = parser.parse_args()
    if args.steps <= 0 or args.episodes <= 0:
        parser.error("--steps and --episodes must be positive")
    folder = Path(args.out)
    resume_pair = None
    if args.mode == "train":
        if args.resume:
            try:
                resume_pair = latest_pair(folder)
            except FileNotFoundError as exc:
                parser.error(str(exc))
        elif (folder / "final.zip").exists() or list((folder / "checkpoints").glob("td3_*_steps.zip")):
            parser.error("Run already exists. Use --resume or choose another --out directory.")
        folder.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    bridge = None
    env = None
    try:
        bridge = UGVBridge()
        env = UGVEnv(bridge)
        if args.mode == "train":
            monitor_path = folder / "train.monitor.csv"
            env = Monitor(env, filename=str(monitor_path),
                          info_keywords=("success", "collision"),
                          override_existing=not (args.resume and monitor_path.exists()))
            print(f"Observation: {env.observation_space}; action: {env.action_space}", flush=True)
            train(env, folder, args, resume_pair)
        else:
            evaluate(env, folder, args)
    finally:
        if env is not None:
            env.close()
        if bridge is not None:
            bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()