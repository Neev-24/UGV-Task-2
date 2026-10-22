#!/usr/bin/env python3
import argparse
import os
import glob
import rclpy
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from ugv_rl.ugv_env import UGVBridge, UGVEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = f"runs/ugv_{args.algo}"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Initialize ROS 2 and Environment first so env and AlgoClass exist
    rclpy.init()
    bridge = UGVBridge()
    env = Monitor(UGVEnv(bridge), filename=os.path.join(out_dir, "monitor"))

    AlgoClass = PPO if args.algo == "ppo" else SAC

    # 2. Determine model path (check final.zip or latest checkpoint if resuming)
    model_path = os.path.join(out_dir, "final.zip")
    if args.resume and not os.path.exists(model_path):
        pattern = os.path.join(out_dir, f"ugv_{args.algo}_*_steps.zip")
        checkpoints = glob.glob(pattern)
        if checkpoints:
            # Sort by step number extracted from filename
            model_path = max(checkpoints, key=lambda p: int(p.split("_")[-2]))
            print(f"Resuming from latest checkpoint: {model_path}")

    try:
        if args.resume and os.path.exists(model_path):
            print(f"Resuming {args.algo.upper()} from {model_path}...")
            model = AlgoClass.load(model_path, env=env)
        else:
            print(f"Starting fresh {args.algo.upper()} training...")
            extra_kwargs = {"learning_starts": 300, "batch_size": 128} if args.algo == "sac" else {}
            model = AlgoClass(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                policy_kwargs={"net_arch": [128, 128]},
                verbose=1,
                **extra_kwargs,
            )

        checkpoint_cb = CheckpointCallback(
            save_freq=2000,
            save_path=out_dir,
            name_prefix=f"ugv_{args.algo}",
        )

        model.learn(
            total_timesteps=args.steps,
            callback=checkpoint_cb,
            reset_num_timesteps=not args.resume,
        )
        
        final_save_path = os.path.join(out_dir, "final")
        model.save(final_save_path)
        print(f"Saved trained model to {final_save_path}.zip")

    finally:
        env.close()
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()