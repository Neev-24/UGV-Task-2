#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import rclpy

from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import PointStamped

from ugv_rl.ugv_env import UGVBridge, UGVEnv


def valid_goal(x, y, obstacles):
    """Reject points outside the arena or inside an active pillar."""
    if not math.isfinite(x) or not math.isfinite(y):
        return False

    if not (-1.9 < x < 4.9 and -3.4 < y < 3.4):
        return False

    return all(
        math.hypot(x - ox, y - oy) > 0.20
        for ox, oy in obstacles
    )


def move_goal_marker(bridge, x, y):
    """Move the visible goal marker in Gazebo."""

    client = bridge.set_state_client

    if not client.wait_for_service(timeout_sec=2.0):
        raise RuntimeError(
            "/gazebo/set_entity_state service is unavailable."
        )

    request = SetEntityState.Request()

    request.state.name = "goal_marker"
    request.state.reference_frame = "world"

    request.state.pose.position.x = float(x)
    request.state.pose.position.y = float(y)
    request.state.pose.position.z = 0.0
    request.state.pose.orientation.w = 1.0

    future = client.call_async(request)

    rclpy.spin_until_future_complete(
        bridge,
        future,
        timeout_sec=2.0,
    )

    if (
        not future.done()
        or future.result() is None
        or not future.result().success
    ):
        raise RuntimeError(
            "Gazebo could not move the goal_marker model."
        )


class ClickGoalRunner:

    def __init__(self, bridge, env, model):

        self.bridge = bridge
        self.env = env
        self.model = model

        self.pending_goal = None
        self.active = False
        self.obs = None

        self.subscription = bridge.create_subscription(
            PointStamped,
            "/ugv/click_goal",
            self.on_click,
            10,
        )

    def on_click(self, msg):

        if msg.header.frame_id != "world":
            print(
                f"Ignoring goal in frame {msg.header.frame_id!r}; "
                "expected world.",
                flush=True,
            )
            return

        x = float(msg.point.x)
        y = float(msg.point.y)

        if not valid_goal(x, y, self.env.obstacles):
            print(
                f"Ignoring click at ({x:.2f}, {y:.2f}): "
                "outside the arena or inside a pillar.",
                flush=True,
            )
            return

        self.pending_goal = (x, y)

    def apply_pending_goal(self):

        if self.pending_goal is None:
            return

        x, y = self.pending_goal
        self.pending_goal = None
        self.active = False

        self.bridge.move(0.0, 0.0)

        self.env.goal = (x, y)
        self.env.steps = 0

        move_goal_marker(self.bridge, x, y)
        self.bridge.wait_for_fresh_data()

        if self.bridge.model_pose is None:
            raise RuntimeError(
                "No world pose for 'ugv' on /gazebo/model_states."
            )

        self.obs, distance, _ = self.env._get_obs()

        self.env.prev_dist = distance
        self.env.prev_angle = abs(
            float(self.obs[25]) * math.pi
        )

        print(
            f"Goal set: x={x:.2f}, y={y:.2f}, "
            f"distance={distance:.2f} m",
            flush=True,
        )

        if self.model is None:
            print(
                "Preview mode: marker updated; UGV will not move.",
                flush=True,
            )

        elif distance < 0.50:
            print(
                "UGV is already near this goal. Click another point.",
                flush=True,
            )

        else:
            self.active = True

    def run(self):

        self.obs, _ = self.env.reset()

        if self.bridge.model_pose is None:
            raise RuntimeError(
                "No world pose for 'ugv' on /gazebo/model_states."
            )

        self.bridge.move(0.0, 0.0)

        print(
            "Ready. Left-click a clear spot in Gazebo.",
            flush=True,
        )

        while rclpy.ok():

            rclpy.spin_once(
                self.bridge,
                timeout_sec=0.01 if self.active else 0.2,
            )

            self.apply_pending_goal()

            if not self.active:
                continue

            action, _ = self.model.predict(
                self.obs,
                deterministic=True,
            )

            self.obs, _, terminated, truncated, info = self.env.step(
                action
            )

            if terminated or truncated:

                self.active = False

                print(
                    f"Finished: "
                    f"success={info.get('success', False)}, "
                    f"collision={info.get('collision', False)}, "
                    f"timeout={bool(truncated and not terminated)}. "
                    "Click another goal.",
                    flush=True,
                )


def main():

    parser = argparse.ArgumentParser(
        description="Test click-to-goal in Gazebo"
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Move only the goal marker",
    )

    parser.add_argument(
        "--model",
        default="runs/ugv_td3_v1/final.zip",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")

    if not args.preview and not Path(args.model).is_file():
        parser.error(
            f"Model not found: {args.model}. "
            "Use --preview to test clicks first."
        )

    rclpy.init()

    bridge = None
    env = None

    try:

        bridge = UGVBridge()
        env = UGVEnv(bridge, max_steps=args.max_steps)

        model = None

        if not args.preview:
            from stable_baselines3 import TD3

            model = TD3.load(
                args.model,
                env=env,
                device="cpu",
            )

            print(
                f"Loaded TD3 model: {args.model}",
                flush=True,
            )

        runner = ClickGoalRunner(bridge, env, model)
        runner.run()

    except KeyboardInterrupt:
        print("\nStopping click-to-goal test.", flush=True)

    finally:

        if env is not None:
            env.close()

        if bridge is not None:
            bridge.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()