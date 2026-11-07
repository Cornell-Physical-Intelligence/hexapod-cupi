"""Surrogate closed loop: the waypoint follower drives c48_long from the surrogate's true pose.

Usage: closed_loop.py <checkpoint.pt> <output dir>
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

import locomotion.omniwalk.task as task_module
import locomotion.rl.ppo as algorithms
from contracts.pose import Pose2D
from locomotion.evaluation.checkpoint import configuration
from locomotion.evaluation.requirements import planar_pose
from locomotion.surrogate.env import (
    DEFAULT_GEOMETRY,
    DEFAULT_MODEL,
    DEFAULT_STANCE,
    SurrogateConfig,
    SurrogateEnv,
)
from locomotion.surrogate.evaluate import DEFAULT_TASK, DeterministicPolicy
from locomotion.surrogate.train import build_wrapper, resolve, wrapper_options
from navigation.waypoint import STAND, WaypointFollower

DT = 0.02
SETTLE_S = 2.0
ROUTES = {
    "forward_1m": [(1.0, 0.0)],
    "diagonal_1m": [(0.7071, 0.7071)],
    "left_1m": [(0.0, 1.0)],
    "backward_1m": [(-1.0, 0.0)],
    "L_1m_1m": [(1.0, 0.0), (1.0, 1.0)],
}


def heading_of(root):
    """Forward heading of the body: anatomical forward is body -Y, so yaw - pi/2."""
    x, y, yaw = planar_pose(root.numpy())[0]
    return x, y, yaw - math.pi / 2


def main():
    checkpoint_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    record = json.loads(checkpoint_path.with_suffix(".json").read_text())
    identity = record["identity"]
    config = configuration(identity["ppo_config"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    seed = int(identity.get("seed", 20260917))
    options = wrapper_options(config)
    results = {}
    for name, goals in ROUTES.items():
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = SurrogateConfig(num_envs=1, seed=seed, episode_seconds=120.0)
        env = SurrogateEnv(
            cfg, None, DEFAULT_MODEL, DEFAULT_GEOMETRY, out / "native" / name,
            reference_metadata=json.loads(DEFAULT_STANCE.read_text()), threads=0,
        )
        task = resolve(DEFAULT_TASK)(env, task_module.TaskConfig(seed=seed), None)
        wrapped = build_wrapper(resolve("locomotion.rl.wrapper:OmniVecEnv"), task, options)
        observations = wrapped.get_observations()
        actor, _ = algorithms.load_actor(checkpoint, config, observations, "cpu")
        policy = DeterministicPolicy(actor, env, options)
        env.commands.zero_()
        env.reset()
        x0, y0, h0 = heading_of(env.current["root"])
        # Goals are given relative to the start pose, in the start's forward frame.
        c, s = math.cos(h0), math.sin(h0)
        world_goals = [(x0 + c * gx - s * gy, y0 + s * gx + c * gy) for gx, gy in goals]
        follower = WaypointFollower(world_goals)
        trace, stops, settle_until, t, started = [], [], None, 0.0, time.monotonic()
        changes, last_cmd, fell = 0, None, False
        for k in range(6000):
            x, y, h = heading_of(env.current["root"])
            cmd = follower.update(Pose2D(x_m=x, y_m=y, heading_rad=h), t)
            vec = [cmd.vx_mps, cmd.vy_mps, cmd.wz_rad_s]
            if vec != last_cmd:
                changes += 1
                last_cmd = vec
            if cmd == STAND and follower._index >= len(world_goals) and settle_until is None:
                settle_until = t + SETTLE_S
            env.commands.copy_(torch.tensor([vec], dtype=torch.float32))
            obs = env.observations(env.current)["obs"]
            action = policy(obs)
            result = env.step(action)
            t += DT
            trace.append((t, x, y, h, *vec))
            if bool(result["terminated"][0]):
                fell = True
                break
            if settle_until is not None and t >= settle_until:
                break
        x, y, h = heading_of(env.current["root"])
        gx, gy = world_goals[-1]
        final_distance = math.hypot(x - gx, y - gy)
        # Distance to each intermediate goal at the control it was retired: from the trace.
        results[name] = {
            "goals_world": world_goals,
            "controls": len(trace),
            "seconds": round(len(trace) * DT, 2),
            "command_changes": changes,
            "fell": fell,
            "truncated": len(trace) >= 6000,
            "final_distance_to_last_goal_m": final_distance,
            "final_heading_error_rad": None,
            "wall_s": round(time.monotonic() - started, 1),
        }
        np.save(out / f"{name}_trace.npy", np.array(trace))
        print(name, json.dumps({k: v for k, v in results[name].items() if k != "goals_world"}), flush=True)
    (out / "closed_loop.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
