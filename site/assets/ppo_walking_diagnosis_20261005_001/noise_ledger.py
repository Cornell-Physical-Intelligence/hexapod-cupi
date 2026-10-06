"""Score one native fixed-mean noise probe under reward versions 1, 2, 3 and 4.

The probe (locomotion/noise_probe.py) held four poses under action standard deviations 0, 0.05,
0.10 and 0.15, with eight replicas per cell, for 600 controls and no optimizer. Its open-loop
means ignore the command, so this script scores each cell as if the probe had held each listed
command. It reports the mean reward per control after the settle window for each version, and
the mean of each version-4 component.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from locomotion import noise_probe, task, task_v2, task_v3, task_v4
from locomotion.env import inverse_rotate, navigation
from locomotion.env_config import JOINT_NAMES

REMOTE = "/srv/cupi/hexapod/runs/james/ppo_noise_probe_20261004_002/run/standing/probe"
INPUTS = {"telemetry.npz": "910cdefc24d28af9a14a6d6db456d0701891a819191dea4e2fea094b9715725e",
          "plan.json": "afc42132e21ef0dd22f4b25bd2e2812d4fc128bf062cedf1fa97cfa59c90e987",
          "summary.json": "0732ff83f01c960c52ecc41d06e8421a5dca2ed48b252922e456a46c51cf9c0b"}
ROBOT = {"robot/hexapod_mkii_updated_v1/stance.json": "830cb07c0fdb3d80af82476d8e6e88f25440cfd2e9255ac1e16327ac826d259c",
         "robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json": "7126935b35398d2a7f03271335a8a1b8f2059354af2b19d9474f22b8e73e4881"}
SOURCES = ("task.py", "task_v2.py", "task_v3.py", "task_v4.py", "env.py", "env_config.py", "noise_probe.py")
SETTLE = 160  # 100 settle controls plus one 60-control stride window
ACTION_SCALE_RAD = .35
COMMANDS = {"forward_0.05": (.05, 0., 0.), "forward_0.025": (.025, 0., 0.), "yaw_0.2": (0., 0., .2), "zero": (0., 0., 0.)}
VERSIONS = ("v1", "v2", "v3", "v4")
HEADLINE = "forward_0.05"
SHARED = ("linear_velocity_nav", "angular_velocity_body", "root_pose_xyzw", "joint_velocity_rad_s",
          "joint_target_rad", "torque_square_sum_400hz", "requested_torque_abs_max_400hz",
          "other_body_force_max_400hz", "tibia_floor_force_world_n", "action")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(arrays, command, neutral, lower, upper, height):
    """Return each replica's mean reward per version, and each version-4 component, after the settle window."""
    controls, n = arrays["action"].shape[:2]
    commands = torch.tensor(command, dtype=torch.float32).expand(n, -1).clone()
    field = lambda name, t: torch.from_numpy(arrays[name][t])
    v4 = task_v4.REWARD_V4_CONFIG
    stride = task_v2.StrideWindow(n, task_v2.REWARD_V2_CONFIG.stride_controls, 3)
    window = task_v2.StrideWindow(n, v4.tracking_window_controls, 3)
    stride.restart(torch.arange(n), commands)
    window.restart(torch.arange(n), commands)
    totals = {}
    for t in range(1, controls):
        tel = {name: field(name, t) for name in SHARED}
        tel["command"] = commands
        fallen = field("terminated", t)
        previous_target, previous_rate = field("joint_target_rad", t - 1), field("joint_velocity_rad_s", t - 1)
        rewards = {"v1": task.measured_reward(tel, commands, previous_target, fallen, task.TaskConfig(), height)}
        # Versions 2 and 3 share their penalties. Version 2 tracks the 60-control mean velocity.
        memory = dict(tel, previous_action=field("action", t - 1), previous_joint_velocity_rad_s=previous_rate)
        tracked = stride.update(torch.cat((tel["linear_velocity_nav"][:, :2], tel["angular_velocity_body"][:, 2:]), -1), commands)
        rewards["v2"] = task_v2.measured_reward_v2(dict(memory, tracked_planar_velocity_nav=tracked[:, :2],
            tracked_yaw_rate_rad_s=tracked[:, 2]), commands, fallen, task_v2.REWARD_V2_CONFIG)
        rewards["v3"] = task_v2.measured_reward_v2(memory, commands, fallen, task_v3.REWARD_V3_CONFIG)
        # TrainingTaskV4.reward_telemetry builds the same version 4 memory.
        pose, q = tel["root_pose_xyzw"], field("joint_position_rad", t)
        displaced = window.update(task_v4.displacement_velocity(pose, field("root_pose_xyzw", t - 1)), commands)
        toe_step = (field("toe_xyz_world", t) - field("toe_xyz_world", t - 1)) / task_v2.CONTROL_DT_S
        toe_velocity = navigation(inverse_rotate(pose[:, None, 3:].expand(-1, 6, -1), toe_step))[..., :2]
        rewards["v4"] = task_v4.measured_reward_v4(dict(tel, tracked_velocity_nav=displaced,
            previous_joint_velocity_rad_s=previous_rate, previous_joint_target_rad=previous_target,
            computed_torque_nm=field("computed_torque_nm", t), toe_velocity_nav=toe_velocity,
            toe_xyz_nav=navigation(field("toe_xyz_body", t))[..., :2],
            # Array row t holds the control that started at episode step t.
            scheduled_swing=task_v4.scheduled_swing(torch.full((n,), t), v4),
            # The task counts the first control after a reset as command age 1, so row t has age t + 1.
            command_age_controls=torch.full((n,), t + 1),
            joint_limit_margin_rad=torch.minimum(q - lower, upper - q),
            executed_action=(tel["joint_target_rad"] - neutral) / ACTION_SCALE_RAD), commands, fallen, v4, height)
        if t < SETTLE:
            continue
        for version, (reward, parts) in rewards.items():
            entry = totals.setdefault(version, {})
            for key, value in {"reward": reward, **(parts if version == "v4" else {})}.items():
                entry[key] = entry.get(key, 0) + value.double()
    return {version: {key: (value / (controls - SETTLE)).numpy() for key, value in entry.items()}
            for version, entry in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    output = args.output or args.workspace / "noise_ledger.json"
    if output.exists():
        raise FileExistsError("Use a fresh output path")
    torch.set_num_threads(1)
    inputs = args.workspace / "inputs"
    missing = [name for name in INPUTS if not (inputs / name).exists()]
    if args.fetch and missing:
        inputs.mkdir(parents=True, exist_ok=True)
        command = "tar -C " + shlex.quote(REMOTE) + " -cf - " + " ".join(map(shlex.quote, missing))
        process = subprocess.Popen(["ssh", "spark", command], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            archive.extractall(inputs, filter="data")
        assert process.wait() == 0
    assert all(sha(inputs / name) == digest for name, digest in INPUTS.items())
    assert all(sha(ROOT / name) == digest for name, digest in ROBOT.items())
    git = lambda *words: subprocess.run(["git", "-C", str(ROOT), *words], capture_output=True, text=True)
    # The record names one commit, so the reward sources and the robot files must match it. The scorer reads no Markdown.
    assert git("diff", "--quiet", "HEAD", "--", "locomotion", "robot", ":(exclude)*.md").returncode == 0
    stance_path, model_path = (ROOT / name for name in ROBOT)
    stance = json.loads(stance_path.read_text())
    assert tuple(stance["joint_names"]) == JOINT_NAMES
    neutral = torch.tensor(stance["nominal_joint_position_rad"], dtype=torch.float32)
    limits = {joint["name"]: (joint["lower"], joint["upper"]) for joint in json.loads(model_path.read_text())["joints"]}
    lower, upper = (torch.tensor([limits[name][side] for name in JOINT_NAMES]) for side in (0, 1))
    plan = json.loads((inputs / "plan.json").read_text())
    summary = json.loads((inputs / "summary.json").read_text())
    with np.load(inputs / "telemetry.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    assert plan["replicas"] == noise_probe.plan(128) and arrays["action"].shape == (noise_probe.CONTROLS, 128, 18)
    # The scorer carries no reset memory, so it needs a probe without a fall.
    assert summary["terminations"] == 0 and not arrays["terminated"].any()
    # The method text states that the quiet settle window ends before the first scored control.
    assert SETTLE >= task_v4.REWARD_V4_CONFIG.quiet_grace_controls
    cells = {}
    for row in plan["replicas"]:
        cells.setdefault((row["pose"], row["standard_deviation"]), []).append(row["replica"])
    # The noise-free neutral cell holds the stance targets, so the stance file is the probe's neutral.
    assert (arrays["joint_target_rad"][:, cells[("neutral", 0.)]] == neutral.numpy()).all()
    scored = {name: score(arrays, command, neutral, lower, upper, stance["root_height_m"])
              for name, command in COMMANDS.items()}
    ledger = []
    for (pose, std), replicas in cells.items():
        rewards = {}
        for name in COMMANDS:
            mean = lambda version, key="reward": float(scored[name][version][key][replicas].mean())
            rewards[name] = {**{version: mean(version) for version in VERSIONS},
                             "v4_components": {key: mean("v4", key) for key in scored[name]["v4"] if key != "reward"}}
        ledger.append({"pose": pose, "standard_deviation": std, "replicas": replicas, "rewards": rewards})
    cell = {(row["pose"], row["standard_deviation"]): row["rewards"][HEADLINE] for row in ledger}
    gap = lambda first, second: {v: cell[(first, .15)][v] - cell[(second, .15)][v] for v in VERSIONS}
    headline = {"command": HEADLINE,
        "neutral": {v: {"std_0.00": cell[("neutral", 0.)][v], "std_0.15": cell[("neutral", .15)][v]} for v in VERSIONS},
        "crouch_minus_neutral_std_0.15": gap("crouch", "neutral"),
        "beyond_bound_minus_femur_raised_std_0.15": gap("beyond_bound", "femur_raised"),
        "v4_action_limit_std_0.00": {pose: cell[(pose, 0.)]["v4_components"]["action_limit"] for pose in noise_probe.POSES}}
    receipt = {
        "schema": "hexapod_noise_probe_reward_ledger_v2", "analysis_sha256": sha(__file__),
        "source_commit": git("rev-parse", "HEAD").stdout.strip(),
        "source_sha256": {"locomotion/" + name: sha(ROOT / "locomotion" / name) for name in SOURCES},
        "robot_sha256": ROBOT,
        "probe": {"remote_directory": REMOTE, "sha256": INPUTS, "seed": plan["seed"], "controls": plan["controls"],
                  "replicas_per_cell": plan["replicas_per_cell"], "terminations": summary["terminations"],
                  "poses": {name: value["leg_action_mean"] for name, value in plan["poses"].items()}},
        "method": "Replay the recorded telemetry through measured_reward (version 1), measured_reward_v2 with the "
            "default stride kernel (version 2) and with task_v3.REWARD_V3_CONFIG (version 3), and measured_reward_v4 "
            "with the default RewardV4Config (version 4). Rebuild each task's memory from the previous recorded "
            "control. Version 4 reads executed_action = (joint_target_rad - neutral) / 0.35 with the stance file's "
            "nominal joint positions as neutral. Array row t holds the control that started at episode step t, "
            "and the contact schedule reads that step. Version 4 reads command_age_controls = t + 1 at row t, the "
            "count that TrainingTaskV4 keeps for one command held since the reset. That age exceeds "
            "quiet_grace_controls at each scored control, so under the zero command the quiet joint-rate and "
            "target-motion terms charge each scored control. Hold each listed command for all 600 controls, average "
            "each replica over controls 160 to 599, then average the eight replicas of a cell.",
        "scope": "The probe held a zero command and fixed action means, and it ran no policy. A scored command changes the "
            "reward and leaves the recorded motion unchanged, so these values compare fixed poses under noise. "
            "They do not measure a trained policy, and one seed gives eight replicas per cell.",
        "native_started_by_analysis": False,
        "settle_controls": SETTLE, "scored_controls": noise_probe.CONTROLS - SETTLE,
        "nominal_height_m": stance["root_height_m"], "commands": COMMANDS,
        "v4_config": asdict(task_v4.REWARD_V4_CONFIG), "headline": headline, "cells": ledger,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=1, allow_nan=False) + "\n")
    print(json.dumps(headline, indent=1))


if __name__ == "__main__":
    main()
