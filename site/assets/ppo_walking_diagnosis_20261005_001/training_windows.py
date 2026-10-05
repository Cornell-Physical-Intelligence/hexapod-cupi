"""Measure training-window speed along the 0.05 m/s command across reward versions and learner paths.

The script reads the retained metric rows of five native PPO attempts with seed 20260917 and 128 robots.
It reports the mean speed over updates 701 to 800 for a 2 by 2 of reward version against learner path,
and ten 200-update windows of the reward-v4 forward attempt. It starts no native process."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


ROOT = "/srv/cupi/hexapod/runs/james"
UNLAUNCHED, REPAIR = "flat_pilot_20260930_003/ppo_mlp", "flat_pilot_20261002_001/access_repair.json"
ATTEMPTS = {
    "reward_v2_stock": "flat_pilot_20261002_001/ppo_mlp",
    "reward_v1_stock": "ppo_reward_v1_control_20261004_001",
    "reward_v1_corrected": "ppo_v1_corrected_path_20261004_001",
    "reward_v2_corrected": "ppo_v2_corrected_path_20261004_001",
    "reward_v4_forward": "ppo_v4_forward_20261004_001",
}
METRICS, CONFIG, TASK = "/run/standing/metrics.jsonl", "/run/standing/ppo_config.json", "source/locomotion/task.py"
SHA256 = {
    UNLAUNCHED + "/PACK.json": "fb16fce9c3b3d41934510866b4ca32274f784c9cf81c742a432f2c69d9c3155a",
    REPAIR: "32f33f77c106bd31ae6d4313ac89dab2d469c29adf287055de47f32e8338aced",
    "flat_pilot_20261002_001/ppo_mlp/PACK.json": "9f0160071b769b076aa2d9cb6b832d98a550efde1042193f9195b530c5b04388",
    "flat_pilot_20261002_001/ppo_mlp" + CONFIG: "7eca7a6673b51fe3eeb8d9d3816944549d52cb836b09c62358f253e793d6d2c2",
    "flat_pilot_20261002_001/ppo_mlp" + METRICS: "c672fa6aea911a665d001ef2d8bdf1e59e71c78019323a0506c3bdf94a76abfa",
    "ppo_reward_v1_control_20261004_001/PACK.json": "f2ba953cc03e378b010edf62b76c0d57cb867b96544ad4cc0c52e34f3913b2bd",
    "ppo_reward_v1_control_20261004_001/preparation.json": "983b7d336b72d29877e09e64bd92e22d93c658e5651d4e71557cd1201a674cdf",
    "ppo_reward_v1_control_20261004_001" + CONFIG: "7eca7a6673b51fe3eeb8d9d3816944549d52cb836b09c62358f253e793d6d2c2",
    "ppo_reward_v1_control_20261004_001" + METRICS: "07104299477bb75277684b5acbc11ad76fee2b22ea5e15bd255fd94e264197d9",
    "ppo_v1_corrected_path_20261004_001/PACK.json": "5a106a0a89762b1780e805e4f770781b2f2ddc809d9bd90a50a8a1b57d40495c",
    "ppo_v1_corrected_path_20261004_001/preparation.json": "f3f4e81b812b4003db8667842e0a0104b6c8cf6f3307c26fe8eb1c51e713de7d",
    "ppo_v1_corrected_path_20261004_001" + CONFIG: "b8ea0d791e3c27b7307ec35d6d0eb5fd38a2282ba2d5e3c7ce9e047a0843163e",
    "ppo_v1_corrected_path_20261004_001" + METRICS: "b7dd7e039c5542c590252632514a97e1a6cfbae6928e55ba01554bcbfe561589",
    "ppo_v2_corrected_path_20261004_001/PACK.json": "c671fb43e783c19396b2fcdb845f3e1d75dce96a8ab9d96160bacffa1343a65b",
    "ppo_v2_corrected_path_20261004_001/preparation.json": "ea8736684cdddc914f61163add32812661fa4994dc4867c179e29b8b98787231",
    "ppo_v2_corrected_path_20261004_001" + CONFIG: "b8ea0d791e3c27b7307ec35d6d0eb5fd38a2282ba2d5e3c7ce9e047a0843163e",
    "ppo_v2_corrected_path_20261004_001" + METRICS: "adba82c70a70ac66491f4a8ed8deb012ee3596af40ea392c8f862cbf008adb0e",
    "ppo_v4_forward_20261004_001/PACK.json": "fc6b54f9c507b87a0499684c36d7a5e5e1e062047fdb91df55178e67e8d59fa8",
    "ppo_v4_forward_20261004_001/preparation.json": "96debb3fe9925069e1237c446efc12a61025ec619d9eec7f46be872276b592f6",
    "ppo_v4_forward_20261004_001" + CONFIG: "c425bcc9ee5ed1b7e70e460f56a66672a0fd47787940eaea00793808c6aa9161",
    "ppo_v4_forward_20261004_001" + METRICS: "5ea4ebcb38324c8d269999bd68401654e86787709d9a793423aed799ec956e96",
}
OPTIONS = ("learner", "networks", "action_mean", "observation_normalization", "observation_scaling", "command_segments",
           "learning_rate_max", "action_std_final", "gait_clock", "episode_seconds", "allocation_profile")
SPEED = "moving_signed_command_direction_speed_mps"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    """Average the values that are present and return None for an empty set."""
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def classes(rows, name, field):
    return [row["task"]["interval_metrics"]["command_classes"][name][field] for row in rows]


def near_bound(rows, part):
    """Average the near-bound share of the policy mean over the six joints of one leg segment."""
    names = [name for name in rows[0]["actions"]["per_joint"] if "_" + part + "_" in name]
    assert len(names) == 6
    return mean([row["actions"]["per_joint"][name]["mean_near_bound_fraction"] for row in rows for name in names])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    output = args.output or args.workspace / "training_windows.json"
    if output.exists():
        raise FileExistsError("Use a fresh output path")
    inputs, listing = args.workspace / "inputs", args.workspace / "inputs/unlaunched_listing.txt"
    missing = [name for name in SHA256 if not (inputs / name).exists()]
    if args.fetch and missing:
        inputs.mkdir(parents=True, exist_ok=True)
        command = "tar -C " + shlex.quote(ROOT) + " -cf - " + " ".join(map(shlex.quote, missing))
        process = subprocess.Popen(["ssh", "spark", command], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            archive.extractall(inputs, filter="data")
        assert process.wait() == 0
    if args.fetch and not listing.exists():
        with listing.open("xb") as stream:
            subprocess.run(["ssh", "spark", "ls -1 " + shlex.quote(ROOT + "/" + UNLAUNCHED)], stdout=stream, check=True)
    for name, digest in SHA256.items():
        assert sha(inputs / name) == digest, name
    read = lambda name: json.loads((inputs / name).read_text())
    # The earlier pack holds the same reward-v2 request and no run directory.
    other, chosen, repair = read(UNLAUNCHED + "/PACK.json"), read(ATTEMPTS["reward_v2_stock"] + "/PACK.json"), read(REPAIR)
    shared = ("mode", "learner", "networks", "reward_version", "updates", "seed", "source_freeze_sha256")
    assert all(other[name] == chosen[name] for name in shared) and chosen["updates"] == 2000
    assert listing.read_text().split() == ["PACK.json", "binding.json", "source"] and repair["source_bytes_unchanged"]
    assert repair["failed_attempt"] == ROOT + "/flat_pilot_20260930_003/tripod_reference"
    assert "ppo_mlp" in repair["fresh_attempts"]
    records, history, cells = {}, {}, {}
    for key, attempt in ATTEMPTS.items():
        pack, config = read(attempt + "/PACK.json"), read(attempt + CONFIG)
        source = REPAIR if key == "reward_v2_stock" else attempt + "/preparation.json"
        rows = [json.loads(line) for line in (inputs / (attempt + METRICS)).read_text().splitlines()]
        controls = 128 * config["num_steps_per_env"]
        assert [row["update"] for row in rows] == list(range(1, pack["updates"] + 1))
        assert all(row["task"]["interval_metrics"]["environment_controls"] == controls for row in rows)
        assert rows[-1]["transitions"] == controls * len(rows) and pack["files"][TASK] == chosen["files"][TASK]
        assert pack["seed"] == config["seed"] == 20260917 and pack["learner"] == "ppo"
        (label,) = {row["task"]["reward_version"] for row in rows}
        actor, algorithm = config["actor"], config["algorithm"]
        stock = (actor["obs_normalization"], actor["distribution_cfg"]["class_name"],
                 algorithm["class_name"]) == (True, "GaussianDistribution", "PPO")
        assert stock == key.endswith("_stock")
        history[key] = rows
        records[key] = {
            "attempt": attempt, "metrics_sha256": SHA256[attempt + METRICS], "metrics_rows": len(rows),
            "seed": pack["seed"], "robots": 128, "updates": pack["updates"], "reward_version": pack["reward_version"],
            "reward_label_in_metric_rows": label, "learner_path": "stock" if stock else "corrected",
            "source_commit": read(source)["source_commit"], "source_commit_record": source,
            "source_freeze_sha256": pack["source_freeze_sha256"], "ppo_config_sha256": SHA256[attempt + CONFIG],
            "command_classes_drawn": [name for name in rows[0]["task"]["interval_metrics"]["command_classes"]
                                      if sum(classes(rows, name, "environment_controls"))],
            "pack_options": {name: pack[name] for name in OPTIONS if name in pack},
            "ppo_config": {"observation_normalization": actor["obs_normalization"], "algorithm": algorithm["class_name"],
                           "action_distribution": actor["distribution_cfg"]["class_name"],
                           "learning_rate": algorithm["learning_rate"], "learning_rate_max": algorithm.get("learning_rate_max"),
                           "exploration": config.get("exploration"), "environment_wrapper": config.get("environment_wrapper")},
        }
    assert records["reward_v2_stock"]["ppo_config_sha256"] == records["reward_v1_stock"]["ppo_config_sha256"]
    assert records["reward_v1_corrected"]["ppo_config_sha256"] == records["reward_v2_corrected"]["ppo_config_sha256"]
    for key in list(ATTEMPTS)[:4]:
        selected = history[key][700:800]
        speeds = classes(selected, "linear_0.05", SPEED)
        cells[key] = {"attempt": ATTEMPTS[key], "reward_version": records[key]["reward_version"],
                      "learner_path": records[key]["learner_path"], "linear_0.05_speed_mps": mean(speeds),
                      "updates_with_speed": sum(value is not None for value in speeds),
                      "linear_0.05_controls": sum(classes(selected, "linear_0.05", "environment_controls"))}
    rows, windows = history["reward_v4_forward"], []
    assert all(row["actions"]["mean_near_bound_threshold"] == .95 for row in rows)
    assert records["reward_v4_forward"]["command_classes_drawn"] == ["zero", "linear_0.05"]
    for first in range(1, 2001, 200):
        selected = rows[first - 1:first + 199]
        intervals = [row["task"]["interval_metrics"] for row in selected]
        updates = [row["policy_update"] for row in selected]
        windows.append({
            "first_update": first, "last_update": first + 199,
            "linear_0.05_speed_mps": mean(classes(selected, "linear_0.05", SPEED)),
            "linear_0.05_task_reward_mean": mean(classes(selected, "linear_0.05", "task_reward_mean")),
            "zero_task_reward_mean": mean(classes(selected, "zero", "task_reward_mean")),
            "mean_near_bound_fraction": {part: near_bound(selected, part) for part in ("coxa", "femur", "tibia")},
            "terminations": sum(row["terminations"] for row in intervals),
            "termination_reason_rows": {reason: sum(row["termination_reasons"]["rows"][reason] for row in intervals)
                                        for reason in intervals[0]["termination_reasons"]["rows"]},
            "learning_rate_mean": mean([row["learning_rate"] for row in selected]),
            "after_update_kl_mean": mean([update["after"]["kl_mean"] for update in updates]),
            "value_explained_variance_mean": mean([update["value"]["explained_variance"] for update in updates]),
        })
    speed_400 = [mean(classes(rows[first:first + 400], "linear_0.05", SPEED)) for first in range(0, 2000, 400)]
    report = {
        "schema": "ppo_training_windows_v1", "analysis_sha256": sha(__file__), "remote_root": ROOT,
        "input_sha256": SHA256, "metric_source_sha256": {TASK: chosen["files"][TASK]},
        "method": {
            "speed": "Read task.interval_metrics.command_classes['linear_0.05']." + SPEED + " from each update. "
                     "Average the updates of a window with equal weight and skip null values.",
            "class_reward": "Average command_classes[class].task_reward_mean over the updates and skip null values.",
            "near_bound": "actions.per_joint[joint].mean_near_bound_fraction gives the share of collected rollout rows "
                          "with an absolute policy mean of 0.95 or more. Average six joints of one segment, then the updates.",
            "terminations": "Sum interval terminations and termination_reasons.rows over the updates of a window.",
            "robots": "Each metric row holds 128 times num_steps_per_env environment controls."},
        "scope": "The record measures training rollouts with exploration noise, one seed per cell. It holds no evaluation "
                 "probe and establishes no Stage 2 result. The stock reward-v2 cell uses updates 701 to 800 of a 2000-update "
                 "attempt from an earlier source commit; the other three cells ran 800 updates. The reward-v4 forward "
                 "attempt draws two command classes, zero and linear_0.05; each attempt record lists its drawn classes. "
                 "One task.py computes the speed in all five attempts.",
        "identification": {
            "selected": ATTEMPTS["reward_v2_stock"], "rejected": UNLAUNCHED,
            "basis": "Both packs request PPO with MLP networks, reward version 2, 2000 updates, seed 20260917 and one "
                     "source freeze. The rejected directory lists PACK.json, binding.json and source, and holds no run. "
                     "access_repair.json names flat_pilot_20260930_003/tripod_reference as the failed attempt and "
                     "ppo_mlp as a fresh attempt. The selected attempt holds 2000 metric rows, and its ppo_config.json "
                     "matches the reward-v1 stock control byte for byte: empirical observation normalization, "
                     "GaussianDistribution and stock PPO.",
            "layout": "The selected attempt keeps its run under ppo_mlp/run/standing."},
        "attempts": records, "two_by_two": {"first_update": 701, "last_update": 800, "cells": cells},
        "reward_v4_forward_windows": windows, "reward_v4_forward_speed_400_update_windows_mps": speed_400,
        "native_started_by_analysis": False}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"two_by_two_mm_s": {key: round(cell["linear_0.05_speed_mps"] * 1000, 2) for key, cell in cells.items()},
                      "v4_speed_mm_s": [round(row["linear_0.05_speed_mps"] * 1000, 1) for row in windows]}))


if __name__ == "__main__":
    main()
