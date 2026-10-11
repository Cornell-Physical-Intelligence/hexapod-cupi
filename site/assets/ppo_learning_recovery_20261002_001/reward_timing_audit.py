"""Compare reward timing on retained native traces without native computation."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


AUDIT_SHA256 = "24b31232c268375290e90722f8a9d57a6b8361c6aea6e072ccd3320e223b4ee0"
TRAINING_FREEZE_SHA256 = "0dfec66ede2d7498f9b0f3cc87f7015039368523d89a0c4c5b28ace0108101e4"
TRACKING = ("linear_tracking", "yaw_tracking")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, expected, verified):
    if path.is_symlink() or sha(path) != expected:
        raise ValueError("Input hash mismatch or symlink: " + str(path))
    verified[str(path.resolve())] = expected


def load_arrays(path):
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key] for key in bundle.files}


def tracking_means(trace, env, reward, mask):
    after = torch.as_tensor(trace["amp_state_after"], dtype=torch.float32)
    commands = torch.as_tensor(trace["command"], dtype=torch.float32)
    linear = env.navigation(after[..., 36:39])
    angular = after[..., 39:42]
    window = reward.StrideWindow(1, reward.REWARD_V2_CONFIG.stride_controls, 3)
    window.restart(torch.tensor([0]), commands[0])
    tracked = []
    for index in range(len(commands)):
        velocity = torch.cat((linear[index, :, :2], angular[index, :, 2:3]), -1)
        tracked.append(window.update(velocity, commands[index]))
    tracked = torch.stack(tracked)
    components = {
        "B_immediate": reward.tracking_components(
            linear[:, 0, :2], angular[:, 0, 2], commands[:, 0],
            replace(reward.REWARD_V2_CONFIG, tracking_kernel="scaled")),
        "C_stride_60": reward.tracking_components(
            tracked[:, 0, :2], tracked[:, 0, 2], commands[:, 0], reward.REWARD_V2_CONFIG),
    }
    return {kernel: {key: float(value[mask].double().mean()) for key, value in terms.items()}
            for kernel, terms in components.items()}


def one_step_signal(reward, ppo_config):
    command = torch.tensor([[.05, 0., 0.]], dtype=torch.float32)
    config = reward.REWARD_V2_CONFIG
    zero = torch.zeros(1, 3)
    base = reward.tracking_components(zero[:, :2], zero[:, 2], command, config)
    base_linear = float(base["linear_tracking"][0])
    rows = []
    for speed in (0., .001, .005, .01):
        window = reward.StrideWindow(1, config.stride_controls, 3)
        window.restart(torch.tensor([0]), command)
        for _ in range(config.stride_controls):
            window.update(zero, command)
        endpoint = torch.tensor([[speed, 0., 0.]], dtype=torch.float32)
        averaged = window.update(endpoint, command)
        immediate = float(reward.tracking_components(endpoint[:, :2], endpoint[:, 2], command, config)["linear_tracking"][0])
        stride = float(reward.tracking_components(averaged[:, :2], averaged[:, 2], command, config)["linear_tracking"][0])
        immediate_gain, stride_gain = immediate - base_linear, stride - base_linear
        rows.append({"endpoint_speed_mps": speed, "B_linear_reward": immediate,
                     "C_linear_reward": stride, "B_gain_from_stationary": immediate_gain,
                     "C_gain_from_stationary": stride_gain,
                     "B_gain_divided_by_C_gain": None if stride_gain == 0 else immediate_gain / stride_gain})
    gamma, lam = ppo_config["algorithm"]["gamma"], ppo_config["algorithm"]["lam"]
    return {"command": [.05, 0., 0.], "stationary_linear_reward": base_linear,
            "convention": "Fill the 60-control window with zero velocity at an unchanged 0.05 m/s command. Replace its oldest zero with one current endpoint velocity. Compare that first response with the immediate kernel; hold yaw at zero. This calculation varies tracking only and does not simulate an action or a physical response.",
            "rows": rows,
            "gae": {"gamma": gamma, "lambda": lam, "factor_after_60_controls": (gamma * lam) ** config.stride_controls,
                    "limit": "This factor describes the explicit GAE weighting. Value bootstrapping prevents interpreting it as the exact fraction of policy credit that survives."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True,
                        help="Retained bounded-action audit workspace, including native captures and frozen sources")
    parser.add_argument("--output", type=Path, required=True, help="Fresh JSON receipt path")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Use a fresh output path")
    base = args.base.resolve()
    audit_path = base / "reward_audit.json"
    verified = {}
    verify(audit_path, AUDIT_SHA256, verified)
    audit = json.loads(audit_path.read_text())
    verify(base / "reward_audit.py", audit["analysis_sha256"], verified)
    source, data = base / "reward_audit_source", base / "reward_audit_inputs"
    for name, digest in audit["source_files"].items():
        verify(source / "locomotion" / name, digest, verified)
    for name, digest in audit["input_metadata_sha256"].items():
        verify(data / name, digest, verified)
    for case, record in audit["results"].items():
        for name, digest in record["input_sha256"].items():
            verify(data / case / name, digest, verified)
    training_source = base / "training_comparison_inputs/bounded_mean/source"
    freeze_path = training_source / "FREEZE_SHA256.json"
    verify(freeze_path, TRAINING_FREEZE_SHA256, verified)
    freeze = json.loads(freeze_path.read_text())
    ppo_path = training_source / "locomotion/ppo.py"
    verify(ppo_path, freeze["locomotion/ppo.py"], verified)
    spec = importlib.util.spec_from_file_location("reward_timing_frozen_ppo", ppo_path)
    ppo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ppo)
    ppo_config = ppo.ppo_config(20260917, action_mean="tanh")
    sys.path.insert(0, str(source))
    env = importlib.import_module("locomotion.env")
    reward = importlib.import_module("locomotion.task_v2")
    assert Path(reward.__file__).resolve() == (source / "locomotion/task_v2.py").resolve()
    torch.set_num_threads(1)
    results = {}
    for case, original in audit["results"].items():
        trace = load_arrays(data / case / "control_trace.npz")
        mask = (trace["time_s"] > 2. + 1e-9) & (trace["time_s"] <= 20. + 1e-9)
        assert len(mask) == 1000 and int(mask.sum()) == 900
        assert not trace["reset"].any() and not trace["terminated"].any() and not trace["truncated"].any()
        means = tracking_means(trace, env, reward, mask)
        representations = {}
        for representation, previous in original["representations"].items():
            for term in TRACKING:
                assert abs(means["C_stride_60"][term] - previous["component_means"][term]) < 1e-8
            kernels = {}
            for kernel, tracking in means.items():
                components = {**previous["component_means"], **tracking}
                noncollision = sum(value for key, value in components.items() if key != "collisions")
                kernels[kernel] = {"component_means": components,
                    "reward_mean_reconstructed_collision": sum(components.values()),
                    "reward_mean_collision_agnostic_interval": [noncollision - reward.REWARD_V2_CONFIG.collision_weight, noncollision]}
            representations[representation] = kernels
        results[case] = {"remote_source": original["remote_source"],
            "input_sha256": original["input_sha256"], "native_identity": original["native_identity"],
            "admitted_dataset_entry": original["admitted_dataset_entry"], "screen_pass": original["screen_pass"],
            "requested_command": original["requested_command"], "window_seconds": original["window_seconds"],
            "scored_controls": 900, "history_controls": 1000,
            "first_scored_endpoint_s": float(trace["time_s"][mask][0]),
            "last_scored_endpoint_s": float(trace["time_s"][mask][-1]),
            "target_recurrence_max_abs_error_rad": original["target_recurrence_max_abs_error_rad"],
            "original_and_clamped_targets_identical": original["original_and_clamped_targets_identical"],
            "along_command_speed_mps": original["along_command_speed_mps"],
            "representations": representations}
    comparisons = {}
    for representation in ("original_raw", "clamped_counterfactual"):
        comparisons[representation] = {}
        for kernel in ("B_immediate", "C_stride_60"):
            failed = results["failed_ppo"]["representations"][representation][kernel]["reward_mean_collision_agnostic_interval"]
            comparisons[representation][kernel] = {}
            for case in ("tripod", "walk_phase0", "walk_phase05"):
                good = results[case]["representations"][representation][kernel]["reward_mean_collision_agnostic_interval"]
                comparisons[representation][kernel][case] = {
                    "walk_minus_failed_reward_interval": [good[0] - failed[1], good[1] - failed[0]],
                    "walking_ranks_above_failure_despite_unknown_collision": good[0] > failed[1]}
    receipt = {"schema": "hexapod_reward_timing_audit_v1", "analysis_sha256": sha(__file__),
        "source_commit": audit["source_commit"], "source_files": audit["source_files"],
        "parent_exact_capture_audit_sha256": AUDIT_SHA256,
        "verified_files": {str(Path(path).relative_to(base)): digest for path, digest in verified.items()},
        "learning_config": {"training_source_freeze_sha256": TRAINING_FREEZE_SHA256,
            "ppo_source_sha256": freeze["locomotion/ppo.py"], "ppo": ppo_config,
            "control_dt_s": reward.CONTROL_DT_S, "rollout_controls": ppo_config["num_steps_per_env"],
            "rollout_duration_s": ppo_config["num_steps_per_env"] * reward.CONTROL_DT_S,
            "stride_controls": reward.REWARD_V2_CONFIG.stride_controls,
            "stride_duration_s": reward.REWARD_V2_CONFIG.stride_controls * reward.CONTROL_DT_S},
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "device": "cpu"},
        "configurations": {"B_immediate": reward.reward_declaration(replace(reward.REWARD_V2_CONFIG, tracking_kernel="scaled")),
                           "C_stride_60": reward.reward_declaration(reward.REWARD_V2_CONFIG)},
        "method": {
            "endpoint": "Read completed-control native AMP features: body-origin linear velocity transformed to navigation axes and body angular velocity. Each endpoint follows eight 400 Hz physics substeps; controls run at 50 Hz.",
            "window": "Reconstruct C from control zero with the frozen 60-control StrideWindow. Compute both kernels before selecting endpoints 2 < t <= 20 s with a 1e-9 s tolerance. The 900 scored endpoints run from 2.02 to 20 s; no case resets or terminates.",
            "penalties": "Reuse exact component means from the hash-pinned parent audit. That audit reconstructs eight-substep torque sums, requested torque maxima, previous raw actions and joint rates from retained native records. This script verifies its source and all retained input hashes; it recomputes tracking only.",
            "collision": "Keep the parent audit's full collision penalty interval [−0.05, 0] per control. The native force matrix is absent, so named contact patches cannot prove its exact collision result.",
            "counterfactual": "Report original raw actions and the clamped-action counterfactual. The parent audit verifies identical target recurrence for both. The counterfactual changes the action-rate term on a fixed trajectory; it does not represent a newly trained policy.",
            "totals": "Sum float64 means of float32 component outputs. This changes the reduction order relative to a per-control float32 sum. C tracking means must match the parent exact audit within 1e-8.",
            "interpretation": "Both kernels rank the retained admitted walks above the failed policy at the 0.05 m/s command. B gives a larger immediate tracking response and penalizes within-stride oscillation. These fixed trajectories do not establish PPO convergence, human walking acceptance for a new policy, or the frozen pilot decision.",
            "learning_context": "The frozen actor receives five proprioceptive frames, the current command and the prior held-target offset divided by 0.35 rad. The environment computes that target feature after action clipping, joint limits and slew limiting. The critic adds current linear velocity. Neither input contains the 60-control reward state. PPO collects 24 controls per rollout. This comparison motivates a timing experiment; it does not prove the cause of failure.",
            "experiment_scope": "The next comparison selects observation normalization none and retains reward v2. Immediate tracking remains a separate future reward version if evidence warrants that experiment. This receipt starts no run and changes no production reward version."},
        "results": results, "comparisons": comparisons, "one_step_signal": one_step_signal(reward, ppo_config),
        "native_started": False, "production_files_changed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output),
                      "verified_file_count": len(verified), "comparisons": comparisons,
                      "one_step_signal": receipt["one_step_signal"]}, indent=2))


if __name__ == "__main__":
    main()
