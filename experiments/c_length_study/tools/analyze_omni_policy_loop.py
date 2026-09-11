#!/usr/bin/env python3
"""Reproduce the standing-loop and observation-noise analysis on saved evidence.

The CPU actor reconstruction uses the saved ELU MLP and mean/std normalizer.
Its match to recorded actions is reported; it is an analysis approximation,
not a replacement for an Isaac closed-loop noise comparison.
"""

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.c_length_study.tools.report_omni_diagnostics import direction_smoothness, trace_analysis


ROOT = Path(__file__).resolve().parents[3]


def analyze(folder, checkpoint, training_plan):
    report = json.loads((folder / "diagnostics.json").read_text())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != report["checkpoint_sha256"]:
        raise ValueError("Diagnostic checkpoint identity mismatch")
    data = np.load(folder / "diagnostic_trace.npz", allow_pickle=False)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved["actor_state_dict"]
    network = nn.Sequential(nn.Linear(315, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(),
                            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 18))
    network.load_state_dict({key.removeprefix("mlp."): value for key, value in state.items() if key.startswith("mlp.")})
    network.eval()
    defaults = training_plan["variants"]["f050_t060"]["stances"][0]["joint_positions_rad"]
    q0 = np.array([defaults[name] for name in data["joint_names"]])
    gravity = data["projected_gravity"][:, 0]
    gravity_nav = np.stack((-gravity[:, 1], gravity[:, 0], gravity[:, 2]), -1)
    frame = np.concatenate((data["gyro_navigation_rad_s"][:, 0]*.25, gravity_nav,
                            data["command"][:, 0]*[5, 5, 2.5], data["joint_position_rad"][:, 0]-q0,
                            data["joint_velocity_rad_s"][:, 0]*.05, data["action"][:, 0]), -1)
    indices = [index for index in range(105, len(frame))
               if np.all(data["age_s"][index-5:index+1, 0] >= 2)
               and not np.any(data["terminated"][index-5:index+1, 0])]
    if not indices:
        raise ValueError("No contiguous steady standing histories")
    observations = torch.tensor(np.stack([frame[index-5:index].ravel() for index in indices]), dtype=torch.float32)
    mean = state["obs_normalizer._mean"]
    std = state["obs_normalizer._std"]
    actual = torch.tensor(data["action"][indices, 0])
    with torch.inference_mode():
        reconstructed = network((observations-mean)/std).clamp(-1, 1)
        noise_scale = torch.zeros(63)
        noise_scale[:3] = .00375
        noise_scale[3:6] = .01
        noise_scale[9:27] = .005
        noise_scale[27:45] = .0025
        groups = {"all": noise_scale,
                  "gyro": torch.cat((noise_scale[:3], torch.zeros(60))),
                  "gravity": torch.cat((torch.zeros(3), noise_scale[3:6], torch.zeros(57))),
                  "joint_position": torch.cat((torch.zeros(9), noise_scale[9:27], torch.zeros(36))),
                  "joint_velocity": torch.cat((torch.zeros(27), noise_scale[27:45], torch.zeros(18)))}
        torch.manual_seed(157)
        sample = observations[::50].repeat_interleave(256, 0)
        contributions = {}
        for name, scale in groups.items():
            predictions = network((sample + torch.randn_like(sample)*scale.repeat(5)-mean)/std).clamp(-1, 1)
            predictions = predictions.reshape(-1, 256, 18)
            contributions[name] = float(predictions.std(1).square().mean().sqrt())
    target_step = np.diff(data["joint_target_rad"][100:, 0], axis=0)
    stand = report["scenarios"][0]["windows"]["post_settle"]
    return {"checkpoint_sha256": digest, "checkpoint_iteration": saved["iter"],
            "saved_distribution_parameter_mean": float(state["distribution.std_param"].mean()),
            "optimizer_learning_rate": saved["optimizer_state_dict"]["param_groups"][0]["lr"],
            "stand_metrics": {key: stand[key] for key in ("planar_error_mps", "yaw_error_rad_s", "torque_saturation_fraction", "positive_mechanical_power_w")},
            "stand_smoothness": direction_smoothness(folder / "diagnostic_trace.npz", report)[0],
            "stand_spectrum": trace_analysis(folder / "diagnostic_trace.npz", report),
            "normalized_action_at_clip_fraction": float((np.abs(data["action"][100:, 0]) >= .9999).mean()),
            "target_step_at_slew_limit_fraction": float((np.abs(target_step) >= .02999).mean()),
            "raw_reward_terms": stand["mean_raw_reward_terms"],
            "actor_reconstruction": {
                "recorded_action_rms": float(actual.square().mean().sqrt()),
                "recorded_action_temporal_std_rms": float(actual.std(0).square().mean().sqrt()),
                "noiseless_prediction_rmse": float((reconstructed-actual).square().mean().sqrt()),
                "prediction_recorded_correlation": float(torch.corrcoef(torch.stack((reconstructed.ravel(), actual.ravel())))[0, 1]),
                "noise_induced_action_std_rms": contributions,
                "limitation": "Saved-MLP CPU approximation on recorded histories; noise perturbations are not closed-loop physics trials."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "artifacts/omni_diagnostics_2026-09-09/repair_002")
    parser.add_argument("--baseline-checkpoint", type=Path, default=ROOT / "artifacts/omni_flat_2026-09-09/omni_flat_002/stage_002/policy.pt")
    parser.add_argument("--training-plan", type=Path, default=ROOT / "artifacts/omni_flat_2026-09-09/inputs/training_plan.json")
    args = parser.parse_args()
    plan = json.loads(args.training_plan.read_text())
    results = {"baseline": analyze(args.directory / "baseline", args.baseline_checkpoint, plan),
               "pilot": analyze(args.directory / "stage_000", args.directory / "stage_000/policy.pt", plan)}
    (args.directory / "policy_loop_analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({name: {"clip_fraction": row["normalized_action_at_clip_fraction"],
                             "slew_limit_fraction": row["target_step_at_slew_limit_fraction"],
                             "actor_reconstruction": row["actor_reconstruction"]} for name, row in results.items()}, indent=2))


if __name__ == "__main__":
    main()
