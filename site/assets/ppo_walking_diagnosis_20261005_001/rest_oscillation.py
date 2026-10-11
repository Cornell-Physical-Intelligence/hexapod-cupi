"""Record how one native PPO policy oscillates at rest and the feedback gains of its actor.

The script reads one checkpoint and one retained evaluation of it. It copies the forward, quiet and stop screens, measures
the action, target and joint motion of the stop and quiet traces under a zero command, replays the actor on the recorded
observations and reads same-joint gains from its Jacobian. Run it from the repository root with PYTHONPATH set to that root.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

import numpy as np
import torch

from locomotion.env_config import JOINT_NAMES
from locomotion.ppo import ACTOR_SCALES, scale_observation

RUN = "ppo_v4_action_wall_20261005_001"
REMOTE = "/srv/cupi/hexapod/runs/james/"
STANDING = RUN + "/run/standing/"
EVALUATION = RUN + "_evaluate_stop_video/run/standing/evaluation/"
SEED, UPDATE, CONTROL_HZ, ACTION_SCALE_RAD = 20260917, 2000, 50, .35
CASES = {"forward": "learning:translate_0.05_0deg", "quiet_20s": "learning:quiet_20s",
         "forward_to_stop": "learning:forward_0.05_to_stop"}
REST = {"forward_to_stop": (550, 1050), "quiet_20s": (300, 1000)}  # first control, one past the last control
SCREEN = ("planar_error_mps", "mean_velocity_mps", "max_joint_velocity_rms_rad_s", "max_target_step_abs_p95_rad_per_20ms")
WIDTHS = {"policy_action": 18, "policy_observation": 231, "joint_target_rad": 18, "joint_velocity_rad_s": 18,
          "joint_position_rad": 18, "command": 3}
# PINS holds the sha256 of each remote record that the script reads. BATCH_PINS holds the files of each case's batch.
PINS = {
    STANDING + "checkpoint_update%06d.pt" % UPDATE: "fe3eb69ef3ee4675d49373e8803a0f4207a387c85be4abf80ddcc163ece30371",
    STANDING + "checkpoint_update%06d.json" % UPDATE: "8fa2c812fe31f7f50abdcd029d6bafcba878fd1afc7f7c58b32847fff15f2859",
    EVALUATION + "summary.json": "835dc7c3bc7bc2df4ceec08b7fc21d7c0024d7a680586c0f5dfe67c52e3a499e",
    EVALUATION + "allocation.json": "a987c286624aaba15d8d08218a15f83d36bf3e40e3af0fc942ce00582ebccee9"}
BATCH_PINS = {
    "forward": {"report.json": "87a3a31c46eb914e5b5a1a49d89166f082f804625b6e5f9942b459ae75e3dfd2"},
    "quiet_20s": {"report.json": "663d9e6913bb629f448f863b0cd4d89ffa5d5cd5bb88b9033c9e19b800b19339",
                  "control_trace.npz": "402a1bab5b127446ddfc4481eacfc807a6f41ad108eefd19c87ffdf0f4fc35c8"},
    "forward_to_stop": {"report.json": "0a2a1e7f2f47f041348a1c72420374407a8847a2fa54153897ee193113f64646",
                        "control_trace.npz": "f33a278fc95204a150a81e30a25ad4a8458e1d160f406ee62394c4a9d5ff83f1"}}
METHODS = {
    "scope": "One checkpoint, one seed and one retained evaluation with one trial per case. The evaluator takes the actor mean "
             "and adds no sampling noise, so the record measures no trial-to-trial spread.",
    "batch": "Each case's batch index is its position in the allocation's selected_case_ids. Its report must name that case "
             "alone, the pinned checkpoint, the summary result and, for stop and quiet, the pinned trace.",
    "screens": "Copy pass, failed_bounds, the score window and four metrics from summary.json. A null metric means the "
               "evaluator wrote no value. The summary scores from window_start_control, before each rest window starts.",
    "rest_motion": "The script asserts a zero command over stop controls 550 to 1049 and quiet controls 300 to 999. action_std "
                   "is the population standard deviation of policy_action per joint. For the joint with the largest one, x is "
                   "the action minus its window mean, the lag-1 autocorrelation is sum(x[t] x[t+1]) / sum(x[t]^2), the peak is "
                   "the largest rFFT power bin above DC at bin * 50 / controls Hz, and its share divides by the sum of those "
                   "bins. As in locomotion/evaluation.py, the target step is the largest per-joint 95th percentile of "
                   "|diff(joint_target_rad)| and the joint velocity is the largest per-joint RMS.",
    "replay": "The script asserts the checkpoint record's hidden sizes 256, 256, 128, ELU, no observation normalization, "
              "BoundedMeanGaussian, fixed observation scaling, gait clock and no action smoothing. The actor is linear layers "
              "mlp.0, mlp.2, mlp.4, mlp.6 with ELU after the first three and tanh after the last. Its input is "
              "scale_observation(policy_observation, 'fixed') plus two zero columns, because the evaluation policy in "
              "locomotion/train.py appends clock_features, which a zero command sets to zero. Over the rest window |replayed "
              "mean - policy_action| must stay below 1e-5.",
    "layout": "locomotion/env.py builds the observation as cat(history.flatten(1), commands, previous_action) from five "
              "42-value frames. LocomotionEnv.step sets history = cat(history[:, 1:], new frame), so frame 4 is the latest. "
              "_proprio orders a frame as body rates (3), gravity (3), q - neutral (18), dq (18). previous_action holds "
              "(target - neutral) / 0.35 of the previous control. The script asserts ACTOR_SCALES of locomotion/ppo.py: 1/0.35 "
              "for q - neutral, 0.5 for dq, 1 for previous_action. layout_check holds the largest gaps between frame f of "
              "observation row t and joint state row t - 5 + f, and between previous_action and target row t - 1.",
    "gains": "Take the Jacobian of the replayed mean with respect to the scaled 233-value input at the middle control of the "
             "stop window. In row j the previous-target gain is column 213 + j, the position gain of frame f is column 42f + 6 "
             "+ j per action unit of joint offset, and the velocity gain is 0.5 times column 42f + 24 + j in action units per "
             "rad/s. Each frame reports the mean, minimum and maximum over the 18 joints. two_control_mean evaluates |cos(pi f "
             "/ 50)|, the gain of a mean of two consecutive actions, at the peak frequency of the stop window.",
    "limits": "The gains use the same-joint Jacobian entries at one state. The record holds no model of the motor and joint "
              "response, so it establishes no cause for the oscillation."}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(enabled, names, destination):
    """Copy the missing named files from the Spark run directory through a tar stream when the caller enables it."""
    missing = [name for name in names if enabled and not (destination / name).exists()]
    if missing:
        destination.mkdir(parents=True, exist_ok=True)
        command = "tar -C " + shlex.quote(REMOTE) + " -cf - " + " ".join(map(shlex.quote, missing))
        process = subprocess.Popen(["ssh", "spark", command], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            archive.extractall(destination, filter="data")
        assert process.wait() == 0


def window(path, neutral, actor, start, stop):
    """Check one control trace, then measure the rest motion and replay the actor over controls start to stop - 1."""
    with np.load(path, allow_pickle=False) as data:
        trace = {key: data[key].squeeze(1) for key in WIDTHS}  # squeeze fails unless the trace holds one replica
        assert not data["reset"].any() and not data["terminated"].any()
    assert all(trace[key].shape[1:] == (width,) and np.isfinite(trace[key]).all() for key, width in WIDTHS.items())
    # Observation row t pairs with trace row t - 5 + frame. Frame 4 also pairs with the previous target.
    gap = lambda first, key, frame, offset, scale: float(np.abs(
        trace["policy_observation"][5:, first:first + 18] - (trace[key][frame:frame - 5] - offset) / scale).max())
    layout = {"previous_executed_target": gap(213, "joint_target_rad", 4, neutral, ACTION_SCALE_RAD),
              "frame_position_rad": max(gap(42 * frame + 6, "joint_position_rad", frame, neutral, 1) for frame in range(5)),
              "frame_velocity_rad_s": max(gap(42 * frame + 24, "joint_velocity_rad_s", frame, 0, 1) for frame in range(5))}
    zero_from = int(np.flatnonzero(np.r_[True, trace["command"].any(axis=1)]).max())
    observation, recorded = (torch.from_numpy(trace[key][start:stop]) for key in ("policy_observation", "policy_action"))
    scaled = torch.cat((scale_observation(observation, "fixed"), torch.zeros(stop - start, 2)), dim=-1)
    error = float((actor(scaled).detach() - recorded).abs().max())
    assert max(layout.values()) < 1e-6 and error < 1e-5 and zero_from <= start and not observation[:, 210:213].any()
    action, target, rate = (trace[key][start:stop].astype(np.float64)
                            for key in ("policy_action", "joint_target_rad", "joint_velocity_rad_s"))
    spread = action.std(axis=0)
    joint = int(spread.argmax())
    x = action[:, joint] - action[:, joint].mean()
    power = np.abs(np.fft.rfft(x)) ** 2
    peak = int(power[1:].argmax()) + 1
    step, rms = np.quantile(np.abs(np.diff(target, axis=0)), .95, axis=0), np.sqrt((rate ** 2).mean(axis=0))
    return scaled, {
        "first_control": start, "last_control": stop - 1, "command_zero_from_control": zero_from,
        "action_std": dict(zip(JOINT_NAMES, map(float, spread))),
        "largest_action_std": float(spread[joint]), "largest_action_std_joint": JOINT_NAMES[joint],
        "lag1_autocorrelation": float((x[1:] * x[:-1]).sum() / (x * x).sum()),
        "peak_frequency_hz": peak * CONTROL_HZ / len(x), "peak_share_of_non_dc_power": float(power[peak] / power[1:].sum()),
        "max_target_step_abs_p95_rad_per_20ms": float(step.max()), "max_target_step_joint": JOINT_NAMES[int(step.argmax())],
        "max_joint_velocity_rms_rad_s": float(rms.max()), "max_joint_velocity_rms_joint": JOINT_NAMES[int(rms.argmax())],
        "policy_replay_max_abs_error": error, "layout_check": layout}


def gains(actor, scaled):
    """Read the same-joint gains from the Jacobian of the action mean at one scaled input."""
    jacobian = torch.autograd.functional.jacobian(actor, scaled).numpy().astype(np.float64)
    scales, joint, frames = np.array(ACTOR_SCALES), np.arange(18), []
    spread = lambda values: {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}
    assert jacobian.shape == (18, 233) and (scales[213:] == 1).all()
    for frame in range(5):
        q, dq = 42 * frame + 6 + joint, 42 * frame + 24 + joint
        assert np.allclose(scales[q], 1 / ACTION_SCALE_RAD) and (scales[dq] == .5).all()
        frames.append({"frame": frame, "controls_before_latest_frame": 4 - frame, "position_gain": spread(jacobian[joint, q]),
                       "velocity_gain_action_per_rad_s": spread(.5 * jacobian[joint, dq])})
    previous = jacobian[joint, 213 + joint]
    return {"previous_executed_target_gain": {**spread(previous), "by_joint": dict(zip(JOINT_NAMES, map(float, previous)))},
            "history_frames": frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output, inputs = args.output or args.workspace / "rest_oscillation.json", args.workspace / "inputs"
    if output.exists():
        raise FileExistsError("Use a fresh output path")
    fetch(args.fetch, PINS, inputs)
    weights, record, summary, allocation = (inputs / item for item in PINS)
    record, summary, allocation = (json.loads(path.read_text()) for path in (record, summary, allocation))
    batch = {label: EVALUATION + "batch_%03d/" % allocation["selected_case_ids"].index(case) for label, case in CASES.items()}
    pins = {**PINS, **{batch[label] + name: pin for label in CASES for name, pin in BATCH_PINS[label].items()}}
    fetch(args.fetch, pins, inputs)
    assert {item: sha(inputs / item) for item in pins} == pins
    identity, physics, checkpoint = record["identity"], record["identity"]["physics_config"], sha(weights)
    network, wrapper = identity["ppo_config"]["actor"], identity["ppo_config"]["environment_wrapper"]
    assert record["updates"] == UPDATE and record["checkpoint_sha256"] == checkpoint == allocation["checkpoint_sha256"]
    assert identity["seed"] == SEED == allocation["seed"] and summary["allocation"] == allocation
    assert identity["config"]["joint_names"] == list(JOINT_NAMES) and physics["action_scale_rad"] == ACTION_SCALE_RAD
    assert network["hidden_dims"] == [256, 256, 128] and network["activation"] == "elu" and not network["obs_normalization"]
    assert network["distribution_cfg"]["class_name"] == "locomotion.action_distribution:BoundedMeanGaussian"
    assert wrapper == {"observation_scaling": "fixed", "command_segments": "bootstrap", "gait_clock": 60}
    results, screens = {row["case_id"]: row for row in summary["results"]}, {}
    for label, case in CASES.items():
        report, result = json.loads((inputs / batch[label] / "report.json").read_text()), results[case]
        assert (report["assigned_case_ids"], report["checkpoint_sha256"], report["results"]) == ([case], checkpoint, [result])
        assert BATCH_PINS[label].get("control_trace.npz") in (None, report["files"]["control_trace.npz"])
        screens[label] = {**{key: result[key] for key in ("pass", "failed_bounds", "window_start_control", "window_controls")},
                          **{key: result["metrics"].get(key) for key in SCREEN}}
    actor = torch.nn.Sequential(torch.nn.Linear(233, 256), torch.nn.ELU(), torch.nn.Linear(256, 256), torch.nn.ELU(),
                                torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 18), torch.nn.Tanh())
    state = torch.load(weights, map_location="cpu", weights_only=True)["actor_state_dict"]
    actor.load_state_dict({key[4:]: value for key, value in state.items() if key.startswith("mlp.")})
    neutral, motion = np.array(physics["neutral_joint_position_rad"]), {}
    for label, (start, stop) in REST.items():
        scaled, motion[label] = window(inputs / batch[label] / "control_trace.npz", neutral, actor, start, stop)
        if label == "forward_to_stop":
            gain = {"control": (start + stop) // 2, **gains(actor, scaled[(stop - start) // 2])}
    frequency = motion["forward_to_stop"]["peak_frequency_hz"]
    receipt = {"schema": "hexapod_ppo_rest_oscillation_v1", "analysis_sha256": sha(__file__), "training_attempt": RUN,
               "remote_directory": REMOTE, "seed": SEED, "update": UPDATE, "checkpoint_sha256": checkpoint, "cases": CASES,
               "sha256": pins, "methods": METHODS, "screens": screens, "rest_motion": motion, "gains": gain,
               "two_control_mean": {"frequency_hz": frequency, "gain": float(abs(np.cos(np.pi * frequency / CONTROL_HZ)))},
               "native_started_by_analysis": False}
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: receipt[key] for key in ("screens", "rest_motion", "gains", "two_control_mean")}))


if __name__ == "__main__":
    main()
