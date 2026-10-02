"""Audit frozen reward v2 on retained native trajectories; use CPU execution."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

REVISION = "13f55d4efa40ac8497812a96c7b7af4741fd7666"
CASES = {
    "failed_ppo": "/srv/cupi/hexapod/runs/james/flat_pilot_20261002_001/ppo_mlp_video_001/run/standing/evaluation/batch_000",
    "tripod": "/srv/cupi/hexapod/runs/james/flat_pilot_20261002_001/tripod_reference/run/standing/trial_000",
    "walk_phase0": "/srv/cupi/hexapod/evidence/legacy_runs/restart_20260914/amp_20260923_001/replay_08_phase0/run/standing/evaluation",
    "walk_phase05": "/srv/cupi/hexapod/evidence/legacy_runs/restart_20260914/amp_20260923_001/replay_08_phase05_continuation004/run/standing/evaluation",
}
EXPECTED_TRACES = {
    "failed_ppo": "474c935cf0e9072e6877e7189917b21a67c29ecb4e5b986f5fa24e358252704c",
    "tripod": "1c2465cd232b4939874cb95be36f0d06206431f8e62abf3c5066c276e9aa24ff",
    "walk_phase0": "c4b6ce4e2d1ee7613dd29379adae12bef1b43aaf2b6bc1d1299e190ac4d26720",
    "walk_phase05": "8ff49a31959bb94641d963b4870bfa096c468a3468692e1e3d9195ed51b742f1",
}
MODULES = ("__init__.py", "task.py", "task_v2.py", "env.py", "env_config.py", "amp.py")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load(path):
    return json.loads(Path(path).read_text())


def fetch():
    DATA.mkdir(exist_ok=True)
    for label, remote in CASES.items():
        out = DATA / label
        if out.exists():
            continue
        out.mkdir()
        names = ["control_trace.npz", "report.json", "declaration.json", "native400hz"]
        command = "tar -C " + shlex.quote(remote) + " -cf - " + " ".join(map(shlex.quote, names))
        process = subprocess.Popen(["ssh", "spark", command], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            archive.extractall(out, filter="data")
        if process.wait():
            raise RuntimeError("SSH transfer failed: " + label)
        parent = str(Path(remote).parent)
        if label == "failed_ppo":
            parent = str(Path(parent).parent)
        subprocess.run(["scp", "-q", "spark:" + parent + "/state.json", str(out / "native_state.json")], check=True)
        print("Retrieved", label, flush=True)
    for name, remote in {
        "model.json": "/srv/cupi/hexapod/inputs/flat_pilot_20260930_001/asset/source/model.json",
        "stance.json": "/srv/cupi/hexapod/inputs/flat_pilot_20260930_001/prior/stance.json",
    }.items():
        if not (DATA / name).exists():
            subprocess.run(["scp", "-q", "spark:" + remote, str(DATA / name)], check=True)
    package = SOURCE / "locomotion"
    package.mkdir(parents=True, exist_ok=True)
    for name in MODULES:
        content = subprocess.check_output(["git", "show", REVISION + ":locomotion/" + name], cwd=WORKTREE)
        path = package / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError("Frozen audit source differs: " + name)
        path.write_bytes(content)
    manifest = subprocess.check_output(["git", "show", REVISION + ":locomotion/priors/datasets/amp_demonstrations_001/manifest.json"], cwd=WORKTREE)
    (DATA / "amp_manifest.json").write_bytes(manifest)


def arrays(path):
    with np.load(path, allow_pickle=False) as bundle:
        return {name: bundle[name] for name in bundle.files}


def collision_from_patches(path, count):
    result = np.zeros(count, dtype=np.float64)
    nearest = math.inf
    rows = 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            assert row["sequence"] == rows
            sums = {}
            for patch in row["patches"]:
                assert patch["env"] == 0
                if patch["body"].endswith("_tibia"):
                    continue
                body = patch["body"]
                sums[body] = sums.get(body, np.zeros(3)) + patch["normal_force_n"] * np.asarray(patch["normal_world"])
            norms = [float(np.linalg.norm(v)) for v in sums.values()]
            if norms:
                nearest = min(nearest, min(abs(v - 1) for v in norms))
                result[rows] = max(norms)
            rows += 1
    assert rows == count
    return result, (None if nearest == math.inf else nearest)


def analyze(label, env, reward):
    directory = DATA / label
    trace = arrays(directory / "control_trace.npz")
    report = load(directory / "report.json")
    capture = load(directory / "native400hz/capture.json")
    declaration = load(directory / "declaration.json")
    state = load(directory / "native_state.json")
    assert sha(directory / "control_trace.npz") == EXPECTED_TRACES[label]
    assert report["files"]["control_trace.npz"] == EXPECTED_TRACES[label]
    assert report["files"]["declaration.json"] == sha(directory / "declaration.json")
    assert report["acquisition_complete"] and report["failure"] is None
    assert capture["failure"] is None and capture["steps"] == 8000
    assert declaration["cases"][0]["command"] == [0.05, 0.0, 0.0]
    identity = state["identity"]
    assert identity["model_sha256"] == sha(DATA / "model.json")
    assert identity["stance_sha256"] == sha(DATA / "stance.json")
    assert identity["config"]["action_scale_rad"] == .35
    assert identity["config"]["target_slew_rad"] == .04
    assert capture["joint_names"] == list(env.JOINT_NAMES)
    if "task_v2.py" in identity["source_files"]:
        assert identity["source_files"]["task_v2.py"] == sha(SOURCE / "locomotion/task_v2.py")
    admission = None
    if label.startswith("walk_"):
        manifest = load(DATA / "amp_manifest.json")
        assert manifest["admitted"] is True
        clips = [clip for clip in manifest["clips"] if clip["trace_sha256"] == EXPECTED_TRACES[label]]
        assert len(clips) == 1
        clip = clips[0]
        assert clip["report_sha256"] == sha(directory / "report.json")
        assert clip["state_sha256"] == sha(directory / "native_state.json")
        admission = {key: clip[key] for key in ("path", "trace_sha256", "report_sha256", "state_sha256", "start_phase")}
    verified = {}
    for name, digest in capture["files"].items():
        path = directory / "native400hz" / name
        assert Path(name).name == name and not path.is_symlink()
        assert sha(path) == digest, str(path)
        verified["native400hz/" + name] = digest
    for name in ("control_trace.npz", "declaration.json", "report.json", "native_state.json", "native400hz/capture.json"):
        verified[name] = sha(directory / name)
    chunks = [arrays(directory / "native400hz" / name) for name in capture["substep_files"]]
    keys = ("sequence", "explicit_counter", "control_index", "substep_index", "time_s", "applied_torque_nm", "computed_torque_nm", "native_input_pre_nm", "joint_target_rad", "joint_velocity_rad_s", "command")
    raw = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in keys}
    assert np.array_equal(raw["sequence"], np.arange(8000))
    assert np.array_equal(raw["explicit_counter"], capture["initial_counter"] + np.arange(8000) + 1)
    assert np.array_equal(raw["control_index"], np.arange(8000) // 8)
    assert np.array_equal(raw["substep_index"], np.arange(8000) % 8)
    assert np.allclose(raw["time_s"], (np.arange(8000) + 1) * .0025, atol=1e-12, rtol=0)
    assert np.array_equal(raw["native_input_pre_nm"], raw["applied_torque_nm"])
    assert np.array_equal(raw["joint_velocity_rad_s"][7::8], trace["joint_velocity_rad_s"])
    assert np.array_equal(raw["command"][7::8], trace["command"])
    assert np.allclose(trace["time_s"], (np.arange(1000) + 1) * .02, atol=1e-12, rtol=0)
    assert not trace["reset"].any() and not trace["terminated"].any() and not trace["truncated"].any()
    command = torch.as_tensor(trace["command"], dtype=torch.float32)
    assert torch.equal(command, command[:1].expand_as(command))
    after = torch.as_tensor(trace["amp_state_after"], dtype=torch.float32)
    before = torch.as_tensor(trace["amp_state_before"], dtype=torch.float32)
    assert torch.equal(after[:-1], before[1:])
    assert np.array_equal(after[..., 18:36].numpy(), trace["joint_velocity_rad_s"])
    native_linear = env.navigation(after[..., 36:39])
    native_angular = after[..., 39:42]
    critic_error = float(np.max(np.abs(native_linear[:-1].numpy() - trace["critic_observation"][1:, :, -3:])))
    assert critic_error == 0
    gyro_error = float(np.max(np.abs(native_angular.numpy() - trace["gyro_body_rad_s"])))
    assert gyro_error < 1e-5
    model = load(DATA / "model.json")
    joint_map = {j["name"]: j for j in model["joints"]}
    lower = torch.tensor([joint_map[name]["lower"] for name in env.JOINT_NAMES], dtype=torch.float32)
    upper = torch.tensor([joint_map[name]["upper"] for name in env.JOINT_NAMES], dtype=torch.float32)
    neutral = torch.tensor(load(DATA / "stance.json")["nominal_joint_position_rad"], dtype=torch.float32)
    actions = torch.as_tensor(trace["policy_action"], dtype=torch.float32)
    bounded = actions.clamp(-1, 1)
    recurrence_errors = {}
    emitted = {}
    for representation, values in (("original_raw", actions), ("clamped_counterfactual", bounded)):
        held = neutral[None].clone()
        records = []
        for action in values:
            held = env.emitted_target(action, held, neutral, lower, upper)
            records.append(held)
        targets = torch.stack(records)
        emitted[representation] = targets
        delta = float((targets - torch.as_tensor(trace["joint_target_rad"])).abs().max())
        recurrence_errors[representation] = delta
        assert delta <= 2e-7, (label, representation, delta)
    assert torch.equal(emitted["original_raw"], emitted["clamped_counterfactual"])
    loads = torch.as_tensor(raw["applied_torque_nm"], dtype=torch.float32).reshape(1000, 8, 1, 18)
    requested = torch.as_tensor(raw["computed_torque_nm"], dtype=torch.float32).reshape(1000, 8, 1, 18)
    torque_square = torch.zeros(1000, 1, 18)
    requested_max = torch.zeros_like(torque_square)
    for substep in range(8):
        torque_square += loads[:, substep].square()
        requested_max = torch.maximum(requested_max, requested[:, substep].abs())
    patch_forces, nearest = collision_from_patches(directory / "native400hz/contacts.jsonl", 8000)
    control_force = torch.tensor(patch_forces.reshape(1000, 8).max(1)[:, None], dtype=torch.float32)
    tracked = []
    stride = reward.StrideWindow(1, reward.REWARD_V2_CONFIG.stride_controls, 3)
    stride.restart(torch.tensor([0]), command[0])
    for index in range(1000):
        values = torch.cat((native_linear[index, :, :2], native_angular[index, :, 2:3]), -1)
        tracked.append(stride.update(values, command[index]))
    tracked = torch.stack(tracked)
    previous_rate = torch.cat((before[0:1, :, 18:36], after[:-1, :, 18:36]))
    base = {
        "linear_velocity_nav": native_linear[:, 0], "angular_velocity_body": native_angular[:, 0],
        "joint_velocity_rad_s": after[:, 0, 18:36], "previous_joint_velocity_rad_s": previous_rate[:, 0],
        "torque_square_sum_400hz": torque_square[:, 0], "requested_torque_abs_max_400hz": requested_max[:, 0],
        "other_body_force_max_400hz": control_force[:, 0], "command": command[:, 0],
        "tracked_planar_velocity_nav": tracked[:, 0, :2], "tracked_yaw_rate_rad_s": tracked[:, 0, 2],
    }
    mask = (trace["time_s"] > 2. + 1e-9) & (trace["time_s"] <= 20. + 1e-9)
    assert int(mask.sum()) == 900
    output = {}
    for representation, values in (("original_raw", actions), ("clamped_counterfactual", bounded)):
        previous = torch.cat((torch.zeros_like(values[:1]), values[:-1]))
        telemetry = {**base, "action": values[:, 0], "previous_action": previous[:, 0]}
        total, components = reward.measured_reward_v2(telemetry, command[:, 0], torch.as_tensor(trace["terminated"][:, 0]))
        means = {key: float(value[mask].double().mean()) for key, value in components.items()}
        noncollision = sum(value for key, value in means.items() if key != "collisions")
        output[representation] = {
            "component_means": means,
            "reward_mean_reconstructed_collision": float(total[mask].double().mean()),
            "reward_mean_collision_agnostic_interval": [noncollision - reward.REWARD_V2_CONFIG.collision_weight, noncollision],
            "raw_action_abs_max": float(values[mask].abs().max()),
            "raw_action_outside_unit_fraction": float((values[mask].abs() > 1).double().mean()),
        }
    return {
        "remote_source": CASES[label], "input_sha256": verified,
        "admitted_dataset_entry": admission,
        "native_identity": {key: identity.get(key) for key in ("model_sha256", "usd_sha256", "physics_source_files", "physics_config", "stance_sha256", "geometry_sha256", "geometry_extrema_sha256")},
        "screen_pass": report["results"][0]["pass"],
        "requested_command": [0.05, 0, 0], "window_seconds": {"greater_than": 2, "less_than_or_equal": 20},
        "scored_controls": 900, "history_controls": 1000,
        "target_recurrence_max_abs_error_rad": recurrence_errors,
        "original_and_clamped_targets_identical": True,
        "critic_velocity_crosscheck_max_error_mps": critic_error,
        "gyro_crosscheck_max_error_rad_s": gyro_error,
        "reconstructed_non_tibia_collision_controls": int((control_force[mask] > 1).sum()),
        "nearest_recorded_non_tibia_resultant_to_threshold_n": nearest,
        "along_command_speed_mps": float(native_linear[mask, 0, 0].double().mean()),
        "root_height_mean_m": float(after[mask, 0, 42].double().mean()),
        "representations": output,
    }


def main():
    global DATA, SOURCE, WORKTREE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--workspace", type=Path, required=True,
                        help="External directory for raw captures and frozen reward source")
    parser.add_argument("--repository", type=Path, default=Path.cwd(),
                        help="Git checkout containing the frozen source commit")
    parser.add_argument("--output", type=Path,
                        help="Receipt path; defaults to WORKSPACE/reward_audit.json")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    DATA = workspace / "reward_audit_inputs"
    SOURCE = workspace / "reward_audit_source"
    WORKTREE = args.repository.resolve()
    output = args.output or workspace / "reward_audit.json"
    if output.exists():
        raise FileExistsError("Use a fresh receipt path: " + str(output))
    if args.fetch:
        fetch()
    sys.path.insert(0, str(SOURCE))
    env = importlib.import_module("locomotion.env")
    reward = importlib.import_module("locomotion.task_v2")
    torch.set_num_threads(1)
    results = {label: analyze(label, env, reward) for label in CASES}
    # Preserve each native physics identity; admit no source-equivalence shortcut.
    reference = results["failed_ppo"]["native_identity"]
    identity_matches = {label: result["native_identity"] == reference for label, result in results.items()}
    assert all(identity_matches.values())
    comparisons = {}
    for representation in ("original_raw", "clamped_counterfactual"):
        failed = results["failed_ppo"]["representations"][representation]["reward_mean_collision_agnostic_interval"]
        comparisons[representation] = {}
        for label in CASES:
            if label == "failed_ppo":
                continue
            good = results[label]["representations"][representation]["reward_mean_collision_agnostic_interval"]
            comparisons[representation][label] = {
                "walk_minus_failed_reward_interval": [good[0] - failed[1], good[1] - failed[0]],
                "walking_ranks_above_failure_despite_unknown_collision": good[0] > failed[1],
                "failure_ranks_above_walking_despite_unknown_collision": failed[0] > good[1],
            }
    receipt = {
        "schema": "hexapod_reward_v2_exact_capture_audit_v1", "source_commit": REVISION,
        "analysis_sha256": sha(__file__), "source_files": {name: sha(SOURCE / "locomotion" / name) for name in MODULES},
        "input_metadata_sha256": {name: sha(DATA / name) for name in ("model.json", "stance.json", "amp_manifest.json")},
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "device": "cpu"},
        "reward": reward.reward_declaration(reward.REWARD_V2_CONFIG),
        "method": {
            "history": "Start at control zero. Use recorded actions, previous rates, native AMP velocities and the frozen 60-control stride function; select endpoints 2<t<=20 after computing history.",
            "torque": "Use each native applied-torque sample and requested-torque sample. Accumulate eight float32 squares in control order; take each joint's absolute requested maximum across all eight substeps.",
            "collision": "Sum named contact-patch force vectors per non-tibia body, then take the largest body norm across eight substeps. The original force matrix is absent; preserve a full [−0.05,0] collision reward interval for every control, regardless of reconstructed events.",
            "counterfactual": "Clamp recorded raw actions to [−1,1], preserving identical target execution. This is a bounded-command reward counterfactual, not a new tanh-policy rollout.",
            "scope": "These fixed-trajectory rewards test ranking at one command. They do not establish learning success or the flat_pilot_v1 decision.",
            "precision": "CPU float32 reproduces reward operations. Reduction differences from the original CUDA runtime remain possible; no training reward was recorded for these evaluation trajectories.",
        },
        "native_identity_matches_failed_ppo": identity_matches, "results": results, "comparisons": comparisons,
        "native_started": False,
    }
    save(output, receipt)
    print(json.dumps({"reward_means": {k: v["representations"] for k, v in results.items()}, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
