"""Measure how the no-noise forward gait changes between checkpoints of one native PPO run.

The script reads four retained evaluations of ppo_v4_forward_20261004_001 at updates 600,
1000, 1400 and 2000. It records the forward, quiet and stop screens from each summary. From
each forward control trace it records action saturation, coxa action spread, toe fore-aft
travel, toe contact and toe height over two control windows.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

import numpy as np


RUN = "ppo_v4_forward_20261004_001"
REMOTE = "/srv/cupi/hexapod/runs/james/"
EVALUATION = "run/standing/evaluation/"
SEED = 20260917
FORWARD, QUIET, STOP = "learning:translate_0.05_0deg", "learning:quiet_20s", "learning:forward_0.05_to_stop"
FORWARD_METRICS = ("mean_velocity_mps", "planar_error_mps", "yaw_error_rad_s", "tilt_rms_degrees",
                   "computed_demand_over_rating_fraction")
QUIET_METRICS = ("max_joint_velocity_rms_rad_s", "max_target_step_abs_p95_rad_per_20ms")
LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")
JOINTS = ("coxa", "femur", "tibia")
WINDOWS = {"controls_400_999": (400, 1000), "controls_0_59": (0, 60)}
# Each entry pins the update and the sha256 of the five remote files that the script reads.
PINS = {
    RUN + "_evaluate_u0600": (600, {
        "binding.json": "e1beabb0e11d62cca30b5fb0183f7d9e64eb42cf9aedc307ec874cc597b04b14",
        "summary.json": "ea0c81fb068436f9ef503fc89d65c9c4f51b421e1c0892149766c0c483548e3d",
        "allocation.json": "90b15a88ca29b1f2f487ea74edaeedc30007afd987d7199a7f38fc3d66865a4b",
        "report.json": "4dea720bda33669fcae2b60e710303711c18d859eb6d51d28ae83bd0bd8376ca",
        "control_trace.npz": "35c43ce4aad083e1a6bac03c87858afa1feb951f22ba9db7942e69fbb336d83d"}),
    RUN + "_evaluate_u1000": (1000, {
        "binding.json": "c9cd13a9758a7be9a641df61c279a2a2a5fa131a99adddb46e36853cdb8bc843",
        "summary.json": "49537011ef1ca82cbd7757e44e4af585698da59cd331f937d35afce9cb48e78a",
        "allocation.json": "50a922cac9404faf87af8a668c7d09cd31354507182162e4826c9863abe7a914",
        "report.json": "f5a33167a03e6993211810b8e2383ce99e918bae75b3b91d69e57cd9815be268",
        "control_trace.npz": "f000461c48ecc4fa97a028df1c4f5b32a47a706ce7cf1f7ca0a61e7549ebb8ba"}),
    RUN + "_evaluate_u1400": (1400, {
        "binding.json": "a0caf200488ee2522ee54d10b3c34bfdb80946ce111afdb5c4aff5a43c67073f",
        "summary.json": "800423a4ebad0d96e503c3830b8b095ec8208960f7320abf1de256ed2b91467d",
        "allocation.json": "1fa0ded1a294c118d8de854a8b35c581766c3f4b98c701a3c83844bcc36a44ca",
        "report.json": "374a7cce13d4134de830f0c33fb69a994e0b8ac12b0d7bfee7bd0efa4e6b3659",
        "control_trace.npz": "1b6056db25aed20f31e5b290a52a1ef9cac63b9575895c75828383afda190754"}),
    RUN + "_evaluate": (2000, {
        "binding.json": "48afeeb57eeef3e01def7cb83465066679f9442eaf63efa05228454305b4e4ae",
        "summary.json": "b5531d4fc77cf22fd6c897195f3dadb0b9850f54517cdbf789d027f3bfe1b5d7",
        "allocation.json": "0771e460f0b0a4144d153ad7454ac26d235658ebab04080239b8a2c3396965ed",
        "report.json": "5b3951a6fb4ab044d92dfcf150010ce7ba4e79ceee141e28e8e9e12ae927489d",
        "control_trace.npz": "89212933eb96aa718897ea2bfa9d1cfec7ceab8867298fada16e1ea49549fa2d"}),
}
METHODS = {
    "scope": "Four retained evaluations of one training run and one seed. The evaluator takes the "
             "actor mean and adds no sampling noise. Each checkpoint has one forward trial, so the "
             "record shows differences between checkpoints and measures no trial-to-trial spread.",
    "batch": "Take the forward batch index from the position of the forward case in the allocation's "
             "selected_case_ids, then require that the batch report names the same case, the same "
             "checkpoint and the hash of the control trace.",
    "screens": "Copy pass, failed_bounds, recorded_controls and the named metrics from summary.json. "
               "A null metric means the evaluator wrote no value for that trial.",
    "actions": "policy_action holds 18 values in the order lf, lm, lr, rf, rm, rr, each coxa, femur, "
               "tibia. The share counts samples with absolute value above 0.95 across six legs and "
               "all controls in the window. The coxa spread is the population standard deviation of "
               "each coxa action over the window.",
    "toes": "Subtract the root position from each toe position in the world xy plane. Rotate the "
            "difference by minus the root yaw, with yaw = atan2(2(wz+xy), 1-2(yy+zz)) from the XYZW "
            "quaternion. The fore coordinate is the negative rotated y component because navigation "
            "forward is world -Y at zero yaw. The range is maximum minus minimum over the window. "
            "The height is the maximum world z of the toe point, which is no clearance measurement.",
    "contact": "The contact fraction is the mean of distal_contact for each leg over the window.",
    "limits": "A fore-aft range covers stance and swing travel together. The record establishes no "
              "cause for a difference between checkpoints.",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def fetch(remote, names, destination):
    """Copy the missing named files from one Spark directory through a tar stream."""
    missing = [name for name in names if not (destination / name).exists()]
    if not missing:
        return
    destination.mkdir(parents=True, exist_ok=True)
    command = "tar -C " + shlex.quote(remote) + " -cf - " + " ".join(map(shlex.quote, missing))
    process = subprocess.Popen(["ssh", "spark", command], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        archive.extractall(destination, filter="data")
    assert process.wait() == 0


def screen(result, keys):
    """Return the pass state and the named metrics of one summary result."""
    metrics = result.get("metrics", {})
    return {"pass": result["pass"], "failed_bounds": result["failed_bounds"],
            "recorded_controls": result["recorded_controls"], **{key: metrics.get(key) for key in keys}}


def gait(trace, start, stop):
    """Measure actions and toe motion over controls start to stop - 1."""
    action = trace["policy_action"][start:stop, 0].astype(np.float64).reshape(-1, 6, 3)
    pose = trace["root_pose_xyzw"][start:stop, 0].astype(np.float64)
    toe = trace["toe_xyz_world_m"][start:stop, 0].astype(np.float64)
    contact = trace["distal_contact"][start:stop, 0]
    x, y, z, w = pose[:, 3:].T
    turn = -np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))[:, None]
    relative = toe[:, :, :2] - pose[:, None, :2]
    fore = -(relative[:, :, 0] * np.sin(turn) + relative[:, :, 1] * np.cos(turn))
    by_leg = lambda values: {leg: float(value) for leg, value in zip(LEGS, values)}
    return {
        "first_control": start, "last_control": stop - 1,
        "action_abs_above_0p95_share": {joint: float((np.abs(action[:, :, index]) > 0.95).mean())
                                        for index, joint in enumerate(JOINTS)},
        "coxa_action_std": by_leg(action[:, :, 0].std(axis=0)),
        "toe_fore_aft_range_mm": by_leg(1000 * np.ptp(fore, axis=0)),
        "toe_contact_fraction": by_leg(contact.mean(axis=0)),
        "toe_height_max_mm": by_leg(1000 * toe[:, :, 2].max(axis=0)),
    }


def analyze(name, update, pins, workspace, do_fetch):
    """Check one evaluation's records and measure its forward trace."""
    remote, local = REMOTE + name, workspace / "inputs" / name
    records = ["binding.json", EVALUATION + "summary.json", EVALUATION + "allocation.json"]
    if do_fetch:
        fetch(remote, records, local)
    binding, summary, allocation = (read(local / item) for item in records)
    batch = "batch_%03d" % allocation["selected_case_ids"].index(FORWARD)
    captures = [EVALUATION + batch + "/report.json", EVALUATION + batch + "/control_trace.npz"]
    if do_fetch:
        fetch(remote, captures, local)
    hashes = {item: sha(local / item) for item in records + captures}
    assert {Path(item).name: digest for item, digest in hashes.items()} == pins
    report = read(local / captures[0])
    arguments = binding["command_args"]
    option = lambda flag: arguments[arguments.index(flag) + 1]
    checkpoint = option("--checkpoint-sha256")
    assert int(option("--updates")) == update and int(option("--seed")) == SEED == allocation["seed"]
    assert option("--checkpoint").endswith("checkpoint_update%06d.pt" % update)
    assert binding["input_files"][REMOTE + RUN + "/run/standing/checkpoint_update%06d.pt" % update] == checkpoint
    assert summary["allocation"] == allocation and allocation["checkpoint_sha256"] == checkpoint
    assert report["assigned_case_ids"] == [FORWARD] and report["checkpoint_sha256"] == checkpoint
    assert report["files"]["control_trace.npz"] == hashes[captures[1]]
    results = {row["case_id"]: row for row in summary["results"]}
    assert all(results[case]["lineage"]["checkpoint_sha256"] == checkpoint for case in (FORWARD, QUIET, STOP))
    forward = results[FORWARD]
    assert report["results"] == [forward]
    with np.load(local / captures[1], allow_pickle=False) as data:
        trace = {key: data[key] for key in ("policy_action", "root_pose_xyzw", "toe_xyz_world_m", "distal_contact",
                                            "velocity_navigation_mps", "command", "reset", "terminated")}
    assert trace["policy_action"].shape == (1000, 1, 18) and trace["root_pose_xyzw"].shape == (1000, 1, 7)
    assert trace["toe_xyz_world_m"].shape == (1000, 1, 6, 3) and trace["distal_contact"].shape == (1000, 1, 6)
    assert all(np.isfinite(trace[key]).all() for key in ("policy_action", "root_pose_xyzw", "toe_xyz_world_m"))
    assert np.allclose(trace["command"], [0.05, 0, 0]) and not trace["reset"].any() and not trace["terminated"].any()
    # The summary's mean velocity must come from this trace over the summary's own window.
    first = forward["window_start_control"]
    velocity = trace["velocity_navigation_mps"][first:first + forward["window_controls"], 0].mean(axis=0)
    assert np.allclose(velocity, forward["metrics"]["mean_velocity_mps"], rtol=0, atol=1e-9)
    return {"update": update, "evaluation": name, "remote_directory": remote, "eval_scope": option("--eval-scope"),
            "checkpoint_sha256": checkpoint, "forward_batch": batch, "sha256": hashes,
            "forward": screen(forward, FORWARD_METRICS), "quiet_20s": screen(results[QUIET], QUIET_METRICS),
            "forward_to_stop": screen(results[STOP], QUIET_METRICS),
            "forward_trace": {label: gait(trace, *bounds) for label, bounds in WINDOWS.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    output = args.output or args.workspace / "checkpoint_drift.json"
    if output.exists():
        raise FileExistsError("Use a fresh output path")
    rows = [analyze(name, update, pins, args.workspace, args.fetch) for name, (update, pins) in PINS.items()]
    receipt = {"schema": "hexapod_ppo_checkpoint_drift_v1", "analysis_sha256": sha(__file__),
               "training_attempt": RUN, "seed": SEED,
               "cases": {"forward": FORWARD, "quiet_20s": QUIET, "forward_to_stop": STOP},
               "methods": METHODS, "evaluations": rows, "native_started_by_analysis": False}
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{"update": row["update"], "forward": row["forward"], "quiet_20s": row["quiet_20s"],
                       "forward_to_stop": row["forward_to_stop"],
                       "controls_400_999": row["forward_trace"]["controls_400_999"]} for row in rows]))


if __name__ == "__main__":
    main()
