"""Measure joint torque, mechanical joint power and foot load in retained native PPO evaluations.

The script reads the 400 Hz captures of seven evaluations of six policies at update 2000. It measures each forward
probe, the standing windows and a 50 Hz series of the passing policy, and the motor constants that the repository holds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
REMOTE, EVALUATION = "/srv/cupi/hexapod/runs/james/", "run/standing/evaluation/"
UPDATE, PHYSICS_HZ, DECIMATION, CAP_NM, GRAVITY, STANCE_N = 2000, 400, 8, 1.6, 9.81, 1.0
FORWARD, QUIET, STOP = "learning:translate_0.05_0deg", "learning:quiet_20s", "learning:forward_0.05_to_stop"
# Each case maps to the time when its 0.05 m/s command ends and the time when its window starts.
WINDOWS = {FORWARD: (20.0, 2.0), QUIET: (0.0, 4.0), STOP: (8.0, 10.0)}
LEGS, JOINTS = ("lf", "lm", "lr", "rf", "rm", "rr"), ("coxa", "femur", "tibia")
NAMES = [leg + "_" + joint for leg in LEGS for joint in ("coxa_yaw", "femur_pitch", "tibia_pitch")]
ARRAYS = ("time_s", "command", "applied_torque_nm", "computed_torque_nm", "joint_velocity_rad_s",
          "interval_angle_rate_rad_s", "root_pose_xyzw", "distal_force_world_n", "nonfoot_force_world_n")
PASSING, SPREAD = "ppo_v4_settle_seed20260917_20261006_001", "ppo_v4_%s_20261005_001_evaluate"
CASES = {PASSING + "_evaluate": (FORWARD, QUIET), PASSING + "_evaluate_stop_video": (STOP,)}
# Each pin is the sha256 of the sorted path-to-sha256 map of the files read from one evaluation. The passing policy
# has two evaluations with the cases above, and each other evaluation supplies its forward case.
PINS = {
    PASSING + "_evaluate": "a1e7ec496290c3ecfe4fca6c688b820fececbe047903113642c291debf7b0c8a",
    PASSING + "_evaluate_stop_video": "149fdc6f0b6e948d8225f20bb044151a897c0f63e362a9a8e6788b20c9c14e50",
    SPREAD % "vnoise_seed20260917": "3c3f2cff4ccf8db8ef8933566bcda7a6e22aa2a6217a1fceb59cdc528c9fa80d",
    SPREAD % "vnoise_seed20260918": "34deee2cfdd21dba4148d3384c5a9b469d3d129b5bf7a86fb958ea771e614123",
    SPREAD % "vnoise_seed20260919": "c6063c14442f28e940efbd531dd14f36a96eee31832c9a64b93e613b790574a4",
    SPREAD % "restload_seed20260917": "a464305da09da16d1927e185945f24301c90b412d6393695fca3a1cbc8a40a84",
    SPREAD % "restload_seed20260918": "3b9e0b1ce8eb7adfb9d93a68ecf85482874d8b2046d43702f7720556c21cd5ca",
}
ENV, SPEC = "locomotion/env.py", "docs/RS05_SPEC_REVIEW.md"
# Each entry gives a quantity, its recorded value, the file and the text in that file that states the value.
RECORDED = (
    ("robot mass", "7.466088235225788 kg", "robot/active_model.json", '"mass_kg": 7.466088235225788'),
    ("gravity in the simulation", "9.81 m/s^2", ENV, "gravity=(0., 0., -9.81)"),
    ("software torque cap in motor_force", "1.6 N m", ENV, "clamp(min=0., max=1.6)"),
    ("provisional 48 V speed-torque curve in motor_force, speed points", "0, 70, 275, 340, 450, 477, 480 rpm", ENV,
     "xp = torch.tensor([0., 70., 275., 340., 450., 477., 480.]"),
    ("provisional 48 V speed-torque curve in motor_force, torque points", "5.5, 5.5, 4, 3, 1.6, 0.5, 0 N m", ENV,
     "yp = torch.tensor([5.5, 5.5, 4., 3., 1.6, .5, 0.]"),
    ("rated and allowed supply voltage", "48 V rated, 15 to 60 V allowed", SPEC, "supply voltage | 48 V / 15"),
    ("internal reduction", "7.75:1; vendor torque and speed refer to the motor output", SPEC, "| 7.75:1 / 191 g |"),
    ("peak output torque", "5.5 N m", SPEC, "Peak output torque | 5.5 N"),
    ("rated output torque while rotating", "1.6 N m at 100 rpm on a 70 mm plate", SPEC, "while rotating | 1.6 N"),
    ("continuous stall torque", "1.2 N m", SPEC, "Continuous stall torque | 1.2 N"),
    ("stalled overload points", "1.2 N m rated, 1.6 N m for 175 s, 5.5 N m for 1 s", SPEC, "Stalled | 1.2 N"),
    ("no-load speed", "480 rpm", SPEC, "No-load speed | 480 rpm"),
    ("maximum phase current", "11 A peak, no battery-line current", SPEC, "The maximum phase current is 11 A peak"),
)
# Each entry gives a quantity and the pattern that the script searches for in the scanned files.
ABSENT = (
    ("motor torque constant", r"torque[ _-]?constant|N.{0,2}m ?/ ?A\b|\bK_?t\b"),
    ("winding resistance", r"resistance|\bohms?\b"),
    ("motor driver efficiency", r"(driver|inverter|motor)[^.|]{0,40}\befficien|\befficien[^.|]{0,40}(driver|motor)"),
    ("idle electronics power", r"(idle|standby)[^.|]{0,60}(power|watt)|(power|watt)[^.|]{0,60}(idle|standby)"),
    ("onboard computer power", r"jetson|\borin\b"),
    ("battery voltage, capacity and current limit", r"batter"),
)
METHODS = {
    "scope": "Seven retained native evaluations of six training attempts at update 2000. The evaluator takes the actor "
        "mean and adds no sampling noise. Each policy has one trial for each probe, so the record measures no "
        "trial-to-trial spread.",
    "inputs": "Take each batch index from the position of the case in the allocation's selected_case_ids. Require that "
        "the batch report names the case and the checkpoint and holds the hash of force_metrics.json, that "
        "force_metrics.json holds the hash of capture.json, and that capture.json holds the hash of each substep file. "
        "files_sha256 is the sha256 of the compact JSON map from file path to sha256, with sorted keys. Each forward "
        "window must reproduce the RMS torque of each joint and the mean total support in force_metrics.json.",
    "windows": "Each capture holds one row for each 400 Hz physics step, and row i ends at (i + 1) / 400 s. Each "
        "window starts at the summary's window_start_control times 8 and ends with the trial: 2 to 20 s of the forward "
        "probe, 4 to 20 s of the quiet probe and 10 to 21 s of the stop probe, whose command drops to zero at 8 s.",
    "power": "p_j = tau_j * omega_j, with tau the applied torque and omega the recorded joint velocity after the "
        "physics step. Positive power is sum_j max(p_j, 0), signed power is sum_j p_j and absolute power is sum_j "
        "|p_j|. The braking share is mean(sum_j max(-p_j, 0)) / mean(sum_j |p_j|). Each mean covers the 400 Hz rows of "
        "the window. positive_peak_w is the largest single row and positive_peak_20ms_w is the largest mean over one "
        "control step of 8 rows. Positive power gives the braking joints no credit for returned energy.",
    "work_rate_check": "The same block with omega = interval_angle_rate_rad_s, the joint angle change over one step "
        "divided by 0.0025 s. The applied torque stays constant over one step, so tau times this rate is the motor "
        "work in the step divided by the step time. The recorded velocity after the step differs from this rate row by "
        "row, so the record keeps both blocks.",
    "torque": "RMS is sqrt(mean(tau^2)) and peak is max(|tau|) over the window, for each joint and pooled over the six "
        "legs for each joint type. torque_sum_mean_square_n2m2 is sum_j mean(tau_j^2) over the 18 joints. Winding heat "
        "is this sum divided by the square of one fixed motor constant in N m per square root watt. cap_share is the "
        "share of rows with |tau| >= 1.6 N m - 1e-6, pooled over the six legs. peak_requested_nm is the largest "
        "request before the cap.",
    "speed": "Forward speed is the root displacement from the last row before the window to the last row of the trial, "
        "projected on the forward axis (sin yaw, -cos yaw) at the window start, divided by the window time. Navigation "
        "forward is world -Y at zero yaw, and yaw = atan2(2(wz+xy), 1-2(yy+zz)) from the XYZW quaternion.",
    "transport": "Energy per metre is mean positive power / forward speed in J/m. Cost of transport is mean positive "
        "power / (m g v), with m from mass_kg in robot/active_model.json and g = 9.81 m/s^2 from locomotion/env.py. A "
        "window with a zero command has no value for these two fields.",
    "feet": "distal_force_world_n holds the normal contact force on each foot in the world frame, and the capture "
        "records no friction force. The normal force is the +Z component. Stance rows have a normal force above 1 N, "
        "the threshold in locomotion/force_metrics.py. Total support sums the +Z force over the six feet and the "
        "nonfoot contact groups, and support_over_weight divides its mean by m g.",
    "series_50hz": "Each point is the mean over the 8 rows of one control step at the end time of the step, rounded to "
        "4 significant digits. Torque RMS for a joint type is sqrt(mean(tau^2)) over the six legs and the 8 rows.",
    "battery_inputs": "The script scans ARCHITECTURE.md, locomotion/env.py, locomotion/env_config.py and each .md, "
        ".json, .py and .urdf file under robot/ and docs/. Each recorded value must appear in its file as quoted. Each "
        "absent quantity lists the text around each match of its search pattern.",
    "limits": "The physics is the simulated robot with the provisional motor model, and the record holds no hardware "
        "measurement. Mechanical joint power is no electrical power: a motor that holds torque at zero speed draws "
        "current and makes heat with zero mechanical power. The record adds no winding loss, driver loss, computer "
        "load or sensor load. Each forward probe commands 0.05 m/s on flat ground with no payload.",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def fetch(remote, names, destination):
    """Copy the missing named files from one Spark directory through a tar stream; a null remote copies nothing."""
    missing = [name for name in names if not (destination / name).exists()]
    if not missing or remote is None:
        return
    destination.mkdir(parents=True, exist_ok=True)
    command = "tar -C " + shlex.quote(remote) + " -cf - " + " ".join(map(shlex.quote, missing))
    process = subprocess.Popen(["ssh", "spark", command], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        archive.extractall(destination, filter="data")
    assert process.wait() == 0


def measure(trace, first, mass):
    """Measure torque, mechanical power, speed and foot load from row first to the end of the trial."""
    tau = trace["applied_torque_nm"][first:, 0].astype(np.float64)
    seconds, square, moving = len(tau) / PHYSICS_HZ, tau ** 2, bool(trace["command"][first:].any())
    pose = trace["root_pose_xyzw"][[first - 1, -1], 0].astype(np.float64)
    x, y, z, w = pose[:, 3:].T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    step = pose[1, :2] - pose[0, :2]
    speed = float(step @ (np.sin(yaw[0]), -np.cos(yaw[0]))) / seconds
    named = lambda names, values: dict(zip(names, map(float, values)))

    def power(rate):
        joint = tau * rate[first:, 0]
        positive, absolute = np.maximum(joint, 0).sum(axis=1), np.abs(joint).sum(axis=1)
        return {"positive_mean_w": float(positive.mean()), "positive_peak_w": float(positive.max()),
                "positive_peak_20ms_w": float(positive.reshape(-1, DECIMATION).mean(axis=1).max()),
                "signed_mean_w": float(joint.sum(axis=1).mean()), "absolute_mean_w": float(absolute.mean()),
                "braking_share": float((absolute - positive).mean() / absolute.mean()),
                "energy_per_metre_j_m": float(positive.mean()) / speed if moving else None,
                "cost_of_transport": float(positive.mean()) / (mass * GRAVITY * speed) if moving else None}

    normal = trace["distal_force_world_n"][first:, 0, :, 2]
    support = float((normal.sum(axis=1) + trace["nonfoot_force_world_n"][first:, 0, :, 2].sum(axis=1)).mean())
    feet = {leg: {"mean_n": float(force.mean()), "peak_n": float(force.max()),
                  "stance_mean_n": float(force[force > STANCE_N].mean()) if (force > STANCE_N).any() else None,
                  "stance_share": float((force > STANCE_N).mean())} for leg, force in zip(LEGS, normal.T)}
    return {"first_row": first, "rows": len(tau), "start_s": first / PHYSICS_HZ, "seconds": seconds,
            "mechanical_power": power(trace["joint_velocity_rad_s"]),
            "work_rate_check": power(trace["interval_angle_rate_rad_s"]),
            "torque_rms_nm": named(JOINTS, np.sqrt(square.reshape(-1, 6, 3).mean(axis=(0, 1)))),
            "torque_peak_nm": named(JOINTS, np.abs(tau).reshape(-1, 6, 3).max(axis=(0, 1))),
            "torque_sum_mean_square_n2m2": float(square.mean(axis=0).sum()),
            "cap_share": named(JOINTS, (np.abs(tau) >= CAP_NM - 1e-6).reshape(-1, 6, 3).mean(axis=(0, 1))),
            "peak_requested_nm": float(np.abs(trace["computed_torque_nm"][first:]).max()),
            "forward_speed_mps": speed, "net_displacement_m": float(np.linalg.norm(step)),
            "heading_change_rad": float(yaw[1] - yaw[0]),
            "support_mean_n": support, "support_over_weight": support / (mass * GRAVITY),
            "foot_normal_force_n": feet, "torque_rms_by_joint_nm": named(NAMES, np.sqrt(square.mean(axis=0))),
            "torque_peak_by_joint_nm": named(NAMES, np.abs(tau).max(axis=0))}


def series(trace):
    """Average each control step of 8 rows over the complete trial for a plot."""
    tau = trace["applied_torque_nm"][:, 0].astype(np.float64)
    joint = tau * trace["joint_velocity_rad_s"][:, 0]
    mean = lambda values: values.reshape(-1, DECIMATION, *values.shape[1:]).mean(axis=1)
    rounded = lambda values: [float("%.4g" % value) for value in values]
    square, normal = mean(tau ** 2), mean(trace["distal_force_world_n"][:, 0, :, 2])
    return {"time_s": rounded(trace["time_s"][DECIMATION - 1::DECIMATION]),
            "positive_power_w": rounded(mean(np.maximum(joint, 0).sum(axis=1))),
            "signed_power_w": rounded(mean(joint.sum(axis=1))),
            "torque_sum_square_n2m2": rounded(square.sum(axis=1)),
            "torque_rms_nm": dict(zip(JOINTS, map(rounded, np.sqrt(square.reshape(-1, 6, 3).mean(axis=1)).T))),
            "foot_normal_force_n": dict(zip(LEGS, map(rounded, normal.T)))}


def analyze(name, pin, workspace, do_fetch, mass):
    """Check one evaluation's records, then measure its cases and plot the passing forward and stop cases."""
    remote, local = REMOTE + name if do_fetch else None, workspace / "inputs" / name
    files = ["binding.json", EVALUATION + "allocation.json", EVALUATION + "summary.json"]
    fetch(remote, files, local)
    binding, allocation, summary = (read(local / item) for item in files)
    option = lambda flag: binding["command_args"][binding["command_args"].index(flag) + 1]
    attempt, checkpoint = name.split("_evaluate")[0], option("--checkpoint-sha256")
    stored = "/run/standing/checkpoint_update%06d.pt" % UPDATE
    assert int(option("--updates")) == UPDATE and int(option("--seed")) == allocation["seed"]
    assert option("--checkpoint").endswith(Path(stored).name) and summary["allocation"] == allocation
    assert binding["input_files"][REMOTE + attempt + stored] == checkpoint == allocation["checkpoint_sha256"]
    results, measured, plots = {row["case_id"]: row for row in summary["results"]}, {}, {}
    for case in CASES.get(name, (FORWARD,)):
        batch = EVALUATION + "batch_%03d/" % allocation["selected_case_ids"].index(case)
        heads = [batch + "report.json", batch + "force_metrics.json", batch + "native400hz/capture.json"]
        fetch(remote, heads, local)
        report, recorded, capture = (read(local / item) for item in heads)
        substeps = [batch + "native400hz/" + item for item in capture["substep_files"]]
        fetch(remote, substeps, local)
        result, (moving_until, start) = results[case], WINDOWS[case]
        assert report["assigned_case_ids"] == [case] and report["checkpoint_sha256"] == checkpoint
        assert report["results"] == [result] and report["failure"] is None and capture["failure"] is None
        assert report["files"]["force_metrics.json"] == sha(local / heads[1]) and capture["joint_names"] == NAMES
        assert recorded["source_files"]["native400hz/capture.json"] == sha(local / heads[2])
        assert all(capture["files"][Path(item).name] == sha(local / item) for item in substeps)
        trace = {key: np.concatenate([np.load(local / item)[key] for item in substeps]) for key in ARRAYS}
        rows, first = len(trace["time_s"]), result["window_start_control"] * DECIMATION
        assert rows == capture["steps"] == report["recorded_physics_steps"] == result["recorded_controls"] * DECIMATION
        assert first == start * PHYSICS_HZ and first + result["window_controls"] * DECIMATION == rows
        assert trace["applied_torque_nm"].shape[1:] == (1, 18) and trace["distal_force_world_n"].shape[1:] == (1, 6, 3)
        assert all(len(trace[key]) == rows and np.isfinite(trace[key]).all() for key in ARRAYS)
        assert np.allclose(trace["time_s"], np.arange(1, rows + 1) / PHYSICS_HZ, rtol=0, atol=1e-9)
        expected = 0.05 * (np.arange(1, rows + 1) <= moving_until * PHYSICS_HZ)
        assert np.allclose(trace["command"][:, 0], expected[:, None] * [1, 0, 0], rtol=0, atol=1e-6)
        assert np.abs(trace["applied_torque_nm"]).max() <= CAP_NM + 1e-5
        row = measure(trace, first, mass)
        if case == FORWARD:  # The evaluator's own force metrics must agree on this window.
            window = recorded["cases"][0]["windows"]["commanded_locomotion_after_settle"]
            theirs = [window["motor_torque"]["applied"]["per_joint_abs_nm"][item]["rms"] for item in NAMES]
            assert recorded["cases"][0]["case_id"] == case and window["samples"] == row["rows"]
            assert np.allclose(theirs, list(row["torque_rms_by_joint_nm"].values()), rtol=1e-9, atol=0)
            assert np.isclose(window["total_vertical_support_n"]["mean"], row["support_mean_n"], rtol=1e-9, atol=0)
        measured[case] = {"batch": batch.split("/")[-2], "screen_pass": result["pass"],
                          "screen_failed_bounds": result["failed_bounds"],
                          "screen_mean_velocity_mps": result["metrics"]["mean_velocity_mps"], **row}
        if attempt == PASSING and case != QUIET:
            plots[case] = series(trace)
        files += heads + substeps
    hashes = {item: sha(local / item) for item in files}
    aggregate = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert aggregate == pin, (name, aggregate)
    return {"evaluation": name, "training_attempt": attempt, "remote_directory": REMOTE + name, "update": UPDATE,
            "seed": allocation["seed"], "checkpoint_sha256": checkpoint, "files_sha256": aggregate,
            "sha256": hashes, "cases": measured}, plots


def battery():
    """Check each recorded constant against its file and list each mention of an absent constant."""
    paths = [REPOSITORY / item for item in ("ARCHITECTURE.md", ENV, "locomotion/env_config.py")]
    paths += sorted(path for folder in ("robot", "docs") for path in (REPOSITORY / folder).rglob("*")
                    if path.suffix in (".md", ".json", ".py", ".urdf"))
    texts = {str(path.relative_to(REPOSITORY)): path.read_text() for path in paths}
    assert all(quote in texts[name] for _, _, name, quote in RECORDED)
    absent = [{"quantity": quantity, "search_pattern": pattern,
               "mentions": [{"file": name, "line": number, "text": line[max(found.start() - 90, 0):found.end() + 90]}
                            for name, text in texts.items() for number, line in enumerate(text.splitlines(), 1)
                            if (found := re.search(pattern, line, re.IGNORECASE))]} for quantity, pattern in ABSENT]
    assert not absent[0]["mentions"] and not absent[1]["mentions"]
    return {"files_scanned": len(texts),
            "source_sha256": {name: sha(REPOSITORY / name) for name in sorted({row[2] for row in RECORDED})},
            "recorded": [dict(zip(("quantity", "value", "file", "field"), row)) for row in RECORDED],
            "absent": absent, "electrical_power_estimate": None,
            "electrical_power_reason": "No scanned file records a torque constant or a winding resistance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    output = args.output or args.workspace / "power_loads.json"
    if output.exists():
        raise FileExistsError("Use a fresh output path")
    mass = read(REPOSITORY / "robot/active_model.json")["mass_kg"]
    pairs = [analyze(name, pin, args.workspace, args.fetch, mass) for name, pin in PINS.items()]
    rows, plots = [row for row, _ in pairs], {case: plot for _, group in pairs for case, plot in group.items()}
    receipt = {"schema": "hexapod_ppo_power_loads_v1", "analysis_sha256": sha(__file__),
               "passing_training_attempt": PASSING, "cases": {"forward": FORWARD, "quiet_20s": QUIET, "stop": STOP},
               "constants": {"mass_kg": mass, "gravity_m_s2": GRAVITY, "weight_n": mass * GRAVITY,
                             "torque_cap_nm": CAP_NM, "physics_hz": PHYSICS_HZ, "stance_threshold_n": STANCE_N},
               "methods": METHODS, "evaluations": rows, "battery_inputs": battery(),
               "native_started_by_analysis": False}
    # The series stays on one line so that the file stays under 400 KB.
    text = json.dumps(receipt, indent=2, allow_nan=False)[:-2] + ',\n  "series_50hz": '
    output.write_text(text + json.dumps(plots, separators=(",", ":"), allow_nan=False) + "\n}\n")
    assert read(output)["series_50hz"] == plots and output.stat().st_size < 400_000
    print(json.dumps([{"attempt": row["training_attempt"], "case": case, **item["mechanical_power"]}
                      for row in rows for case, item in row["cases"].items()]))


if __name__ == "__main__":
    main()
