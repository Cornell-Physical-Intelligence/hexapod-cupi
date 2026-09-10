#!/usr/bin/env python3
"""Fetch completed motor probes and compare matched pre-reset measurements.

Read-only Spark access. No container, service, GPU, or training actions occur.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE = "/home/orionh/HEXAPOD_runs/mock_length_study_20260909/omni_diagnostics_001"


def fetch_completed(destination, remote, host):
    destination.mkdir(parents=True, exist_ok=True)
    # Remote paths come from our campaign, not shell interpolation of arbitrary
    # report content. scp is used only for the expected read-only file paths.
    if not remote.startswith("/home/orionh/HEXAPOD_runs/") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/_-." for c in remote):
        raise ValueError("Unexpected remote campaign path")
    campaign_file = destination / "diagnostic_campaign.json"
    subprocess.run(["scp", "-q", "-o", "ConnectTimeout=10", f"{host}:{remote}/diagnostic_campaign.json", str(campaign_file)], check=True)
    campaign = json.loads(campaign_file.read_text())
    fetched = []
    for index, entry in enumerate(campaign["comparisons"]):
        if entry["status"] != "completed":
            continue
        folder = destination / f"comparison_{index:02d}"
        folder.mkdir(exist_ok=True)
        base = f"{remote}/comparison_{index:02d}/f050_t060/evaluate"
        for name in ("diagnostics.json", "diagnostic_trace.npz", "state.json", "environment.yaml"):
            if not (folder / name).exists():
                subprocess.run(["scp", "-q", "-o", "ConnectTimeout=10", f"{host}:{base}/{name}", str(folder / name)], check=True)
        report = json.loads((folder / "diagnostics.json").read_text())
        state = json.loads((folder / "state.json").read_text())
        if not report["complete"] or state["status"] != "completed":
            raise ValueError("Only completed diagnostic results can be compared")
        if not report["checkpoint_sha256"] == state["checkpoint_sha256"] == campaign["checkpoint_sha256"]:
            raise ValueError("Checkpoint identity mismatch")
        fetched.append(folder.name)
    return {"status": campaign["status"], "completed": fetched}


def label(report):
    overrides = report["overrides"]
    if overrides.get("external_forces_every_iteration"):
        return "External forces each iteration"
    if "target_slew_rad_per_20ms" in overrides:
        return f"Slew {overrides['target_slew_rad_per_20ms']:.2f} rad/20 ms"
    return "Original controller"


def longest_run(mask):
    indices = np.where(mask)[0]
    if not len(indices):
        return indices
    runs = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    return max(runs, key=len)


def spectrum(signal, dt):
    if len(signal) < 50:
        return {"samples": len(signal), "dominant_frequency_hz": None, "power_above_5hz_fraction": None}
    centered = np.asarray(signal) - np.mean(signal)
    power = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
    frequencies = np.fft.rfftfreq(len(centered), dt)
    power[0] = 0
    return {"samples": len(signal), "dominant_frequency_hz": float(frequencies[np.argmax(power)]),
            "power_above_5hz_fraction": float(power[frequencies > 5].sum() / max(power.sum(), 1e-30))}


def trace_analysis(trace, report):
    data = np.load(trace, allow_pickle=False)
    steady = data["age_s"][:, 0] >= report["options"]["settle_s"]
    stable = longest_run(steady & ~data["terminated"][:, 0])
    if not len(stable):
        return {"stand_trace_steady_samples": 0}
    computed = data["computed_torque_nm"][stable, 0]
    scores = (np.abs(computed) > 1.6).mean(0)
    worst = int(np.argmax(scores))
    gyro = data["gyro_navigation_rad_s"][stable, 0, 2]
    fd = data["finite_difference_heading_rate_rad_s"][stable, 0]
    return {"stand_trace_steady_samples": len(stable), "stand_trace_duration_s": len(stable) * report["control_sample_dt_s"],
            "worst_stand_joint": str(data["joint_names"][worst]),
            "worst_stand_joint_saturation_fraction": float(scores[worst]),
            "gyro_z_spectrum": spectrum(gyro, report["control_sample_dt_s"]),
            "fd_heading_spectrum": spectrum(fd, report["control_sample_dt_s"]),
            "computed_torque_spectrum": spectrum(computed[:, worst], report["control_sample_dt_s"]),
            "gyro_z_std_rad_s": float(gyro.std()), "fd_heading_rate_std_rad_s": float(fd.std()),
            "gyro_z_minus_fd_heading_rms_rad_s": float(np.sqrt(np.mean((gyro - fd)**2)))}


def direction_smoothness(trace, report):
    """One recorded replica per direction; no stitching across resets."""
    data = np.load(trace, allow_pickle=False)
    dt = report["control_sample_dt_s"]
    trace_replicas = report["options"]["trace_envs_per_scenario"]
    result = []
    for scenario_index, scenario in enumerate(report["scenarios"]):
        env = scenario_index * trace_replicas
        valid = (data["age_s"][:, env] >= report["options"]["settle_s"]) & ~data["terminated"][:, env]
        steps = longest_run(valid)
        row = {"scenario": scenario["name"], "recorded_replica": 0,
               "samples": len(steps), "duration_s": len(steps) * dt,
               "all_replica_terminations": scenario["terminations"]}
        if len(steps) < 2:
            result.append(row)
            continue
        q = data["joint_position_rad"][steps, env]
        target = data["joint_target_rad"][steps, env]
        velocity = data["joint_velocity_rad_s"][steps, env]
        target_step = np.diff(target, axis=0)
        fd_joint_velocity = np.diff(q, axis=0) / dt
        fd_joint_acceleration = np.diff(q, n=2, axis=0) / dt**2
        computed = data["computed_torque_nm"][steps, env]
        applied = data["applied_torque_nm"][steps, env]
        position = data["position_world_m"][steps, env]
        quat = data["quaternion_world_wxyz"][steps, env]
        w, x, y, z = quat.T
        heading = np.unwrap(np.arctan2(-1 + 2*(x*x+z*z), 2*(w*z-x*y)))
        row.update(
            joint_velocity_rms_rad_s=float(np.sqrt(np.mean(velocity**2))),
            joint_velocity_abs_p95_rad_s=float(np.quantile(np.abs(velocity), .95)),
            fd_joint_velocity_rms_rad_s=float(np.sqrt(np.mean(fd_joint_velocity**2))),
            fd_joint_acceleration_rms_rad_s2=float(np.sqrt(np.mean(fd_joint_acceleration**2))),
            joint_position_range_max_rad=float(np.ptp(q, axis=0).max()),
            target_step_rms_rad=float(np.sqrt(np.mean(target_step**2))),
            target_step_abs_p95_rad=float(np.quantile(np.abs(target_step), .95)),
            target_total_variation_mean_rad_s=float(np.abs(target_step).sum(0).mean() / ((len(q)-1)*dt)),
            applied_torque_step_rms_nm=float(np.sqrt(np.mean(np.diff(applied, axis=0)**2))),
            requested_torque_abs_p95_nm=float(np.quantile(np.abs(computed), .95)),
            requested_torque_abs_max_nm=float(np.abs(computed).max()),
            planar_excursion_max_m=float(np.linalg.norm(position[:, :2] - position[0, :2], axis=1).max()),
            heading_excursion_max_deg=float(np.degrees(np.abs(heading - heading[0]).max())),
            vertical_position_range_m=float(np.ptp(position[:, 2])),
            joints={str(name): {"velocity_rms_rad_s": float(np.sqrt(np.mean(velocity[:, j]**2))),
                               "position_range_rad": float(np.ptp(q[:, j])),
                               "target_step_abs_p95_rad": float(np.quantile(np.abs(target_step[:, j]), .95)),
                               "saturation_fraction": float((np.abs(computed[:, j]) > 1.6).mean())}
                    for j, name in enumerate(data["joint_names"])})
        result.append(row)
    return result


def summarize(folder):
    report = json.loads((folder / "diagnostics.json").read_text())
    windows = [row["windows"]["post_settle"] for row in report["scenarios"]]
    valid = [window for window in windows if window["samples"]]
    return {
        "comparison": folder.name, "label": label(report), "overrides": report["overrides"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "diagnostics_sha256": hashlib.sha256((folder / "diagnostics.json").read_bytes()).hexdigest(),
        "trace_sha256": hashlib.sha256((folder / "diagnostic_trace.npz").read_bytes()).hexdigest(),
        "terminations": sum(row["terminations"] for row in report["scenarios"]),
        "truncations": sum(row["truncations"] for row in report["scenarios"]),
        "termination_reasons": {key: sum(row["termination_reasons"][key] for row in report["scenarios"])
                                for key in report["scenarios"][0]["termination_reasons"]},
        "scenarios_with_post_settle_samples": len(valid),
        "median_post_settle": {key: float(np.median([row[key] for row in valid])) for key in
                               ("torque_saturation_fraction", "planar_error_mps", "yaw_error_rad_s",
                                "positive_mechanical_power_w", "tilt_rms_deg")},
        "max_applied_torque_nm": max(row["applied_torque_abs_max_nm"] for row in valid),
        "max_post_settle_computed_torque_nm": max(row["computed_torque_abs_max_nm"] for row in valid),
        "observation_audit": report["observation_audit"],
        "stand": report["scenarios"][0],
        "stand_trace": trace_analysis(folder / "diagnostic_trace.npz", report),
        "direction_smoothness": direction_smoothness(folder / "diagnostic_trace.npz", report),
    }, report


def plots(folders, reports, summaries, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.5), layout="constrained")
    metrics = [("torque_saturation_fraction", "Requested torque > 1.6 Nm (%)", 100),
               ("planar_error_mps", "Planar tracking error (m/s)", 1),
               ("yaw_error_rad_s", "Yaw tracking error (rad/s)", 1),
               ("positive_mechanical_power_w", "Positive mechanical power (W)", 1)]
    x = np.arange(len(reports[0]["scenarios"]))
    for ax, (key, title, scale) in zip(axes.flat, metrics):
        for i, report in enumerate(reports):
            values = [row["windows"]["post_settle"].get(key, np.nan) * scale for row in report["scenarios"]]
            ax.plot(x, values, marker="o", markersize=4, label=label(report))
        ax.set(title=title, xticks=x, xticklabels=[row["name"].replace("_", " ") for row in reports[0]["scenarios"]])
        ax.tick_params(axis="x", rotation=38)
        ax.grid(alpha=.2)
    axes[0, 0].axhline(.5, color="black", linestyle="--", alpha=.5, label="Qualification saturation limit")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Matched 12-second diagnostic probes · post-settle per episode · no qualification claim", fontsize=16)
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)

    # Use the baseline's worst stand joint for every comparison, by name.
    joint_name = summaries[0]["stand_trace"].get("worst_stand_joint")
    if joint_name is None:
        return
    fig, axes = plt.subplots(3, len(folders), figsize=(6 * len(folders), 9), squeeze=False, layout="constrained")
    for i, (folder, report) in enumerate(zip(folders, reports)):
        data = np.load(folder / "diagnostic_trace.npz", allow_pickle=False)
        j = list(data["joint_names"]).index(joint_name)
        t = data["time_s"]
        chosen = (t >= 4) & (t <= 6) & (data["age_s"][:, 0] >= 2)
        axes[0, i].plot(t[chosen], data["computed_torque_nm"][chosen, 0, j], label="Requested", lw=1)
        axes[0, i].plot(t[chosen], data["applied_torque_nm"][chosen, 0, j], label="Applied", lw=1)
        axes[0, i].axhline(1.6, color="black", linestyle="--", alpha=.5)
        axes[0, i].axhline(-1.6, color="black", linestyle="--", alpha=.5)
        axes[0, i].set_title(label(report))
        axes[0, i].set_ylabel(f"Torque (Nm)\n{joint_name}")
        axes[1, i].plot(t[chosen], data["joint_target_rad"][chosen, 0, j], label="Target", lw=1)
        axes[1, i].plot(t[chosen], data["joint_position_rad"][chosen, 0, j], label="Position", lw=1)
        axes[1, i].set_ylabel("Joint angle (rad)")
        axes[2, i].plot(t[chosen], data["gyro_navigation_rad_s"][chosen, 0, 2], label="Reported gyro z", lw=1)
        axes[2, i].plot(t[chosen], data["finite_difference_heading_rate_rad_s"][chosen, 0], label="Pose difference", lw=1)
        axes[2, i].set_ylabel("Angular velocity (rad/s)")
        axes[2, i].set_xlabel("Simulation time (s)")
        for ax in axes[:, i]:
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
    fig.suptitle("Standing: same policy, same named joint, separate controlled probes", fontsize=16)
    fig.savefig(output / "standing_motor_trace.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "artifacts/omni_diagnostics_2026-09-09")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--host", default="spark")
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        print(json.dumps(fetch_completed(args.directory, args.remote, args.host)))
    folders = sorted(folder for folder in args.directory.glob("comparison_*")
                     if (folder / "diagnostics.json").exists() and (folder / "diagnostic_trace.npz").exists())
    if not folders:
        print("No completed diagnostic probes available yet.")
        return
    records = [summarize(folder) for folder in folders]
    summaries = [record[0] for record in records]
    reports = [record[1] for record in records]
    if len({report["checkpoint_sha256"] for report in reports}) != 1:
        raise ValueError("Comparisons must use the same checkpoint")
    report = {"kind": "controlled_diagnostics_not_qualification", "comparisons": summaries,
              "completion_contract": {
                  "user_requirement": "Stage 2 is incomplete until quiet standing and every direction/path are comparably smooth to the accepted forward benchmark and current straight trial.",
                  "unchanged_existing_gates": "All held-out direction, yaw, combined-command and continuous-transition tracking/contact/torque gates remain required.",
                  "visual_review_required": ["all 16 translation bearings including reverse, strafe and diagonals", "both pure yaw signs", "gentle and tight arcs in both directions", "S curves and fixed-heading curves", "start, reversal, stop and quiet standing"],
                  "proposed_quiet_stand_gates": {"scope": "undisturbed 30 s stand after 2 s settling; disturbances evaluated separately", "max_planar_excursion_m": .01,
                      "max_heading_excursion_deg": 2., "max_joint_velocity_rms_rad_s": .03,
                      "max_joint_position_range_rad": .02, "max_target_step_abs_p95_rad_per_20ms": .002,
                      "max_requested_torque_saturation_fraction": .005, "terminations": 0},
                  "stop_to_stand_proposal": "After commanded deceleration reaches zero, enter quiet-stand bounds within 2 s and sustain for 10 s; recover from disturbances without rigid freezing.",
                  "smoothness_quantification": "Compare per-direction joint motion, target increments, applied torque increments and body pose jitter against instrumented forward reference at comparable motion demand; no aggregate median can waive a failed direction.",
                  "limitation": "New quiet-stand thresholds are proposed engineering acceptance bounds, not hardware-calibrated or user-approved test standards. Forward reference needs matched instrumentation before relative smoothness thresholds can be certified."},
              "notes": ["Same held checkpoint; each probe has its own full standing admission.",
                        "Window thresholds restart after every reset; terminal samples are retained in post-settle windows.",
                        "Spectrum uses longest uninterrupted post-settle stand trace, not stitched reset segments.",
                        "Finite difference averages motion across a control step; gyro samples its endpoint.",
                        "50 Hz control traces cannot resolve frequencies above 25 Hz.",
                        "Scenario medians do not establish a pass; full direction/transition gates remain unchanged."]}
    (args.directory / "comparison_report.json").write_text(json.dumps(report, indent=2) + "\n")
    plots(folders, reports, summaries, args.directory)
    lines = ["# Omnidirectional PPO motor diagnostics", "", "These short controlled probes diagnose the previous policy; they are not qualification runs.", "",
             "| Probe | Terminations | Median saturation | Planar error | Yaw error | Positive power |",
             "|---|---:|---:|---:|---:|---:|"]
    for summary in summaries:
        m = summary["median_post_settle"]
        lines.append(f"| {summary['label']} | {summary['terminations']} | {100*m['torque_saturation_fraction']:.2f}% | {m['planar_error_mps']:.4f} m/s | {m['yaw_error_rad_s']:.4f} rad/s | {m['positive_mechanical_power_w']:.3f} W |")
    lines += ["", "![Scenario comparison](comparison.png)", "", "![Standing trace](standing_motor_trace.png)", "",
              "Raw reports and compressed traces retain runtime joint names, checkpoint hashes, per-episode age, all termination causes and requested versus applied torque.", "",
              "## Completion requires smooth movement and quiet standing", "",
              "The accepted forward animation and current straight trial are visual references. Every bearing, reverse/strafe/diagonal, both yaw signs, both arc directions, path bends, reversals and stops require review. Passing a median or a forward clip cannot complete Stage 2.", "",
              "The JSON contains direction-by-direction joint velocity, target increments, pose excursion and applied torque increments. Quiet-stand acceptance bounds are proposed there for a separate 30-second undisturbed test, plus stop-to-stand and disturbance recovery; this 12-second probe cannot establish those gates.", ""]
    (args.directory / "README.md").write_text("\n".join(lines))
    print(json.dumps({"report": str(args.directory / "comparison_report.json"), "completed_probes": len(summaries)}))


if __name__ == "__main__":
    main()
