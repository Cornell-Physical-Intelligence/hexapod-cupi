"""CPU inspection of actual native policy recordings; never changes a verdict.

Usage: python -m experiments.paper_walk.analyze --directory BATCH_OR_SUITE
       --output FRESH_DIRECTORY [--checkpoint POLICY.pt] [--compare analysis.json]
No simulator, remote execution, invented missing channels, or artifact imports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "canonical_native_policy_analysis_v1"
DT, NATIVE_DT = .02, .0025
PHYSICS_KEYS = ("physics_source_files", "physics_config", "prior_metadata_sha256",
                "model_sha256", "usd_sha256", "geometry_sha256", "geometry_extrema_sha256")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verified_files(root, manifest):
    checked = {}
    for name, digest in manifest.items():
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("Missing or unsafe recorded file: " + str(path))
        actual = sha(path)
        if actual != digest:
            raise ValueError("Recorded file hash differs: " + str(path))
        checked[name] = actual
    return checked


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key].copy() for key in data.files}
    for key, value in result.items():
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError("Nonfinite/non-numeric recorded channel: " + key)
    return result


def load_recording(directory):
    root = Path(directory)
    report = read(root/"report.json")
    verified = verified_files(root, report.get("files", {}))
    control_path = root/"control_trace.npz"
    controls = load_arrays(control_path) if control_path.is_file() else {}
    if controls and "control_trace.npz" not in verified:
        raise ValueError("Control trace is not bound by the recording report")
    if report.get("controls", 0) and not controls:
        raise ValueError("Recorded controls are missing")
    if controls:
        count = len(controls["time_s"])
        if count != report["controls"] or any(len(x) != count for x in controls.values()):
            raise ValueError("Control channel length differs from recording")
        if count > 1 and not np.allclose(np.diff(controls["time_s"]), DT, atol=1e-9, rtol=0):
            raise ValueError("Control timeline is not contiguous50Hz")
    capture_path = root/"native400hz"/"capture.json"
    capture, native = None, {}
    if capture_path.is_file():
        capture = read(capture_path)
        checked = verified_files(capture_path.parent, capture.get("files", {}))
        chunks = []
        for name in capture["substep_files"]:
            if name not in checked:
                raise ValueError("Native chunk lacks a recorded hash: " + name)
            chunks.append(load_arrays(capture_path.parent/name))
        if chunks:
            if any(set(x) != set(chunks[0]) for x in chunks):
                raise ValueError("Native channel schema changes within an attempt")
            native = {k: np.concatenate([x[k] for x in chunks]) for k in chunks[0]}
            count = len(native["sequence"])
            if count != capture["steps"] or not np.array_equal(native["sequence"], np.arange(count)):
                raise ValueError("Native sample count/sequence differs")
            if any(len(x) != count for x in native.values()):
                raise ValueError("Native channel lengths differ")
            if not np.array_equal(native["substep_index"], np.arange(count)%8):
                raise ValueError("Native hold/substep ordering differs")
            if count > 1 and not np.all(np.diff(native["explicit_counter"]) == 1):
                raise ValueError("Native physics counter is not contiguous")
        elif capture["steps"]:
            raise ValueError("Nonempty native acquisition lacks chunks")
        verified.update({"native400hz/"+k: v for k, v in checked.items()})
    binding = None
    for parent in list(root.parents)[:4]:
        state_path = parent/"state.json"
        if state_path.is_file():
            identity = read(state_path).get("identity", {})
            if all(key in identity for key in PHYSICS_KEYS):
                binding = {key: identity[key] for key in PHYSICS_KEYS}
                if binding["model_sha256"] != report.get("model_sha256") or binding["usd_sha256"] != report.get("usd_sha256"):
                    raise ValueError("Acquisition and recording model identities differ")
            break
    integrity = {"report_sha256": sha(root/"report.json"), "verified_files": verified,
        "capture_manifest_sha256": sha(capture_path) if capture_path.is_file() else None,
        "capture_manifest_bound_by_parent_report": "native400hz/capture.json" in verified,
        "scope": "Listed file bytes verified against saved manifests; manifests are not digital signatures."}
    return report, capture, controls, native, integrity, binding


def spectrum(values, dt):
    values = np.asarray(values, dtype=float)
    if len(values) < 8:
        return {"status": "missing", "reason": "Fewer than8 contiguous samples"}
    values = values.reshape(len(values), -1)
    centered = values-values.mean(0)
    power = np.square(np.abs(np.fft.rfft(centered*np.hanning(len(values))[:, None], axis=0))).mean(-1)
    frequency = np.fft.rfftfreq(len(values), dt)
    power[0] = 0
    total = float(power.sum())
    return {"status": "measured", "samples": len(values), "sample_rate_hz": 1/dt,
        "ac_rms": float(np.sqrt(np.mean(centered**2))),
        "dominant_frequency_hz": float(frequency[power.argmax()]) if total > 1e-24 else None,
        "power_above_5hz_fraction": float(power[frequency > 5].sum()/total) if total > 1e-24 else None,
        "method": "Full-trial mean removal and Hann-window FFT; no dropped failure or selected quiet suffix."}


def longest_runs(mask, dt):
    mask = np.asarray(mask, bool)
    current = np.zeros(mask.shape[1:], dtype=int)
    best = current.copy()
    for row in mask:
        current = np.where(row, current+1, 0)
        best = np.maximum(best, current)
    return best*dt


def analyze_trace(controls, native, env=0, joint_names=None):
    """All-trial measured diagnostics. Missing channels are explicit, never zeros."""
    missing = []
    def channel(data, name):
        if name not in data:
            missing.append(name)
            return None
        value = data[name]
        return np.asarray(value[:, env], dtype=float) if value.ndim >= 2 else np.asarray(value, dtype=float)
    times = channel(controls, "time_s")
    result = {"controls": len(times) if times is not None else 0, "missing_metrics": missing,
              "scope": "Diagnostics of the complete recorded attempt; no acceptance thresholds or verdicts changed."}
    if times is None or not len(times):
        result["status"] = "no_control_samples"
        return result
    result["status"] = "measured"
    command = channel(controls, "command")
    velocity = channel(controls, "velocity_navigation_mps")
    gyro = channel(controls, "gyro_body_rad_s")
    if command is not None and velocity is not None and gyro is not None:
        error = velocity[:, :2]-command[:, :2]
        yaw_error = gyro[:, 2]-command[:, 2]
        speed = np.linalg.norm(command[:, :2], axis=1)
        projection = np.sum(velocity[:, :2]*command[:, :2], axis=1)/np.maximum(speed, 1e-8)
        result["tracking"] = {"mean_requested_navigation_velocity_mps": command[:, :2].mean(0).tolist(),
            "mean_achieved_navigation_velocity_mps": velocity[:, :2].mean(0).tolist(),
            "mean_signed_velocity_error_mps": error.mean(0).tolist(),
            "mean_absolute_velocity_error_mps": abs(error).mean(0).tolist(),
            "planar_error_rmse_mps": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
            "mean_requested_speed_mps": float(speed.mean()),
            "mean_achieved_speed_mps": float(np.linalg.norm(velocity[:, :2], axis=1).mean()),
            "mean_signed_direction_speed_mps": float(projection[speed > 1e-8].mean()) if np.any(speed > 1e-8) else None,
            "mean_signed_yaw_error_rad_s": float(yaw_error.mean()), "yaw_error_rmse_rad_s": float(np.sqrt(np.mean(yaw_error**2))),
            "measurement_point": "Native COM for exact evaluator velocity_navigation_mps; estimator is compared separately at root origin."}
    root = channel(controls, "root_pose_xyzw")
    if root is not None:
        norm = np.linalg.norm(root[:, 3:], axis=1)
        if np.max(abs(norm-1)) > 2e-5:
            raise ValueError("Invalid measured root quaternion")
        tilt = np.degrees(np.arccos(np.clip(1-2*(root[:, 3]**2+root[:, 4]**2), -1, 1)))
        result["deck"] = {"height_mean_m": float(root[:, 2].mean()), "height_std_m": float(root[:, 2].std()),
            "height_peak_to_peak_m": float(np.ptp(root[:, 2])), "tilt_rms_deg": float(np.sqrt(np.mean(tilt**2))),
            "tilt_peak_deg": float(tilt.max()), "planar_excursion_max_m": float(np.linalg.norm(root[:, :2]-root[0, :2], axis=1).max())}
        result["deck_height_spectrum"] = spectrum(root[:, 2], DT)
    q, dq, target = [channel(controls, k) for k in ("joint_position_rad", "joint_velocity_rad_s", "joint_target_rad")]
    if q is not None and dq is not None and target is not None:
        delta = np.diff(target, axis=0)
        result["joints"] = {"names": joint_names, "position_range_rad": np.ptp(q, axis=0).tolist(),
            "velocity_rms_rad_s": np.sqrt(np.mean(dq**2, axis=0)).tolist(),
            "target_tracking_rms_rad": np.sqrt(np.mean((q-target)**2, axis=0)).tolist(),
            "target_step_abs_max_rad": abs(delta).max(0).tolist() if len(delta) else None,
            "target_step_abs_p95_rad": np.quantile(abs(delta), .95, axis=0).tolist() if len(delta) else None,
            "slew_limit_occupancy_fraction": (abs(delta) >= .04-1e-6).mean(0).tolist() if len(delta) else None}
        result["target_step_spectrum"] = spectrum(delta, DT)
    requested = channel(native, "computed_torque_nm")
    applied = channel(native, "applied_torque_nm")
    if requested is not None and applied is not None:
        over = abs(requested) > 1.6
        result["torque_400hz"] = {"samples": len(applied), "joint_names": joint_names,
            "requested_saturation_fraction": over.mean(0).tolist(),
            "requested_saturation_longest_burst_s": longest_runs(over, NATIVE_DT).tolist(),
            "requested_abs_peak_nm": abs(requested).max(0).tolist(),
            "applied_abs_peak_nm": abs(applied).max(0).tolist(),
            "applied_rms_nm": np.sqrt(np.mean(applied**2, axis=0)).tolist(),
            "first_requested_saturation_s": float((np.flatnonzero(over.any(1))[0]+1)*NATIVE_DT) if over.any() else None}
    support = channel(native, "distal_contact")
    toes = channel(native, "toe_xyz_world_m")
    if support is not None:
        support = support.astype(bool)
        result["support_400hz"] = {"toe_contact_fraction": support.mean(0).tolist(),
            "support_count_histogram": np.bincount(support.sum(1), minlength=7).tolist(),
            "no_toe_support_samples": int((support.sum(1) == 0).sum())}
        if toes is not None and len(toes) > 1:
            contact_pairs = support[1:] & support[:-1]
            speed = np.linalg.norm(np.diff(toes[..., :2], axis=0), axis=-1)/NATIVE_DT
            result["contact_reference_motion"] = {"measurement": "World motion of each toe-link reference point during consecutive actual distal contact samples; a slip proxy, not the contact patch velocity.",
                "eligible_pairs_by_toe": contact_pairs.sum(0).tolist(),
                "mean_planar_speed_mps": [float(speed[:, i][contact_pairs[:, i]].mean()) if contact_pairs[:, i].any() else None for i in range(6)],
                "p95_planar_speed_mps": [float(np.quantile(speed[:, i][contact_pairs[:, i]], .95)) if contact_pairs[:, i].any() else None for i in range(6)]}
    result["failures"] = {}
    for key, data, dt in (("terminated", controls, DT), ("truncated", controls, DT),
                          ("reset", controls, DT), ("nonfoot_contact", native, NATIVE_DT)):
        flags = channel(data, key)
        if flags is not None:
            found = np.flatnonzero(flags.astype(bool))
            result["failures"][key] = {"samples": len(found), "first_time_s": float((found[0]+1)*dt) if len(found) else None,
                "event_start_times_s": ((np.flatnonzero(flags.astype(bool) & ~np.r_[False, flags[:-1].astype(bool)])+1)*dt).tolist()}
    for key in ("joint_velocity_rad_s", "joint_position_rad"):
        value = channel(native, key)
        if value is not None:
            result[key+"_spectrum_400hz"] = spectrum(value, NATIVE_DT)
    estimated = channel(controls, "estimated_velocity_navigation_mps")
    critic = channel(controls, "critic_observation")
    if estimated is not None and critic is not None:
        result["estimator"] = {"root_origin_velocity_rmse_mps": np.sqrt(np.mean((estimated-critic[:, -3:])**2, axis=0)).tolist(),
            "measurement_point": "Pre-policy body-root origin navigation velocity in critic[:, -3:], not COM tracking velocity."}
    for name in ("motion_prior_style_reward", "motion_prior_discriminator_score"):
        value = channel(controls, name)
        if value is not None:
            result[name] = {"mean": float(value.mean()), "min": float(value.min()), "max": float(value.max())}
    result["missing_metrics"] = sorted(set(missing))
    return result


def reconstruct_actor(controls, checkpoint, expected_sha):
    if sha(checkpoint) != expected_sha:
        raise ValueError("Analysis checkpoint differs from actual recording")
    if "policy_observation" not in controls or "policy_action" not in controls:
        return {"status": "missing", "reason": "Recorded231-dimensional pre-policy observations/actions unavailable"}
    import torch
    from . import learner
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("schema") != learner.SCHEMA:
        raise ValueError("Checkpoint is not the canonical PPO/AMP schema")
    if saved.get("learner_sha256") != sha(learner.__file__):
        return {"status": "unavailable", "reason": "Checkpoint learner source differs; no artifact source imported or compatibility assumed"}
    config = learner.Config(**saved["config"])
    model = learner.ActorCritic(config).cpu().eval()
    model.load_state_dict(saved["model"], strict=True)
    obs = controls["policy_observation"].reshape(-1, 231)
    actual = controls["policy_action"].reshape(-1, 18)
    values = []
    with torch.inference_mode():
        for start in range(0, len(obs), 256):
            values.append(model.actor(torch.as_tensor(obs[start:start+256], dtype=torch.float32))[0].numpy())
    predicted = np.concatenate(values)
    difference = predicted-actual
    return {"status": "reconstructed", "checkpoint_sha256": expected_sha, "observations": len(obs),
        "actor_state_dict_strict": True, "absolute_tolerance": 1e-5, "relative_tolerance": 1e-5,
        "within_cpu_float_tolerance": bool(np.allclose(predicted, actual, atol=1e-5, rtol=1e-5)),
        "action_error_abs_max": float(abs(difference).max()), "action_error_rmse": float(np.sqrt(np.mean(difference**2))),
        "scope": "CPU mean-action reconstruction from recorded histories and saved normalizers; no closed-loop or observation-noise trial reproduced."}


def comparison(current, previous):
    if not current.get("comparison_identity") or current.get("comparison_identity") != previous.get("comparison_identity"):
        return {"status": "incomparable", "reason": "Missing or different model, physics, command cases, duration, or measurement-point identity"}
    changes = []
    for a, b in zip(current["recordings"], previous["recordings"]):
        for x, y in zip(a["replicas"], b["replicas"]):
            row = {"case_id": x["case_id"], "metric_changes": {}}
            for group, key in (("tracking", "planar_error_rmse_mps"), ("tracking", "yaw_error_rmse_rad_s"),
                               ("deck", "height_std_m"), ("deck", "tilt_rms_deg")):
                left, right = x.get(group, {}).get(key), y.get(group, {}).get(key)
                if left is not None and right is not None:
                    row["metric_changes"][key] = {"previous": right, "current": left, "difference": left-right}
            changes.append(row)
    return {"status": "matched_identity", "changes": changes,
            "scope": "Measured before/after descriptive differences only; no uncertainty estimate or changed acceptance verdict."}


def plots(controls, native, directory, index):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "unavailable", "reason": "Matplotlib is not installed; no dependency installed by this analysis"}
    if not controls:
        return {"status": "missing", "reason": "No measured control samples"}
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    time = controls["time_s"]
    if "command" in controls and "velocity_navigation_mps" in controls:
        for i, label in enumerate(("forward", "left")):
            axes[0].plot(time, controls["command"][:, 0, i], "--", label=label+" requested")
            axes[0].plot(time, controls["velocity_navigation_mps"][:, 0, i], label=label+" achieved")
        axes[0].set_ylabel("COM velocity (m/s)"); axes[0].legend(ncol=2)
    if "command" in controls and "gyro_body_rad_s" in controls:
        axes[1].plot(time, controls["command"][:, 0, 2], "--", label="requested yaw")
        axes[1].plot(time, controls["gyro_body_rad_s"][:, 0, 2], label="achieved yaw")
        axes[1].set_ylabel("Yaw rate (rad/s)"); axes[1].legend()
    if "root_pose_xyzw" in controls:
        root = controls["root_pose_xyzw"][:, 0]
        axes[2].plot(time, root[:, 2], label="measured root height")
        axes[2].set_ylabel("Height (m)"); axes[2].legend()
        tilt_axis = axes[2].twinx()
        tilt = np.degrees(np.arccos(np.clip(1-2*(root[:, 3]**2+root[:, 4]**2), -1, 1)))
        tilt_axis.plot(time, tilt, color="tab:orange", alpha=.6)
        tilt_axis.set_ylabel("Tilt (degrees)")
    axes[2].set_xlabel("Time in complete recorded attempt (s)")
    name = f"recording_{index:03d}_motion.png"
    fig.savefig(directory/name, dpi=160); plt.close(fig)
    files = [name]
    if native:
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), constrained_layout=True)
        for ax, key, title in ((axes[0], "joint_position_rad", "Joint position (rad)"),
                              (axes[1], "joint_velocity_rad_s", "Joint velocity (rad/s)")):
            if key in native:
                values = native[key][:, 0]
                picture = ax.imshow(values.T, aspect="auto", origin="lower", interpolation="nearest",
                    extent=(0, len(values)*NATIVE_DT, -.5, 17.5), cmap="RdBu_r")
                ax.set_ylabel("Joint index"); ax.set_xlabel("Time (s)"); ax.set_title(title)
                fig.colorbar(picture, ax=ax, fraction=.025)
            else:
                ax.text(.5, .5, key+" unavailable", ha="center", transform=ax.transAxes)
        if "computed_torque_nm" in native:
            fraction = (abs(native["computed_torque_nm"][:, 0]) > 1.6).mean(0)*100
            axes[2].bar(np.arange(18), fraction)
            axes[2].set_ylabel("Requested saturation (%)")
            axes[2].set_xlabel("Joint index (recorded leg-major order)")
        if "distal_contact" in native:
            contact = native["distal_contact"][:, 0]
            axes[3].imshow(contact.T, aspect="auto", origin="lower", interpolation="nearest", cmap="Greys",
                vmin=0, vmax=1, extent=(0, len(contact)*NATIVE_DT, -.5, 5.5))
            axes[3].set_yticks(range(6), ("LF", "LM", "LR", "RF", "RM", "RR"))
            axes[3].set_ylabel("Actual distal contact"); axes[3].set_xlabel("Time (s)")
        name = f"recording_{index:03d}_joints_torque_support.png"
        fig.savefig(directory/name, dpi=160); plt.close(fig); files.append(name)
        if "joint_velocity_rad_s" in native and len(native["joint_velocity_rad_s"]) >= 8:
            values = native["joint_velocity_rad_s"][:, 0].astype(float)
            centered = values-values.mean(0)
            power = abs(np.fft.rfft(centered*np.hanning(len(values))[:, None], axis=0))**2
            power = power.mean(-1); power[0] = 0
            frequency = np.fft.rfftfreq(len(values), NATIVE_DT)
            fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
            if power.sum() > 1e-24:
                ax.semilogy(frequency[1:], np.maximum(power[1:]/power.sum(), 1e-16))
            else:
                ax.text(.5, .5, "Constant joint velocity: no AC spectral power", ha="center", transform=ax.transAxes)
            ax.axvline(5., color="tab:orange", linestyle="--", label="5Hz diagnostic boundary")
            ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Relative joint-velocity spectral power")
            ax.set_title("Full-trial 400Hz samples, mean removed, Hann window"); ax.legend()
            name = f"recording_{index:03d}_joint_velocity_spectrum.png"
            fig.savefig(directory/name, dpi=160); plt.close(fig); files.append(name)
    return {"status": "written", "files": files, "scope": "Replica0 measured timelines and spectra; no smoothing or removed failure samples"}


def analyze(directory, output, *, checkpoint=None, compare=None):
    directory, output = Path(directory), Path(output)
    if directory.is_file(): directory = directory.parent
    batches = [directory] if (directory/"report.json").is_file() else [p.parent for p in sorted(directory.glob("batch_*/report.json"))]
    if not batches:
        raise ValueError("No actual recording report found in the supplied batch/suite directory")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": SCHEMA, "analyzer_sha256": sha(__file__), "recordings": [],
        "stage2_verdict_changed": False, "historical_reference_scope": "Historical Stage2 videos are visual targets on different models; never a matched quantitative baseline."}
    identities = []
    for index, batch in enumerate(batches):
        original, capture, control, native, integrity, binding = load_recording(batch)
        names = capture.get("joint_names") if capture else None
        cases = original.get("cases", [])
        row = {"directory": str(batch.resolve()), "integrity": integrity, "original_results": original.get("results", []),
            "original_failure": original.get("failure"), "original_acquisition_complete": original.get("acquisition_complete"),
            "replicas": [{"case_id": case["case_id"], **analyze_trace(control, native, e, names)} for e, case in enumerate(cases)]}
        row["actor_reconstruction"] = reconstruct_actor(control, checkpoint, original["checkpoint_sha256"]) if checkpoint else {"status": "not_requested"}
        row["plots"] = plots(control, native, output, index)
        report["recordings"].append(row)
        identities.append({"physics": binding, "cases": cases,
            "assigned_case_ids": original.get("assigned_case_ids"),
            "recorded_controls": original.get("controls"), "recorded_physics_steps": original.get("recorded_physics_steps"),
            "episode_timeout_seconds": original.get("episode_timeout_seconds"),
            "velocity_measurement_point": original.get("velocity_measurement_point")})
    report["comparison_identity"] = identities if all(x["physics"] is not None and x["velocity_measurement_point"] for x in identities) else None
    if compare is not None:
        report["comparison"] = comparison(report, read(compare))
    (output/"analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    lines = ["# Native policy recording analysis", "", "The original numerical verdicts are preserved. These diagnostics do not grant Stage2 acceptance.", ""]
    for recording in report["recordings"]:
        lines += [str(recording["directory"]), ""]
        if recording["original_failure"]:
            lines += ["Recorded acquisition failure: "+recording["original_failure"], ""]
        for row in recording["replicas"]:
            lines += ["- "+row["case_id"]+": "+str(row["controls"])+" controls; "+row["status"]+"."]
            if "tracking" in row:
                t = row["tracking"]
                lines += [f"  Planar tracking RMSE {t['planar_error_rmse_mps']:.5f}m/s; yaw RMSE {t['yaw_error_rmse_rad_s']:.5f}rad/s."]
            if "torque_400hz" in row:
                t = row["torque_400hz"]
                lines += [f"  Worst-joint requested saturation {100*max(t['requested_saturation_fraction']):.3f}%; peak applied torque {max(t['applied_abs_peak_nm']):.6f}N·m."]
            if row["missing_metrics"]:
                lines += ["  Missing channels: "+", ".join(row["missing_metrics"])+"."]
        lines += ["", "Plots: "+recording["plots"]["status"]+". Actor reconstruction: "+recording["actor_reconstruction"]["status"]+".", ""]
    lines += ["Toe-reference motion during contact is a slip proxy. Historical videos remain model-specific visual references. No closed-loop noise experiment is inferred.", ""]
    (output/"analysis.md").write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    report = analyze(args.directory, args.output, checkpoint=args.checkpoint, compare=args.compare)
    print(json.dumps({"recordings": len(report["recordings"]), "output": str(args.output), "stage2_verdict_changed": False}))


if __name__ == "__main__":
    main()
