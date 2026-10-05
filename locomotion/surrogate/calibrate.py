"""Compare the surrogate with retained native captures.

You pass the directory of native reference copies with ``--native-ref`` and a directory for the
results with ``--output``. The repository holds neither; README.md names the Spark origin of each
reference file. Each result needs a new name: no subcommand replaces a file. Five subcommands (run
from the repository root with ``uv run python -m locomotion.surrogate.calibrate ...``):

  model  One ``ContactModel`` against three references: neutral standing at 400 Hz
         (``standing_one/standing/substeps_000.npz`` and ``substeps_009.npz``), iid Gaussian
         exploration noise at std 0.15 (native training updates 6 to 20 in
         ``native_noise_updates_6_20.json``) and an open-loop replay of the native tripod action
         trace (``tripod_trial_000/control_trace.npz``).
  sweep  For each ``NAME=FIELD=VALUE;FIELD=VALUE`` variant in turn: the standing and tripod checks,
         a 20-update training and its first updates against ``metrics_A.jsonl``.
  probe  Replay the native fixed-mean noise probe (``noise_probe/plan.json`` and
         ``telemetry.npz``) with the plan's seed and compare the 16 cells.
  start  The first updates of a surrogate training run against a native compact extract.
  extract  Write the two training extracts of a reference directory from native ``metrics.jsonl``
         files: ``metrics_<NAME>.jsonl`` for each ``--run NAME=PATH`` and
         ``native_noise_updates_6_20.json`` for the runs together.

Each native number in a result comes from a reference file. A result is design evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from locomotion import noise_probe
from locomotion.surrogate import replay
from locomotion.surrogate.env import contact_overrides, make_env, repository_state
from locomotion.surrogate.train import thread_count
from locomotion.task import ProximityGuard, TaskConfig
from locomotion.task_v2 import TrainingTaskV2
from locomotion.task_v3 import TrainingTaskV3

# The native extract holds runs A to D. Runs A to C trained reward v2 and run D trained reward v3,
# so the tracking terms of each reward version compare with its own runs.
NOISE_RUNS = {"v2": ("A", "B", "C"), "v3": ("D",)}
# The eight statistics of noise_probe.summarize that the native summary reports per cell.
METRICS = ("planar_speed_rms_mps", "vertical_velocity_rms_mps", "roll_pitch_rate_rms_rad_s", "yaw_rate_rms_rad_s",
           "joint_velocity_rms_rad_s", "executed_target_step_rms_rad", "root_height_mean_m", "applied_torque_rms_nm")
SHORT = ("planar", "vert", "rollpitch", "yaw", "jointvel", "tstep", "height", "torque")
# Offline reward scoring assumes this command; the probe's open-loop means ignore the command.
SCORE_COMMAND = (.05, 0., 0.)
REWARDS = ("v1", "v2", "v3", "v4")
# Per-cell extras that the printed table adds to the eight summary statistics.
TAILS = (("tibia>3", "tibia_rate_above_3_rad_s_fraction"), ("acc norm", "joint_acceleration_norm_mean_rad_s2"),
         ("acc rms", "joint_acceleration_rms_rad_s2"), ("touchdown/s", "touchdowns_per_foot_per_second"),
         ("loaded8", "tibia_loaded_all_substeps_fraction"), ("collapse/min", "tibia_collapse_events_per_robot_minute"),
         ("saturation", "requested_saturation_fraction_400hz"))


def new_file(path):
    """A result path that must not exist yet: each result keeps a fresh name."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; choose a new label or a new output directory")
    return path


# ----- one contact model against standing, noise and the tripod trace ----------------------
def standing(contact, reference, controls=1000):
    """Zero action from the reset; the native capture is one robot, 1000 controls."""
    env = make_env(1, threads=0, contact=contact, episode_seconds=60.)
    env.commands.zero_()
    rows = []

    def capture(env, state, q, dq, requested, applied, ceiling, target, forces, substep, native_pre):
        rows.append((float(state["root"][0, 2]), forces[0, env.toe_index, 2].numpy().copy(),
                     float(state["toe_world"][0, :, 2].min())))
    env.capture = capture
    initial_toe = float(env.current["toe_world"][0, :, 2].mean())
    for _ in range(controls):
        env.step(torch.zeros(1, 18))
    t = env.telemetry
    with np.load(Path(reference) / "standing_one/standing/substeps_000.npz") as z:
        native_z = z["root_pose_xyzw"][:, 0, 2]
        native_force = z["distal_force_world_n"][:, 0, :, 2]
    with np.load(Path(reference) / "standing_one/standing/substeps_009.npz") as z:
        native_q = z["joint_position_rad"][-1, 0]
        native_final_z = float(z["root_pose_xyzw"][-1, 0, 2])
        # The same statistic as the surrogate's: the norm over joints of the torque RMS of the last control.
        native_torque = float(np.linalg.norm(np.sqrt((z["applied_torque_nm"][-8:, 0].astype(np.float64) ** 2).mean(0))))
    z_trace = np.array([r[0] for r in rows])
    force = np.array([r[1] for r in rows])
    q = t["joint_position_rad"][0].numpy()
    root = t["root_pose_xyzw"][0].numpy()
    yaw = 2 * np.arctan2(root[5], root[6])
    env.close()
    return {"initial_toe_height_mm": 1000 * initial_toe,
            "settled_toe_height_mm": 1000 * float(t["toe_xyz_world"][0, :, 2].mean()),
            "toe_drop_mm": 1000 * (initial_toe - float(t["toe_xyz_world"][0, :, 2].mean())),
            "first_contact_ms": 2.5 * (1 + int(np.argmax(force.max(1) > 0))),
            "native_first_contact_ms": 2.5 * (1 + int(np.argmax(native_force.max(1) > 0))),
            "root_z_rmse_first_200_substeps_mm": 1000 * float(np.sqrt(np.mean((z_trace[:200] - native_z[:200]) ** 2))),
            "root_z_min_mm": 1000 * float(z_trace[:200].min()), "native_root_z_min_mm": 1000 * float(native_z[:200].min()),
            "settled_root_z_m": float(root[2]), "native_settled_root_z_m": native_final_z,
            "femur_q": float(q[1::3].mean()), "native_femur_q": float(native_q[1::3].mean()),
            "tibia_q": float(q[2::3].mean()), "native_tibia_q": float(native_q[2::3].mean()),
            "joint_position_gap_to_native_rad": {"largest": float(np.abs(q - native_q).max()),
                                                 "mean": float(np.abs(q - native_q).mean())},
            "foot_load_n": [float(x) for x in t["tibia_floor_force_world_n"][0, :, 2]],
            "torque_norm_nm": float(torch.linalg.vector_norm((t["torque_square_sum_400hz"] / 8).sqrt())),
            "native_torque_norm_nm": native_torque,
            "planar_drift_mm": 1000 * float(np.hypot(root[0], root[1])), "yaw_drift_deg": float(np.degrees(yaw)),
            "joint_speed_rms_last_200": float(t["joint_velocity_rad_s"].square().mean().sqrt())}


def noise(contact, task_class=TrainingTaskV2, replicas=128, controls=480, std=.15, seed=20260917, threads=2, mean=0.):
    """iid Gaussian actions around ``mean``; interval metrics for update 1 and updates 6 to 20."""
    env = make_env(replicas, threads=threads, contact=contact, substep_state=False)
    task = task_class(env, TaskConfig(seed=seed))
    task.reset()
    generator = torch.Generator().manual_seed(seed)
    velocity, yaw_rate = [], []
    report = {}
    for control in range(controls):
        output = task.step(mean + std * torch.randn(replicas, 18, generator=generator))
        done = output["terminated"] | output["truncated"]
        if control >= 120:
            t = env.telemetry
            velocity.append(t["linear_velocity_nav"].numpy().copy())
            yaw_rate.append(t["angular_velocity_body"][:, 2].numpy().copy())
        if bool(done.any()):
            task.reset(done.nonzero().flatten())
        if control == 23:
            report["update_1"] = task.status(reset_interval=True)["interval_metrics"]
        if control == 119:
            task.status(reset_interval=True)
    status = task.status(reset_interval=True)["interval_metrics"]
    velocity, yaw_rate = np.stack(velocity), np.stack(yaw_rate)

    def summary(m):
        slew = m["actual_target_slew"]["by_joint"]
        return {"achieved_planar_speed_mps": m["achieved_planar_speed_mps"],
                "requested_saturation_fraction": m["requested_saturation_fraction"],
                "limiter_fraction_coxa_femur_tibia": [float(np.mean([slew[n]["at_limit_fraction"] for n in env.joint_names[k::3]])) for k in range(3)],
                "terminations": m["terminations"], "termination_reasons": m["termination_reasons"]["rows"],
                "nonfoot_event_fraction": m["nonfoot_event_fraction"], "task_reward_mean": m["task_reward_mean"],
                "components": {k: v["mean"] for k, v in m["reward_components"].items()}}
    result = {"update_1": summary(report["update_1"]), "updates_6_20": summary(status),
              "planar_velocity_std_xy_mps": [float(velocity[..., 0].std()), float(velocity[..., 1].std())],
              "vertical_velocity_rms_mps": float(np.sqrt((velocity[..., 2] ** 2).mean())),
              "yaw_rate_std_rad_s": float(yaw_rate.std()),
              "speed_clamp_events": env.speed_clamp_events}
    env.close()
    return result


def native_noise_ranges(reference):
    """Minimum and maximum of each native noise statistic over the runs of the extract; {} without the file."""
    path = Path(reference) / "native_noise_updates_6_20.json"
    if not path.exists():
        return {}
    runs = json.loads(path.read_text())
    span = lambda values: [min(values), max(values)]
    over = lambda names, read: span([read(runs[name]) for name in names if name in runs])
    every = list(runs)
    result = {"runs": every, "tracking_runs": {version: [n for n in names if n in runs] for version, names in NOISE_RUNS.items()},
              "updates_6_20": {"achieved_planar_speed_mps": over(every, lambda r: r["aspd"]),
                               "requested_saturation_fraction": over(every, lambda r: r["sat"]),
                               "components": {key: over(every, lambda r, key=key: r["comp"][key])
                                              for key in ("vertical_velocity", "roll_pitch_rate", "joint_torque",
                                                          "joint_acceleration", "action_rate")}},
              "update_1": {"achieved_planar_speed_mps": over(every, lambda r: r["u1"]["aspd"]),
                           "requested_saturation_fraction": over(every, lambda r: r["u1"]["sat"]),
                           "components": {key: over(every, lambda r, key=key: r["u1"]["comp"][key])
                                          for key in ("vertical_velocity", "joint_acceleration")}}}
    for version, names in NOISE_RUNS.items():
        if any(name in runs for name in names):
            result["updates_6_20"]["tracking_" + version] = {
                key: over(names, lambda r, key=key: r["comp"][key]) for key in ("linear_tracking", "yaw_tracking")}
    return result


def tripod_replay(contact, reference, controls=1000):
    """Open loop: the native trial's recorded actions without its foot-force feedback."""
    with np.load(Path(reference) / "tripod_trial_000/control_trace.npz") as z:
        actions = z["policy_action"][:, 0]
        native = {k: z[k][:, 0] for k in ("root_pose_xyzw", "velocity_navigation_mps", "gyro_body_rad_s", "joint_position_rad",
                                         "applied_torque_squared_sum_400hz", "distal_contact", "toe_xyz_world_m", "velocity_world_mps")}
    env = make_env(1, threads=0, contact=contact, episode_seconds=60.)
    env.commands[:] = torch.tensor([.05, 0., 0.])
    rows = []
    contact_rows = []

    def capture(env, state, q, dq, requested, applied, ceiling, target, forces, substep, native_pre):
        contact_rows.append((state["toe_cap_force_n"][0].numpy() > 1.).copy())
    env.capture = capture
    env.substep_state = False
    terminated = False
    for control in range(controls):
        output = env.step(torch.from_numpy(actions[control])[None])
        t = env.telemetry
        rows.append({"nav": t["linear_velocity_nav"][0].numpy().copy(), "gyro": t["angular_velocity_body"][0].numpy().copy(),
                     "root": t["root_pose_xyzw"][0].numpy().copy(), "q": t["joint_position_rad"][0].numpy().copy(),
                     "tsq": t["torque_square_sum_400hz"][0].numpy().copy(), "toe": t["toe_xyz_world"][0].numpy().copy(),
                     "other": float(t["other_body_force_max_400hz"][0]), "slew": (t["joint_target_rad"][0] - env.neutral).numpy().copy()})
        if bool(output["terminated"].any()):
            terminated = True
            break
    env.close()
    data = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    sel = slice(100, len(rows))
    nav = data["nav"][sel]
    error = np.linalg.norm(nav[:, :2] - np.array([.05, 0.]), axis=-1)
    native_nav, native_gyro = native["velocity_navigation_mps"][sel], native["gyro_body_rad_s"][sel]
    rms = np.sqrt(data["tsq"][sel] / 8)
    native_rms = np.sqrt(native["applied_torque_squared_sum_400hz"][sel] / 8)
    contacts = np.array(contact_rows)[800:]
    x, y, zq, w = data["root"][sel, 3:].T
    tilt = np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))
    return {"controls": len(rows), "terminated": terminated,
            "mean_forward_speed_mps": float(nav[:, 0].mean()), "native_mean_forward_speed_mps": float(native_nav[:, 0].mean()),
            "mean_left_speed_mps": float(nav[:, 1].mean()), "mean_yaw_rate_rad_s": float(data["gyro"][sel, 2].mean()),
            "planar_error_mps": float(error.mean()),
            "native_planar_error_mps": float(np.linalg.norm(native_nav[:, :2] - np.array([.05, 0.]), axis=-1).mean()),
            "yaw_error_rad_s": float(np.abs(data["gyro"][sel, 2]).mean()),
            "native_yaw_error_rad_s": float(np.abs(native_gyro[:, 2]).mean()),
            "vertical_velocity_rms_mps": float(np.sqrt((nav[:, 2] ** 2).mean())),
            "native_vertical_velocity_rms_mps": float(np.sqrt((native_nav[:, 2] ** 2).mean())),
            "tilt_rms_deg": float(np.degrees(np.sqrt((tilt ** 2).mean()))),
            "root_height_mean_m": float(data["root"][sel, 2].mean()), "native_root_height_mean_m": float(native["root_pose_xyzw"][sel, 2].mean()),
            "torque_rms_coxa_femur_tibia_nm": [float(np.sqrt((rms[:, k::3] ** 2).mean())) for k in range(3)],
            "native_torque_rms_coxa_femur_tibia_nm": [float(np.sqrt((native_rms[:, k::3] ** 2).mean())) for k in range(3)],
            "toe_contact_fraction": [float(x) for x in contacts.mean(0)], "native_toe_contact_fraction": [float(x) for x in native["distal_contact"][sel].mean(0)],
            "toe_height_max_mm": [float(x) for x in 1000 * data["toe"][sel, :, 2].max(0)],
            "native_toe_height_max_mm": [float(x) for x in 1000 * native["toe_xyz_world_m"][sel, :, 2].max(0)],
            "joint_position_rmse_to_native_rad": float(np.sqrt(((data["q"] - native["joint_position_rad"][:len(rows)]) ** 2).mean())),
            "nonfoot_controls": int((data["other"] > 1.).sum()),
            "final_displacement_forward_left_m": [float(-data["root"][-1, 1]), float(data["root"][-1, 0])],
            "native_final_displacement_forward_left_m": [float(-native["root_pose_xyzw"][-1, 1]), float(native["root_pose_xyzw"][-1, 0])]}


def model_checks(contact, reference, label, *, skip=(), threads=2):
    """The standing, tripod and noise checks of one contact model."""
    result = {"label": label, "contact": asdict(contact), "repository": repository_state()}
    started = time.perf_counter()
    if "standing" not in skip:
        result["standing"] = standing(contact, reference)
    if "tripod" not in skip:
        result["tripod_replay"] = tripod_replay(contact, reference)
    if "noise" not in skip:
        result["noise_v2"] = noise(contact, TrainingTaskV2, threads=threads)
    if "noise_v3" not in skip:
        result["noise_v3"] = noise(contact, TrainingTaskV3, threads=threads, controls=240)
    result["native_noise_ranges"] = native_noise_ranges(reference)
    result["wall_seconds"] = time.perf_counter() - started
    return result


def model_lines(result):
    """One printed line per check: each surrogate value beside its native value or native range."""
    label, lines = result["label"], []
    s, n, t = result.get("standing"), result.get("noise_v2"), result.get("tripod_replay")
    ranges = result.get("native_noise_ranges") or {}
    span = lambda value: "" if value is None else f" ({value[0]:.4f}..{value[1]:.4f})"
    native = lambda *keys: span(_dig(ranges, keys))
    if s:
        lines.append(f"[{label}] stand z {s['settled_root_z_m']:.5f} (native {s['native_settled_root_z_m']:.5f}) femur {s['femur_q']:.4f}/{s['native_femur_q']:.4f} "
                     f"tibia {s['tibia_q']:.4f}/{s['native_tibia_q']:.4f} torque {s['torque_norm_nm']:.3f}/{s['native_torque_norm_nm']:.3f} zrmse {s['root_z_rmse_first_200_substeps_mm']:.2f}mm "
                     f"zmin {s['root_z_min_mm']:.2f}/{s['native_root_z_min_mm']:.2f} drift {s['planar_drift_mm']:.2f}mm yaw {s['yaw_drift_deg']:.3f}deg")
    if t:
        lines.append(f"[{label}] tripod fwd {t['mean_forward_speed_mps']:.4f} (native {t['native_mean_forward_speed_mps']:.4f}) left {t['mean_left_speed_mps']:.4f} yaw {t['mean_yaw_rate_rad_s']:.4f} "
                     f"perr {t['planar_error_mps']:.4f} (native {t['native_planar_error_mps']:.4f}) vz {t['vertical_velocity_rms_mps']:.4f} torque {np.round(t['torque_rms_coxa_femur_tibia_nm'], 3).tolist()} "
                     f"native {np.round(t['native_torque_rms_coxa_femur_tibia_nm'], 3).tolist()} contact {np.round(t['toe_contact_fraction'], 2).tolist()} term {t['terminated']} nonfoot {t['nonfoot_controls']}")
    if n:
        u, c = n["updates_6_20"], n["updates_6_20"]["components"]
        lines.append(f"[{label}] noise aspd {u['achieved_planar_speed_mps']:.4f}{native('updates_6_20', 'achieved_planar_speed_mps')} std_xy {np.round(n['planar_velocity_std_xy_mps'], 4).tolist()} yaw {n['yaw_rate_std_rad_s']:.4f} "
                     f"limiter {np.round(u['limiter_fraction_coxa_femur_tibia'], 3).tolist()} sat {u['requested_saturation_fraction']:.4f}{native('updates_6_20', 'requested_saturation_fraction')}")
        lines.append(f"[{label}]   " + " ".join(f"{short} {c[key]:.4f}{native('updates_6_20', 'components', key)}" for short, key in (
                         ("vert", "vertical_velocity"), ("rp", "roll_pitch_rate"), ("torque", "joint_torque"),
                         ("acc", "joint_acceleration"), ("rate", "action_rate")))
                     + f" lin {c['linear_tracking']:.4f}{native('updates_6_20', 'tracking_v2', 'linear_tracking')}"
                     + f" yawtrk {c['yaw_tracking']:.4f}{native('updates_6_20', 'tracking_v2', 'yaw_tracking')}"
                     + f" term {u['terminations']} nonfoot {u['nonfoot_event_fraction']:.4f}")
        c1 = n["update_1"]["components"]
        lines.append(f"[{label}]   update1 vert {c1['vertical_velocity']:.4f}{native('update_1', 'components', 'vertical_velocity')} "
                     f"acc {c1['joint_acceleration']:.4f}{native('update_1', 'components', 'joint_acceleration')} "
                     f"aspd {n['update_1']['achieved_planar_speed_mps']:.4f}{native('update_1', 'achieved_planar_speed_mps')} "
                     f"sat {n['update_1']['requested_saturation_fraction']:.4f}{native('update_1', 'requested_saturation_fraction')} term {n['update_1']['terminations']}")
    if result.get("noise_v3"):
        c = result["noise_v3"]["updates_6_20"]["components"]
        lines.append(f"[{label}]   v3 lin {c['linear_tracking']:.4f}{native('updates_6_20', 'tracking_v3', 'linear_tracking')} "
                     f"yaw {c['yaw_tracking']:.4f}{native('updates_6_20', 'tracking_v3', 'yaw_tracking')}")
    lines.append(f"[{label}] wall {result['wall_seconds']:.1f}s")
    return lines


def _dig(value, keys):
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value


# ----- the first updates of a training run against a native extract ------------------------
def compact_rows(metrics):
    """The compact per-update rows of one ``metrics.jsonl``; native and surrogate runs share the layout."""
    rows = []
    for line in Path(metrics).read_text().splitlines():
        row = json.loads(line)
        m = row["task"]["interval_metrics"]
        slew = m["actual_target_slew"]["by_joint"]
        names = list(slew)
        rows.append({"u": row["update"], "std": row["mean_action_std"], "r": m["task_reward_mean"],
                     "comp": {k: v["mean"] for k, v in m["reward_components"].items()},
                     "sat": m["requested_saturation_fraction"], "spd": m["moving_signed_command_direction_speed_mps"],
                     "aspd": m["achieved_planar_speed_mps"], "term": m["terminations"],
                     **{"slew_" + tag: float(np.mean([slew[n]["at_limit_fraction"] or 0. for n in names[k::3]]))
                        for k, tag in enumerate("cft")}})
    return rows


def surrogate_rows(run):
    """The compact per-update rows of a surrogate run directory."""
    return compact_rows(Path(run) / "metrics.jsonl")


def noise_extract(rows, first=6, last=20):
    """One run's entry of ``native_noise_updates_6_20.json``: window means and the first update."""
    span, start = window(rows, first, last), window(rows, 1, 1)
    components = lambda values: {key[5:]: value for key, value in values.items() if key.startswith("comp.")}
    return {"std": span["std"], "aspd": span["aspd"], "slew": [span["slew_c"], span["slew_f"], span["slew_t"]],
            "sat": span["sat"], "r": span["r"], "comp": components(span),
            "u1": {"aspd": start["aspd"], "sat": start["sat"], "comp": components(start)}}


def extract(runs, output, first=6, last=20):
    """Write the training extracts of a reference directory from ``{label: native metrics.jsonl}``.

    ``NOISE_RUNS`` names the labels that the noise check reads, and the training-start check reads
    ``metrics_A.jsonl``. Returns the written paths.
    """
    output = Path(output)
    targets = {label: new_file(output / f"metrics_{label}.jsonl") for label in runs}
    noise = new_file(output / "native_noise_updates_6_20.json")
    rows = {label: compact_rows(path) for label, path in runs.items()}
    output.mkdir(parents=True, exist_ok=True)
    for label, target in targets.items():
        target.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows[label]))
    noise.write_text(json.dumps({label: noise_extract(values, first, last) for label, values in rows.items()}, indent=1) + "\n")
    return [*targets.values(), noise]


def window(rows, first, last):
    """Means over the updates ``first`` to ``last`` of compact rows."""
    sel = [r for r in rows if first <= r["u"] <= last]
    out = {k: float(np.mean([r[k] for r in sel])) for k in ("std", "r", "sat", "aspd", "slew_c", "slew_f", "slew_t")}
    out.update({"comp." + k: float(np.mean([r["comp"][k] for r in sel])) for k in sel[0]["comp"]})
    return out


def training_start(run, native, first=6, last=20):
    """Update 1 and updates ``first`` to ``last`` of a surrogate run beside a native compact extract."""
    mine = surrogate_rows(run)
    theirs = [json.loads(line) for line in Path(native).read_text().splitlines()]
    return {label: {"surrogate": window(mine, a, b), "native": window(theirs, a, b)}
            for label, a, b in (("update 1", 1, 1), (f"updates {first}-{last}", first, last))}


def start_lines(result):
    lines = []
    for label, pair in result.items():
        lines.append(label)
        for key, value in pair["surrogate"].items():
            if key in pair["native"]:
                reference = pair["native"][key]
                ratio = value / reference if abs(reference) > 1e-9 else float("nan")
                lines.append(f"  {key:28s} surrogate {value:+.4f}  native {reference:+.4f}  ratio {ratio:.3f}")
    return lines


# ----- the native fixed-mean noise probe, cell by cell --------------------------------------
def extras(arrays, replicas, settle):
    """Per-cell statistics beyond noise_probe.summarize, from fields both simulators record."""
    take = lambda name: arrays[name][settle:][:, replicas].astype(np.float64)
    target = arrays["joint_target_rad"][:, replicas].astype(np.float64)
    step = np.abs(np.diff(target, axis=0))[settle:]
    rate, torque = take("joint_velocity_rad_s"), take("torque_square_sum_400hz") / 8
    velocity, pose = take("linear_velocity_nav"), take("root_pose_xyzw")
    first = arrays["root_pose_xyzw"][settle, replicas].astype(np.float64)
    x, y, z, w = pose[..., 3], pose[..., 4], pose[..., 5], pose[..., 6]
    tilt = np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))
    heading = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    previous_rate = arrays["joint_velocity_rad_s"][settle - 1:-1][:, replicas].astype(np.float64)
    acceleration = (rate - previous_rate) / .02
    force = np.linalg.norm(arrays["tibia_floor_force_world_n"][:, replicas].astype(np.float64), axis=-1) > 1.
    touchdown = (force[1:] & ~force[:-1])[settle - 1:]
    tibia, lag = rate[..., 2::3], (take("joint_target_rad") - take("joint_position_rad"))[..., 2::3]
    collapsed = lag > .25
    minutes = len(rate) * .02 / 60 * len(replicas)
    return {
        "tibia_rate_above_3_rad_s_fraction": float((np.abs(tibia) > 3).mean()),
        "joint_velocity_abs_max_rad_s": float(np.abs(rate).max()),
        "joint_acceleration_norm_mean_rad_s2": float(np.linalg.norm(acceleration, axis=-1).mean()),
        "joint_acceleration_rms_rad_s2": float(np.sqrt((acceleration ** 2).mean())),
        "touchdowns_per_foot_per_second": float(touchdown.mean() / .02),
        "tibia_lag_max_rad": float(lag.max()),
        "tibia_collapse_events_per_robot_minute": float((collapsed[1:] & ~collapsed[:-1]).sum() / minutes),
        "limiter_fraction": float((np.abs(step - .04) <= 2e-6).mean()),
        "requested_saturation_fraction_400hz": float(take("saturation_count_400hz").mean() / 8),
        "joint_velocity_rms_coxa_femur_tibia_rad_s": [float(np.sqrt((rate[..., k::3] ** 2).mean())) for k in range(3)],
        "applied_torque_rms_coxa_femur_tibia_nm": [float(np.sqrt(torque[..., k::3].mean())) for k in range(3)],
        "planar_velocity_std_forward_left_mps": [float(velocity[..., 0].std()), float(velocity[..., 1].std())],
        "tibia_loaded_all_substeps_fraction": float((take("tibia_floor_force_min_norm_400hz") > 1.).mean()),
        "tibia_endpoint_force_mean_n": float(np.linalg.norm(take("tibia_floor_force_world_n"), axis=-1).mean()),
        "nonfoot_control_fraction": float((take("other_body_force_max_400hz") > 1.).mean()),
        "tilt_rms_deg": float(np.degrees(np.sqrt((tilt ** 2).mean()))),
        "root_height_std_m": float(pose[..., 2].std()),
        "planar_drift_end_m": float(np.linalg.norm(pose[-1, :, :2] - first[:, :2], axis=-1).mean()),
        "heading_drift_end_abs_deg": float(np.degrees(np.abs(np.arctan2(np.sin(heading[-1] - heading[0]), np.cos(heading[-1] - heading[0])))).mean()),
    }


def reward_components(arrays, settle):
    """Reward terms of versions 1 to 4 per control and replica, under SCORE_COMMAND.

    ``replay.score_arrays`` runs the repository task classes on the record, so each version's
    memory rules follow the checked-out revision. Returns
    {version: {component: [control, replica]}} for controls from ``settle`` onward, and
    {version: message} for each version that needs a field the record lacks. The native probe
    record of 2026-10-04 holds no ``computed_torque_nm``, for example.
    """
    controls, replicas = arrays["action"].shape[:2]
    data = {key: value.astype(np.float32) if value.dtype != bool else value for key, value in arrays.items()}
    data["command"] = np.broadcast_to(np.asarray(SCORE_COMMAND, dtype=np.float32), (controls, replicas, 3)).copy()
    # A fallen replica restarts at neutral in the probe; the scorer memory restarts with it.
    data["reset_before"] = np.concatenate((np.ones((1, replicas), dtype=bool), arrays["terminated"][:-1]))
    scores, unscored = {}, {}
    for name in REWARDS:
        try:
            scores.update(replay.score_arrays(data, [name]))
        except ValueError as error:
            unscored[name] = str(error)
    return {name: {key: value[settle:] for key, value in components.items()} for name, components in scores.items()}, unscored


def load(directory):
    plan = json.loads((Path(directory) / "plan.json").read_text())
    with np.load(Path(directory) / "telemetry.npz", allow_pickle=False) as file:
        arrays = {key: file[key] for key in file.files}
    return plan, arrays


def ratio(value, reference):
    return None if abs(reference) < 1e-9 else value / reference


def compare(native_dir, surrogate_dir):
    """Cell-by-cell comparison of two probe directories that share one plan."""
    native_plan, native = load(native_dir)
    plan, mine = load(surrogate_dir)
    if native_plan["replicas"] != plan["replicas"] or native_plan["seed"] != plan["seed"]:
        raise ValueError("The surrogate replay used another plan than the native probe")
    settle = plan["settle_controls"]
    action_difference = float(np.abs(native["action"] - mine["action"]).max())
    cells = {"native": noise_probe.summarize(native, plan["replicas"]), "surrogate": noise_probe.summarize(mine, plan["replicas"])}
    scored = {"native": reward_components(native, settle), "surrogate": reward_components(mine, settle)}
    rewards = {side: value[0] for side, value in scored.items()}
    # A reward version enters the comparison where both records support it.
    versions = [name for name in REWARDS if all(name in value for value in rewards.values())]
    rows = []
    for a, b in zip(cells["native"], cells["surrogate"]):
        replicas = a["replicas"]
        row = {"pose": a["pose"], "standard_deviation": a["standard_deviation"], "replicas": replicas,
               "metrics": {key: {"native": a[key], "surrogate": b[key], "ratio": ratio(b[key], a[key]),
                                 "difference": b[key] - a[key]} for key in METRICS},
               "terminations": {"native": a["terminations"], "surrogate": b["terminations"]},
               "planar_velocity_mean_mps": {"native": a["planar_velocity_mean_mps"], "surrogate": b["planar_velocity_mean_mps"]},
               "extras": {"native": extras(native, replicas, settle), "surrogate": extras(mine, replicas, settle)},
               "reward_component_means": {version: {key: {"native": float(rewards["native"][version][key][:, replicas].mean()),
                                                          "surrogate": float(rewards["surrogate"][version][key][:, replicas].mean())}
                                                    for key in rewards["native"][version]} for version in versions}}
        rows.append(row)
    noisy = [row for row in rows if row["standard_deviation"] > 0]
    spread = {key: {"min_ratio": min(row["metrics"][key]["ratio"] for row in noisy),
                    "max_ratio": max(row["metrics"][key]["ratio"] for row in noisy),
                    "mean_abs_log_ratio": float(np.mean([abs(np.log(row["metrics"][key]["ratio"])) for row in noisy]))}
              for key in METRICS}
    return {"schema": "hexapod_surrogate_probe_compare_v1", "surrogate": True,
            "native_probe": str(native_dir), "surrogate_probe": str(surrogate_dir),
            "seed": plan["seed"], "controls": plan["controls"], "settle_controls": settle,
            "max_abs_action_difference": action_difference, "score_command": list(SCORE_COMMAND),
            "reward_versions": versions, "rewards_unscored": {side: value[1] for side, value in scored.items()},
            "ratio_spread_over_12_noisy_cells": spread, "cells": rows,
            "scope": "CPU MuJoCo surrogate against one native probe; design evidence."}


def table(report):
    lines = ["cell (pose, std)        " + "".join(f"{name:>22s}" for name in SHORT),
             "                        " + "".join(f"{'surrogate/native':>22s}" for _ in SHORT)]
    for row in report["cells"]:
        cells = "".join(f"{m['surrogate']:>11.4f}/{m['native']:<10.4f}" for m in (row["metrics"][key] for key in METRICS))
        lines.append(f"{row['pose']:13s} {row['standard_deviation']:<5g}    " + cells
                     + f"  term {row['terminations']['surrogate']}/{row['terminations']['native']}")
    lines.append("ratio surrogate/native over the 12 noisy cells (min, max): "
                 + ", ".join(f"{name} {s['min_ratio']:.2f}-{s['max_ratio']:.2f}"
                             for name, s in zip(SHORT, (report["ratio_spread_over_12_noisy_cells"][key] for key in METRICS))))
    lines.append("tails and feet (surrogate/native): " + ", ".join(name for name, _ in TAILS))
    for row in report["cells"]:
        if row["standard_deviation"] > 0:
            lines.append(f"  {row['pose']:13s} {row['standard_deviation']:<5g} " + "  ".join(
                f"{row['extras']['surrogate'][key]:.4g}/{row['extras']['native'][key]:.4g}" for _, key in TAILS))
    versions = report["reward_versions"]
    for side, missing in report["rewards_unscored"].items():
        lines += [f"reward {name} has no score on the {side} record: {message}" for name, message in missing.items()]
    lines.append(f"reward totals per control at command {report['score_command']} (surrogate/native): " + ", ".join(versions))
    for row in report["cells"]:
        values = row["reward_component_means"]
        lines.append(f"  {row['pose']:13s} {row['standard_deviation']:<5g} " + "  ".join(
            f"{values[version]['reward']['surrogate']:+.3f}/{values[version]['reward']['native']:+.3f}" for version in versions))
    for version, keys in (("v2", ("vertical_velocity", "roll_pitch_rate", "joint_torque", "joint_acceleration", "action_rate", "linear_tracking", "yaw_tracking")),
                          ("v4", None)):
        if version not in versions:
            continue
        first = report["cells"][0]["reward_component_means"][version]
        keys = [key for key in (keys or first) if key in first and key != "reward"
                and any(abs(row["reward_component_means"][version][key][side]) > 5e-4 for row in report["cells"] for side in ("native", "surrogate"))]
        lines.append(f"reward {version} component means at command {report['score_command']} (surrogate/native):")
        for row in report["cells"]:
            values = row["reward_component_means"][version]
            lines.append(f"  {row['pose']:13s} {row['standard_deviation']:<5g} "
                         + " ".join(f"{key} {values[key]['surrogate']:+.3f}/{values[key]['native']:+.3f}" for key in keys))
    return "\n".join(lines)


def probe(native, output, contact, *, threads=2):
    """Run ``noise_probe.run`` on the surrogate with the native plan's seed and write the comparison."""
    native, output = Path(native), Path(output)
    native_plan = json.loads((native / "plan.json").read_text())
    seed = int(native_plan["seed"])
    if native_plan["replicas"] != noise_probe.plan(len(native_plan["replicas"])):
        raise ValueError("The native plan differs from locomotion.noise_probe.plan at this revision")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    # Native probe mode: 128 replicas, 20 s episodes, zero command, reset by noise_probe.run.
    env = make_env(len(native_plan["replicas"]), threads=min(threads, 6), contact=contact, episode_seconds=20.,
                   substep_state=False, seed=seed, output=output / "native")
    guard = ProximityGuard(env, output / "probe_guard")
    noise_probe.run(env, output / "probe", seed=seed, guard=guard)
    report = compare(native, output / "probe")
    report.update(contact_model=asdict(contact), speed_clamp_events=env.speed_clamp_events,
                  peak_joint_speed_rad_s=env.peak_joint_speed, touchdown_events=env.touchdown_events,
                  touchdown_overshoot_events=env.overshoot_events, repository=repository_state(),
                  wall_seconds=time.perf_counter() - started)
    env.close()
    (output / "compare.json").write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    (output / "compare.txt").write_text(table(report) + "\n")
    return report


def sweep(reference, output, variants, *, threads=2):
    """One variant after the other: standing and tripod checks, a 20-update training and its start."""
    from locomotion.surrogate import train
    reference, output = Path(reference), Path(output)
    parsed = []
    for item in variants:
        name, separator, spec = item.partition("=")
        if not separator or not name or name in dict(parsed):
            raise ValueError("Expected distinct NAME=FIELD=VALUE;FIELD=VALUE variants, got " + item)
        # A variant name that holds a result stops the sweep before its first run.
        new_file(output / ("sweep_" + name + ".json"))
        new_file(output / ("sweep_" + name + "_train"))
        parsed.append((name, spec))
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, spec in parsed:
        sets = [x for x in spec.split(";") if x]
        result = model_checks(contact_overrides(sets), reference, "sweep_" + name, skip=("noise", "noise_v3"), threads=threads)
        run = output / ("sweep_" + name + "_train")
        code = train.main(["--output", str(run), "--updates", "20", "--threads", str(threads)]
                          + sum((["--contact", x] for x in sets), []))
        state = json.loads((run / "state.json").read_text())
        result["training"] = {"returncode": code, "status": state["status"], "errors": state["errors"],
                              "wall_seconds": state.get("wall_seconds"), "speed_clamp_events": state.get("speed_clamp_events")}
        if code == 0:
            result["training_start_vs_native_run_A"] = training_start(run, reference / "metrics_A.jsonl")
        (output / ("sweep_" + name + ".json")).write_text(json.dumps(result, indent=1) + "\n")
        print(f"== {name}: {spec}")
        print("\n".join(model_lines(result) + start_lines(result.get("training_start_vs_native_run_A", {}))), flush=True)
        results[name] = result
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    sub = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", type=Path, required=True, help="Directory for the results, outside the repository.")
    common.add_argument("--threads", type=int, default=2, help="Physics and torch threads for this process (1 to 6).")
    model = sub.add_parser("model", parents=[common])
    model.add_argument("--native-ref", type=Path, required=True, help="Directory of native reference copies.")
    model.add_argument("--label", required=True)
    model.add_argument("--set", action="append", default=[], help="ContactModel field, e.g. impratio=10")
    model.add_argument("--skip", nargs="*", default=[], choices=["standing", "noise", "tripod", "noise_v3"])
    variants = sub.add_parser("sweep", parents=[common])
    variants.add_argument("--native-ref", type=Path, required=True)
    variants.add_argument("--variant", action="append", required=True, metavar="NAME=FIELD=VALUE;FIELD=VALUE")
    cells = sub.add_parser("probe", parents=[common])
    cells.add_argument("--native", type=Path, required=True, help="Directory with the native plan.json and telemetry.npz.")
    cells.add_argument("--contact", action="append", default=[], metavar="FIELD=VALUE", help="ContactModel override.")
    cells.add_argument("--quiet", action="store_true", help="Print the ratio line alone.")
    start = sub.add_parser("start")
    start.add_argument("run", type=Path, help="Surrogate run directory with metrics.jsonl.")
    start.add_argument("native", type=Path, help="Native compact per-update extract (jsonl).")
    start.add_argument("--first", type=int, default=6)
    start.add_argument("--last", type=int, default=20)
    rows = sub.add_parser("extract")
    rows.add_argument("--run", action="append", required=True, metavar="NAME=PATH",
                      help="A run label and its native metrics.jsonl; the checks read the labels A to D.")
    rows.add_argument("--output", type=Path, required=True, help="The reference directory, outside the repository.")
    args = parser.parse_args(argv)
    if args.mode == "start":
        print("\n".join(start_lines(training_start(args.run, args.native, args.first, args.last))))
        return 0
    if args.mode == "extract":
        runs = dict(item.partition("=")[::2] for item in args.run)
        if len(runs) != len(args.run) or not all(runs) or not all(runs.values()):
            raise ValueError("Expected distinct NAME=PATH runs")
        print("\n".join(str(path) for path in extract(runs, args.output)))
        return 0
    # The same count reaches torch and the MuJoCo thread pool.
    torch.set_num_threads(thread_count(args.threads))
    if args.mode == "model":
        target = new_file(args.output / f"{args.label}.json")
        result = model_checks(contact_overrides(args.set), args.native_ref, args.label, skip=args.skip, threads=args.threads)
        args.output.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=1) + "\n")
        print("\n".join(model_lines(result)))
    elif args.mode == "sweep":
        sweep(args.native_ref, args.output, args.variant, threads=args.threads)
    else:
        report = probe(args.native, args.output, contact_overrides(args.contact), threads=args.threads)
        text = table(report)
        print(next(line for line in text.splitlines() if line.startswith("ratio surrogate/native")) if args.quiet else text)
        # One float32 rounding step separates the CPU and GPU evaluations of mean + std * noise.
        if report["max_abs_action_difference"] > 1e-6:
            print("WARNING: surrogate action samples differ from the native record by", report["max_abs_action_difference"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
