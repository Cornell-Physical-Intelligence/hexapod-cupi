"""Check the surrogate on this machine and measure it against native reference copies.

  uv run python -m locomotion.surrogate.self_check --output <new dir> [--native-ref <dir>] [--skip-training] [--throughput]

Without ``--native-ref`` the check needs no native data: it reads the model contract, lays a tibia
shaft on the floor for the repository contact classifier, and runs short trainings and one
evaluation in this process. With ``--native-ref`` it adds the comparisons of
``locomotion.surrogate.calibrate``: standing, the tripod trace, exploration noise, the noise probe
and the start of a training run. The check writes ``self_check.json`` and its run directories
under ``--output``.

The record reaches the disk after each stage with ``complete`` false, so an error in a later stage
keeps the earlier measurements. The last write sets ``complete``, ``failures`` and ``pass``. The
command exits 1 where ``failures`` names a check: the shaft classification, a training whose
status differs from ``expected_status``, or the evaluation. The native comparisons are measurements
and set no bound. The record holds no machine path; a failed run keeps its traceback in its own
``state.json``.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from locomotion.env_config import MASS_KG
from locomotion.surrogate import calibrate, evaluate as evaluator, train as trainer
from locomotion.surrogate.env import (DEFAULT_GEOMETRY, DEFAULT_MODEL, DEFAULT_STANCE, LEGS, ROOT, ContactModel,
                                      make_env, repository_state, sha)
from locomotion.surrogate.train import thread_count
from locomotion.task_v2 import TrainingTaskV2
from locomotion.task_v3 import TrainingTaskV3

# The native reference files that the comparisons read, relative to --native-ref.
REFERENCE_FILES = ("standing_one/standing/substeps_000.npz", "standing_one/standing/substeps_009.npz",
                   "tripod_trial_000/control_trace.npz", "noise_probe/plan.json", "noise_probe/telemetry.npz",
                   "native_noise_updates_6_20.json", "metrics_A.jsonl")


def throughput(replicas=128, threads=2, controls=300):
    """Environment controls per second under iid noise at std 0.15."""
    env = make_env(replicas, threads=threads, substep_state=False)
    env.commands.zero_()
    generator = torch.Generator().manual_seed(0)
    actions = [.15 * torch.randn(replicas, 18, generator=generator) for _ in range(controls)]
    for action in actions[:20]:
        env.step(action)
    started = time.perf_counter()
    for action in actions[20:]:
        output = env.step(action)
        done = output["terminated"] | output["truncated"]
        if bool(done.any()):
            env.reset(done.nonzero().flatten())
    env.close()
    return (controls - 20) * replicas / (time.perf_counter() - started)


def shaft_contact():
    """Lay tibia shafts on the floor and read the repository classifier: the non-foot path's shaft category.

    The search takes the first leg pose whose lowest collision point belongs to a shaft hull and
    not to a toe cap, sets every leg to it with that point 0.3 mm inside the floor, and advances
    one substep with zero torque. ``_classify_patches`` must then report shaft force and no toe force.
    """
    from locomotion.env import _DiagnosticGeometry, _classify_patches
    env = make_env(1, threads=0)
    m, d = env.mj_model, mujoco.MjData(env.mj_model)
    shaft, toe = {}, {}
    for leg in LEGS:
        for kind, store in (("shaft", shaft), ("toe", toe)):
            geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_tibia_{kind}")
            mesh = m.geom_dataid[geom]
            store[leg] = (geom, m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]].astype(float))

    def lowest(geoms):
        return min(float((d.geom_xpos[g] + vertices @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min()) for g, vertices in geoms.values())

    found = None
    for femur in np.linspace(-1.4, 1.3, 28):
        for tibia in np.linspace(0., 3.0, 31):
            d.qpos[:] = env._qpos0
            d.qpos[env._qpos_columns] = np.tile([0., femur, tibia], 6)
            mujoco.mj_kinematics(m, d)
            low_shaft, low_toe = lowest(shaft), lowest(toe)
            others = min(float(d.geom_xpos[g][2]) for g in range(m.ngeom) if g not in [x[0] for x in shaft.values()] + [x[0] for x in toe.values()] and m.geom_bodyid[g] > 0)
            if low_shaft < low_toe - .004 and low_shaft < others - .02:
                found = (float(femur), float(tibia), low_shaft)
                break
        if found:
            break
    if found is None:
        env.close()
        return {"pass": False, "reason": "no leg pose puts a shaft below its toe cap"}
    femur, tibia, low = found
    env._state[0, 1 + env._qpos_columns] = np.tile([0., femur, tibia], 6)
    env._state[0, 3] = env._state[0, 3] - low - .0003
    env._state[0, 26:] = 0.
    poses = env._read(full=True)["link"].numpy()
    env._advance(torch.zeros(1, 18))
    with np.load(env.geometry_extrema_path, allow_pickle=False) as data:
        geometry = _DiagnosticGeometry(env.geometry_meta, {key: data[key] for key in data.files}, env.native_body_names)
    packets = [np.array(x.numpy(), copy=True) for x in env.contact.get_contact_data(env.cfg.physics_dt)]
    contact = _classify_patches(packets, env.sensor_map, poses, geometry, 1)
    categories = sorted({patch["category"] for patch in contact["patches"] if patch["normal_force_n"] > 0})
    force = float(np.linalg.norm(contact["nonfoot_force_world"][0, 3]))
    env.close()
    toe_force = float(np.linalg.norm(contact["distal_force_world"][0], axis=-1).sum())
    return {"pass": "shaft" in categories and force > 1. and toe_force == 0. and bool(contact["nonfoot_contact"][0]),
            "leg_pose_femur_tibia_rad": [femur, tibia],
            "patch_categories_with_force": categories, "shaft_force_n": force, "nonfoot_contact": bool(contact["nonfoot_contact"][0]),
            "toe_cap_force_n": toe_force}


def quiet(function, arguments):
    """Call an entry point in this process without its printed output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return function(arguments)


def portable(text, output):
    """A message without the output directory and the repository root, so the record holds no machine path."""
    for path, label in ((Path(output).resolve(), "<output>"), (Path(output), "<output>"), (ROOT, "<repository>")):
        text = text.replace(str(path), label)
    return text


def train(output, updates, threads, *extra, expected="completed"):
    """A short training through ``train.main``; the run directory must not exist."""
    started = time.perf_counter()
    code = quiet(trainer.main, ["--output", str(output), "--updates", str(updates), "--threads", str(threads), *extra])
    state = json.loads((output / "state.json").read_text())
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()] if (output / "metrics.jsonl").exists() else []
    return {"status": state["status"], "expected_status": expected,
            "errors": [portable(error, output.parent) for error in state["errors"]], "updates": len(rows), "returncode": code,
            "wall_seconds": time.perf_counter() - started,
            "collection_seconds_per_update": float(np.mean([r["collection_seconds"] for r in rows[1:]])) if len(rows) > 1 else None,
            "learning_seconds_per_update": float(np.mean([r["learning_seconds"] for r in rows[1:]])) if len(rows) > 1 else None,
            "task_reward_mean_last": rows[-1]["task"]["interval_metrics"]["task_reward_mean"] if rows else None,
            "reward_version": rows[-1]["task"]["reward_version"] if rows else None,
            "metrics_row_keys": sorted(rows[-1]) if rows else None,
            "files": sorted(p.name for p in output.iterdir())}


def probe_cells(native, output, threads):
    """Replay the native noise probe and keep the cell table without the per-cell extras."""
    report = calibrate.probe(native, output, ContactModel(), threads=threads)
    versions = report["reward_versions"]
    return {"max_abs_action_difference": report["max_abs_action_difference"],
            "reward_versions": versions, "rewards_unscored": report["rewards_unscored"],
            "ratio_spread_over_12_noisy_cells": report["ratio_spread_over_12_noisy_cells"],
            "cells": [{"pose": row["pose"], "standard_deviation": row["standard_deviation"],
                       "terminations": row["terminations"],
                       **{key: [value["surrogate"], value["native"]] for key, value in row["metrics"].items()},
                       "tibia_joint_velocity_rms_rad_s": [row["extras"][side]["joint_velocity_rms_coxa_femur_tibia_rad_s"][2] for side in ("surrogate", "native")],
                       "limiter_fraction": [row["extras"][side]["limiter_fraction"] for side in ("surrogate", "native")],
                       "tibia_loaded_all_substeps_fraction": [row["extras"][side]["tibia_loaded_all_substeps_fraction"] for side in ("surrogate", "native")],
                       **({"reward_v2_joint_acceleration": [row["reward_component_means"]["v2"]["joint_acceleration"][side] for side in ("surrogate", "native")]}
                          if "v2" in versions else {}),
                       **{"reward_" + version + "_total": [row["reward_component_means"][version]["reward"][side] for side in ("surrogate", "native")]
                          for version in versions},
                       **{key: [row["extras"][side][key] for side in ("surrogate", "native")] for _, key in calibrate.TAILS}}
                      for row in report["cells"]],
            "touchdown_events": report["touchdown_events"], "touchdown_overshoot_events": report["touchdown_overshoot_events"],
            "peak_joint_speed_rad_s": report["peak_joint_speed_rad_s"], "speed_clamp_events": report["speed_clamp_events"],
            "order": "each pair is [surrogate, native]", "table": "probe/compare.txt"}


def evaluate(checkpoint, output, threads):
    """Evaluate a checkpoint on the three focus probes without video."""
    started = time.perf_counter()
    try:
        code = quiet(evaluator.main, ["--checkpoint", str(checkpoint), "--output", str(output), "--video", "none",
                                      "--threads", str(threads)])
    except Exception as error:
        return {"returncode": 1, "error": portable(repr(error), output.parent)}
    if code:
        return {"returncode": code}
    summary = json.loads((output / "surrogate_summary.json").read_text())
    return {"returncode": code, "wall_seconds": time.perf_counter() - started, "observation_scaling": summary["observation_scaling"],
            "task": summary["task"], "policy_path": summary["policy_path"],
            "cases": {row["case_id"]: {"pass": row["pass"], "failed_bounds": row["failed_bounds"],
                                       "consensus": row.get("contact_consensus", {}).get("verdict"),
                                       "planar_error_mps": row["metrics"].get("planar_error_mps")}
                      for row in summary["cases"]}}


def failures(report):
    """The names of the checks of a record that did not hold; an empty list means that each check held.

    The shaft classification must pass, each training must end with its ``expected_status`` and the
    evaluation must return 0. A probe verdict of the evaluated smoke policy is no check: three
    updates train no gait.
    """
    failed = []
    if not report.get("shaft_contact_classification", {}).get("pass"):
        failed.append("shaft_contact_classification")
    for key, row in report.items():
        for name, run in ((key, row), (key + ".run", row.get("run") if isinstance(row, dict) else None)):
            if isinstance(run, dict) and "expected_status" in run and run.get("status") != run["expected_status"]:
                failed.append(name)
    if report.get("evaluation_smoke_learner_options", {}).get("returncode", 0) != 0:
        failed.append("evaluation_smoke_learner_options")
    return failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True, help="New directory, outside the repository.")
    parser.add_argument("--native-ref", type=Path, help="Directory of native reference copies; adds the native comparisons.")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--throughput", action="store_true", help="Measure controls per second; run it on an idle machine.")
    parser.add_argument("--threads", type=int, default=2, help="Physics and torch threads for this process (1 to 6).")
    args = parser.parse_args(argv)
    # The same count reaches torch, the MuJoCo thread pool, the trainer and the evaluator.
    threads = thread_count(args.threads)
    torch.set_num_threads(threads)
    args.output.mkdir(parents=True, exist_ok=False)
    contact = ContactModel()
    env = make_env(1, threads=0)
    package = Path(__file__).resolve().parent
    report = {"schema": "hexapod_surrogate_self_check_v3", "surrogate": True, "complete": False, "pass": None, "failures": None,
              "backend": "mujoco " + mujoco.__version__, "repository": repository_state(),
              "surrogate_files": {p.name: sha(p) for p in sorted(package.glob("*.py"))},
              "model_sha256": sha(DEFAULT_MODEL), "stance_sha256": sha(DEFAULT_STANCE), "geometry_sha256": sha(DEFAULT_GEOMETRY),
              "contact_model": json.loads(json.dumps(contact.__dict__)),
              "total_mass_kg": float(env.mj_model.body_mass.sum()), "native_total_mass_kg": MASS_KG,
              "bodies": int(env.mj_model.nbody - 1), "joints": int(env.mj_model.nu),
              "observation_widths": [int(env.reset()["obs"].shape[1]), int(env.reset()["critic"].shape[1])]}
    env.close()

    def record(key, value):
        """Store one stage and write the record, so an error in a later stage keeps this one."""
        report[key] = value
        (args.output / "self_check.json").write_text(json.dumps(report, indent=2) + "\n")
        return value

    record("shaft_contact_classification", shaft_contact())
    reference = args.native_ref
    if reference is not None:
        record("native_reference_files", {
            name: {"sha256": sha(reference / name), "bytes": (reference / name).stat().st_size}
            for name in REFERENCE_FILES if (reference / name).exists()})
        record("standing", calibrate.standing(contact, reference))
        record("tripod_open_loop_replay", calibrate.tripod_replay(contact, reference))
        record("noise_zero_mean_v2", calibrate.noise(contact, TrainingTaskV2, threads=threads))
        record("noise_zero_mean_v3", calibrate.noise(contact, TrainingTaskV3, threads=threads, controls=240))
        record("native_noise_ranges", calibrate.native_noise_ranges(reference))
        record("noise_probe_vs_native", probe_cells(reference / "noise_probe", args.output / "probe", threads))
    if args.throughput:
        record("throughput_controls_per_second", {f"env_128_replicas_{threads}_threads": throughput(128, threads),
                                                  "env_128_replicas_1_thread": throughput(128, 1, 120),
                                                  "env_1_replica": throughput(1, 0, 400)})
    if not args.skip_training:
        record("training_smoke_v2", train(args.output / "smoke_v2", 3, threads))
        record("training_smoke_v3", train(args.output / "smoke_v3", 3, threads, "--task", "locomotion.task_v3:TrainingTaskV3",
                                          "--action-mean", "tanh", "--observation-normalization", "none"))
        # The default reward v4 holds a contact-schedule term, which needs the gait clock of its period.
        from locomotion.task_v4 import REWARD_V4_CONFIG
        clock = str(REWARD_V4_CONFIG.schedule_period_controls) if REWARD_V4_CONFIG.schedule_weight else "0"
        record("training_smoke_v4", train(args.output / "smoke_v4", 3, threads, "--task", "locomotion.task_v4:TrainingTaskV4",
                                          "--gait-clock", clock))
        # Without the clock the trainer must stop a reward that holds the schedule term.
        record("training_smoke_v4_without_clock", train(
            args.output / "smoke_v4_no_clock", 1, threads, "--task", "locomotion.task_v4:TrainingTaskV4",
            expected="failed" if REWARD_V4_CONFIG.schedule_weight else "completed"))
        options = train(
            args.output / "smoke_options", 3, threads,
            # A variant at the default of a field that this check reads, so the factory form runs at any revision.
            "--task", f"locomotion.task_v4:variant(schedule_weight={REWARD_V4_CONFIG.schedule_weight})",
            "--action-mean", "tanh", "--observation-normalization", "none", "--observation-scaling", "fixed",
            "--command-segments", "bootstrap", "--learning-rate-max", "3e-4", "--action-std", "0.1",
            "--gait-clock", clock, "--action-std-final", "0.05", "--action-noise-correlation", "0.5",
            "--action-smoothing", "mean2", "--velocity-noise", "0.5")
        options["ppo_config"] = {
            key: value for key, value in json.loads((args.output / "smoke_options/ppo_config.json").read_text()).items()
            if key in ("environment_wrapper", "exploration", "algorithm", "actor")}
        metrics = args.output / "smoke_options/metrics.jsonl"
        options["action_std_by_update"] = [json.loads(line)["mean_action_std"]
                                           for line in (metrics.read_text().splitlines() if metrics.exists() else [])]
        record("training_smoke_learner_options", options)
        record("evaluation_smoke_learner_options", evaluate(args.output / "smoke_options/checkpoint_update000003.pt",
                                                            args.output / "smoke_options_eval", threads))
        if reference is not None:
            start = train(args.output / "start_A", 20, threads)
            record("training_start_vs_native_run_A", {"run": start, **({
                key.replace(" ", "_").replace("-", "_"): value
                for key, value in calibrate.training_start(args.output / "start_A", reference / "metrics_A.jsonl").items()}
                if start["status"] == "completed" else {})})
        for key in ("training_smoke_v2", "training_smoke_v3", "training_smoke_v4", "training_smoke_learner_options"):
            row = report[key]
            if row["collection_seconds_per_update"]:
                row["collection_controls_per_second"] = 3072 / row["collection_seconds_per_update"]
    report.update(complete=True, failures=failures(report))
    record("pass", not report["failures"])
    print(f"mass {report['total_mass_kg']:.6f} kg (ledger {MASS_KG:.6f})")
    print("shaft contact classification", report["shaft_contact_classification"])
    if reference is not None:
        s, t = report["standing"], report["tripod_open_loop_replay"]
        print("\n".join(calibrate.model_lines({"label": "default", "standing": s, "tripod_replay": t,
                                               "noise_v2": report["noise_zero_mean_v2"], "noise_v3": report["noise_zero_mean_v3"],
                                               "native_noise_ranges": report["native_noise_ranges"], "wall_seconds": 0.})[:-1]))
        print(f"standing: foot loads {np.round(s['foot_load_n'], 2).tolist()} N, toe drop {s['toe_drop_mm']:.2f} mm from "
              f"{s['initial_toe_height_mm']:.2f} mm, first contact {s['first_contact_ms']} ms (native capture {s['native_first_contact_ms']} ms)")
        spread = report["noise_probe_vs_native"]["ratio_spread_over_12_noisy_cells"]
        print("native noise probe, surrogate/native ratio over the 12 noisy cells: "
              + ", ".join(f"{name} {spread[key]['min_ratio']:.2f}-{spread[key]['max_ratio']:.2f}" for name, key in zip(calibrate.SHORT, calibrate.METRICS)))
        versions = report["noise_probe_vs_native"]["reward_versions"]
        print("rewards without a score:", report["noise_probe_vs_native"]["rewards_unscored"])
        print("reward totals per control at a 0.05 m/s command, surrogate/native (" + ", ".join(versions) + "):")
        for row in report["noise_probe_vs_native"]["cells"]:
            print(f"  {row['pose']:13s} std {row['standard_deviation']:<4g} " + "  ".join(
                f"{row['reward_' + v + '_total'][0]:+.3f}/{row['reward_' + v + '_total'][1]:+.3f}" for v in versions))
    if "throughput_controls_per_second" in report:
        print("throughput", {k: round(v) for k, v in report["throughput_controls_per_second"].items()})
    if "evaluation_smoke_learner_options" in report:
        print("evaluation_smoke_learner_options", report["evaluation_smoke_learner_options"])
    for key in ("training_smoke_v2", "training_smoke_v3", "training_smoke_v4",
                "training_smoke_v4_without_clock", "training_smoke_learner_options"):
        if key in report:
            print(key, {k: report[key][k] for k in ("status", "expected_status", "updates", "collection_seconds_per_update",
                                                    "learning_seconds_per_update", "reward_version")})
    if "updates_6_20" in report.get("training_start_vs_native_run_A", {}):
        a = report["training_start_vs_native_run_A"]["updates_6_20"]
        print("training start vs native A (updates 6-20): " + ", ".join(f"{k.replace('comp.', '')} {a['surrogate'][k]:.4f}/{a['native'][k]:.4f}" for k in a["surrogate"] if k in a["native"]))
    print("self check passed" if report["pass"] else "self check FAILED: " + ", ".join(report["failures"]))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
