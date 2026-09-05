#!/usr/bin/env python3
"""Two co-origin v5 robots: live filtered/negative collision controls, never admission.

Run each --case in a fresh root-owned Kit process. This script neither launches
containers nor changes production task configuration or collision-filter code.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = next((parent for parent in Path(__file__).resolve().parents
             if (parent / "packages/hexapod_core").is_dir()), None)


def load_source(source_dir=None):
    """Resolve the explicit read-only production checkout before importing it."""
    global ROOT, contract, identity
    root = Path(source_dir).resolve() if source_dir is not None else ROOT
    if root is None or not (root / "packages/hexapod_core/hexapod_core/fourbar_v1.py").is_file():
        raise ValueError("A physical case requires --source-dir pointing to the frozen source checkout")
    ROOT = root
    sys.path[:0] = [str(ROOT / path) for path in ("tools", "isaaclab", "packages/hexapod_core", "packages/hexapod_env")]
    from hexapod_core import fourbar_v1 as source_contract
    from mkii_training_contract import identity as source_identity
    if (Path(source_contract.__file__).resolve() != ROOT / "packages/hexapod_core/hexapod_core/fourbar_v1.py"
            or Path(source_identity.__code__.co_filename).resolve() != ROOT / "tools/mkii_training_contract.py"):
        raise ValueError("Imported physical contract or source identity came from a different checkout")
    contract, identity = source_contract, source_identity


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_numerical_recipe(report, cfg, multiplier):
    """Bind the actual configured solver values to the resolved runtime report."""
    props = cfg.robot.spawn.articulation_props
    recipe = dict(contract.numerical_recipe(multiplier),
        physics_dt_s=cfg.sim.dt, decimation=cfg.decimation,
        solver_type=cfg.sim.physics.solver_type,
        enable_external_forces_every_iteration=cfg.sim.physics.enable_external_forces_every_iteration,
        solver_position_iterations=props.solver_position_iteration_count,
        solver_velocity_iterations=props.solver_velocity_iteration_count)
    if "numerical_recipe" in report and report["numerical_recipe"] != recipe:
        raise ValueError("Configured numerical recipe changed during environment construction")
    report.update(numerical_recipe=recipe, solver_multiplier=multiplier,
        solver_iterations=[props.solver_position_iteration_count, props.solver_velocity_iteration_count])
    return contract.validate_numerical_recipe_report(report, multiplier)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

CASES = ("filtered", "unfiltered_negative")
CAPACITY = 512


def parser(add_launcher_args=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case", choices=CASES)
    selection.add_argument("--compare", nargs=2, type=Path, metavar=("FILTERED_REPORT", "NEGATIVE_REPORT"))
    p.add_argument("--physics-steps", type=int, default=256)
    p.add_argument("--source-dir", type=Path, default=ROOT)
    p.add_argument("--solver-multiplier", type=int, choices=(1, 2), default=2)
    p.add_argument("--report", type=Path, required=True)
    if add_launcher_args:
        add_launcher_args(p)
        p.set_defaults(visualizer=[])
    return p


def add_negative_allowlinks(stage):
    """Artifact-only negative control, authored before physics initialization."""
    paths = [f"/World/collisions/group{i}" for i in range(2)]
    evidence = []
    for source, other in ((paths[0], paths[1]), (paths[1], paths[0])):
        rel = stage.GetPrimAtPath(source).GetRelationship("physics:filteredGroups")
        before = [str(path) for path in rel.GetTargets()]
        if set(before) != {source, "/World/collisions/global_group"}:
            raise ValueError("Negative control did not start from verified isolation")
        rel.AddTarget(other)
        after = [str(path) for path in rel.GetTargets()]
        if set(after) != {source, other, "/World/collisions/global_group"}:
            raise ValueError("Negative control cross-environment allowlink was not authored")
        evidence.append({"group": source, "before": before, "after": after})
    return evidence


def mimic_reference_audit(stage, environment_paths):
    if "contract" not in globals():
        load_source()
    rows = []
    for environment in environment_paths:
        robot = environment + "/Robot"
        for name in contract.PASSIVE_JOINT_NAMES:
            path = robot + "/Physics/" + name
            prim = stage.GetPrimAtPath(path)
            schemas = set(prim.GetAppliedSchemas()) if prim else set()
            authored = prim.GetMetadata("apiSchemas") if prim else None
            if authored:
                schemas.update(authored.GetAppliedItems())
            if not prim or "PhysxMimicJointAPI:rotZ" not in schemas:
                raise ValueError(f"Actual cloned mimic joint cannot be resolved: {path}")
            targets = [str(target) for target in prim.GetRelationship("physxMimicJoint:rotZ:referenceJoint").GetTargets()]
            expected = robot + "/Physics/" + name.split("_", 1)[0] + "_tibia_lever_pivot"
            if targets != [expected] or not stage.GetPrimAtPath(expected):
                raise ValueError(f"Cloned mimic reference escapes or differs from its articulation: {path}: {targets}")
            rows.append({"joint": path, "reference_joint": targets[0], "articulation_namespace": robot})
    return {"resolved_relationships_verified": True, "joints": rows,
            "scope": "Actual composed USD relationships; native solver remapping is not directly queried"}


def grade_control(case, metrics, requested_steps):
    errors = []
    if type(metrics.get("samples")) is not int or metrics.get("samples") != requested_steps or metrics.get("finite") is not True:
        errors.append("Incomplete or nonfinite physical samples")
    count, force = metrics.get("max_pair_contact_count"), metrics.get("max_pair_force_n")
    if (type(count) is not int or count < 0 or type(force) not in (int, float)
            or not math.isfinite(force) or force < 0):
        return [*errors, "Invalid native contact-count/force metrics"]
    if metrics.get("contact_buffer_capacity_reached") is not False:
        errors.append("Pair-contact buffer coverage cannot be guaranteed")
    if case == "filtered":
        if count != 0 or force > 1e-6:
            errors.append("Filtered robots report cross-robot body-pair contact")
        support = metrics.get("ground_support_observed_per_robot")
        if not isinstance(support, list) or len(support) != 2 or any(value is not True for value in support):
            errors.append("Filtered robots did not independently report ground support")
    elif case == "unfiltered_negative":
        if count < 1 or force <= 1e-3:
            errors.append("Negative control did not demonstrate actual cross-robot contact counts and force")
    else:
        errors.append("Unknown fixture case")
    return errors


def compare_cases(filtered, negative):
    errors = []
    for result, case in ((filtered, "filtered"), (negative, "unfiltered_negative")):
        if (result.get("case") != case or result.get("fixture_complete") is not True
                or result.get("control_expectation_met") is not True or result.get("errors") != []
                or result.get("pass") is not False or result.get("simulation_training_admission") is not False
                or result.get("hardware_admission") is not False):
            errors.append(f"{case}: missing, failed or admitting control report")
        errors += [f"{case}: {error}" for error in grade_control(case, result.get("metrics", {}), result.get("physics_steps_requested"))]
    for key in ("source_identity", "fixture_source_sha256", "physics_steps_requested", "numerical_recipe",
                "solver_multiplier", "solver_iterations", "initial_reset_roots_m"):
        if key not in filtered or filtered.get(key) != negative.get(key):
            errors.append(f"Controlled cases differ or omit {key}")
    runtimes = [copy.deepcopy(result.get("runtime_manifest", {})) for result in (filtered, negative)]
    for runtime in runtimes:
        runtime.pop("resolved_collision_isolation", None)
    if not runtimes[0] or runtimes[0] != runtimes[1]:
        errors.append("Physical/motor runtime differs beyond the controlled filter override")
    return errors


def finish(report, path, env=None):
    report.update({"pass": False, "simulation_training_admission": False, "hardware_admission": False})
    write_json(path, report)
    print("ROBOT_PAIR_CONTROL_RESULT " + json.dumps({key: report.get(key) for key in
        ("case", "fixture_complete", "control_expectation_met", "pair_control_success", "errors")}), flush=True)
    if env is not None:
        env.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    early, _ = parser().parse_known_args(argv)
    if early.report.exists():
        raise ValueError("Need a new report path")
    early.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "hexapod.live_collision_pair_fixture.v1", "case": early.case,
        "fixture_complete": False, "control_expectation_met": False, "errors": [],
        "physics_steps_requested": early.physics_steps, "fixture_source_sha256": digest(Path(__file__)),
        "scope": "Diagnostic robot-pair contact controls only; no training or hardware admission"}
    env, rows, columns = None, [], []
    try:
        if early.compare:
            cases = [json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                     for path in early.compare]
            for path, case in zip(early.compare, cases):
                trace = case.get("trace", {})
                if trace.get("file") != "trace.npz" or digest(path.parent / "trace.npz") != trace.get("sha256"):
                    raise ValueError("Control trace bytes do not match the recorded evidence")
            report["errors"] = compare_cases(*cases)
            report["pair_control_success"] = not report["errors"]
            report["input_reports"] = [{"path": str(path), "sha256": digest(path)} for path in early.compare]
            finish(report, early.report)
            return int(bool(report["errors"]))
        load_source(early.source_dir)
        if not 32 <= early.physics_steps <= 512 or early.physics_steps % contract.DECIMATION:
            raise ValueError("Need 32..512 physics steps divisible by the policy decimation")
        report["source_identity"] = identity(ROOT)
        usd = ROOT / contract.ASSET_BUNDLES["mkii_fourbar_v5"]["usd_path_relative"]
        audit = subprocess.run([sys.executable, str(ROOT / "tools/audit_mkii_fourbar_usd.py"), str(usd)],
                               capture_output=True, text=True, timeout=180)
        report["cpu_asset_audit"] = json.loads(audit.stdout)
        if audit.returncode or report["cpu_asset_audit"].get("pass") is not True:
            raise ValueError("Actual v5 CPU asset audit failed")
        from hexapod_env.tasks.mkii_fourbar_v1.register import register_mkii_fourbar_v1
        register_mkii_fourbar_v1()
        from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli
        from validate_mkii_fourbar import apply_numerical_recipe, tensor
        saved = sys.argv
        try:
            sys.argv = [__file__, *argv]
            args, overrides = setup_preset_cli(parser(add_launcher_args))
            if overrides:
                raise ValueError(f"Unreviewed config overrides: {overrides}")
            sys.argv = [__file__]
            cfg, _ = resolve_task_config(contract.TASK_ID, "")
        finally:
            sys.argv = saved
        cfg.scene.num_envs, cfg.seed = 2, 0
        cfg.robot.spawn.usd_path = str(usd)
        cfg.standing_only, cfg.reset_joint_jitter_rad = True, 0.
        report["numerical_recipe"] = apply_numerical_recipe(cfg, args.solver_multiplier)
        if "pxr" in sys.modules:
            raise ValueError("Standalone USD was imported before Kit")
        with launch_simulation(cfg, args):
            try:
                import numpy as np
                import torch
                import warp as wp
                from isaaclab_physx.physics import PhysxManager
                from hexapod_env.tasks.mkii_fourbar_v1.env import HexapodMkiiFourbarEnv
                from hexapod_env.tasks.mkii_fourbar_v1.collision_isolation import verify_collision_isolation

                class PairFixture(HexapodMkiiFourbarEnv):
                    def _setup_scene(self):
                        super()._setup_scene()
                        self.fixture_negative_override = []
                        if args.case == "unfiltered_negative":
                            self.fixture_negative_override = add_negative_allowlinks(self.scene.stage)
                            try:
                                verify_collision_isolation(self.scene.stage, physics_scene_path=self.scene.physics_scene_path,
                                    env_prim_paths=self.scene.env_prim_paths, global_prim_paths=[self.cfg.terrain.prim_path])
                            except ValueError:
                                pass
                            else:
                                raise ValueError("Negative control still passed production collision isolation")
                            self.collision_isolation_report["runtime_identity"] = dict(
                                self.collision_isolation_report["runtime_identity"],
                                mode="DIAGNOSTIC_ONLY_mutual_cross_environment_allowlinks",
                                authored_topology_verified=False, cross_environment_allowlinks=True,
                                environment_allowlist="own_group_global_group_and_other_environment")

                env = PairFixture(cfg=cfg)
                if not str(env.device).startswith("cuda") or env._physics_handles_decimation is not False:
                    raise ValueError("Fixture requires the explicit PhysX GPU physics-step backend")
                report["runtime_manifest"] = env.runtime_manifest
                record_numerical_recipe(report, env.cfg, args.solver_multiplier)
                report["isolation_topology"] = env.collision_isolation_report
                report["negative_override_before_physics_initialization"] = env.fixture_negative_override
                report["mimic_reference_audit"] = mimic_reference_audit(env.scene.stage, env.scene.env_prim_paths)
                if env._robot.num_bodies != 31 or env._robot.num_joints != 30 or env.num_envs != 2:
                    raise ValueError("Fixture did not resolve two actual 31-body/30-coordinate robots")
                origins = tensor(env._terrain.env_origins)
                report["original_terrain_origins_m"] = origins.clone().cpu().tolist()
                if not bool((origins[:, 2] == 0.).all()):
                    raise ValueError("Fixture requires the unchanged world Z=0 support plane")
                origins[:, :2] = 0.
                env.reset(seed=0)
                roots = tensor(env._robot.data.root_pos_w).clone()
                defaults = tensor(env._robot.data.default_root_pose)[:, :3]
                if not torch.allclose(roots, defaults + origins, rtol=0, atol=1e-5):
                    raise ValueError("Co-origin reset roots do not match the actual terrain origins")
                body_separation = torch.linalg.vector_norm(tensor(env._robot.data.body_link_pos_w)[0] -
                                                           tensor(env._robot.data.body_link_pos_w)[1], dim=-1).max()
                if not torch.isfinite(body_separation) or float(body_separation) > 1e-5:
                    raise ValueError("Initial actual robot bodies are not co-origin")
                report["initial_reset_roots_m"] = roots.cpu().tolist()
                report["initial_max_body_separation_m"] = float(body_separation)
                paths = [f"/World/envs/env_{i}/Robot/Geometry/body" for i in range(2)]
                physics = PhysxManager.get_physics_sim_view()
                views = []
                for source, target in ((paths[0], paths[1]), (paths[1], paths[0])):
                    rigid = physics.create_rigid_body_view(source)
                    target_rigid = physics.create_rigid_body_view(target)
                    if list(rigid.prim_paths) != [source] or list(target_rigid.prim_paths) != [target]:
                        raise ValueError("Native contact pair body paths did not resolve exactly")
                    view = physics.create_rigid_contact_view(source, filter_patterns=[target], max_contact_data_count=CAPACITY)
                    if view.filter_count != 1:
                        raise ValueError("Native contact pair did not resolve exactly one target filter")
                    views.append(view)
                report["native_pair_bindings"] = [{"source_body": paths[i], "target_filter": paths[1-i]}
                                                   for i in range(2)]
                actions = torch.zeros(2, 18, device=env.device)
                pair_forces, pair_counts, ground_forces = [], [], []
                columns = [f"pair_{i}_force_{axis}" for i in range(2) for axis in "xyz"] + ["pair_0_count", "pair_1_count"]
                columns += [f"env_{i}_foot_{leg}_ground_force_{axis}" for i in range(2) for leg in contract.LEGS for axis in "xyz"]
                columns += [f"env_{i}_root_{axis}" for i in range(2) for axis in "xyz"]
                for step in range(args.physics_steps):
                    if step % contract.DECIMATION == 0:
                        env._pre_physics_step(actions)
                    env._sim_step_counter += 1
                    env._apply_action()
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(dt=contract.PHYSICS_DT_S)
                    forces, counts = [], []
                    for view in views:
                        matrix = wp.to_torch(view.get_contact_force_matrix(dt=contract.PHYSICS_DT_S))
                        contact_data = view.get_contact_data(dt=contract.PHYSICS_DT_S)
                        count = wp.to_torch(contact_data[4])
                        if matrix.shape != (1, 1, 3) or count.shape != (1, 1):
                            raise ValueError("Unsupported native pair-contact tensor layout")
                        forces.append(matrix.reshape(3).clone())
                        counts.append(count.reshape(()).clone())
                    force, count = torch.stack(forces), torch.stack(counts)
                    ground = torch.stack([tensor(sensor.data.force_matrix_w)[:, 0, 0].clone()
                                          for sensor in env._feet_contact_sensors], dim=1)
                    if ground.shape != (2, 6, 3) or bool((count < 0).any()):
                        raise ValueError("Pair counts or independent ground-force layout invalid")
                    root_positions = tensor(env._robot.data.root_pos_w).clone()
                    if (not bool(torch.isfinite(torch.cat([force.flatten(), ground.flatten(), root_positions.flatten(),
                            tensor(env._robot.data.joint_pos).flatten(), tensor(env._robot.data.joint_vel).flatten()])).all())
                            or bool(env.motor_telemetry("invalid_input").any())):
                        raise ValueError("Nonfinite state or invalid motor input during fixture")
                    pair_forces.append(force.detach().cpu().numpy().copy())
                    pair_counts.append(count.detach().cpu().numpy().copy())
                    ground_forces.append(ground.detach().cpu().numpy().copy())
                    rows.append(torch.cat([force.flatten(), count.to(force.dtype), ground.flatten(), root_positions.flatten()]).detach().cpu().numpy().copy())
                forces, counts, ground = np.stack(pair_forces), np.stack(pair_counts), np.stack(ground_forces)
                report["metrics"] = {"samples": len(rows), "finite": True,
                    "max_pair_contact_count": int(counts.max()), "max_pair_force_n": float(np.linalg.norm(forces, axis=-1).max()),
                    "contact_buffer_capacity_reached": bool((counts >= CAPACITY).any()),
                    "ground_support_observed_per_robot": (np.linalg.norm(ground, axis=-1) > 1.).any(axis=(0, 2)).tolist(),
                    "peak_ground_force_per_robot_n": np.linalg.norm(ground, axis=-1).max(axis=(0, 2)).tolist()}
                report["fixture_complete"] = True
                report["errors"] += grade_control(args.case, report["metrics"], args.physics_steps)
                report["control_expectation_met"] = not report["errors"]
            except BaseException as error:
                report["errors"].append(f"{type(error).__name__}: {error}")
                raise
            finally:
                try:
                    if rows:
                        trace_path = early.report.with_name("trace.npz")
                        with trace_path.open("xb") as stream:
                            np.savez_compressed(stream, values=np.stack(rows), columns=np.asarray(columns), physics_dt_s=contract.PHYSICS_DT_S)
                        report["trace"] = {"file": trace_path.name, "sha256": digest(trace_path), "samples": len(rows), "columns": columns}
                    if identity(ROOT) != report["source_identity"]:
                        raise ValueError("Production source identity changed during fixture")
                except BaseException as error:
                    report["errors"].append(f"Evidence persistence {type(error).__name__}: {error}")
                    report["control_expectation_met"] = False
                finally:
                    finish(report, early.report, env)
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        if message not in report["errors"]:
            report["errors"].append(message)
        if not early.report.exists():
            finish(report, early.report)
        return 1
    return 0 if report["control_expectation_met"] and not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
