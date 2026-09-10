#!/usr/bin/env python3
"""CPU/USD audit or short Isaac geometry/ray/contact smoke; never starts PPO.

The physical probes test fixture import, not robot locomotion or camera coverage.
Run only through the owner's guarded GPU scheduler after Stage2 priority work.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_geometry_helpers():
    # Match the working study launcher's order: no numpy/USD/physics imports
    # before SimulationApp. Keep the CPU-only path available without Isaac.
    global np, audit_usd, bind_fixture_material, collision_meshes
    global load_catalog, probe_locations, vertical_surface_heights
    import numpy as np
    from terrain_fixture_checks import (audit_usd, bind_fixture_material, collision_meshes,
                                        load_catalog, probe_locations, vertical_surface_heights)


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def numpy_value(value):
    # Isaac Lab 3 uses TensorViews; older builds expose torch.Tensor directly.
    if hasattr(value, "torch"):
        value = value.torch
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def audit(records):
    rows = []
    for entry, usd, vertices, faces in records:
        row = {"id": entry["id"], **audit_usd(usd, vertices, faces)}
        samples = np.array([[-1., 0.], [0., 0.], [1., 0.], [1.75, 0.]])
        heights = vertical_surface_heights(vertices, faces, samples)
        if abs(heights[0]) > 1e-7 or np.isfinite(heights[-1]):
            raise ValueError(f"Start pad or outside-fixture geometry incorrect: {entry['id']}")
        if entry["family"] == "pit" and abs(heights[1] + entry["pit_depth_m"]) > 1e-7:
            raise ValueError(f"Pit centre is sealed: {entry['id']}")
        row.update(cpu_passed=True, sample_heights_m=[float(h) if np.isfinite(h) else None for h in heights],
                   physical_validation=False)
        rows.append(row)
    return rows


def run_isaac(args, records, state):
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns
    from isaaclab_physx.physics import PhysxCfg
    from pxr import Gf, UsdGeom, UsdPhysics
    import omni.physx

    sim = SimulationContext(sim_utils.SimulationCfg(dt=args.dt, device=args.device,
        physics=PhysxCfg(), render_interval=10, enable_scene_query_support=True))
    stage = sim.stage
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    objects, contact_sensors, fixtures = [], [], []
    radius = .01
    for index, (entry, usd, vertices, faces) in enumerate(records):
        origin = np.array([(index % 6) * 4., (index // 6) * 4., 0.])
        prefix = f"/World/Fixtures/f_{index:03d}"
        parent = UsdGeom.Xform.Define(stage, prefix)
        parent.AddTranslateOp().Set(Gf.Vec3d(*origin))
        terrain_path = prefix + "/Terrain"
        prim = stage.DefinePrim(terrain_path, "Mesh")
        prim.GetReferences().AddReference(str(usd))
        prim = collision_meshes(stage, terrain_path)[0]
        bind_fixture_material(stage, prim, friction=1., contact_offset_m=.001)

        anchor_path = f"/World/SensorAnchors/a_{index:03d}"
        anchor = RigidObject(RigidObjectCfg(prim_path=anchor_path,
            spawn=sim_utils.SphereCfg(radius=.005,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=.001),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(origin + [0., 0., .8]))))
        objects.append(anchor)
        sensor = RayCaster(RayCasterCfg(prim_path=anchor_path, update_period=args.dt,
            ray_alignment="world", mesh_prim_paths=[terrain_path],
            pattern_cfg=patterns.GridPatternCfg(resolution=.3, size=(2.4, 2.4)),
            max_distance=2., debug_vis=False))
        probes = []
        locations = probe_locations(entry)
        heights = vertical_surface_heights(vertices, faces, locations)
        for probe_index, (xy, height) in enumerate(zip(locations, heights)):
            path = f"/World/Probes/p_{index:03d}_{probe_index}"
            initial = origin + [xy[0], xy[1], float(height) + radius + .04]
            probe = RigidObject(RigidObjectCfg(prim_path=path,
                spawn=sim_utils.SphereCfg(radius=radius, activate_contact_sensors=True,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=.05, angular_damping=10.,
                        solver_position_iteration_count=8, solver_velocity_iteration_count=2),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001, rest_offset=0.),
                    mass_props=sim_utils.MassPropertiesCfg(mass=.05),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1.,
                        restitution=0., friction_combine_mode="multiply", restitution_combine_mode="multiply")),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(initial))))
            contact = ContactSensor(ContactSensorCfg(prim_path=path, update_period=args.dt,
                history_length=3, filter_prim_paths_expr=[terrain_path], track_pose=True,
                track_contact_points=True, max_contact_data_count_per_prim=8))
            objects.append(probe)
            contact_sensors.append(contact)
            probes.append((probe, contact, initial))
        fixtures.append((entry, vertices, faces, origin, terrain_path, sensor, probes))

    # New scene: only the explicitly enumerated terrain meshes and probes may collide.
    expected_terrain = {f[4] for f in fixtures}
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Plane":
            raise RuntimeError("Unexpected plane could seal pits or catch off-fixture probes")
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            path = str(prim.GetPath())
            if path not in expected_terrain and not path.startswith("/World/Probes/"):
                raise RuntimeError(f"Unexpected collider: {path}")
    sim.reset()
    for obj in objects:
        obj.reset()
    for contact in contact_sensors:
        contact.reset()
    for item in fixtures:
        item[5].reset()

    # Query before the spheres land so rays cannot mistake a probe for terrain.
    sim.step(render=False)
    query = omni.physx.get_physx_scene_query_interface()
    rows = []
    for entry, vertices, faces, origin, terrain_path, sensor, probes in fixtures:
        sensor.update(args.dt)
        hits = numpy_value(sensor.data.ray_hits_w)[0]
        pattern, _ = sensor.cfg.pattern_cfg.func(sensor.cfg.pattern_cfg, args.device)
        expected_xy = numpy_value(pattern)[:, :2] + origin[:2]
        expected_z = vertical_surface_heights(vertices, faces, expected_xy - origin[:2])
        finite = np.isfinite(hits).all(axis=1)
        ray_error = np.abs(hits[:, 2] - expected_z)
        sensor_pass = bool(finite.all() and np.max(np.abs(hits[:, :2] - expected_xy)) < 1e-4
                           and np.max(ray_error) < 2e-4)
        # Offset the PhysX query points from probe centres to exclude probe hits.
        query_xy = np.array([[-1., .11], [0., .031], [.51, -.21], [1.75, 0.]])
        query_expected = vertical_surface_heights(vertices, faces, query_xy)
        query_rows = []
        for xy, height in zip(query_xy, query_expected):
            result = query.raycast_closest(tuple(origin + [xy[0], xy[1], .8]), (0., 0., -1.), 2.)
            actual_hit = bool(result.get("hit", False))
            hit_z = float(result["position"][2]) if actual_hit else None
            expected_hit = bool(np.isfinite(height))
            correct = actual_hit == expected_hit
            if expected_hit and actual_hit:
                correct = correct and abs(hit_z - float(height)) < 2e-4
                correct = correct and str(result.get("collision", "")) == terrain_path
            query_rows.append({"xy_local_m": xy.tolist(), "expected_height_m": float(height) if expected_hit else None,
                "height_m": hit_z, "collision": str(result.get("collision", "")), "passed": bool(correct)})
        rows.append({"id": entry["id"], "ray_caster_rays": len(hits), "ray_caster_passed": sensor_pass,
                     "ray_caster_max_height_error_m": float(ray_error.max()) if finite.all() else None,
                     "physx_queries": query_rows, "probes": []})

    force_windows = [[[] for _ in item[-1]] for item in fixtures]
    for step in range(args.steps):
        sim.step(render=False)
        for obj in objects:
            obj.update(args.dt)
        for contact in contact_sensors:
            contact.update(args.dt)
        if step >= args.steps - 40:
            for i, item in enumerate(fixtures):
                for j, (_, contact, _) in enumerate(item[-1]):
                    force = numpy_value(contact.data.force_matrix_w)
                    if not np.isfinite(force).all():
                        raise RuntimeError("Non-finite contact sensor force")
                    force_windows[i][j].append(float(np.linalg.norm(force.reshape(-1, 3), axis=1).sum()))
    for i, (entry, vertices, faces, origin, terrain_path, sensor, probes) in enumerate(fixtures):
        for j, (probe, contact, initial) in enumerate(probes):
            position = numpy_value(probe.data.root_pos_w)[0]
            velocity = numpy_value(probe.data.root_lin_vel_w)[0]
            height = vertical_surface_heights(vertices, faces, (position[:2] - origin[:2])[None])[0]
            gap = float(position[2] - radius - height)
            contact_fraction = float(np.mean(np.asarray(force_windows[i][j]) > .02))
            passed = bool(np.isfinite(position).all() and np.isfinite(velocity).all()
                and np.isfinite(height) and abs(gap) <= .003 and contact_fraction >= .8
                and abs(float(velocity[2])) < .05)
            # The centre pit probe must reach the negative floor, never hover at z=0.
            if entry["family"] == "pit" and j == 1:
                passed = passed and position[2] < -.08
            rows[i]["probes"].append({"initial_position_m": initial.tolist(), "final_position_m": position.tolist(),
                "surface_height_m": float(height) if np.isfinite(height) else None,
                "sphere_bottom_gap_m": gap if np.isfinite(gap) else None,
                "contact_fraction_last_40_steps": contact_fraction,
                "mean_contact_force_n": float(np.mean(force_windows[i][j])), "passed": bool(passed)})
        rows[i]["passed"] = (rows[i]["ray_caster_passed"] and all(r["passed"] for r in rows[i]["physx_queries"])
                              and all(r["passed"] for r in rows[i]["probes"]))
    state.update(status="completed", runtime_rows=rows, all_fixture_smokes_passed=all(r["passed"] for r in rows),
        simulated_seconds=args.steps * args.dt, robot_validation_performed=False,
        sensor_scope="Ideal static terrain-only RayCaster; no camera/LiDAR or robot self-occlusion validation",
        contact_material="Nominal friction 1.0, restitution 0; not hardware calibration",
        ready_for_terrain_training=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "artifacts/terrain_readiness_2026-09-09/terrain_catalog.json")
    parser.add_argument("--fixtures", help="Comma-separated fixture IDs; default all catalog fixtures")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true", help="USD topology audit; requires usd-core, does not load Isaac")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=.005)
    initial, _ = parser.parse_known_args()
    if not initial.cpu_only:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 80 or not math.isfinite(args.dt) or not 0 < args.dt <= .005:
        parser.error("At least 80 steps and a positive timestep <=5 ms are required")
    if args.output.exists():
        parser.error("Use a fresh output directory to preserve previous evidence")
    args.output.mkdir(parents=True)
    state = {"status": "initializing", "started_unix": time.time(), "cpu_only": args.cpu_only,
             "catalog_sha256": digest(args.catalog), "harness_sha256": digest(__file__),
             "ready_for_terrain_training": False, "robot_validation_performed": False}
    app = None
    try:
        faulthandler.enable()
        faulthandler.dump_traceback_later(45, repeat=True)
        save(args.output / "validation.json", state)
        if not args.cpu_only:
            print("TERRAIN_PHASE starting_isaac", file=sys.stderr, flush=True)
            app = AppLauncher(args).app
        state["status"] = "loading_geometry_helpers"
        save(args.output / "validation.json", state)
        print("TERRAIN_PHASE loading_geometry_helpers", file=sys.stderr, flush=True)
        load_geometry_helpers()
        records = load_catalog(args.catalog, args.fixtures.split(",") if args.fixtures else None)
        state["status"] = "auditing_usd"
        save(args.output / "validation.json", state)
        print("TERRAIN_PHASE auditing_usd", file=sys.stderr, flush=True)
        state["cpu_rows"] = audit(records)
        state["status"] = "cpu_audit_completed_isaac_pending"
        save(args.output / "validation.json", state)
        if not args.cpu_only:
            print("TERRAIN_PHASE starting_physics_smoke", file=sys.stderr, flush=True)
            run_isaac(args, records, state)
        state["finished_unix"] = time.time()
        save(args.output / "validation.json", state)
        print(json.dumps({"status": state["status"], "fixtures": len(records),
                          "runtime_passed": state.get("all_fixture_smokes_passed"), "output": str(args.output)}))
        return 0 if args.cpu_only or state["all_fixture_smokes_passed"] else 1
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        save(args.output / "validation.json", state)
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if app is not None:
            app.close()


if __name__ == "__main__":
    sys.exit(main())
