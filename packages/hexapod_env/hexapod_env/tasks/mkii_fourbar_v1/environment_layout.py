"""Explicit flat-world layout candidate; importing this module needs only stdlib.

Coincident origins change numerical placement, not robot or ground properties.
Authored isolation is verified separately from live collision response. The
historical grid retains its unchanged runtime manifest.
"""
from __future__ import annotations

import copy
import json
import math
import re

GRID_LAYOUT = "grid_2m_v1"
COINCIDENT_LAYOUT = "coincident_flat_origin_v1"
LAYOUTS = (GRID_LAYOUT, COINCIDENT_LAYOUT)
ENVIRONMENT_VARIABLE = "HEXAPOD_MKII_ENVIRONMENT_LAYOUT"
RESET_POSITION_TOLERANCE_M = 1e-5  # Native float32 readback, not a physical gate.


def select_environment_layout(environ):
    value = environ.get(ENVIRONMENT_VARIABLE, GRID_LAYOUT)
    if value not in LAYOUTS:
        raise ValueError(f"Unsupported MKII environment layout: {value!r}")
    return value


def configure_environment_layout(cfg):
    """Run before DirectRLEnv constructs InteractiveScene or clones anything."""
    selected = getattr(cfg, "environment_layout", GRID_LAYOUT)
    if selected not in LAYOUTS:
        raise ValueError("Unsupported MKII environment layout")
    if selected == COINCIDENT_LAYOUT:
        if cfg.terrain.terrain_type != "plane" or cfg.terrain.prim_path != "/World/ground":
            raise ValueError("Coincident origins require the unchanged shared flat ground")
        cfg.scene.env_spacing = 0.0
        cfg.terrain.env_spacing = 0.0
    return selected


def layout_runtime_descriptor():
    """Count-independent identity; concrete environment rows live in layout_report."""
    return {
        "schema": "hexapod.environment_layout.v1",
        "layout_id": COINCIDENT_LAYOUT,
        "origin_world_m": [0.0, 0.0, 0.0],
        "scene_env_spacing_m": 0.0,
        "terrain_env_spacing_m": 0.0,
        "terrain_type": "plane",
        "ground_prim_path": "/World/ground",
        "placement_applied": "configuration_before_interactive_scene_construction",
        "authored_environment_transforms": "identity",
        "terrain_origins": "all_zero",
        "native_sensor_row_mapping": "exact_articulation_link_paths",
        "collider_ownership": "all_authored_colliders_in_exactly_own_environment_group",
        "mimic_reference_ownership": "all_twelve_references_remain_in_own_articulation",
        "reset_readback": "selected_native_root_transforms_checked_after_each_reset",
        "reset_position_tolerance_m": RESET_POSITION_TOLERANCE_M,
        "live_collision_response_claim": False,
        "scope": "flat_physics_and_contact_sensors_only_no_visual_visibility_claim",
    }


def validate_layout_runtime(runtime, requested_layout):
    """Host-safe strict selection check, including the unchanged historical grid."""
    if not isinstance(runtime, dict) or requested_layout not in LAYOUTS:
        raise ValueError("Invalid runtime/layout selection")
    actual = runtime.get("resolved_environment_layout")
    if requested_layout == GRID_LAYOUT:
        if "resolved_environment_layout" in runtime:
            raise ValueError("Grid request resolved to a different layout")
        return {"layout_id": GRID_LAYOUT, "historical_manifest_without_layout_field": True}
    expected = layout_runtime_descriptor()
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("Coincident layout runtime identity or startup proof differs")
    return copy.deepcopy(expected)


def validate_layout_evidence(evidence, num_envs):
    """Host-side completeness check of the full-batch initial reset snapshot."""
    if (type(num_envs) is not int or num_envs < 1 or not isinstance(evidence, dict)
            or evidence.get("schema") != "hexapod.environment_layout_evidence.v1"
            or evidence.get("layout_id") != COINCIDENT_LAYOUT or evidence.get("num_envs") != num_envs
            or evidence.get("startup_scene_verified") is not True):
        raise ValueError("Coincident layout startup evidence is incomplete")
    paths = [f"/World/envs/env_{i}" for i in range(num_envs)]
    if (evidence.get("environment_prim_paths") != paths
            or evidence.get("terrain_origins_m") != [[0., 0., 0.]]*num_envs
            or evidence.get("scene_clone_poses_xyzw") != [[0., 0., 0., 0., 0., 0., 1.]]*num_envs):
        raise ValueError("Coincident layout actual scene/reset origins differ")
    identity = [[float(i == j) for j in range(4)] for i in range(4)]
    if evidence.get("authored_environment_world_matrices") != [identity]*num_envs:
        raise ValueError("Coincident layout authored transforms differ")
    mapping = evidence.get("native_row_mapping", {})
    names = mapping.get("body_names_in_native_order", [])
    environments = mapping.get("environment_paths_in_native_row_order", [])
    links = mapping.get("native_link_paths", [])
    roots = mapping.get("articulation_prim_paths", [])
    sensors = mapping.get("sensor_body_paths", {})
    if (mapping.get("verified") is not True or len(names) != 31 or len(set(names)) != 31
            or len(environments) != num_envs or set(environments) != set(paths)
            or len(roots) != num_envs or len(links) != num_envs or set(sensors) != set(names)):
        raise ValueError("Coincident native sensor/articulation mapping is incomplete")
    for env, root, row in zip(environments, roots, links):
        if (not isinstance(root, str) or not (root == env+"/Robot" or root.startswith(env+"/Robot/"))
                or len(row) != 31 or len(set(row)) != 31
                or any(not isinstance(path, str) or not path.startswith(env+"/Robot/") for path in row)):
            raise ValueError("Coincident native body paths escape their environment")
    if any(sensors[name] != [row[index] for row in links] for index, name in enumerate(names)):
        raise ValueError("Coincident sensor rows differ from articulation rows")
    ownership = evidence.get("physical_ownership", {})
    records = ownership.get("environments", [])
    if (ownership.get("verified") is not True or ownership.get("environment_count") != num_envs
            or len(records) != num_envs or [r.get("environment_prim_path") for r in records] != paths):
        raise ValueError("Coincident physical ownership proof is incomplete")
    for row in records:
        env = row["environment_prim_path"]
        bodies = row.get("rigid_body_paths", [])
        colliders = row.get("collider_paths", [])
        refs = row.get("mimic_references", [])
        if (set(bodies) != set(links[environments.index(env)]) or not colliders or len(set(colliders)) != len(colliders)
                or any(not any(path == body or path.startswith(body+"/") for body in bodies) for path in colliders)
                or len(refs) != 12 or len({r.get("joint") for r in refs}) != 12
                or any(not r.get("joint", "").startswith(env+"/Robot/Physics/")
                       or not r.get("reference_joint", "").startswith(env+"/Robot/Physics/") for r in refs)):
            raise ValueError("Coincident collider/mimic ownership rows are inconsistent")
    reset = evidence.get("latest_reset") or {}
    ids = reset.get("environment_row_indices")
    if (reset.get("verified") is not True or ids != list(range(num_envs))
            or any(type(i) is not int for i in ids)
            or type(evidence.get("reset_checks")) is not int or evidence["reset_checks"] < 1
            or type(evidence.get("reset_rows_checked")) is not int or evidence["reset_rows_checked"] < num_envs):
        raise ValueError("Coincident report lacks a complete initial full-batch reset")
    expected, actual = reset.get("expected_root_poses_xyzw"), reset.get("native_root_poses_xyzw")
    for values in (expected, actual):
        if (not isinstance(values, list) or len(values) != num_envs or any(not isinstance(row, list) or len(row) != 7
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in row) for row in values)):
            raise ValueError("Coincident reset pose arrays are malformed or nonfinite")
    for commanded, observed in zip(expected, actual):
        if (commanded[:2] != [0., 0.] or max(abs(a-b) for a, b in zip(commanded[:3], observed[:3])) > RESET_POSITION_TOLERANCE_M
                or min(max(abs(a-b) for a, b in zip(commanded[3:], observed[3:])),
                       max(abs(a+b) for a, b in zip(commanded[3:], observed[3:]))) > 1e-6):
            raise ValueError("Coincident initial native reset pose differs from its command")
    peak = evidence.get("max_reset_position_error_m")
    if type(peak) not in (int, float) or not math.isfinite(peak) or not 0. <= peak <= RESET_POSITION_TOLERANCE_M:
        raise ValueError("Coincident cumulative reset readback verification failed")
    return {"layout_id": COINCIDENT_LAYOUT, "num_envs": num_envs, "complete_initial_reset_verified": True}


def verify_scene_layout(raw):
    """Read actual USD transforms and both SDK origin buffers after scene setup."""
    import torch
    from pxr import Gf, UsdGeom

    cfg, scene = raw.cfg, raw.scene
    if (cfg.environment_layout != COINCIDENT_LAYOUT or cfg.scene.env_spacing != 0.0
            or cfg.terrain.env_spacing != 0.0 or cfg.terrain.terrain_type != "plane"
            or cfg.terrain.prim_path != "/World/ground"):
        raise ValueError("Coincident layout configuration changed during scene construction")
    paths = list(scene.env_prim_paths)
    if paths != [f"/World/envs/env_{i}" for i in range(raw.num_envs)]:
        raise ValueError("Coincident layout does not contain the complete environment namespaces")
    origins = _tensor(raw._terrain.env_origins)
    poses = _tensor(scene._default_env_pose)
    if (tuple(origins.shape) != (raw.num_envs, 3) or tuple(poses.shape) != (raw.num_envs, 7)
            or not bool(torch.isfinite(origins).all() & torch.isfinite(poses).all())
            or not torch.equal(origins, torch.zeros_like(origins))
            or not torch.equal(poses[:, :6], torch.zeros_like(poses[:, :6]))
            or not torch.equal(poses[:, 6], torch.ones_like(poses[:, 6]))):
        raise ValueError("Scene clone poses and terrain reset origins must both be zero/identity")
    cache, transforms = UsdGeom.XformCache(), []
    for path in paths:
        prim = scene.stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError("Coincident environment prim is missing")
        matrix = cache.GetLocalToWorldTransform(prim)
        if matrix != Gf.Matrix4d(1.0):
            raise ValueError(f"Authored environment transform is not identity: {path}")
        transforms.append([[float(matrix[i][j]) for j in range(4)] for i in range(4)])
    return {
        "schema": "hexapod.environment_layout_evidence.v1", "layout_id": COINCIDENT_LAYOUT,
        "num_envs": raw.num_envs, "environment_prim_paths": paths,
        "scene_clone_poses_xyzw": poses.clone().cpu().tolist(),
        "terrain_origins_m": origins.clone().cpu().tolist(),
        "authored_environment_world_matrices": transforms,
        "startup_scene_verified": True, "reset_checks": 0, "reset_rows_checked": 0,
        "max_reset_position_error_m": 0.0, "latest_reset": None,
        "live_collision_response_verified": False,
    }


def verify_native_sensor_rows(raw, body_paths):
    """Bind each sensor row to the actual native articulation/link row order."""
    roots = list(raw._robot.root_view.prim_paths)
    links = [list(row) for row in raw._robot.root_view.link_paths]
    names = list(raw._robot.body_names)
    if len(roots) != raw.num_envs or len(set(roots)) != raw.num_envs or len(links) != raw.num_envs:
        raise ValueError("Native articulation paths have an invalid environment layout")
    if len(names) != 31 or set(names) != set(body_paths):
        raise ValueError("Native articulation links differ from the 31 physical bodies")
    environments = []
    for index, (root, row) in enumerate(zip(roots, links)):
        match = re.fullmatch(r"(/World/envs/env_[0-9]+)/Robot(?:/.*)?", root)
        if not match:
            raise ValueError("Native articulation root is outside its environment namespace")
        env = match.group(1)
        expected = [f"{env}/Robot/{body_paths[name]}" for name in names]
        if row != expected:
            raise ValueError(f"Native link row/path mapping differs in articulation row {index}")
        environments.append(env)
    if set(environments) != set(raw.scene.env_prim_paths) or len(set(environments)) != raw.num_envs:
        raise ValueError("Native articulation rows omit or duplicate environment namespaces")
    sensors = {}
    if set(raw._body_contact_sensors) != set(names):
        raise ValueError("Contact sensors do not cover every physical body")
    for name, sensor in raw._body_contact_sensors.items():
        actual = list(sensor.body_physx_view.prim_paths)
        expected = [row[names.index(name)] for row in links]
        if actual != expected:
            raise ValueError(f"Contact sensor rows do not match native articulation rows: {name}")
        sensors[name] = actual
    return {"verified": True, "articulation_prim_paths": roots,
            "environment_paths_in_native_row_order": environments,
            "body_names_in_native_order": names, "native_link_paths": links,
            "sensor_body_paths": sensors}


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


def record_reset_readback(raw, env_ids, expected_pose):
    """Check native readback after writes; retain bounded exact reset evidence."""
    import torch
    import warp as wp

    origins = _tensor(raw._terrain.env_origins)
    if not torch.equal(origins, torch.zeros_like(origins)):
        raise ValueError("Coincident reset origins changed after startup")
    ids = _tensor(env_ids).to(dtype=torch.long)
    actual = wp.to_torch(raw._robot.root_view.get_root_transforms()).index_select(0, ids).clone()
    expected = _tensor(expected_pose)
    if (tuple(actual.shape) != (len(ids), 7) or actual.shape != expected.shape
            or not bool(torch.isfinite(actual).all() & torch.isfinite(expected).all())):
        raise ValueError("Native reset root readback is malformed or nonfinite")
    error = (actual[:, :3]-expected[:, :3]).abs().amax() if len(ids) else actual.new_tensor(0.)
    # XYZW q and -q encode the same orientation. This is a reset readback check,
    # not a relaxation of any geometry or solver acceptance threshold.
    quaternion_error = torch.minimum((actual[:, 3:]-expected[:, 3:]).abs().amax(-1),
                                     (actual[:, 3:]+expected[:, 3:]).abs().amax(-1))
    if float(error) > RESET_POSITION_TOLERANCE_M or bool((quaternion_error > 1e-6).any()):
        raise ValueError("Native reset pose differs from the commanded coincident pose")
    report = raw.layout_report
    report["reset_checks"] += 1
    report["reset_rows_checked"] += len(ids)
    report["max_reset_position_error_m"] = max(report["max_reset_position_error_m"], float(error))
    report["latest_reset"] = {"environment_row_indices": ids.cpu().tolist(),
        "expected_root_poses_xyzw": expected.clone().cpu().tolist(),
        "native_root_poses_xyzw": actual.cpu().tolist(), "verified": True}
