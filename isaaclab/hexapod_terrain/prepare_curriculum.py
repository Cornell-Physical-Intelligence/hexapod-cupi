"""Prepare deterministic mild terrain/reset artifacts; no simulator or PPO.

Run with PYTHONPATH=tools:isaaclab python -m hexapod_terrain.prepare_curriculum.
Original fixtures remain immutable. Derived height scales and measured slopes
are recorded explicitly; nominal slope metadata never substitutes for geometry.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from experiments.terrain.tools.terrain_fixture_checks import digest, load_catalog, vertical_surface_heights
from experiments.terrain.tools.terrain_readiness import mesh_usda, validate_mesh
from .fixture_adapter import MildTerrainSpec
from .robot_smoke import check_start_footprint

LEVELS = (
    dict(level=0, name="continuous_small", families=["ramp", "smooth_rough"], height_scale=.25,
         speed_range_mps=[.05, .08], yaw_rate_limit_rad_s=.12),
    dict(level=1, name="continuous_medium", families=["ramp", "smooth_rough"], height_scale=.5,
         speed_range_mps=[.05, .10], yaw_rate_limit_rad_s=.15),
    dict(level=2, name="mild_mixed", families=["ramp", "smooth_rough", "step", "ridge"], height_scale=1.,
         speed_range_mps=[.05, .12], yaw_rate_limit_rad_s=.20),
)
START_XY = np.array([-1.075, 0.])
START_JITTER_M = .02
FLAT_PAD_BOUNDS = np.array([[-1.49, -1.49], [-.66, 1.49]])


def surface_geometry_metrics(vertices, faces):
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    usable = np.abs(normals[:, 2]) > 1e-12
    tangent = np.linalg.norm(normals[usable, :2], axis=1) / np.abs(normals[usable, 2])
    return dict(max_nonvertical_surface_slope_deg=float(np.degrees(np.arctan(tangent.max(initial=0.)))),
                max_abs_vertex_height_m=float(np.abs(vertices[:, 2]).max()),
                vertical_wall_triangle_count=int((~usable).sum()))


def derive_mild_fixture(entry, vertices, faces, level):
    if entry["family"] == "pit" or entry.get("curriculum") == "avoidance_only":
        raise ValueError("Pits cannot become walking fixtures by shrinking their height")
    if entry["family"] not in level["families"]:
        raise ValueError("Fixture family is outside this curriculum level")
    original = surface_geometry_metrics(vertices, faces)
    scale = level["height_scale"]
    if entry["family"] in ("ramp", "smooth_rough"):
        tangent = math.tan(math.radians(original["max_nonvertical_surface_slope_deg"]))
        # 0.01 degree margin covers seven-decimal USDA vertex rounding.
        if tangent > 0:
            scale = min(scale, math.tan(math.radians(4.99)) / tangent)
    derived = vertices.copy()
    derived[:, 2] *= scale
    metrics = surface_geometry_metrics(derived, faces)
    metadata = dict(entry)
    metadata.pop("sha256", None)
    metadata.pop("usda", None)
    metadata.update(id=f"mild{level['level']}_{entry['id']}", source_id=entry["id"],
        source_usda_sha256=entry["sha256"], level=level["level"], source_difficulty=entry["difficulty"],
        applied_vertical_scale=scale, requested_vertical_scale=level["height_scale"],
        source_geometry_metrics=original, geometry_metrics=metrics, start_xy_m=START_XY.tolist(),
        physical_validation=False, material_parameters="Nominal only; runtime binding and hardware calibration required",
        collision_approximation="none; identical triangle topology; vertex Z scaled explicitly")
    metadata.pop("difficulty", None)
    for field in ("slope_x", "slope_y", "max_abs_height_m", "step_height_m", "ridge_height_m"):
        if field in metadata:
            metadata[field] *= scale
    metadata.update(validate_mesh(derived, faces))
    if entry["family"] in ("ramp", "smooth_rough") and metrics["max_nonvertical_surface_slope_deg"] > 5.:
        raise ValueError("Continuous mild fixture exceeds measured 5-degree local-slope cap")
    return metadata, derived, faces.copy()


def conservative_footprints(package, manifest, record, stance, catalog):
    xml = ET.parse(Path(package) / record["urdf"]).getroot()
    source = load_catalog(catalog, ["train_ramp_1103"])[0]
    spec = MildTerrainSpec(Path(catalog), source[0]["id"], start_body_yaw_rad=0.)
    rows = check_start_footprint(package, xml, manifest, record, stance, spec, source)
    rectangles = []
    for row in rows:
        low, high = np.asarray(row["xy_bounds_m"]) - np.asarray(source[0]["start_xy_m"])
        rectangles.append([[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]])
    return np.asarray(rectangles), [row["leg"] for row in rows]


def transformed_footprint_corners(footprints_body_xy, root_xy, body_yaw):
    c, s = np.cos(body_yaw), np.sin(body_yaw)
    rotation = np.array([[c, -s], [s, c]])
    return np.einsum("...j,ij->...i", footprints_body_xy, rotation) + root_xy


def validate_reset_geometry(reset, footprints_body_xy, fixture_record, *, expected_root_height_m):
    entry, _, vertices, faces = fixture_record
    if entry["family"] == "pit" or entry.get("curriculum") == "avoidance_only":
        raise ValueError("An avoidance fixture cannot host a walking reset")
    if any(reset.get(field) != entry.get(source) for field, source in
           (("fixture_id", "id"), ("split", "split"), ("level", "level"))):
        raise ValueError("Reset identity, split or level does not match its fixture")
    pose = np.asarray(reset["root_position_course_m"], dtype=float)
    yaw = float(reset["body_yaw_in_course_rad"])
    if pose.shape != (3,) or not np.isfinite(pose).all() or not math.isfinite(yaw):
        raise ValueError("Reset pose must be finite XYZ and yaw")
    if not math.isfinite(expected_root_height_m) or expected_root_height_m <= 0 or abs(pose[2] - expected_root_height_m) > 1e-7:
        raise ValueError("Reset root height must match the admitted flat stance over this zero-height pad")
    quaternion = np.asarray(reset["root_quaternion_course_wxyz"], dtype=float)
    expected = np.array([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or min(
            np.linalg.norm(quaternion - expected), np.linalg.norm(quaternion + expected)) > 1e-7:
        raise ValueError("Reset quaternion must match its course-local body yaw")
    points = transformed_footprint_corners(footprints_body_xy, pose[:2], yaw)
    if not ((points >= FLAT_PAD_BOUNDS[0]).all() and (points <= FLAT_PAD_BOUNDS[1]).all()):
        raise ValueError("Full foot footprint is outside the inset flat starting pad")
    heights = vertical_surface_heights(vertices, faces, points.reshape(-1, 2))
    if not np.isfinite(heights).all() or np.max(np.abs(heights)) > 1e-6:
        raise ValueError("Reset footprint has unknown or non-flat geometric support")
    return dict(passed=True, sampled_corners=len(heights), minimum_start_pad_margin_m=float(np.minimum(
        points - FLAT_PAD_BOUNDS[0], FLAT_PAD_BOUNDS[1] - points).min()))


def select_reset_records(manifest, *, split, level):
    """Explicit split selection: training never draws from held-out reset rows."""
    if split not in ("train", "heldout") or level not in {item["level"] for item in manifest["levels"]}:
        raise ValueError("Select an existing level and explicit train/heldout split")
    allowed = set(manifest["fixture_ids_by_split"][split])
    opposite = set(manifest["fixture_ids_by_split"]["heldout" if split == "train" else "train"])
    if allowed & opposite:
        raise ValueError("Train and held-out fixture IDs overlap")
    rows = [r for r in manifest["resets"] if r["split"] == split and r["level"] == level]
    if not rows or any(r["fixture_id"] not in allowed or r["fixture_id"] in opposite for r in rows):
        raise ValueError("Reset references an unknown or wrong-split fixture")
    return rows


def prepare(catalog, package, plan_path, output, *, variant="f050_t060", stance_index=0, seed=20260909, resets_per_fixture=32):
    catalog, package, plan_path, output = map(Path, (catalog, package, plan_path, output))
    if output.exists():
        raise ValueError("Use a fresh output directory; never overwrite geometry snapshots")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or resets_per_fixture < 1:
        raise ValueError("Nonnegative integer seed and positive reset count required")
    source = load_catalog(catalog)
    asset_manifest = json.loads((package / "manifest.json").read_text())
    record = next(r for r in asset_manifest["variants"] if r["variant"] == variant)
    if not np.isclose(record["femur_length_m"], .0725) or not np.isclose(record["tibia_length_m"], .126):
        raise ValueError("This curriculum uses the selected full C study geometry")
    if digest(package / record["urdf"]) != record["sha256"]:
        raise ValueError("Selected study URDF hash mismatch")
    plan = json.loads(plan_path.read_text())
    stance = plan["variants"][variant]["stances"][stance_index]
    footprints, legs = conservative_footprints(package, asset_manifest, record, stance, catalog)
    derived_records, resets = [], []
    # Separate independent random streams even if an upstream catalog reuses a
    # numeric seed across splits. Fixture IDs are checked independently as well.
    generators = {split: np.random.default_rng(np.random.SeedSequence([seed, salt]))
                  for split, salt in (("train", 1103), ("heldout", 17057))}
    for level in LEVELS:
        for entry, path, vertices, faces in source:
            if entry["family"] not in level["families"]:
                continue
            metadata, derived, topology = derive_mild_fixture(entry, vertices, faces, level)
            derived_records.append((metadata, None, derived, topology))
            generator = generators[entry["split"]]
            for index in range(resets_per_fixture):
                xy = START_XY + generator.uniform(-START_JITTER_M, START_JITTER_M, 2)
                yaw = float(generator.uniform(-math.pi, math.pi))
                reset = dict(id=f"{metadata['id']}_reset{index:03d}", split=entry["split"], level=level["level"],
                    fixture_id=metadata["id"], body_yaw_in_course_rad=yaw,
                    root_position_course_m=[float(xy[0]), float(xy[1]), stance["suggested_reset_root_height_m"]],
                    root_quaternion_course_wxyz=[math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)],
                    course_heading_of_body_forward_rad=math.atan2(math.sin(yaw - math.pi / 2), math.cos(yaw - math.pi / 2)))
                reset["start_geometry_check"] = validate_reset_geometry(reset, footprints, derived_records[-1],
                    expected_root_height_m=stance["suggested_reset_root_height_m"])
                resets.append(reset)
    output.mkdir(parents=True)
    fixtures = []
    for metadata, _, vertices, faces in derived_records:
        path = output / "terrains" / (metadata["id"] + ".usda")
        path.parent.mkdir(exist_ok=True)
        path.write_text(mesh_usda(vertices, faces))
        np.savez_compressed(path.with_suffix(".npz"), vertices=vertices, faces=faces)
        fixtures.append(dict(metadata, usda=str(path.relative_to(output)), sha256=digest(path),
                             npz_sha256=digest(path.with_suffix(".npz"))))
    def save(path, value):
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    new_catalog = dict(version=2, status="derived_cpu_geometry_checked_isaac_pending", fixtures=fixtures,
        source_catalog_sha256=digest(catalog), physical_validation=False,
        import_requirements=["No backing plane", "No convexification", "Explicit materials/contact offsets",
                             "Re-run Isaac fixture and exact full-robot standing admissions on these derivatives"])
    save(output / "terrain_catalog.json", new_catalog)
    result = dict(version="mild_terrain_reset_curriculum_v1", ready_for_training=False,
        long_training_started=False, actor_or_task_registered=False,
        source_catalog_sha256=digest(catalog), derived_catalog_sha256=digest(output / "terrain_catalog.json"),
        seed=seed, random_generator="numpy PCG64 with independent SeedSequence streams by split",
        numpy_version=np.__version__, variant=variant, source_urdf_sha256=record["sha256"],
        source_study_manifest_sha256=digest(package / "manifest.json"), source_plan_sha256=digest(plan_path),
        stance_index=stance_index, nominal_joint_positions_rad=stance["joint_positions_rad"],
        joint_order="Runtime joint_names; resolve by name", joint_reset_noise_rad=0.,
        reset_height_m=stance["suggested_reset_root_height_m"], terrain_height_at_start_m=0.,
        body_axes="forward=-Y,left=+X,up=+Z", course_axes="X course forward,Y left,Z up",
        footprint=dict(method="conservative body-XY rectangles enclosing exact distal collision vertices at nominal named stance",
                       legs=legs, rectangles_body_xy_m=footprints.tolist(), flat_start_bounds_course_m=FLAT_PAD_BOUNDS.tolist()),
        course=dict(bounds_xy_m=[[-1.5, -1.5], [1.5, 1.5]], goal_x_m=.85, proposed_episode_length_s=60.,
                    bounds_rule="Entire required body/foot swept envelope must stay inside; outside is unknown, never clipped or backed by a plane",
                    goal_rule="Stop within observed/qualified support; avoid extending targets beyond course boundaries"),
        levels=[dict(level, flat_replay_fraction=.25, local_continuous_slope_cap_deg=5.,
                     command_contract=["v_forward", "v_left", "yaw_rate"],
                     command_bearing="continuous uniform 360 degrees; both pure-yaw signs and mixed twist",
                     advancement="Proposed: held-out per-family/per-bearing traversals and stops, no hard-limit violation, unchanged flat gates; no automatic promotion yet")
                for level in LEVELS],
        fixture_ids_by_split={split: [e["id"] for e in fixtures if e["split"] == split] for split in ("train", "heldout")},
        avoidance_only_source_fixtures={split: [e["id"] for e, *_ in source if e["family"] == "pit" and e["split"] == split]
                                       for split in ("train", "heldout")},
        resets=resets,
        observation_scope="Teacher geometry truth only. Deployed map must retain height/usable/age/uncertainty; do not feed perfect geometry to actor implicitly",
        remaining_gates=["Isaac fixture/contact/sensor validation for derived assets", "Exact full robot standing/reset admissions",
            "Terrain-relative rewards, force feasibility and clearance limits", "Collision atlas/environment origin integration",
            "Realistic sensor and actuator/payload calibration", "Independent held-out qualification"],
        sources={name: digest(path) for name, path in (
            ("prepare_curriculum.py", Path(__file__)), ("robot_smoke.py", Path(__file__).with_name("robot_smoke.py")))})
    for split in ("train", "heldout"):
        for level in LEVELS:
            select_reset_records(result, split=split, level=level["level"])
    save(output / "curriculum.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--study-package", type=Path, required=True)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    result = prepare(args.catalog, args.study_package, args.training_plan, args.output, seed=args.seed)
    print(json.dumps(dict(output=str(args.output), fixtures=sum(map(len, result["fixture_ids_by_split"].values())),
                          resets=len(result["resets"]), ready_for_training=False)))


if __name__ == "__main__":
    main()
