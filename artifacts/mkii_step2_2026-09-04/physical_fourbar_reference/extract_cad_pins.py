"""Read actual assembly visual frames; propose, but do not author, physical hinges.

CPU only. This deliberately does not read the obsolete loop_closure cut frame.
Matrices use column vectors: p_export = T_export_from_link @ p_link.
Run from the repository with the original Onshape assembly export available.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np

LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")
WASHER = "m5_nylon_lubricant_filled_nylon_washer__5_5mm_id__10mm_od__1mm_t.stl"
SEEDS = {"femur": "femur_plate.stl", "tibia": "machined_tibia.stl",
         "tibia_push_lever": "tibia_push_lever.stl", "tibia_pushrod": "tibia_pushrod.stl"}


def unit(value):
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all() or np.linalg.norm(value) < 1e-12:
        raise ValueError("Invalid direction")
    return value / np.linalg.norm(value)


def angle(a, b):
    return math.degrees(math.acos(float(np.clip(abs(unit(a) @ unit(b)), 0, 1))))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def filename(visual):
    mesh = visual.find("geometry/mesh")
    if mesh is None:
        raise ValueError("Expected CAD mesh visual")
    if mesh.get("scale") is not None and not np.allclose([float(x) for x in mesh.get("scale").split()], 1, atol=0, rtol=0):
        raise ValueError("Unexpected CAD mesh scaling")
    return Path(mesh.get("filename")).name


def stl_evidence(path):
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * n:
        raise ValueError("Expected binary STL")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    vertices = np.frombuffer(data, offset=84, dtype=dtype, count=n)["vertices"].reshape(-1, 3).astype(float)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    result = {"sha256": sha(path), "triangles": n, "bounds_min_m": low.tolist(), "bounds_max_m": high.tolist()}
    if path.name == WASHER:
        # Centered circular washer: its 10 mm diameter and 1.1 mm thickness
        # provide independent evidence that CAD local Z is its bore normal.
        extents = high - low
        if not (np.allclose(extents[:2], .010, atol=2e-7, rtol=0) and abs(extents[2] - .0011) < 2e-7):
            raise ValueError("Washer mesh no longer supports the local-Z pin-axis inference")
        if np.linalg.norm((low + high) / 2) > 2e-7:
            raise ValueError("Washer origin is no longer its geometric center")
        _, vectors = np.linalg.eigh(np.cov(vertices.T))
        result["pca_thin_axis_abs_dot_local_z"] = abs(float(vectors[:, 0] @ [0, 0, 1]))
        if result["pca_thin_axis_abs_dot_local_z"] < 1 - 1e-10:
            raise ValueError("Washer normal changed")
    return result


def line_metrics(a, b, axis):
    delta = np.asarray(b["point_export_m"]) - a["point_export_m"]
    axial = float(delta @ axis)
    return {"transverse_gap_mm": float(np.linalg.norm(delta - axial * axis) * 1000),
            "signed_axial_offset_mm": axial * 1000,
            "axis_misalignment_deg": angle(a["axis_export_unit"], b["axis_export_unit"])}


def validate_measurements(pins, canonical, axis):
    errors = []
    for pin_name, pin in pins.items():
        point = np.asarray(pin["point_export_m"])
        target = canonical[pin_name[0]]
        delta = target - point
        radial = np.linalg.norm(delta - axis * (delta @ axis)) * 1000
        if radial > .005:
            errors.append(f"{pin_name}: canonical pin line moves more than 0.005 mm transversely")
        if angle(pin["axis_export_unit"], axis) > .005:
            errors.append(f"{pin_name}: canonical axis changes more than 0.005 degrees")
    return errors


def extract(repo, export):
    sys.path.insert(0, str(repo / "tools"))
    from audit_mkii_linkage import forward_kinematics
    from audit_mkii_stance import _origin, _rotation, _vector

    urdf_path = repo / "robot/hexapod_mkii_assy/urdf/hexapod_mkii_linkage.urdf"
    report_path = repo / "robot/hexapod_mkii_assy/assembly_report.json"
    source_path = export / "robot.urdf"
    root = ET.parse(urdf_path).getroot()
    report = json.loads(report_path.read_text())
    source = ET.parse(source_path).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    cad = {f"{leg}_{kind}": report["legs"][leg]["cad_pose_joint_angles_rad"][kind]
           for leg in LEGS for kind in ("coxa_yaw", "femur_pitch", "tibia_pitch")}
    approximate, _ = forward_kinematics(root, cad, enforce_limits=False)
    source_by_mesh = defaultdict(list)
    for i, visual in enumerate(source.findall("link/visual")):
        source_by_mesh[report["mesh_names"].get(filename(visual), filename(visual))].append((i, _origin(visual)))

    result = {
        "schema": "hexapod.cad_fourbar_pin_proposal.v1",
        "status": "CPU geometry inference; no physical USD authored or simulated",
        "coordinate_convention": "Column-vector 4x4 transforms. Export frame equals body URDF frame; meters and radians unless field suffix says otherwise.",
        "sources": {str(p.relative_to(repo)): {"sha256": sha(p)} for p in (
            urdf_path, report_path, repo / "tools/audit_mkii_linkage.py",
            repo / "tools/audit_mkii_stance.py", Path(__file__).resolve())},
        "external_export": {"path_at_extraction": str(source_path), "sha256": sha(source_path), "visual_instances": sum(map(len, source_by_mesh.values()))},
        "method": [
            "Use reported CAD angles and existing mimic FK ONLY to identify which source mesh belongs to each leg; do not use it to infer hinge locations.",
            "Recover exact body pose from its unique structural visual: T_export_from_link = T_export_visual @ inverse(T_link_visual).",
            "Read centered washer visual origins and local Z bore normals, lever CAD origin and local Z, and knee bearing-insert CAD origin and local Z.",
            "Match each pin visual against the original full assembly export and record independent transform residuals.",
            "Project pin points along a common signed motor-axis direction into the plane through A; axial offsets are hinge-origin gauge choices, not body displacement.",
            "Record a candidate exact parallelogram with AB=DC=0.030 m and AD=BC=0.0775 m; retain measured ground and lever directions, and quantify every transverse regularization.",
        ],
        "canonicalization_limits": {"max_transverse_change_mm": .005, "max_axis_change_deg": .005,
                                    "scope": "Numerical CAD export regularization guard; not a manufacturing or field performance tolerance."},
        "mesh_evidence": {}, "legs": {},
        "mass_kg_by_link": {name: float(link.find("inertial/mass").get("value")) for name, link in links.items()},
    }
    used_meshes = set(SEEDS.values()) | {WASHER, "bearing_insert.stl", "motor_0001755650_00mini_000.stl", "motor_6706__300001_1_1_00mini_000.stl"}
    result["mesh_evidence"] = {mesh: stl_evidence(repo / "robot/hexapod_mkii_assy/meshes" / mesh) for mesh in sorted(used_meshes)}
    all_frames = {}
    maxima = defaultdict(float)
    for leg in LEGS:
        bodies = {}
        for suffix, mesh in SEEDS.items():
            name = f"{leg}_{suffix}"
            matches = [(i, v) for i, v in enumerate(links[name].findall("visual")) if filename(v) == mesh]
            if len(matches) != 1 or len(source_by_mesh[mesh]) != 6:
                raise ValueError(f"Ambiguous structural seed: {name}/{mesh}")
            visual_index, visual = matches[0]
            local = _origin(visual)
            expected = approximate[name] @ local
            candidates = sorted(source_by_mesh[mesh], key=lambda item: np.linalg.norm(item[1][:3, 3] - expected[:3, 3]))
            distance = float(np.linalg.norm(candidates[0][1][:3, 3] - expected[:3, 3]))
            next_distance = float(np.linalg.norm(candidates[1][1][:3, 3] - expected[:3, 3]))
            if distance > .002 or next_distance < .05:
                raise ValueError(f"CAD instance matching is not unambiguous: {name}")
            source_index, source_pose = candidates[0]
            exact = source_pose @ np.linalg.inv(local)
            bodies[name] = {"export_from_link_matrix": exact.tolist(), "structural_seed_mesh": mesh,
                            "link_visual_index_zero_based": visual_index, "source_visual_index_zero_based": source_index,
                            "source_export_from_mesh_matrix": source_pose.tolist(),
                            "link_from_mesh_matrix": local.tolist(),
                            "identification_fk_origin_error_mm": distance * 1000,
                            "next_candidate_distance_mm": next_distance * 1000}
            all_frames[name] = exact

        def pin(suffix, mesh, occurrence=0):
            name = f"{leg}_{suffix}"
            candidates = [(i, v) for i, v in enumerate(links[name].findall("visual")) if filename(v) == mesh]
            index, visual = candidates[occurrence]
            local, body = _origin(visual), all_frames[name]
            world = body @ local
            nearest = min(source_by_mesh[mesh], key=lambda item: np.linalg.norm(item[1][:3, 3] - world[:3, 3]))
            origin_error = float(np.linalg.norm(nearest[1][:3, 3] - world[:3, 3]) * 1000)
            normal_error = angle(nearest[1][:3, 2], world[:3, 2])
            if origin_error > .002 or normal_error > .002:
                raise ValueError(f"Pin visual does not independently register to CAD: {name}/{mesh}")
            maxima["pin_visual_source_origin_residual_mm"] = max(maxima["pin_visual_source_origin_residual_mm"], origin_error)
            maxima["pin_visual_source_axis_residual_deg"] = max(maxima["pin_visual_source_axis_residual_deg"], normal_error)
            return {"link": name, "mesh": mesh, "link_visual_index_zero_based": index,
                    "source_visual_index_zero_based": nearest[0], "source_export_from_mesh_matrix": nearest[1].tolist(),
                    "point_link_m": local[:3, 3].tolist(), "axis_link_unit": unit(local[:3, 2]).tolist(),
                    "point_export_m": world[:3, 3].tolist(), "axis_export_unit": unit(world[:3, 2]).tolist(),
                    "source_origin_residual_mm": origin_error, "source_axis_residual_deg": normal_error}

        pins = {"A_lever": pin("tibia_push_lever", "tibia_push_lever.stl"),
                "B_lever": pin("tibia_push_lever", WASHER), "C_tibia": pin("tibia", WASHER),
                "D_tibia": pin("tibia", "bearing_insert.stl")}
        rods = [pin("tibia_pushrod", WASHER, i) for i in range(2)]
        rods.sort(key=lambda p: np.linalg.norm(np.array(p["point_export_m"]) - pins["B_lever"]["point_export_m"]))
        pins["B_rod"], pins["C_rod"] = rods

        axis = unit(pins["A_lever"]["axis_export_unit"])
        old = joints[f"{leg}_tibia_lever_pivot"]
        old_axis = approximate[f"{leg}_femur"][:3, :3] @ _origin(old)[:3, :3] @ _vector(old.find("axis").get("xyz"))
        if axis @ old_axis < 0:
            axis = -axis
        a = np.array(pins["A_lever"]["point_export_m"])
        motor_output = pin("tibia_push_lever", "motor_0001755650_00mini_000.stl")
        bearings = [pin("femur", "motor_6706__300001_1_1_00mini_000.stl", i) for i in range(2)]
        bearings.sort(key=lambda p: line_metrics(pins["A_lever"], p, axis)["transverse_gap_mm"])
        motor_evidence = {"output_hub_on_lever": motor_output, "bearing_on_femur": bearings[0],
                          "output_hub_vs_A": line_metrics(pins["A_lever"], motor_output, axis),
                          "bearing_vs_A": line_metrics(pins["A_lever"], bearings[0], axis)}
        for key in ("output_hub_vs_A", "bearing_vs_A"):
            if motor_evidence[key]["transverse_gap_mm"] > .005 or motor_evidence[key]["axis_misalignment_deg"] > .005:
                raise ValueError(f"Motor output/bearing evidence no longer supports A: {leg}")
        def project(point):
            point = np.asarray(point)
            return point - axis * ((point - a) @ axis)
        measured = {name: project(pin_data["point_export_m"]) for name, pin_data in pins.items()}
        b_direction = unit((measured["B_lever"] + measured["B_rod"]) / 2 - a)
        d_direction = unit(measured["D_tibia"] - a)
        canonical = {"A": a, "B": a + .030 * b_direction, "D": a + .0775 * d_direction}
        canonical["C"] = canonical["B"] + canonical["D"] - canonical["A"]
        errors = validate_measurements(pins, canonical, axis)
        if errors:
            raise ValueError(errors)
        per_pin = {}
        for name, data in pins.items():
            delta = canonical[name[0]] - data["point_export_m"]
            per_pin[name] = {"signed_origin_gauge_shift_mm": float(delta @ axis) * 1000,
                             "transverse_regularization_mm": float(np.linalg.norm(delta - axis * (delta @ axis))) * 1000,
                             "axis_regularization_deg": angle(data["axis_export_unit"], axis)}
            maxima["transverse_regularization_mm"] = max(maxima["transverse_regularization_mm"], per_pin[name]["transverse_regularization_mm"])
            maxima["axis_regularization_deg"] = max(maxima["axis_regularization_deg"], per_pin[name]["axis_regularization_deg"])

        world_rotation = np.column_stack((d_direction, unit(np.cross(axis, d_direction)), axis))
        hinge_specs = {"A": ("femur", "tibia_push_lever", "tibia_lever_pivot", True),
                       "B": ("tibia_push_lever", "tibia_pushrod", "tibia_rod_pivot", False),
                       "C": ("tibia_pushrod", "tibia", "tibia_loop_closure", False),
                       "D": ("femur", "tibia", "tibia_pitch", False)}
        hinges = {}
        for letter, (parent, child, suffix, active) in hinge_specs.items():
            world = np.eye(4)
            world[:3, :3], world[:3, 3] = world_rotation, canonical[letter]
            body0, body1 = f"{leg}_{parent}", f"{leg}_{child}"
            local0, local1 = np.linalg.inv(all_frames[body0]) @ world, np.linalg.inv(all_frames[body1]) @ world
            if not np.allclose(all_frames[body0] @ local0, all_frames[body1] @ local1, atol=1e-14, rtol=0):
                raise ValueError("Canonical frame roundtrip failed")
            hinges[letter] = {"joint_name": f"{leg}_{suffix}", "body0": body0, "body1": body1,
                             "actuated": active, "exclude_from_articulation": letter == "C", "axis_token": "Z",
                             "export_from_hinge_matrix": world.tolist(), "body0_from_hinge_matrix": local0.tolist(),
                             "body1_from_hinge_matrix": local1.tolist(), "q_zero": "Raw full-assembly CAD pose, not existing URDF zero or hardware encoder zero"}

        # Exercise the proposed body-local frames independently through a full
        # rotation, including flattened and crossed orientation neighborhoods.
        sweep_max = 0.0
        for q in np.linspace(-math.pi, math.pi, 721):
            rotate = np.eye(4)
            rotate[:3, :3] = _rotation(np.array([0, 0, 1.]), q)
            t = {f"{leg}_femur": all_frames[f"{leg}_femur"]}
            for letter, sign in (("A", 1), ("D", 1), ("B", -1)):
                h = hinges[letter]
                rotation = rotate if sign == 1 else np.linalg.inv(rotate)
                t[h["body1"]] = t[h["body0"]] @ np.array(h["body0_from_hinge_matrix"]) @ rotation @ np.linalg.inv(h["body1_from_hinge_matrix"])
            h = hinges["C"]
            c0 = t[h["body0"]] @ np.array(h["body0_from_hinge_matrix"])
            c1 = t[h["body1"]] @ np.array(h["body1_from_hinge_matrix"])
            sweep_max = max(sweep_max, float(np.linalg.norm(c0[:3, 3] - c1[:3, 3])))
            if abs(abs(c0[:3, 2] @ c1[:3, 2]) - 1) > 1e-12:
                raise ValueError("Canonical sweep lost coaxial alignment")
        if sweep_max > 1e-12:
            raise ValueError("Canonical full-turn kinematic closure failed")
        maxima["canonical_sweep_closure_m"] = max(maxima["canonical_sweep_closure_m"], sweep_max)

        bad_pins = json.loads(json.dumps(pins))
        bad_pins["B_rod"]["point_export_m"] = (np.array(bad_pins["B_rod"]["point_export_m"]) + .001 * d_direction).tolist()
        shift_detected = bool(validate_measurements(bad_pins, canonical, axis))
        bad_pins = json.loads(json.dumps(pins))
        bad_pins["C_rod"]["axis_export_unit"] = unit(axis + .01 * d_direction).tolist()
        tilt_detected = bool(validate_measurements(bad_pins, canonical, axis))
        if not shift_detected or not tilt_detected:
            raise ValueError("Negative control was not detected")

        crank_ground_angle = math.atan2(float(axis @ np.cross(d_direction, b_direction)), float(d_direction @ b_direction))
        singular_canonical = [-crank_ground_angle + k * math.pi for k in (0, 1)]
        cad_knee = report["legs"][leg]["cad_pose_joint_angles_rad"]["tibia_pitch"]

        def length(x, y):
            return float(np.linalg.norm(measured[y] - measured[x])) * 1000
        result["legs"][leg] = {
            "bodies": bodies, "measured_pins": pins,
            "measured_hinge_line_agreement": {"B": line_metrics(pins["B_lever"], pins["B_rod"], axis), "C": line_metrics(pins["C_rod"], pins["C_tibia"], axis)},
            "measured_projected_lengths_mm": {"AB": length("A_lever", "B_lever"), "BC": length("B_rod", "C_rod"), "DC": length("D_tibia", "C_tibia"), "AD": length("A_lever", "D_tibia")},
            "measured_planar_loop_residual_mm": float(np.linalg.norm(measured["A_lever"] + measured["C_tibia"] - measured["B_lever"] - measured["D_tibia"])) * 1000,
            "canonical_common_positive_axis_export": axis.tolist(), "canonical_regularization": per_pin,
            "proposed_hinges": hinges,
            "existing_reported_cad_joint_angles_rad": report["legs"][leg]["cad_pose_joint_angles_rad"],
            "canonical_axis_vs_existing_motor_axis_deg": angle(axis, old_axis),
            "motor_A_evidence": motor_evidence,
            "toggle_diagnostic": {"cad_signed_ground_to_crank_angle_rad": crank_ground_angle,
                "canonical_q_at_nearest_flattened_parallelograms_rad": singular_canonical,
                "approximate_old_urdf_tibia_q_at_toggles_rad": [q + cad_knee for q in singular_canonical],
                "old_coordinate_conversion_scope": "Approximate q_old=q_canonical+reported CAD knee angle; must calibrate an exact frame-angle bridge in the physical asset builder."},
            "checks": {"canonical_frame_roundtrip_pass": True, "full_rotation_samples": 721, "full_rotation_closure_max_m": sweep_max,
                       "detects_1mm_transverse_shift": shift_detected, "detects_0_01rad_axis_tilt": tilt_detected},
        }
    result["summary_maxima"] = dict(maxima)
    result["total_mass_kg"] = sum(result["mass_kg_by_link"].values())
    result["articulation_contract"] = {"rigid_bodies": 31, "tree_revolute_coordinates": 30, "external_revolute_closures": 6,
        "independent_actuators": 18, "active_suffixes": ["coxa_yaw", "femur_pitch", "tibia_lever_pivot"],
        "passive_tree_suffixes": ["tibia_pitch", "tibia_rod_pivot"], "remove_all_12_mimic_relations": True,
        "existing_serial_task_compatible": False}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = extract(args.repo.resolve(), args.export.resolve())
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.out), "summary_maxima": output["summary_maxima"], "mass_kg": output["total_mass_kg"]}, indent=2))
