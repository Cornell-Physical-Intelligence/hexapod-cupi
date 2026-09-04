"""Read-only CPU audit of the CAD tibia four-bar's documented cut hinge.

Endpoints and axes come from leg_reference.json and leg_parts.json, exactly
the source used by the importer to emit its LOOP CLOSURE comments. No endpoint
is fitted to make closure pass. Mimic FK describes visual kinematics; it does
not create a physical closed-loop constraint or validate actuator dynamics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from audit_mkii_stance import _origin, _rotation, _vector

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY = ROOT / "robot/hexapod_mkii_assy"
REFERENCE = ROOT / "robot/hexapod_leg_v3"
LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")
INDEPENDENT = ("coxa_yaw", "femur_pitch", "tibia_pitch")


def cut_reference(reference_path: Path, parts_path: Path) -> dict:
    """Express the recorded physical cut point/axis in each source link frame."""
    reference = json.loads(reference_path.read_text())
    parts = json.loads(parts_path.read_text())
    point = np.array(reference["loop_closure"]["cut_point_m"], dtype=float)
    axis = np.array(reference["loop_closure"]["cut_axis_dir"], dtype=float)
    if point.shape != (3,) or axis.shape != (3,) or not np.isfinite([point, axis]).all():
        raise ValueError("Cut reference must contain finite three-vectors")
    axis /= np.linalg.norm(axis)
    result = {}
    for name in ("femur", "tibia", "tibia_pushrod"):
        frame = np.array(parts["bodies"][name]["frame_in_export"], dtype=float)
        if frame.shape != (4, 4) or not np.isfinite(frame).all():
            raise ValueError(f"Invalid leg reference frame: {name}")
        inverse = np.linalg.inv(frame)
        result[name] = {"point": (inverse @ np.r_[point, 1.0])[:3], "axis": inverse[:3, :3] @ axis}
    return result


def verify_cut_comments(text: str, cut: dict) -> None:
    matches = re.findall(r"(lf|lm|lr|rf|rm|rr) LOOP CLOSURE.*?pushrod point \[(.*?)\].*?tibia point \[(.*?)\]", text)
    if len(matches) != 6 or {row[0] for row in matches} != set(LEGS):
        raise ValueError("Expected one source-derived cut-point comment per leg")
    for leg, rod, tibia in matches:
        for name, value in (("tibia_pushrod", rod), ("tibia", tibia)):
            if not np.allclose(_vector(value), cut[name]["point"], atol=1e-8, rtol=0):
                raise ValueError(f"{leg}: {name} cut comment differs from the leg reference")


def resolve_positions(root: ET.Element, independent: dict[str, float], *, enforce_limits=True) -> dict[str, float]:
    """Apply explicit mimic coefficients recursively; never infer motor indices."""
    joint_list = root.findall("joint")
    joints = {joint.get("name"): joint for joint in joint_list}
    if len(joints) != len(joint_list) or None in joints:
        raise ValueError("Joint names must be unique and present")
    expected = {name for name, joint in joints.items() if joint.find("mimic") is None}
    if set(independent) != expected:
        raise ValueError("Positions must specify every independent joint exactly once")
    positions, active = {}, set()

    def resolve(name):
        if name in positions:
            return positions[name]
        if name not in joints or name in active:
            raise ValueError("Missing mimic source or cyclic mimic relation")
        active.add(name)
        joint = joints[name]
        mimic = joint.find("mimic")
        value = (float(independent[name]) if mimic is None else
                 resolve(mimic.get("joint")) * float(mimic.get("multiplier", "1")) + float(mimic.get("offset", "0")))
        limit = joint.find("limit")
        if (not math.isfinite(value) or limit is None or
                (enforce_limits and not float(limit.get("lower")) <= value <= float(limit.get("upper")))):
            raise ValueError(f"Nonfinite or out-of-limit joint: {name}")
        positions[name] = value
        active.remove(name)
        return value

    for name in joints:
        resolve(name)
    return positions


def forward_kinematics(root: ET.Element, independent: dict[str, float], *, enforce_limits=True) -> tuple[dict, dict]:
    positions = resolve_positions(root, independent, enforce_limits=enforce_limits)
    links = [link.get("name") for link in root.findall("link")]
    joints = root.findall("joint")
    children = [joint.find("child").get("link") for joint in joints]
    if len(set(links)) != len(links) or len(set(children)) != len(children) or set(links) - set(children) != {"body"}:
        raise ValueError("Linkage must be a tree with one body root and unique links")
    transforms = {"body": np.eye(4)}
    pending = list(joints)
    while pending:
        count = len(pending)
        for joint in pending[:]:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in links or child not in links or joint.get("type") != "revolute":
                raise ValueError("Expected known links and revolute joints")
            if parent not in transforms:
                continue
            rotation = np.eye(4)
            rotation[:3, :3] = _rotation(_vector(joint.find("axis").get("xyz")), positions[joint.get("name")])
            transforms[child] = transforms[parent] @ _origin(joint) @ rotation
            pending.remove(joint)
        if len(pending) == count:
            raise ValueError("Disconnected or cyclic linkage")
    if set(transforms) != set(links):
        raise ValueError("Unreachable linkage link")
    return transforms, positions


def closure_measurements(transforms: dict, cut: dict, *, serial=False) -> dict:
    rows = {}
    for leg in LEGS:
        rod_name = "femur" if serial else "tibia_pushrod"
        rod, tibia = transforms[f"{leg}_{rod_name}"], transforms[f"{leg}_tibia"]
        rod_point = (rod @ np.r_[cut[rod_name]["point"], 1.0])[:3]
        tibia_point = (tibia @ np.r_[cut["tibia"]["point"], 1.0])[:3]
        delta = rod_point - tibia_point
        axis = tibia[:3, :3] @ cut["tibia"]["axis"]
        axis /= np.linalg.norm(axis)
        rod_axis = rod[:3, :3] @ cut[rod_name]["axis"]
        rod_axis /= np.linalg.norm(rod_axis)
        axial = float(delta @ axis)
        rows[leg] = {
            "point_residual_mm": float(np.linalg.norm(delta) * 1000),
            "axial_offset_mm": axial * 1000,
            "transverse_offset_mm": float(np.linalg.norm(delta - axial * axis) * 1000),
            "cut_axis_misalignment_deg": math.degrees(math.acos(float(np.clip(abs(axis @ rod_axis), 0, 1)))),
        }
    return rows


def audit(linkage_path: Path = ASSEMBLY / "urdf/hexapod_mkii_linkage.urdf", *, samples=181) -> dict:
    if samples < 2:
        raise ValueError("At least two samples are required to include both limit endpoints")
    text = linkage_path.read_text()
    root = ET.fromstring(text)
    cut = cut_reference(REFERENCE / "leg_reference.json", REFERENCE / "leg_parts.json")
    verify_cut_comments(text, cut)
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    relation_errors = []
    for leg in LEGS:
        for suffix, coefficient in (("tibia_lever_pivot", 1.0), ("tibia_rod_pivot", -1.0)):
            joint = joints[f"{leg}_{suffix}"]
            mimic = joint.find("mimic")
            parent, child = (("femur", "tibia_push_lever") if coefficient == 1 else ("tibia_push_lever", "tibia_pushrod"))
            if joint.find("parent").get("link") != f"{leg}_{parent}" or joint.find("child").get("link") != f"{leg}_{child}":
                relation_errors.append(f"{leg}_{suffix}: incorrect linkage parent or child")
            if (mimic is None or mimic.get("joint") != f"{leg}_tibia_pitch" or
                    float(mimic.get("multiplier", "1")) != coefficient or float(mimic.get("offset", "0")) != 0):
                relation_errors.append(f"{leg}_{suffix}: expected q={coefficient:+g}*{leg}_tibia_pitch")
    names = [f"{leg}_{kind}" for leg in LEGS for kind in INDEPENDENT]
    zero = {name: 0.0 for name in names}
    stance = json.loads((ASSEMBLY / "stance_v2.json").read_text())
    stance_positions = {f"{leg}_{kind}": stance[f"{kind}_rad"] for leg in LEGS for kind in INDEPENDENT}
    assembly_report = json.loads((ASSEMBLY / "assembly_report.json").read_text())
    cad = {f"{leg}_{kind}": assembly_report["legs"][leg]["cad_pose_joint_angles_rad"][kind]
           for leg in LEGS for kind in INDEPENDENT}
    cad_outside = {name: value for name, value in cad.items()
                   if not float(joints[name].find("limit").get("lower")) <= value <= float(joints[name].find("limit").get("upper"))}
    snapshots = {}
    for label, positions in (("zero", zero), ("stance", stance_positions), ("measured_cad_angles_with_mimics", cad)):
        transforms, _ = forward_kinematics(root, positions, enforce_limits=label != "measured_cad_angles_with_mimics")
        snapshots[label] = closure_measurements(transforms, cut)
    collected = {leg: [] for leg in LEGS}
    for sample in range(samples):
        fraction = sample / (samples - 1)
        positions = {}
        for name in names:
            limit = joints[name].find("limit")
            lower, upper = float(limit.get("lower")), float(limit.get("upper"))
            # All three axes traverse their full software interval. Upstream
            # yaw/femur rotations must preserve the local closure residual.
            positions[name] = lower if sample == 0 else upper if sample == samples - 1 else lower + fraction * (upper - lower)
        transforms, _ = forward_kinematics(root, positions)
        for leg, row in closure_measurements(transforms, cut).items():
            collected[leg].append(row)
    summary = {leg: {f"max_{field}": max(abs(row[field]) for row in rows)
                     for field in rows[0]} for leg, rows in collected.items()}
    serial_root = ET.parse(ASSEMBLY / "urdf/hexapod_mkii_serial.urdf").getroot()
    serial_snapshots = {}
    for label, positions in (("zero", zero), ("stance", stance_positions)):
        transforms, _ = forward_kinematics(serial_root, positions)
        serial_snapshots[label] = closure_measurements(transforms, cut, serial=True)
    maximum = max(row["max_point_residual_mm"] for row in summary.values())
    return {
        "linkage_urdf_sha256": hashlib.sha256(linkage_path.read_bytes()).hexdigest(),
        "endpoint_sources": ["robot/hexapod_leg_v3/leg_reference.json:loop_closure", "robot/hexapod_leg_v3/leg_parts.json:bodies.*.frame_in_export"],
        "links": len(root.findall("link")), "joints": len(joints), "independent_joints": len(names),
        "mimic_relation_errors": relation_errors, "animated_mimic_topology_pass": not relation_errors,
        "physical_closed_loop_joint_authored": False,
        "full_limit_sweep_samples": samples,
        "reference_cad_angles_outside_software_limits_rad": cad_outside,
        "snapshots": snapshots, "sweep": summary, "serial_merged_rod_snapshots": serial_snapshots,
        "cut_point_coincidence_tolerance_mm": 0.1,
        "cut_point_coincidence_pass": maximum <= 0.1 and not relation_errors,
        "interpretation": "Recorded cut points are not exactly coincident. Axial offset along a hinge line is distinguished from transverse separation; neither this audit nor mimic animation proves a physical closed-loop constraint.",
        "dynamics_scope": "Serial asset merges push-lever and pushrod mass/inertia/collision/visual geometry into femur at its reference pose. It omits their relative motion, constraint reactions and motor/linkage effective inertia. Linkage mimic FK is visualization only; no physical closure joint is authored.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=ASSEMBLY / "urdf/hexapod_mkii_linkage.urdf")
    parser.add_argument("--samples", type=int, default=181)
    parser.add_argument("--require-point-coincidence", action="store_true", help="exit nonzero when the 0.1 mm documented cut-point test fails")
    args = parser.parse_args()
    result = audit(args.urdf, samples=args.samples)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(int(bool(result["mimic_relation_errors"]) or (args.require_point_coincidence and not result["cut_point_coincidence_pass"])))
