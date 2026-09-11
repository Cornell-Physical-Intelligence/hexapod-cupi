"""CPU-only reset geometry gate for the serial CAD asset.

Evaluates the actual URDF collision primitives with full joint/collision
transforms. Heights refer to the bottom-plate root, not a CAD bounding box or
an assumed dynamic equilibrium. This does not simulate contacts or certify
hardware stops, self-collision clearance during motion, or standing torque.
"""
from __future__ import annotations

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSEMBLY = REPO_ROOT / "robot" / "hexapod_mkii_assy"
LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")
KINDS = ("coxa_yaw", "femur_pitch", "tibia_pitch")


def _vector(text: str) -> np.ndarray:
    result = np.array([float(part) for part in text.split()])
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"Expected a finite 3-vector, got {text!r}")
    return result


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    length = float(np.linalg.norm(axis))
    if not math.isfinite(angle) or not math.isclose(length, 1.0, abs_tol=1e-7):
        raise ValueError("Joint axis must be unit length and angle finite")
    x, y, z = axis / length
    skew = np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))
    return np.eye(3) + math.sin(angle) * skew + (1. - math.cos(angle)) * (skew @ skew)


def _origin(element: ET.Element) -> np.ndarray:
    result = np.eye(4)
    origin = element.find("origin")
    if origin is not None:
        result[:3, 3] = _vector(origin.get("xyz", "0 0 0"))
        roll, pitch, yaw = _vector(origin.get("rpy", "0 0 0"))
        # URDF fixed-axis roll, pitch, yaw: Rz(yaw) Ry(pitch) Rx(roll).
        result[:3, :3] = (
            _rotation(np.array((0., 0., 1.)), yaw)
            @ _rotation(np.array((0., 1., 0.)), pitch)
            @ _rotation(np.array((1., 0., 0.)), roll)
        )
    return result


def forward_kinematics(root: ET.Element, positions: dict[str, float]) -> dict[str, np.ndarray]:
    """Root-relative poses for every link; fail on missing/extra joints or cycles."""
    joint_list = root.findall("joint")
    joints = {joint.get("name"): joint for joint in joint_list}
    if len(joints) != len(joint_list) or set(joints) != set(positions):
        raise ValueError("Joint positions must name every unique URDF joint exactly once")
    link_list = root.findall("link")
    links = {link.get("name") for link in link_list}
    children = [joint.find("child").get("link") for joint in joint_list]
    if len(links) != len(link_list) or len(set(children)) != len(children):
        raise ValueError("Duplicate link names or multiply-parented links")
    if links - set(children) != {"body"}:
        raise ValueError("Expected a single body root")
    transforms = {"body": np.eye(4)}
    pending = dict(joints)
    while pending:
        before = len(pending)
        for name, joint in list(pending.items()):
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            if parent not in links or child not in links:
                raise ValueError(f"Unknown link in {name}")
            if parent not in transforms:
                continue
            if joint.get("type") != "revolute" or joint.find("mimic") is not None:
                raise ValueError(f"Expected independent revolute joint: {name}")
            limit = joint.find("limit")
            q = positions[name]
            if not float(limit.get("lower")) <= q <= float(limit.get("upper")):
                raise ValueError(f"Stance outside URDF limits: {name}={q}")
            rotation = np.eye(4)
            rotation[:3, :3] = _rotation(_vector(joint.find("axis").get("xyz")), q)
            transforms[child] = transforms[parent] @ _origin(joint) @ rotation
            del pending[name]
        if before == len(pending):
            raise ValueError("Disconnected or cyclic URDF joint graph")
    if set(transforms) != links:
        raise ValueError("Unreachable link")
    return transforms


def primitive_bottom_z(transform: np.ndarray, shape: ET.Element) -> float:
    """Exact vertical support bound of a transformed sphere, box or cylinder."""
    if shape.tag == "sphere":
        extent = float(shape.get("radius"))
        dimensions = (extent,)
    elif shape.tag == "box":
        size = _vector(shape.get("size"))
        dimensions = size
        extent = float(np.abs(transform[2, :3]) @ (size / 2.))
    elif shape.tag == "cylinder":
        radius, length = float(shape.get("radius")), float(shape.get("length"))
        dimensions = (radius, length)
        axis_z = float(transform[2, 2])
        extent = abs(axis_z) * length / 2. + radius * math.sqrt(max(0., 1. - axis_z**2))
    else:
        raise ValueError(f"Unsupported collision shape: {shape.tag}")
    if any(not math.isfinite(float(value)) or value <= 0 for value in dimensions):
        raise ValueError("Collision dimensions must be finite and positive")
    return float(transform[2, 3] - extent)


def measure_stance(urdf: Path, stance: dict) -> dict:
    """Measure geometry without accepting configured heights as evidence."""
    data = urdf.read_bytes()
    root = ET.fromstring(data)
    positions = {f"{leg}_{kind}": float(stance[f"{kind}_rad"]) for leg in LEGS for kind in KINDS}
    transforms = forward_kinematics(root, positions)
    feet = {leg: [] for leg in LEGS}
    nonfoot = []
    collision_count = 0
    for link in root.findall("link"):
        name = link.get("name")
        for collision in link.findall("collision"):
            geometry = list(collision.find("geometry"))
            if len(geometry) != 1:
                raise ValueError(f"Expected one collision primitive: {name}")
            shape = geometry[0]
            bottom = primitive_bottom_z(transforms[name] @ _origin(collision), shape)
            collision_count += 1
            if name.endswith("_tibia") and shape.tag == "sphere":
                feet[name.split("_", 1)[0]].append(bottom)
            else:
                nonfoot.append((bottom, name))
    if any(len(bottoms) != 2 for bottoms in feet.values()) or not nonfoot:
        raise ValueError("Expected two foot-pad spheres per leg and non-foot collisions")
    foot_bottoms = {leg: min(bottoms) for leg, bottoms in feet.items()}
    geometric_height = -min(foot_bottoms.values())
    reset_height = float(stance["reset_root_height_m"])
    nominal_height = float(stance["root_height_m"])
    if not all(math.isfinite(value) for value in (reset_height, nominal_height)):
        raise ValueError("Stance heights must be finite")
    return {
        "urdf_sha256": hashlib.sha256(data).hexdigest(),
        "links": len(transforms),
        "joints": len(positions),
        "collision_primitives": collision_count,
        "geometric_ground_contact_root_height_m": geometric_height,
        "configured_nominal_root_height_m": nominal_height,
        "configured_reset_root_height_m": reset_height,
        "foot_bottom_z_at_root_zero_m": foot_bottoms,
        "foot_clearance_at_reset_m": {leg: bottom + reset_height for leg, bottom in foot_bottoms.items()},
        "foot_vertical_spread_m": max(foot_bottoms.values()) - min(foot_bottoms.values()),
        "nonfoot_min_clearance_at_reset_m": min(nonfoot)[0] + reset_height,
        "lowest_nonfoot_link": min(nonfoot)[1],
        "physical_validation": "not_performed",
    }


def measure_reset_jitter(
    urdf: Path,
    stance: dict,
    *,
    half_width_rad: float = 0.03,
    grid_points_per_joint: int = 5,
    soft_limit_factor: float = 0.95,
) -> dict:
    """Sample the inherited per-joint reset jitter without asserting a bound.

    Every serial leg is a separate three-joint branch. Applying each local
    offset triple to all six branches samples each branch's geometry fully;
    it is unnecessary to enumerate 5**18 whole-body combinations for vertical
    clearance alone. This factorization does NOT apply to self-collision,
    support/load sharing or dynamics. Endpoints and interior samples are not
    a proof that no lower clearance exists between samples.

    Hard/soft limit containment is an exact interval check for the complete
    continuous +/- half_width range, independent of the geometric sampling.
    """
    if not math.isfinite(half_width_rad) or half_width_rad < 0:
        raise ValueError("Jitter half-width must be finite and non-negative")
    if not isinstance(grid_points_per_joint, int) or grid_points_per_joint < 3 or grid_points_per_joint % 2 == 0:
        raise ValueError("Jitter grid must have an odd number of points, at least three")
    if not math.isfinite(soft_limit_factor) or not 0 < soft_limit_factor <= 1:
        raise ValueError("Soft-limit factor must be in (0, 1]")
    root = ET.parse(urdf).getroot()
    positions = {f"{leg}_{kind}": float(stance[f"{kind}_rad"]) for leg in LEGS for kind in KINDS}
    forward_kinematics(root, positions)
    intervals = {}
    all_hard, all_soft = True, True
    for joint in root.findall("joint"):
        name = joint.get("name")
        leg, kind = name.split("_", 1)
        expected = {
            "coxa_yaw": ("body", f"{leg}_coxa"),
            "femur_pitch": (f"{leg}_coxa", f"{leg}_femur"),
            "tibia_pitch": (f"{leg}_femur", f"{leg}_tibia"),
        }
        chain = joint.find("parent").get("link"), joint.find("child").get("link")
        if kind not in expected or chain != expected[kind]:
            raise ValueError("Jitter factorization requires six independent three-joint leg branches")
        limit = joint.find("limit")
        low, high = float(limit.get("lower")), float(limit.get("upper"))
        middle, half = (low + high) / 2., (high - low) * soft_limit_factor / 2.
        interval_low, interval_high = positions[name] - half_width_rad, positions[name] + half_width_rad
        hard_ok = low <= interval_low and interval_high <= high
        soft_ok = middle - half <= interval_low and interval_high <= middle + half
        all_hard &= hard_ok; all_soft &= soft_ok
        intervals[name] = {
            "reset_interval_rad": [interval_low, interval_high],
            "soft_limits_rad": [middle - half, middle + half],
            "within_hard_limits": hard_ok,
            "within_soft_limits": soft_ok,
        }
    # Do not silently clamp test poses: clamping would hide a reset defect.
    if not all_hard:
        raise ValueError("Reset jitter interval exceeds URDF hard limits")
    colliders = []
    for link in root.findall("link"):
        name = link.get("name")
        for collision in link.findall("collision"):
            shapes = list(collision.find("geometry"))
            if len(shapes) != 1:
                raise ValueError(f"Expected one collision primitive: {name}")
            shape = shapes[0]
            colliders.append((name, _origin(collision), shape, name.endswith("_tibia") and shape.tag == "sphere"))
    offsets = np.linspace(-half_width_rad, half_width_rad, grid_points_per_joint).tolist()
    per_leg_minimum = {leg: float("inf") for leg in LEGS}
    minimum_foot = {"clearance_m": float("inf")}
    minimum_nonfoot = {"clearance_m": float("inf")}
    root_height = float(stance["reset_root_height_m"])
    for triple in itertools.product(offsets, repeat=3):
        sample_positions = {
            f"{leg}_{kind}": positions[f"{leg}_{kind}"] + triple[index]
            for leg in LEGS for index, kind in enumerate(KINDS)
        }
        transforms = forward_kinematics(root, sample_positions)
        for name, local, shape, is_foot in colliders:
            bottom = primitive_bottom_z(transforms[name] @ local, shape) + root_height
            target = minimum_foot if is_foot else minimum_nonfoot
            if is_foot:
                leg = name.split("_", 1)[0]
                per_leg_minimum[leg] = min(per_leg_minimum[leg], bottom)
            if bottom < target["clearance_m"]:
                target.update({"clearance_m": bottom, "link": name, "joint_offsets_rad": dict(zip(KINDS, triple))})
    nominal = measure_stance(urdf, stance)
    return {
        "joint_jitter_half_width_rad": half_width_rad,
        "nominal_before_jitter_min_foot_clearance_m": min(nominal["foot_clearance_at_reset_m"].values()),
        "method": "Cartesian grid for each independent three-joint leg branch; fixed level root",
        "per_joint_offset_grid_rad": offsets,
        "samples_per_leg": grid_points_per_joint**3,
        "leg_pose_samples": len(LEGS) * grid_points_per_joint**3,
        "whole_body_fk_evaluations": grid_points_per_joint**3,
        "sampled_minimum_foot": minimum_foot,
        "sampled_minimum_nonfoot": minimum_nonfoot,
        "sampled_minimum_foot_clearance_by_leg_m": per_leg_minimum,
        "all_sampled_ground_clearances_positive": min(minimum_foot["clearance_m"], minimum_nonfoot["clearance_m"]) > 0,
        "nominal_5mm_clearance_preserved_at_all_samples": minimum_foot["clearance_m"] >= float(stance["reset_clearance_m"]),
        "continuous_geometry_clearance_proven": False,
        "continuous_joint_intervals_within_hard_limits": all_hard,
        "continuous_joint_intervals_within_soft_limits": all_soft,
        "joint_intervals": intervals,
        "limitations": "No continuous geometric bound, inter-leg collision, contact, dynamic settling, torque or hardware validation; 5 mm applies to the nominal pose before jitter only.",
    }


def audit_stance(urdf: Path, stance: dict) -> dict:
    """Fail closed if source geometry or a derived height has drifted."""
    report = measure_stance(urdf, stance)
    failures = []
    if report["urdf_sha256"] != stance.get("source_urdf_sha256"):
        failures.append("Source URDF hash differs; regenerate and review the stance")
    if report["links"] != 19 or report["joints"] != 18:
        failures.append("Expected 19 links and 18 joints")
    clearance = float(stance["reset_clearance_m"])
    if not math.isfinite(clearance) or clearance <= 0:
        failures.append("Reset clearance must be positive and finite")
    if min(report["foot_clearance_at_reset_m"].values()) < clearance - 1e-10:
        failures.append("At least one foot violates the configured reset clearance")
    if report["nonfoot_min_clearance_at_reset_m"] <= 0:
        failures.append("A non-foot primitive intersects the reset ground plane")
    expected_nominal = math.ceil(report["geometric_ground_contact_root_height_m"] * 1e6) / 1e6
    expected_reset = math.ceil((report["geometric_ground_contact_root_height_m"] + clearance) * 1e6) / 1e6
    if not math.isclose(float(stance["root_height_m"]), expected_nominal, rel_tol=0., abs_tol=1e-10):
        failures.append("Nominal height is not the geometric contact height rounded upward to 1 um")
    if not math.isclose(float(stance["reset_root_height_m"]), expected_reset, rel_tol=0., abs_tol=1e-10):
        failures.append("Reset height is not geometric contact height + clearance rounded upward to 1 um")
    if failures:
        raise ValueError("; ".join(failures))
    jitter = measure_reset_jitter(urdf, stance)
    if not jitter["continuous_joint_intervals_within_soft_limits"]:
        raise ValueError("Reset jitter interval exceeds articulation soft limits")
    if not jitter["all_sampled_ground_clearances_positive"]:
        raise ValueError("At least one sampled reset jitter pose intersects the ground")
    report["reset_jitter_sampling"] = jitter
    report["status"] = "passed_geometry_only"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=ASSEMBLY / "urdf/hexapod_mkii_serial.urdf")
    parser.add_argument("--stance", type=Path, default=ASSEMBLY / "stance_v2.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_stance(args.urdf, json.loads(args.stance.read_text()))
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
