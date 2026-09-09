"""All-angle bounds on the three redundant planar four-bar closure rows.

Reads original physical v3 geometry only. Does not build or admit a D6 asset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import mkii_fourbar_kinematics as kin

ROUNDING_MARGIN = 1e-12
EQUIVALENCE_BOUND_LIMIT = 1e-8


def mm(a, b):
    result = np.einsum("...ij,...jk->...ik", a, b)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite transform")
    return result


def unit(vector):
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or abs(length - 1) > 1e-12:
        raise ValueError("Expected a unit hinge axis")
    return vector / length


def bounds(g):
    """Triangle-inequality bounds valid for all independent A, B, D angles."""
    pA, pB, pD, p0, p1 = [np.asarray(g[key]) for key in ("pA", "pB", "pD", "pC0", "pC1")]
    n, nB, nD, n0, n1 = [unit(np.asarray(g[key])) for key in ("nA", "nB", "nD", "nC0", "nC1")]
    lengths = dict(AB=float(np.linalg.norm(pB-pA)), BC=float(np.linalg.norm(p0-pB)),
                   DC=float(np.linalg.norm(p1-pD)), AD=float(np.linalg.norm(pD-pA)))
    deviation0 = float(np.linalg.norm(n0-nB) + np.linalg.norm(nB-n))
    deviation1 = float(np.linalg.norm(n1-nD) + np.linalg.norm(nD-n))
    baseline = float(abs(n @ (p0-p1)))
    drift = (2*lengths["BC"]*float(np.linalg.norm(np.cross(n, nB)))
             + 2*lengths["DC"]*float(np.linalg.norm(np.cross(n, nD))))
    axial = baseline + drift + deviation0*sum(lengths.values())
    axis = deviation0 + deviation1
    return {"lengths_m": lengths, "baseline_reference_axial_gap_m": baseline,
            "reference_axial_rotation_drift_bound_m": drift,
            "c0_axis_chord_from_reference_bound": deviation0,
            "c1_axis_chord_from_reference_bound": deviation1,
            "all_angle_c0_local_axial_gap_bound_m": axial,
            "all_angle_c_axis_chord_bound": axis,
            "equivalence_within_1e_minus_8": bool(max(axial, axis) < EQUIVALENCE_BOUND_LIMIT)}


def geometry(frames, leg):
    a, b, d, c = [frames[leg+"_"+name] for name in
        ("tibia_lever_pivot", "tibia_rod_pivot", "tibia_pitch", "tibia_loop_closure")]
    ta, tb, td = [kin.relative_pose(frame, 0.) for frame in (a, b, d)]
    transforms = [np.array(a["body0_from_hinge_matrix"]), mm(ta, np.array(b["body0_from_hinge_matrix"])),
                  np.array(d["body0_from_hinge_matrix"]),
                  mm(mm(ta, tb), np.array(c["body0_from_hinge_matrix"])),
                  mm(td, np.array(c["body1_from_hinge_matrix"]))]
    result = {key: value[:3, 3] for key, value in zip(("pA", "pB", "pD", "pC0", "pC1"), transforms)}
    result.update({key: value[:3, 2] for key, value in zip(("nA", "nB", "nD", "nC0", "nC1"), transforms)})
    return result


def relative_batch(frame, angles):
    rotations = np.broadcast_to(np.eye(4), (len(angles), 4, 4)).copy()
    sine, cosine = np.sin(angles), np.cos(angles)
    rotations[:, 0, 0] = rotations[:, 1, 1] = cosine
    rotations[:, 0, 1], rotations[:, 1, 0] = -sine, sine
    return mm(mm(np.array(frame["body0_from_hinge_matrix"]), rotations),
              np.linalg.inv(np.array(frame["body1_from_hinge_matrix"])))


def sample_grid(frames, leg, samples):
    # Independent angles: deliberately do NOT enforce qD=qA or qB=-qA.
    qa, qb, qd = [x.reshape(-1) for x in np.meshgrid(*samples, indexing="ij")]
    a, b, d, c = [frames[leg+"_"+name] for name in
        ("tibia_lever_pivot", "tibia_rod_pivot", "tibia_pitch", "tibia_loop_closure")]
    p0 = mm(mm(relative_batch(a, qa), relative_batch(b, qb)), np.array(c["body0_from_hinge_matrix"]))
    p1 = mm(relative_batch(d, qd), np.array(c["body1_from_hinge_matrix"]))
    difference = p0[:, :3, 3] - p1[:, :3, 3]
    axial = np.einsum("ni,ni->n", difference, p0[:, :3, 2])
    axis = np.linalg.norm(p0[:, :3, 2]-p1[:, :3, 2], axis=-1)
    return {"independent_configurations": len(qa),
            "maximum_c0_local_axial_gap_m": float(np.max(np.abs(axial))),
            "maximum_c_axis_chord": float(np.max(axis)),
            "maximum_full_pin_separation_m": float(np.max(np.linalg.norm(difference, axis=-1)))}


def load_usd_frames(path, contract):
    from pxr import Gf, Usd, UsdPhysics
    stage = Usd.Stage.Open(str(path))
    result = {}
    for name, source in contract["joint_frames"].items():
        joint = UsdPhysics.Joint(stage.GetPrimAtPath("/Robot/Physics/"+name))
        if not joint:
            raise ValueError(f"Missing authored joint {name}")
        frame = dict(source)
        for side in (0, 1):
            quat = getattr(joint, f"GetLocalRot{side}Attr")().Get()
            # Interpret the authored quaternion as a unit rotation.
            rotation = Gf.Rotation(Gf.Quatd(quat).GetNormalized())
            matrix = np.eye(4)
            matrix[:3, :3] = np.array([rotation.TransformDir(Gf.Vec3d(*axis)) for axis in np.eye(3)]).T
            matrix[:3, 3] = getattr(joint, f"GetLocalPos{side}Attr")().Get()
            frame[f"body{side}_from_hinge_matrix"] = matrix.tolist()
        result[name] = frame
    return result


def run():
    np.seterr(all="raise")
    contract = json.loads(kin.CONTRACT.read_text())
    usd = ROOT / contract["usd_path_relative"]
    paths = [kin.CONTRACT, kin.URDF, kin.PINS, usd, Path(__file__).resolve(),
             ROOT / "tools/mkii_fourbar_kinematics.py", ROOT / "tools/audit_mkii_stance.py"]
    hashes = {str(path.relative_to(ROOT)): kin.sha256(path) for path in paths}
    sources = {"kinematic_contract": contract["joint_frames"], "authored_v3_usd": load_usd_frames(usd, contract)}
    result = {}
    for source, frames in sources.items():
        rows = {}
        for leg in kin.LEGS:
            g = geometry(frames, leg)
            row = bounds(g)
            row["zero_geometry_in_femur_frame"] = {key: value.tolist() for key, value in g.items()}
            names = [leg+"_"+name for name in ("tibia_lever_pivot", "tibia_rod_pivot", "tibia_pitch")]
            samples = [np.unique(np.r_[np.linspace(*contract["joint_limits_rad"][name], 17),
                                         contract["default_joint_positions_rad"][name]]) for name in names]
            row["independent_hard_limit_grid"] = sample_grid(frames, leg, samples)
            row["independent_full_rotation_grid"] = sample_grid(frames, leg, [np.linspace(-np.pi, np.pi, 9)]*3)
            for key in ("independent_hard_limit_grid", "independent_full_rotation_grid"):
                if (row[key]["maximum_c0_local_axial_gap_m"] > row["all_angle_c0_local_axial_gap_bound_m"]+ROUNDING_MARGIN
                        or row[key]["maximum_c_axis_chord"] > row["all_angle_c_axis_chord_bound"]+ROUNDING_MARGIN):
                    raise ValueError(f"Grid exceeds analytic bound: {source} {leg}")
            lo, hi = contract["joint_limits_rad"][names[0]]
            toggles = contract["phase_bridge"][leg]["toggle_canonical_q_rad"]
            row["ideal_branch_toggles_rad"] = toggles
            row["ideal_branch_toggle_inside_hard_interval"] = any(lo <= value <= hi for value in toggles)
            row["nearest_toggle_distance_to_hard_interval_rad"] = min(max(lo-value, value-hi, 0.) for value in toggles)
            rows[leg] = row
        result[source] = rows
    if any(kin.sha256(path) != hashes[str(path.relative_to(ROOT))] for path in paths):
        raise ValueError("Source changed during audit")
    return {"schema": "hexapod.fourbar.planar_closure_equivalence.v1",
            "scope": "All-angle geometric bound and independent grids; no dynamic or hardware admission",
            "source_sha256": hashes, "numerical_check_margin": ROUNDING_MARGIN,
            "equivalence_bound_limit_m_and_axis_chord": EQUIVALENCE_BOUND_LIMIT,
            "pass": all(row["equivalence_within_1e_minus_8"] for rows in result.values() for row in rows.values()),
            "sources": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Write a new report; omit for stdout")
    args = parser.parse_args()
    payload = json.dumps(run(), indent=2, allow_nan=False)+"\n"
    if args.out:
        with args.out.open("x") as stream:
            stream.write(payload)
        print(args.out)
    else:
        print(payload, end="")
