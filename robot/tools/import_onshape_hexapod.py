#!/usr/bin/env python3
"""Import the onshape-to-robot full-assembly export as a body plus six proven legs.

The ``hexapod-mkii-assy`` export is ONE link (no ``dof_*`` mate names) holding
every part instance of the robot with exact fused dynamics and per-part CAD
colors, but no structure.  This importer rebuilds the articulated robot from
the canonical single-leg record written by ``import_onshape_leg.py``
(``leg_parts.json`` + ``leg_reference.json``):

1. per leg and per body (leg_base, coxa, femur, tibia, push lever, pushrod)
   rigidly register the leg body's parts onto the export: Kabsch over
   tessellation-independent surface features of parts that are unique within
   the leg, bodies without enough asymmetric unique parts solved through
   their hinge from the registered neighbour (1-DOF search).  Every joint of
   the CAD pose is therefore allowed to differ from the leg export pose;
2. carry every hinge axis through the parent and the child registration and
   verify both map it to the same line; measure the CAD-pose joint angles;
3. assign every export instance to a body by matching it against the
   transformed leg parts (identical mesh: 0.5 mm, same-size fastener whose
   catalogue part changed: 3 mm).  Unmatched instances are chassis parts, or
   new leg parts adopted by keyword;
4. dynamics: every link inertial is the parallel-axis sum of Onshape's exact
   per-part mass properties (mass, centroid, inertia) carried in the export's
   ``robot.pkl``; the assembled total is checked against the export's fused
   inertial and every leg body against the leg record;
5. emit ``hexapod_mkii_linkage.urdf`` (parallelogram closed by mimic joints)
   and ``hexapod_mkii_serial.urdf`` (18-DOF pipeline contract), ASCII-safe
   mesh copies, an assembly report, and a round-trip forward-kinematics check
   that the URDF posed at the measured CAD angles reproduces the export.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import shutil
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent))
from consolidate_leg_urdf import fmt, make_T, matrix_to_rpy, parse_floats  # noqa: E402
from import_onshape_leg import kabsch, norm_stem, stl_mesh, surface_features  # noqa: E402

BODIES = ["leg_base", "coxa", "femur", "tibia", "tibia_push_lever", "tibia_pushrod"]
# name, parent, child, limit key, mimic (joint, multiplier)
LEG_JOINTS = [
    ("coxa_yaw", "leg_base", "coxa", "coxa_yaw", None),
    ("femur_pitch", "coxa", "femur", "femur_pitch", None),
    ("tibia_pitch", "femur", "tibia", "tibia_pitch", None),
    ("tibia_lever_pivot", "femur", "tibia_push_lever", None, ("tibia_pitch", 1.0)),
    ("tibia_rod_pivot", "tibia_push_lever", "tibia_pushrod", None, ("tibia_pitch", -1.0)),
]
# registered neighbour used to chain each body: (joint, neighbour body)
NEIGHBOUR = {"leg_base": ("coxa_yaw", "coxa")}
NEIGHBOUR.update({c: (j, p) for j, p, c, _, _ in LEG_JOINTS if p != "leg_base"})
STEEL, NYLON = 7850.0, 1150.0
CHASSIS_STRUCTURE = ("top_enclosure", "bottom_enclosure", "bottom_plate", "standoff_plate")
LEG_KEYWORDS = {
    "femur": "femur",
    "pushrod": "tibia_pushrod",
    "lever": "tibia_push_lever",
    "tibia": "tibia",
    "silicone": "tibia",
    "coxa": "coxa",
    "first_joint": "coxa",
}
IDENT_TOL = 5.0e-3  # identical mesh (flagged as moved beyond MOVED_TOL)
MOVED_TOL = 0.5e-3
SWAP_TOL = 3.0e-3  # same-size fastener whose catalogue part changed
MOTOR_TOL = 30e-3  # vendor motor internals with a free internal DOF (planet gears)
# numpy 2 on Apple Accelerate raises bogus FP-exception warnings inside matmul
warnings.filterwarnings("ignore", message=".*encountered in matmul")


# ----------------------------------------------------------------- helpers
def to_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def rot_about(point, direction, theta):
    d = direction / np.linalg.norm(direction)
    K = np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return to_T(R, point - R @ point)


def rotz(theta):
    c, s = math.cos(theta), math.sin(theta)
    return to_T(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), np.zeros(3))


def angle_about(R_rel, axis):
    """Signed rotation angle of R_rel about the unit axis."""
    axis = axis / np.linalg.norm(axis)
    seed = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    u = seed - np.dot(seed, axis) * axis
    u /= np.linalg.norm(u)
    v = R_rel @ u
    return math.atan2(np.dot(np.cross(u, v), axis), np.dot(u, v))


def frame_on_axis(point, direction):
    z = direction / np.linalg.norm(direction)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, z)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    x = seed - np.dot(seed, z) * z
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(z, x), z, point
    return T


def mesh_props(tris):
    """Volume, centroid, and unit-density second moment about the mesh origin."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    vol = det.sum() / 6.0
    verts = tris.reshape(-1, 3)
    if abs(vol) < 1e-15:
        return 0.0, verts.mean(axis=0), np.zeros((3, 3))
    cen = ((a + b + c) / 4.0 * det[:, None]).sum(axis=0) / (6.0 * vol)
    s = a + b + c
    S = (
        det[:, None, None]
        / 120.0
        * (
            a[:, :, None] * a[:, None, :]
            + b[:, :, None] * b[:, None, :]
            + c[:, :, None] * c[:, None, :]
            + s[:, :, None] * s[:, None, :]
        )
    ).sum(axis=0)
    if vol < 0:
        vol, S = -vol, -S
    return float(vol), cen, S


def inertia_from_S(S_about_com):
    return np.trace(S_about_com) * np.eye(3) - S_about_com


def S_from_inertia(I_com):
    return 0.5 * np.trace(I_com) * np.eye(3) - I_com


def category(file: str) -> str:
    n = file.lower()
    m = re.search(r"(?:^|[^a-z0-9])m(\d)(?:[^0-9]|$)", n)
    d = f"m{m.group(1)}" if m else "m?"
    if "washer" in n:
        return f"washer_{d}"
    if "nut" in n:
        return f"nut_{d}"
    if "standoff" in n and "plate" not in n:
        return f"standoff_{d}"
    if "shoulder" in n:
        return f"shoulder_{d}"
    if "countersunk" in n:
        return f"countersunk_{d}"
    if "screw" in n:
        return f"screw_{d}"
    return "part:" + norm_stem(file)


def is_fastener(cat: str) -> bool:
    return not cat.startswith("part:")


def safe_mesh_name(file: str) -> str:
    stem = file[:-4] if file.lower().endswith(".stl") else file
    if "fl46blw10" in stem or not stem.isascii():  # vendor motor sub-assembly parts (mojibake names)
        base = stem.split("_fl46blw10")[0]
        base = re.sub(r"_asm.*$", "", base)
        base = "".join(ch for ch in base if ch.isascii())
        stem = "motor_" + base.strip("_")
    stem = re.sub(r"__+[0-9a-f]{32}$", "", stem)
    stem = re.sub(r"_+$", "", stem)
    return stem + ".stl"


class MeshCache:
    def __init__(self, dirs):
        self.dirs = [Path(d) for d in dirs]
        self.cache = {}

    def get(self, file):
        if file not in self.cache:
            path = next(d / file for d in self.dirs if (d / file).exists())
            tris = stl_mesh(path)
            vol, cen, S = mesh_props(tris)
            verts = tris.reshape(-1, 3)
            step = max(1, len(verts) // 400)
            with np.errstate(all="ignore"):
                feats = np.array(surface_features(tris))
            if not np.isfinite(feats).all():
                print(f"  note: {file}: non-finite surface features, using centroid only")
                feats = cen[None, :]
            self.cache[file] = {
                "vol": vol,
                "cen": cen,
                "S": S,
                "verts": verts[::step].copy(),
                "feats": feats,
                "path": path,
            }
        return self.cache[file]


def make_instance(file, xyz, rpy, rgba, mesh):
    T = make_T(xyz, rpy)
    R, t = T[:3, :3], T[:3, 3]
    vol = mesh["vol"]
    cen = R @ mesh["cen"] + t
    M1 = vol * (R @ mesh["cen"])
    S = R @ mesh["S"] @ R.T + np.outer(M1, t) + np.outer(t, M1) + vol * np.outer(t, t)
    return {
        "file": file,
        "stem": norm_stem(file),
        "cat": category(file),
        "xyz": np.array(xyz, dtype=float),
        "rpy": np.array(rpy, dtype=float),
        "T": T,
        "rgba": rgba,
        "vol": vol,
        "cen": cen,
        "feats": mesh["feats"] @ R.T + t,
        "verts": mesh["verts"] @ R.T + t,
        "S": S,  # unit-density second moment about the export origin
    }


def part_dyn(inst, rho):
    """(mass, com, inertia about com) of an instance at density rho, export frame."""
    m = rho * inst["vol"]
    c = inst["cen"]
    S_c = inst["S"] - inst["vol"] * np.outer(c, c)
    return m, c, rho * inertia_from_S(S_c)


def merge_dyn(items):
    """Combine (mass, com, I_about_com) tuples expressed in one frame."""
    mass = sum(m for m, _, _ in items)
    com = sum(m * c for m, c, _ in items) / mass
    I = np.zeros((3, 3))
    for m, c, Ii in items:
        r = c - com
        I += Ii + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return mass, com, I


def transform_dyn(dyn, T):
    m, c, I = dyn
    R = T[:3, :3]
    return m, R @ c + T[:3, 3], R @ I @ R.T


def obb(points):
    """Oriented bounding box of points: (center, rpy, size)."""
    mean = points.mean(axis=0)
    d = points - mean
    _, vec = np.linalg.eigh(d.T @ d / len(d))
    if np.linalg.det(vec) < 0:
        vec[:, 0] = -vec[:, 0]
    proj = d @ vec
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    center = mean + vec @ ((lo + hi) / 2)
    return center, matrix_to_rpy(vec), np.maximum(hi - lo, 1e-3)


# ---------------------------------------------------------------- loading
def load_leg(leg_dir: Path, cache: MeshCache):
    rec = json.loads((leg_dir / "leg_parts.json").read_text())
    ref = json.loads((leg_dir / "leg_reference.json").read_text())
    frames, dyn, parts = {}, {}, {}
    for body, r in rec["bodies"].items():
        frames[body] = np.array(r["frame_in_export"])
        dyn[body] = (float(r["mass_kg"]), np.array(r["com_in_frame"]), np.array(r["inertia_in_frame"]))
        parts[body] = [
            make_instance(p["file"], p["xyz"], p["rpy"], p["rgba"], cache.get(p["file"]))
            for p in r["parts"]
        ]
    axes = {
        "coxa_yaw": (np.array(ref["body_interface"]["yaw_axis_point_m"]), np.array(ref["body_interface"]["yaw_axis_dir"])),
    }
    for j, spec in ref["joints"].items():
        axes[j] = (np.array(spec["axis_point_m"]), np.array(spec["axis_dir"]))
    axes["loop_cut"] = (np.array(ref["loop_closure"]["cut_point_m"]), np.array(ref["loop_closure"]["cut_axis_dir"]))
    return frames, dyn, parts, axes, ref


def load_export(src: Path, cache: MeshCache):
    root = ET.parse(src / "robot.urdf").getroot()
    links = root.findall("link")
    if len(links) != 1 or root.findall("joint"):
        raise SystemExit("expected a single merged link export")
    link = links[0]
    inr = link.find("inertial")
    e = inr.find("inertia")
    fused = (
        float(inr.find("mass").get("value")),
        parse_floats(inr.find("origin").get("xyz"), [0, 0, 0]),
        np.array(
            [
                [float(e.get("ixx")), float(e.get("ixy")), float(e.get("ixz"))],
                [float(e.get("ixy")), float(e.get("iyy")), float(e.get("iyz"))],
                [float(e.get("ixz")), float(e.get("iyz")), float(e.get("izz"))],
            ]
        ),
    )
    insts = []
    for vis in link.findall("visual"):
        mesh = vis.find("geometry/mesh")
        if mesh is None:
            continue
        file = mesh.get("filename").split("/")[-1]
        o = vis.find("origin")
        xyz = parse_floats(o.get("xyz") if o is not None else None, [0, 0, 0])
        rpy = parse_floats(o.get("rpy") if o is not None else None, [0, 0, 0])
        col = vis.find("material/color")
        insts.append(make_instance(file, xyz, rpy, col.get("rgba") if col is not None else None, cache.get(file)))
    return root.get("name"), fused, insts


class _Stub:
    def __setstate__(self, state):
        self.__dict__.update(state)


class _Unpickler(pickle.Unpickler):
    """Loads onshape-to-robot's robot.pkl without importing the package."""

    def find_class(self, module, name):
        if module.startswith("onshape_to_robot"):
            return type(name, (_Stub,), {})
        return super().find_class(module, name)


def load_part_dynamics(src: Path, insts):
    """Attach Onshape's per-part mass, COM and inertia (export frame) to each instance."""
    with open(src / "robot.pkl", "rb") as f:
        rob = _Unpickler(f).load()
    parts = [p for link in rob.links for p in link.parts]
    if len(parts) != len(insts):
        raise SystemExit(f"robot.pkl has {len(parts)} parts, URDF has {len(insts)} visuals")
    for q, p in zip(insts, parts):
        if np.abs(np.asarray(p.T_world_part) - q["T"]).max() > 1e-4:
            raise SystemExit(f"robot.pkl part order differs from the URDF at {p.name}")
        R, t = q["T"][:3, :3], q["T"][:3, 3]
        q["part_name"] = p.name
        q["mass"] = float(p.mass)
        q["com_B"] = R @ np.array(p.com, dtype=float) + t
        q["I_B"] = R @ np.asarray(p.inertia, dtype=float) @ R.T  # about the part COM
    return len(parts)


# ----------------------------------------------------------- registration
def refine(body, R, t, leg_parts, full, gate=2e-3, trim=3e-4):
    """Consensus refinement of a body transform over ALL its identical-stem parts.

    Every leg part is matched to the nearest export instance of the same mesh
    under the current transform (within ``gate``); Kabsch is refit on the
    surface features of the inliers and parts that disagree by more than
    ``trim`` (they moved in the assembly) are excluded and reported.
    """
    by_stem = defaultdict(list)
    for q in full:
        by_stem[q["stem"]].append(q)
    parts = [p for p in leg_parts[body] if p["stem"] in by_stem]
    stats = {"parts": len(parts)}
    for it in range(4):
        P, Q, res, matched = [], [], [], []
        for p in parts:
            moved = R @ p["cen"] + t
            cands = by_stem[p["stem"]]
            d = np.array([np.linalg.norm(q["cen"] - moved) for q in cands])
            j = int(np.argmin(d))
            if d[j] < gate:
                matched.append((p, cands[j], float(d[j])))
        if len(matched) < 3:
            stats.update({"refined": False, "why": f"{len(matched)} matches"})
            return R, t, stats
        r = np.array([m[2] for m in matched])
        thr = max(trim, 3 * float(np.median(r))) if it else gate
        inl = [m for m in matched if m[2] < thr]
        for p, q, _ in inl:
            k = min(len(p["feats"]), len(q["feats"]))
            P.append(p["feats"][:k])
            Q.append(q["feats"][:k])
        P, Q = np.concatenate(P), np.concatenate(Q)
        rank = int(np.linalg.matrix_rank(P - P.mean(axis=0), tol=1e-4)) if len(P) >= 3 else 0
        if len(P) < 3 or rank < 2:
            stats.update({"refined": False, "why": f"rank {rank}"})
            return R, t, stats
        R, t = kabsch(P, Q)
    moved_parts, no_twin = [], 0
    resid = []
    for p in parts:
        moved = R @ p["cen"] + t
        d = min(np.linalg.norm(q["cen"] - moved) for q in by_stem[p["stem"]])
        if d > gate:
            no_twin += 1
            continue
        resid.append(d)
        if d > trim:
            moved_parts.append((p["file"], d * 1000))
    resid = np.array(resid)
    inl = resid[resid <= trim]
    stats.update({
        "refined": True,
        "rms_mm": float(np.sqrt((inl ** 2).mean()) * 1000) if len(inl) else float("nan"),
        "inliers": int(len(inl)),
        "no_identical_counterpart": no_twin,
        "moved": moved_parts,
    })
    return R, t, stats


def register_direct(body, leg_parts, full, counts_leg, counts_full, report):
    """Initial per-leg fits from the parts that are unique within the leg."""
    uniq = [p for p in leg_parts[body] if counts_leg[p["stem"]] == 1 and counts_full.get(p["stem"], 0) == 6]
    if not uniq:
        report.append(f"{body}: no per-leg-unique parts -> hinge solve")
        return {}
    anchor = max(uniq, key=lambda p: p["vol"])
    out = {}
    for a in [q for q in full if q["stem"] == anchor["stem"]]:
        pairs = [(anchor, a)]
        for p in uniq:
            if p is anchor:
                continue
            d0 = np.linalg.norm(p["cen"] - anchor["cen"])
            cands = [
                (abs(np.linalg.norm(q["cen"] - a["cen"]) - d0), q)
                for q in full
                if q["stem"] == p["stem"]
            ]
            cands = [x for x in cands if x[0] < 1e-3]
            if cands:
                pairs.append((p, min(cands, key=lambda x: x[0])[1]))
        P, Q = [], []
        for p, q in pairs:
            k = min(len(p["feats"]), len(q["feats"]))
            P.append(p["feats"][:k])
            Q.append(q["feats"][:k])
        P, Q = np.concatenate(P), np.concatenate(Q)
        rank = int(np.linalg.matrix_rank(P - P.mean(axis=0), tol=1e-4)) if len(P) >= 3 else 0
        if len(P) < 3 or rank < 2:
            report.append(f"{body}: {len(pairs)} unique parts, {len(P)} pts, rank {rank} -> hinge solve")
            return {}
        R, t = kabsch(P, Q)
        res = np.linalg.norm((P @ R.T + t) - Q, axis=1)
        out[id(a)] = {"R": R, "t": t, "seed_rms_mm": float(np.sqrt((res ** 2).mean()) * 1000), "seed_parts": len(pairs)}
    report.append(f"{body}: seeded by Kabsch on {len(uniq)} unique parts "
                  f"(worst seed rms {max(v['seed_rms_mm'] for v in out.values()):.3f} mm)")
    return out


def register_hinge(body, parent_reg, jname, axes, leg_parts, full, counts_full):
    """1-DOF solve of a body about its hinge from the registered neighbour."""
    Rp, tp = parent_reg
    Tp = to_T(Rp, tp)
    ap, ad = axes[jname]
    ap_B = Rp @ ap + tp
    stems = {p["stem"] for p in leg_parts[body] if counts_full.get(p["stem"], 0) > 0}
    cands = np.array([q["cen"] for q in full if q["stem"] in stems and np.linalg.norm(q["cen"] - ap_B) < 0.25])
    olds = np.array([p["cen"] for p in leg_parts[body] if p["stem"] in stems])

    def score(theta):
        T = Tp @ rot_about(ap, ad, theta)
        moved = olds @ T[:3, :3].T + T[:3, 3]
        d = np.linalg.norm(moved[:, None, :] - cands[None, :, :], axis=2).min(axis=1)
        return float(np.clip(d, None, 0.02).mean())

    thetas = np.linspace(-math.pi, math.pi, 1441)
    best = thetas[int(np.argmin([score(x) for x in thetas]))]
    for span in (0.02, 0.002, 0.0002):
        fine = np.linspace(best - span * 10, best + span * 10, 41)
        best = fine[int(np.argmin([score(x) for x in fine]))]
    T = Tp @ rot_about(ap, ad, best)
    return {"R": T[:3, :3], "t": T[:3, 3], "dtheta_deg": math.degrees(best), "seed_fit_mm": score(best) * 1000, "seed_parts": len(olds)}


# ------------------------------------------------------------------ build
def build(args):
    src = Path(args.source)
    leg_dir = Path(args.leg_dir)
    out = Path(args.out)
    pkg = out.name
    cache = MeshCache([src / "assets", leg_dir / "meshes"])
    frames, leg_dyn, leg_parts, axes, ref = load_leg(leg_dir, cache)
    export_name, fused, full = load_export(src, cache)
    load_part_dynamics(src, full)
    limits = {}
    for k, v in json.loads(Path(args.limits).read_text()).items():
        if isinstance(v, dict):
            limits[k] = (float(v["lower"]), float(v["upper"]))
    effort, velocity = float(ref["rs05"]["effort_nm"]), float(ref["rs05"]["velocity_rad_s"])
    print(f"export {export_name}: {len(full)} instances, fused {fused[0]:.5f} kg; leg record: "
          f"{sum(len(v) for v in leg_parts.values())} parts in {len(leg_parts)} bodies")

    counts_leg = Counter(p["stem"] for b in leg_parts for p in leg_parts[b])
    counts_full = Counter(q["stem"] for q in full)
    log = []

    # -- coxa first: it defines the six legs
    regs = {}  # (leg, body) -> (R, t)
    reg_info = {}
    direct = register_direct("coxa", leg_parts, full, counts_leg, counts_full, log)
    if len(direct) != 6:
        raise SystemExit(f"coxa registration found {len(direct)} legs, expected 6")
    yaw_pt, yaw_dir = axes["coxa_yaw"]
    fwd = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}[args.forward]
    f_vec = np.array([fwd[0], fwd[1], 0.0])
    l_vec = np.cross(np.array([0, 0, 1.0]), f_vec)
    named = []
    for v in direct.values():
        R, t, st = refine("coxa", v["R"], v["t"], leg_parts, full)
        p = R @ yaw_pt + t
        named.append(("l" if np.dot(p, l_vec) > 0 else "r", float(np.dot(p, f_vec)), R, t, {"method": "kabsch", **{k: v[k] for k in ("seed_rms_mm", "seed_parts")}, **st}))
    legs = []
    for side in ("l", "r"):
        rows = sorted([x for x in named if x[0] == side], key=lambda x: -x[1])
        for rank, (_, _, R, t, info) in zip(("f", "m", "r"), rows):
            legs.append(side + rank)
            regs[(side + rank, "coxa")] = (R, t)
            reg_info[(side + rank, "coxa")] = info
    if len(legs) != 6:
        raise SystemExit(f"leg naming produced {legs}; check --forward")

    # -- chain the remaining bodies outward from the coxa
    def attach(body):
        jname, parent = NEIGHBOUR[body]
        fits = register_direct(body, leg_parts, full, counts_leg, counts_full, log)
        ap, ad = axes[jname]
        for v in fits.values():
            pc = v["R"] @ ap + v["t"]
            leg = min(legs, key=lambda L: np.linalg.norm(regs[(L, parent)][0] @ ap + regs[(L, parent)][1] - pc))
            if (leg, body) in regs:
                raise SystemExit(f"{body}: two fits claim leg {leg}")
            R, t, st = refine(body, v["R"], v["t"], leg_parts, full)
            regs[(leg, body)] = (R, t)
            reg_info[(leg, body)] = {"method": "kabsch", "seed_rms_mm": v["seed_rms_mm"], "seed_parts": v["seed_parts"], **st}
        for leg in legs:
            if (leg, body) in regs:
                continue
            h = register_hinge(body, regs[(leg, parent)], jname, axes, leg_parts, full, counts_full)
            R, t, st = refine(body, h["R"], h["t"], leg_parts, full)
            regs[(leg, body)] = (R, t)
            reg_info[(leg, body)] = {"method": f"hinge:{jname}", "seed_fit_mm": h["seed_fit_mm"], "dtheta_deg": h["dtheta_deg"], "seed_parts": h["seed_parts"], **st}

    for body in ("leg_base", "femur", "tibia", "tibia_push_lever", "tibia_pushrod"):
        attach(body)
    worst = max(v.get("rms_mm", 0.0) for v in reg_info.values())
    n_moved = sum(len(v.get("moved", [])) for v in reg_info.values())
    print(f"  consensus refinement: worst inlier rms {worst:.4f} mm, {n_moved} part instances moved in the assembly (>0.3 mm)")
    for line in log:
        print("  " + line)

    # -- axis verification and CAD-pose joint angles
    axis_checks = {}
    cad_angles = {}
    up = np.array([0, 0, 1.0])
    for leg in legs:
        for jname, pb, cb, _, _ in LEG_JOINTS:
            ap, ad = axes[jname]
            Rp, tp = regs[(leg, pb)]
            Rc, tc = regs[(leg, cb)]
            pp, dp = Rp @ ap + tp, Rp @ ad
            pc, dc = Rc @ ap + tc, Rc @ ad
            ang = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(np.dot(dp, dc)))))))
            gap = np.linalg.norm((pc - pp) - np.dot(pc - pp, dp) * dp) * 1000
            axis_checks[(leg, jname)] = (ang, gap)
            # URDF joint axis: child-frame z for the leg joints; for the yaw the
            # physical axis oriented upward (positive = counter-clockwise from above)
            urdf_axis = (dp if np.dot(dp, up) > 0 else -dp) if jname == "coxa_yaw" else Rp @ frames[cb][:3, 2]
            cad_angles[(leg, jname)] = angle_about(Rc @ Rp.T, urdf_axis)
    worst_ang = max(v[0] for v in axis_checks.values())
    worst_gap = max(v[1] for v in axis_checks.values())
    print(f"  hinge axes: worst parent/child mismatch {worst_ang:.3f} deg, line gap {worst_gap:.3f} mm")

    # -- link frames in the export frame (CAD pose) and URDF zero pose
    T_link = {}
    for (leg, body), (R, t) in regs.items():
        T_link[(leg, body)] = to_T(R, t) @ frames[body]
    yaw_sign = {}
    for leg in legs:
        z = T_link[(leg, "coxa")][:3, 2]
        yaw_sign[leg] = 1.0 if np.dot(z, up) > 0 else -1.0
        tilt = math.degrees(math.acos(min(1.0, abs(float(np.dot(z, up))))))
        if tilt > 0.5:
            print(f"  WARNING: {leg} yaw axis tilted {tilt:.2f} deg from body z")
    # coxa at zero yaw: the registered coxa with the measured CAD yaw removed
    # (the zero is the leg-record mount pose; cross-checked against the mount
    # registration below)
    T_coxa0 = {leg: T_link[(leg, "coxa")] @ rotz(-yaw_sign[leg] * cad_angles[(leg, "coxa_yaw")]) for leg in legs}
    mount_dev = {}
    for leg in legs:
        D = np.linalg.inv(to_T(*regs[(leg, "leg_base")]) @ frames["coxa"]) @ T_coxa0[leg]
        mount_dev[leg] = (np.linalg.norm(D[:3, 3]) * 1000, math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(D[:3, :3]) - 1) / 2)))))
    # heading = direction from the yaw axis to the hip (femur_pitch) axis, projected on xy
    hip_E = axes["femur_pitch"][0]

    def heading_deg(T_coxa_pose):
        Tc = T_coxa_pose @ np.linalg.inv(frames["coxa"])  # E frame of the leg for this coxa pose
        p = Tc[:3, :3] @ yaw_pt + Tc[:3, 3]
        h = Tc[:3, :3] @ hip_E + Tc[:3, 3]
        return math.degrees(math.atan2(h[1] - p[1], h[0] - p[0]))

    cad_heading, yaw_offset = {}, {}
    for leg in legs:
        p = T_coxa0[leg][:3, 3]
        cad_heading[leg] = heading_deg(T_coxa0[leg])
        radial = math.degrees(math.atan2(p[1], p[0]))
        yaw_offset[leg] = (radial - cad_heading[leg] + 180.0) % 360.0 - 180.0
        if args.yaw_zero == "radial":
            T_coxa0[leg] = rot_about(p, up, math.radians(yaw_offset[leg])) @ T_coxa0[leg]
            cad_angles[(leg, "coxa_yaw")] = angle_about(T_link[(leg, "coxa")][:3, :3] @ T_coxa0[leg][:3, :3].T, up)
    print("  yaw zero: " + args.yaw_zero + "; CAD mount headings relative to radial: "
          + ", ".join(f"{leg} {-yaw_offset[leg]:+.1f} deg" for leg in legs))
    # leg-internal joint transforms MEASURED from the assembly (child frame at
    # zero angle in the parent frame), averaged over the six legs; the leg
    # record's own transforms are kept for comparison
    T_record = {j: np.linalg.inv(frames[p]) @ frames[c] for j, p, c, _, _ in LEG_JOINTS if j != "coxa_yaw"}
    T_internal, joint_dev, record_dev = {}, {}, {}
    for jname, pb, cb, _, _ in LEG_JOINTS[1:]:
        Ts = [np.linalg.inv(T_link[(leg, pb)]) @ T_link[(leg, cb)] @ rotz(-cad_angles[(leg, jname)]) for leg in legs]
        Rsum = sum(T[:3, :3] for T in Ts)
        U, _, Vt = np.linalg.svd(Rsum)
        Rm = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        tm = np.mean([T[:3, 3] for T in Ts], axis=0)
        T_internal[jname] = to_T(Rm, tm)
        for leg, T in zip(legs, Ts):
            D = np.linalg.inv(T_internal[jname]) @ T
            joint_dev[(leg, jname)] = (np.linalg.norm(D[:3, 3]) * 1000, math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(D[:3, :3]) - 1) / 2)))))
        D = np.linalg.inv(T_record[jname]) @ T_internal[jname]
        record_dev[jname] = (D[:3, 3] * 1000, math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(D[:3, :3]) - 1) / 2)))))
    print("  leg-internal joints measured from the assembly (mean of 6 legs); leg-to-leg deviation worst "
          f"{max(v[0] for v in joint_dev.values()):.4f} mm / {max(v[1] for v in joint_dev.values()):.4f} deg")
    for jname, (dt, da) in record_dev.items():
        print(f"    {jname}: assembly vs leg record: shift [{fmt(dt, 3)}] mm, {da:.3f} deg")
    print(f"  mount cross-check (coxa zero-yaw pose from the mount registration vs from the coxa): worst "
          f"{max(v[0] for v in mount_dev.values()):.3f} mm / {max(v[1] for v in mount_dev.values()):.3f} deg")

    # -- assign every export instance
    expected = []
    for (leg, body), (R, t) in regs.items():
        for k, p in enumerate(leg_parts[body]):
            expected.append({"leg": leg, "body": body, "k": k, "stem": p["stem"], "cat": p["cat"], "pt": R @ p["cen"] + t, "file": p["file"]})
    E_pts = np.array([e["pt"] for e in expected])
    E_stem = np.array([e["stem"] for e in expected])
    E_cat = np.array([e["cat"] for e in expected])
    pairs = []
    for qi, q in enumerate(full):
        d = np.linalg.norm(E_pts - q["cen"], axis=1)
        for j in np.nonzero((d < IDENT_TOL) & (E_stem == q["stem"]))[0]:
            pairs.append((0, d[j], qi, int(j)))
        if is_fastener(q["cat"]):
            for j in np.nonzero((d < SWAP_TOL) & (E_cat == q["cat"]) & (E_stem != q["stem"]))[0]:
                pairs.append((1, d[j], qi, int(j)))
        if q["file"][0].isdigit():  # vendor motor internals: free internal DOF (planet gears orbit)
            for j in np.nonzero((d < MOTOR_TOL) & (d >= IDENT_TOL) & (E_stem == q["stem"]))[0]:
                pairs.append((2, d[j], qi, int(j)))
    pairs.sort()
    inst_home = {}
    exp_used = {}
    for prio, d, qi, j in pairs:
        if qi in inst_home or j in exp_used:
            continue
        inst_home[qi] = (expected[j]["leg"], expected[j]["body"], d, prio)
        exp_used[j] = qi
    missing = [expected[j] for j in range(len(expected)) if j not in exp_used]
    assignment = defaultdict(list)  # (leg, body) or ("body", None) -> instances
    new_leg_parts = []
    review = []
    for qi, q in enumerate(full):
        if qi in inst_home:
            leg, body, d, prio = inst_home[qi]
            q["residual_mm"], q["swapped"] = d * 1000, prio == 1
            q["moved"] = prio == 0 and d > MOVED_TOL
            q["internal_dof"] = prio == 2
            assignment[(leg, body)].append(q)
            continue
        d_all = np.linalg.norm(E_pts - q["cen"], axis=1)
        target = next((b for kw, b in LEG_KEYWORDS.items() if kw in q["file"].lower()), None)
        if target is not None:
            mask = np.array([e["body"] == target for e in expected])
            j = int(np.argmin(np.where(mask, d_all, np.inf)))
            leg = expected[j]["leg"]
            assignment[(leg, target)].append(q)
            q["new_part"] = True
            new_leg_parts.append((q, leg, target, d_all[j] * 1000))
        else:
            assignment[("body", None)].append(q)
            q["new_part"] = True
            if d_all.min() < 5e-3:
                review.append((q, expected[int(np.argmin(d_all))], d_all.min() * 1000))
    # -- user-verified attachments the leg record cannot know (new parts that
    # carry no leg keyword): rules {pattern, from, target}; the leg is the one
    # whose target body has the nearest expected part
    overrides = json.loads(Path(args.overrides).read_text()) if args.overrides and Path(args.overrides).exists() else []
    for rule in overrides:
        rx = re.compile(rule["pattern"])
        src_key = ("body", None) if rule.get("from", "body") == "body" else None
        mask = np.array([e["body"] == rule["target"] for e in expected])
        moved_n = 0
        for key in ([src_key] if src_key else [k for k in assignment if k != ("body", None)]):
            keep = []
            for q in assignment[key]:
                if rx.search(q["file"]) and (src_key or key[1] == rule.get("from")):
                    d_all = np.linalg.norm(E_pts - q["cen"], axis=1)
                    j = int(np.argmin(np.where(mask, d_all, np.inf)))
                    assignment[(expected[j]["leg"], rule["target"])].append(q)
                    q["new_part"] = True
                    moved_n += 1
                else:
                    keep.append(q)
            assignment[key] = keep
        print(f"  override {rule['pattern']} -> {rule['target']}: {moved_n} parts")

    # -- yaw motor: only its output side (flange + hub, bolted through the frame)
    # is fixed to the chassis; the motor body, the top plate and everything
    # screwed to them swing with the leg, so they belong to the coxa
    out_rx = re.compile(args.yaw_output_side)
    n_rehomed = 0
    for leg in legs:
        keep, move = [], []
        for q in assignment[(leg, "leg_base")]:
            (keep if out_rx.search(q["file"]) else move).append(q)
        assignment[(leg, "leg_base")] = keep
        assignment[(leg, "coxa")].extend(move)
        n_rehomed += len(move)
    print(f"  yaw motor split: {len(assignment[(legs[0], 'leg_base')])} output-side parts per leg stay on the chassis, "
          f"{n_rehomed // len(legs)} per leg (plate, motor body, screws) rotate with the coxa")
    n_swapped = sum(1 for v in inst_home.values() if v[3] == 1)
    n_moved = sum(1 for q in full if q.get("moved"))
    n_dof = sum(1 for q in full if q.get("internal_dof"))
    print(f"  assignment: {len(inst_home)} matched ({n_swapped} by fastener size class, {n_moved} identical parts moved >0.5 mm, "
          f"{n_dof} motor internals on a free internal DOF), "
          f"{len(new_leg_parts)} new leg parts adopted by keyword, {len(assignment[('body', None)])} chassis parts, "
          f"{len(missing)} leg parts missing from the assembly, {len(review)} chassis parts near a leg (review)")

    # -- dynamics: parallel-axis sums of Onshape's exact per-part mass properties
    def dyn_of(q):
        return (q["mass"], q["com_B"], q["I_B"])

    fused_m, fused_c, fused_I = fused

    # -- actuator mass: the RS05 vendor CAD is a hollow shell (housing, covers,
    # gears: ~60 g).  Hard override: every motor (all parts of the vendor
    # document, assigned to the nearest housing) totals --motor-mass; the
    # missing mass sits on the housing link as a solid cylinder the size of
    # the housing part.  Everything else keeps its CAD mass.
    extras = defaultdict(list)
    motor_report = {"motor_mass_kg": args.motor_mass, "motors": 0, "cad_mass_g": [], "topup_kg": 0.0}
    if args.motor_mass > 0:
        home = {id(q): key for key, insts in assignment.items() for q in insts}
        doc_of = {}
        for q in full:
            if q["file"] not in doc_of:
                meta = src / "assets" / (q["file"][:-4] + ".part")
                doc_of[q["file"]] = json.loads(meta.read_text()).get("documentId") if meta.exists() else None
        housings = [q for q in full if q["file"].startswith(args.motor_housing)]
        vendor_doc = doc_of[housings[0]["file"]] if housings else None
        vendor = [q for q in full if doc_of[q["file"]] == vendor_doc]
        hc = np.array([h["cen"] for h in housings])
        clusters = defaultdict(list)
        for q in vendor:
            clusters[int(np.argmin(np.linalg.norm(hc - q["cen"], axis=1)))].append(q)
        for hi, h in enumerate(housings):
            m_cad = sum(q["mass"] for q in clusters[hi])
            deficit = args.motor_mass - m_cad
            motor_report["cad_mass_g"].append(m_cad * 1000)
            if deficit <= 0:
                continue
            d = h["verts"] - h["cen"]
            _, vec = np.linalg.eigh(d.T @ d / len(d))
            best = None
            for k in range(3):
                ax = vec[:, k]
                along = d @ ax
                r = float(np.linalg.norm(d - np.outer(along, ax), axis=1).max())
                L = float(along.max() - along.min())
                if best is None or math.pi * r * r * L < best[0]:
                    best = (math.pi * r * r * L, ax, r, L)
            _, ax, r, L = best
            Rc = frame_on_axis(h["cen"], ax)[:3, :3]
            I_local = np.diag([deficit * (3 * r * r + L * L) / 12, deficit * (3 * r * r + L * L) / 12, 0.5 * deficit * r * r])
            extras[home[id(h)]].append((deficit, h["cen"].copy(), Rc @ I_local @ Rc.T))
            motor_report["motors"] += 1
            motor_report["topup_kg"] += deficit
        print(f"  actuators: {len(housings)} motors, CAD shell mass {np.mean(motor_report['cad_mass_g']):.1f} g each "
              f"(min {min(motor_report['cad_mass_g']):.1f}, max {max(motor_report['cad_mass_g']):.1f}); overridden to "
              f"{args.motor_mass*1000:.0f} g each on the housing link: +{motor_report['topup_kg']:.3f} kg total")

    link_dyn = {}
    for leg in legs:
        for body in BODIES:
            Tinv = np.linalg.inv(T_link[(leg, body)])
            items = [transform_dyn(dyn_of(q), Tinv) for q in assignment[(leg, body)]]
            items += [transform_dyn(e, Tinv) for e in extras.get((leg, body), [])]
            link_dyn[(leg, body)] = merge_dyn(items)
    chassis_only = merge_dyn([dyn_of(q) for q in assignment[("body", None)]] + extras.get(("body", None), []))
    link_dyn["body"] = merge_dyn([chassis_only] + [transform_dyn(link_dyn[(leg, "leg_base")], T_link[(leg, "leg_base")]) for leg in legs])
    total = merge_dyn([link_dyn["body"]] + [transform_dyn(link_dyn[(leg, b)], T_link[(leg, b)]) for leg in legs for b in BODIES[1:]])
    com_err = np.linalg.norm(total[1] - fused_c) * 1000
    I_err = np.abs(total[2] - fused_I).max() / np.abs(fused_I).max()
    print(f"  mass model: Onshape per-part properties{' + 191 g actuator override' if motor_report['motors'] else ''}; assembled {total[0]:.5f} kg (fused {fused_m:.5f}), COM error {com_err:.3f} mm, "
          f"worst inertia term error {I_err*100:.3f} % of max term; chassis-only {chassis_only[0]:.4f} kg, "
          f"per leg {np.mean([sum(link_dyn[(leg, b)][0] for b in BODIES) for leg in legs]):.4f} kg incl. its yaw output side")
    record_cmp = {}
    for body in BODIES:
        m_ref, c_ref, I_ref = leg_dyn[body]
        ms = [link_dyn[(leg, body)][0] for leg in legs]
        cs = [np.linalg.norm(link_dyn[(leg, body)][1] - c_ref) * 1000 for leg in legs]
        tr = [np.trace(link_dyn[(leg, body)][2]) / np.trace(I_ref) for leg in legs]
        record_cmp[body] = {"leg_record_mass_kg": m_ref, "assembly_mass_kg": float(np.mean(ms)), "mass_delta_g": float((np.mean(ms) - m_ref) * 1000),
                            "com_shift_mm_max": float(max(cs)), "inertia_trace_ratio": float(np.mean(tr))}
    print("  leg record check: " + ", ".join(f"{b} {v['mass_delta_g']:+.1f} g, I-trace x{v['inertia_trace_ratio']:.2f}" for b, v in record_cmp.items()))
    struct_rows = []
    seen = set()
    for q in full:
        key = q["file"]
        if key in seen or q["vol"] < 2e-6 or is_fastener(q["cat"]):
            continue
        seen.add(key)
        n = sum(1 for x in full if x["file"] == key)
        struct_rows.append((safe_mesh_name(key), n, q["mass"] * 1000, q["mass"] / q["vol"]))

    # -- round-trip: URDF posed at the CAD angles must reproduce the registered pose
    fk_err_pos, fk_err_rot = 0.0, 0.0
    fk_detail = {}
    for leg in legs:
        T = {"coxa": T_coxa0[leg] @ rotz(yaw_sign[leg] * cad_angles[(leg, "coxa_yaw")])}
        for jname, pb, cb, _, _ in LEG_JOINTS[1:]:
            T[cb] = T[pb] @ T_internal[jname] @ rotz(cad_angles[(leg, jname)])
        for body in BODIES[1:]:
            D = np.linalg.inv(T_link[(leg, body)]) @ T[body]
            e_pos = np.linalg.norm(D[:3, 3]) * 1000
            e_rot = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(D[:3, :3]) - 1) / 2))))
            fk_detail[f"{leg}_{body}"] = (round(e_pos, 4), round(e_rot, 4))
            fk_err_pos, fk_err_rot = max(fk_err_pos, e_pos), max(fk_err_rot, e_rot)
    print(f"  round-trip FK at CAD angles: worst link origin error {fk_err_pos:.4f} mm, {fk_err_rot:.4f} deg")

    # -- meshes
    (out / "urdf").mkdir(parents=True, exist_ok=True)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    name_map = {}
    for file in sorted({q["file"] for q in full}):
        safe = safe_mesh_name(file)
        n = 2
        while safe in name_map.values():
            safe = safe_mesh_name(file)[:-4] + f"_{n}.stl"
            n += 1
        name_map[file] = safe
        shutil.copy2(cache.get(file)["path"], out / "meshes" / safe)
    for stale in (out / "meshes").glob("*.stl"):
        if stale.name not in name_map.values():
            stale.unlink()

    # -- materials
    materials = {}
    for q in full:
        if q["rgba"] and q["rgba"] not in materials:
            rgb = [float(x) for x in q["rgba"].split()[:3]]
            materials[q["rgba"]] = "cad_" + "".join(f"{int(round(v * 255)):02x}" for v in rgb)

    # -- collision primitives: one per structural part (>= 2 cm3, no fasteners,
    # no motor internals): boxes for plates, covers and the tibia, cylinders
    # for motors (each motor's shell parts merged into one) and round parts,
    # two spheres per silicone foot pad
    STRUCT_MIN_VOL = 2e-6
    ROUND = re.compile(r"^(motor_flange|motor_bearing_holder|screw_head_cap|bearing_insert|first_joint_spacer)")
    MOTOR_SHELL = re.compile(r"^(motor_1_1_|motor_1_2_16|motor_0001755650|0001755650|1_1_|1_2_16)")
    MOTOR_INTERNAL = re.compile(r"^(motor_1_2_05|motor_1_21|motor_6706|motor_0001755652|1_2_05|1_21_|6706|0001755652)")

    def fit_cylinder(pts):
        mean = pts.mean(axis=0)
        d = pts - mean
        _, vec = np.linalg.eigh(d.T @ d / len(d))
        best = None
        for k in range(3):
            ax = vec[:, k]
            along = d @ ax
            radial = np.linalg.norm(d - np.outer(along, ax), axis=1)
            r, lo, hi = float(radial.max()), float(along.min()), float(along.max())
            vol = math.pi * r * r * (hi - lo)
            if best is None or vol < best[0]:
                best = (vol, ax, r, lo, hi)
        _, ax, r, lo, hi = best
        Tc = frame_on_axis(mean + ax * (lo + hi) / 2, ax)
        return ("cylinder", Tc[:3, 3], matrix_to_rpy(Tc[:3, :3]), (max(r, 1e-3), max(hi - lo, 1e-3)))

    def link_shapes(insts, Tinv):
        shapes, shells = [], []
        for q in insts:
            name = name_map[q["file"]]
            if q["vol"] < STRUCT_MIN_VOL or is_fastener(q["cat"]) or MOTOR_INTERNAL.search(name) or name.startswith("silicone_foot"):
                continue
            pts = q["verts"] @ Tinv[:3, :3].T + Tinv[:3, 3]
            if MOTOR_SHELL.search(name):
                shells.append((q, pts))
            elif ROUND.search(name):
                shapes.append(fit_cylinder(pts))
            else:
                c, rpy, size = obb(pts)
                shapes.append(("box", c, rpy, size))
        while shells:  # merge each motor's shell parts into one cylinder
            q0, p0 = shells.pop()
            group, rest = [p0], []
            for q, pts in shells:
                if np.linalg.norm(q["cen"] - q0["cen"]) < 0.035:
                    group.append(pts)
                else:
                    rest.append((q, pts))
            shells = rest
            shapes.append(fit_cylinder(np.concatenate(group)))
        return shapes

    collisions = {"body": link_shapes(assignment[("body", None)], np.eye(4))}
    for leg in legs:
        collisions["body"] += link_shapes(assignment[(leg, "leg_base")], np.eye(4))
    foot_spec = {}
    for leg in legs:
        for body in BODIES[1:]:
            collisions[(leg, body)] = link_shapes(assignment[(leg, body)], np.linalg.inv(T_link[(leg, body)]))
        feet = [q for q in assignment[(leg, "tibia")] if q["file"].startswith("silicone_foot")]
        if feet:
            Tinv = np.linalg.inv(T_link[(leg, "tibia")])
            pts = feet[0]["verts"] @ Tinv[:3, :3].T + Tinv[:3, 3]
            c, rpy, size = obb(pts)
            Rf = np.array(make_T([0, 0, 0], rpy))[:3, :3]
            axis = Rf[:, int(np.argmax(size))]
            half = size.max() / 2
            r = float(np.sort(size)[:2].mean() / 2)
            for sgn in (-1, 1):
                collisions[(leg, "tibia")].append(("sphere", c + sgn * (half - r) * axis, (0, 0, 0), r))
            foot_spec[leg] = {"center_tibia_frame": c.tolist(), "radius": r}
    n_shapes = sum(len(v) for v in collisions.values())
    print(f"  collision primitives: {n_shapes} ({len(collisions['body'])} on the body, "
          f"{sum(len(collisions[(legs[0], b)]) for b in BODIES[1:])} per leg)")

    # -- emit
    limits_used = {"coxa_yaw": limits.get("coxa_yaw", (-0.872665, 0.872665)), "femur_pitch": limits["femur_pitch"], "tibia_pitch": limits["tibia_pitch"]}

    def visual_lines(insts, Tinv, indent="    "):
        lines = []
        for q in insts:
            Tv = Tinv @ q["T"]
            lines.append(f"{indent}<visual>")
            lines.append(f'{indent}  <origin xyz="{fmt(Tv[:3, 3])}" rpy="{fmt(matrix_to_rpy(Tv[:3, :3]))}" />')
            lines.append(f'{indent}  <geometry><mesh filename="package://{pkg}/meshes/{name_map[q["file"]]}" /></geometry>')
            if q["rgba"]:
                lines.append(f'{indent}  <material name="{materials[q["rgba"]]}" />')
            lines.append(f"{indent}</visual>")
        return lines

    def collision_lines(shapes, Tpre=None, indent="    "):
        lines = []
        for kind, c, rpy, param in shapes:
            Tc = make_T(c, rpy)
            if Tpre is not None:
                Tc = Tpre @ Tc
            lines.append(f"{indent}<collision>")
            lines.append(f'{indent}  <origin xyz="{fmt(Tc[:3, 3])}" rpy="{fmt(matrix_to_rpy(Tc[:3, :3]))}" />')
            if kind == "box":
                lines.append(f'{indent}  <geometry><box size="{fmt(param)}" /></geometry>')
            elif kind == "sphere":
                lines.append(f'{indent}  <geometry><sphere radius="{param:.6g}" /></geometry>')
            else:
                lines.append(f'{indent}  <geometry><cylinder radius="{param[0]:.6g}" length="{param[1]:.6g}" /></geometry>')
            lines.append(f"{indent}</collision>")
        return lines

    def inertial_lines(dyn, indent="    "):
        m, c, I = dyn
        return [
            f"{indent}<inertial>",
            f'{indent}  <origin xyz="{fmt(c)}" rpy="0 0 0" />',
            f'{indent}  <mass value="{m:.8g}" />',
            f'{indent}  <inertia ixx="{I[0,0]:.8g}" ixy="{I[0,1]:.8g}" ixz="{I[0,2]:.8g}" iyy="{I[1,1]:.8g}" iyz="{I[1,2]:.8g}" izz="{I[2,2]:.8g}" />',
            f"{indent}</inertial>",
        ]

    def emit(path, include_linkage):
        name = path.stem
        L = ['<?xml version="1.0" ?>', f'<robot name="{name}">']
        L.append(f"  <!-- Generated by robot/tools/import_onshape_hexapod.py from the onshape-to-robot export "
                 f"'{export_name}' and the leg v3 reference. Frame: export frame, z up, forward {args.forward}. "
                 f"Positive coxa_yaw is counter-clockwise about body +z; positive femur/tibia pitch follows the leg reference "
                 f"(motor positions at the shared CAD zero). -->")
        for rgba, mname in materials.items():
            L.append(f'  <material name="{mname}"><color rgba="{rgba}" /></material>')
        # chassis
        L.append('  <link name="body">')
        L += inertial_lines(link_dyn["body"])
        L += visual_lines(assignment[("body", None)], np.eye(4))
        for leg in legs:
            L.append(f"    <!-- {leg} yaw motor output side (flange + hub bolted through the frame) -->")
            L += visual_lines(assignment[(leg, "leg_base")], np.eye(4))
        L += collision_lines(collisions["body"])
        L.append("  </link>")
        merged_into = {} if include_linkage else {"femur": ["tibia_push_lever", "tibia_pushrod"]}
        for leg in legs:
            order = ["coxa", "femur", "tibia"] + (["tibia_push_lever", "tibia_pushrod"] if include_linkage else [])
            for body in order:
                dyn = link_dyn[(leg, body)]
                extras = merged_into.get(body, [])
                if extras:
                    items = [dyn]
                    for e in extras:
                        items.append(transform_dyn(link_dyn[(leg, e)], np.linalg.inv(frames[body]) @ frames[e]))
                    dyn = merge_dyn(items)
                L.append(f'  <link name="{leg}_{body}">')
                L += inertial_lines(dyn)
                L += visual_lines(assignment[(leg, body)], np.linalg.inv(T_link[(leg, body)]))
                L += collision_lines(collisions[(leg, body)])
                for e in extras:
                    Te = np.linalg.inv(frames[body]) @ frames[e]
                    L += visual_lines(assignment[(leg, e)], Te @ np.linalg.inv(T_link[(leg, e)]))
                    L += collision_lines(collisions[(leg, e)], Te)
                L.append("  </link>")
            for jname, pb, cb, limkey, mimic in LEG_JOINTS:
                if mimic is not None and not include_linkage:
                    continue
                parent = "body" if pb == "leg_base" else f"{leg}_{pb}"
                Tj = T_coxa0[leg] if jname == "coxa_yaw" else T_internal[jname]
                axis = f"0 0 {int(yaw_sign[leg])}" if jname == "coxa_yaw" else "0 0 1"
                L.append(f'  <joint name="{leg}_{jname}" type="revolute">')
                L.append(f'    <origin xyz="{fmt(Tj[:3, 3])}" rpy="{fmt(matrix_to_rpy(Tj[:3, :3]))}" />')
                L.append(f'    <axis xyz="{axis}" />')
                L.append(f'    <parent link="{parent}" />')
                L.append(f'    <child link="{leg}_{cb}" />')
                if limkey:
                    lo, hi = limits_used[limkey]
                    L.append(f'    <limit effort="{effort}" velocity="{velocity}" lower="{lo}" upper="{hi}" />')
                else:
                    span = max(abs(x) for x in limits_used["tibia_pitch"])
                    L.append(f'    <limit effort="0.5" velocity="{velocity}" lower="{-span}" upper="{span}" />')
                    L.append(f'    <mimic joint="{leg}_{mimic[0]}" multiplier="{mimic[1]}" offset="0" />')
                L.append('    <dynamics damping="0.01" friction="0.01" />')
                L.append("  </joint>")
            if include_linkage:
                p_cut, _ = axes["loop_cut"]
                Tr = np.linalg.inv(frames["tibia_pushrod"])
                Tt = np.linalg.inv(frames["tibia"])
                L.append(f"  <!-- {leg} LOOP CLOSURE (cut joint): pushrod point [{fmt(Tr[:3,:3] @ p_cut + Tr[:3,3])}] "
                         f"pivots on tibia point [{fmt(Tt[:3,:3] @ p_cut + Tt[:3,3])}]; parallelogram closed by mimics. -->")
        L.append("</robot>")
        path.write_text("\n".join(L) + "\n")

    emit(out / "urdf" / "hexapod_mkii_linkage.urdf", True)
    emit(out / "urdf" / "hexapod_mkii_serial.urdf", False)

    # -- report
    rep = {
        "source": {"export": str(src), "robot_name": export_name, "instances": len(full), "fused_mass_kg": fused_m,
                   "fused_com_m": fused_c.tolist(), "fused_inertia": fused_I.tolist()},
        "conventions": {"body_frame": "export frame (z up)", "forward": args.forward, "yaw_zero": args.yaw_zero,
                        "leg_names": "l/r = left/right, f/m/r = front/middle/rear",
                        "coxa_yaw_positive": "counter-clockwise about body +z",
                        "zero_pose": "femur/tibia: leg reference CAD zero (leg export pose); coxa_yaw zero: "
                                     + ("leg points radially outward from the body centre" if args.yaw_zero == "radial" else "CAD mount placement")},
        "legs": {},
        "registration_log": log,
        "assignment": {"matched": len(inst_home), "swapped_fasteners": n_swapped,
                       "new_leg_parts": [(q["file"], leg, body, round(d, 2)) for q, leg, body, d in new_leg_parts],
                       "chassis_parts": len(assignment[("body", None)]),
                       "missing_leg_parts": [(e["leg"], e["body"], e["file"]) for e in missing],
                       "review": [(q["file"], e["leg"], e["body"], e["file"], round(d, 2)) for q, e, d in review]},
        "mass_model": {"source": "Onshape per-part mass properties (robot.pkl), parallel-axis sums per link"
                                 + (f"; each of the {motor_report['motors']} RS05 actuators overridden from its CAD shell mass to {args.motor_mass*1000:.0f} g on its housing link" if motor_report["motors"] else ""),
                       "actuators": motor_report,
                       "chassis_only_kg": float(chassis_only[0]), "mounts_kg": float(sum(link_dyn[(leg, "leg_base")][0] for leg in legs)),
                       "per_leg_kg": {leg: float(sum(link_dyn[(leg, b)][0] for b in BODIES)) for leg in legs},
                       "leg_record_comparison": record_cmp,
                       "structural_parts": [{"mesh": f, "count": n, "mass_g": m, "density_kg_m3": rho} for f, n, m, rho in sorted(struct_rows, key=lambda r: -r[2] * r[1])]},
        "totals": {"mass_kg": float(total[0]), "com_m": total[1].tolist(), "com_error_mm": float(com_err),
                   "inertia": total[2].tolist(), "inertia_error_frac_of_max": float(I_err)},
        "collision_primitives": n_shapes,
        "round_trip_fk": {"worst_origin_error_mm": fk_err_pos, "worst_rotation_error_deg": fk_err_rot, "per_link_mm_deg": fk_detail},
        "leg_internal_joints": {j: {"origin_xyz_m": T_internal[j][:3, 3].tolist(), "rpy": list(matrix_to_rpy(T_internal[j][:3, :3])),
                                    "assembly_vs_leg_record_shift_mm": record_dev[j][0].tolist(), "assembly_vs_leg_record_deg": record_dev[j][1],
                                    "leg_deviation_from_mean_mm_deg": {leg: joint_dev[(leg, j)] for leg in legs}} for j in T_internal},
        "mount_cross_check_mm_deg": mount_dev,
        "link_masses_kg": {("body" if k == "body" else (f"{k[0]} yaw output side (in body)" if k[1] == "leg_base" else f"{k[0]}_{k[1]}")): float(v[0]) for k, v in link_dyn.items()},
        "mesh_names": name_map,
        "materials": materials,
    }
    chassis_fast = [q for q in assignment[("body", None)] if is_fastener(q["cat"])]
    for leg in legs:
        R, t = regs[(leg, "leg_base")]
        p, d = R @ yaw_pt + t, R @ yaw_dir
        d /= np.linalg.norm(d)
        n_near = sum(1 for q in chassis_fast if np.linalg.norm((q["cen"] - p)[:2]) < 0.06)
        feet = [q for q in assignment[(leg, "tibia")] if q["file"].startswith("silicone_foot")]
        foot_low = float(min(q["verts"][:, 2].min() for q in feet)) if feet else None
        rep["legs"][leg] = {
            "yaw_axis_point_m": p.round(6).tolist(),
            "yaw_axis_up_dir": (-d if d[2] < 0 else d).round(6).tolist(),
            "yaw_axis_tilt_deg": math.degrees(math.acos(min(1.0, abs(float(d[2]))))),
            "azimuth_deg": math.degrees(math.atan2(p[1], p[0])),
            "radius_m": float(math.hypot(p[0], p[1])),
            "heading_deg_zero_yaw": heading_deg(T_coxa0[leg]),
            "cad_mount_heading_deg": cad_heading[leg],
            "cad_heading_minus_radial_deg": -yaw_offset[leg],
            "chassis_fasteners_within_60mm": n_near,
            "registration": {b: {k: v for k, v in reg_info[(leg, b)].items() if k != "moved"} for b in BODIES},
            "moved_parts_registration": {b: [(f, round(d, 3)) for f, d in reg_info[(leg, b)].get("moved", [])] for b in BODIES},
            "moved_parts_assignment": [(b, name_map[q["file"]], round(q["residual_mm"], 3)) for b in BODIES for q in assignment[(leg, b)] if q.get("moved")],
            "axis_checks": {j: {"dir_mismatch_deg": axis_checks[(leg, j)][0], "line_gap_mm": axis_checks[(leg, j)][1]} for j, *_ in LEG_JOINTS},
            "cad_pose_joint_angles_rad": {j: cad_angles[(leg, j)] for j, *_ in LEG_JOINTS},
            "parts": {b: len(assignment[(leg, b)]) for b in BODIES},
            "worst_part_residual_mm": max((q.get("residual_mm", 0.0) for b in BODIES for q in assignment[(leg, b)] if not q.get("internal_dof")), default=0.0),
            "internal_dof_parts": sum(1 for b in BODIES for q in assignment[(leg, b)] if q.get("internal_dof")),
            "foot_lowest_z_cad_pose_m": foot_low,
            "foot_collision": foot_spec.get(leg),
        }
    (out / "assembly_report.json").write_text(json.dumps(rep, indent=1, default=float))
    write_markdown(out / "assembly_report.md", rep, legs)
    print(f"  wrote {out / 'urdf'} , report {out / 'assembly_report.md'}")
    return 0


def write_markdown(path, rep, legs):
    L = ["# Hexapod MKII assembly import report", ""]
    s = rep["source"]
    L += [f"Source: `{s['export']}` (`{s['robot_name']}`), {s['instances']} part instances, "
          f"Onshape fused mass {s['fused_mass_kg']:.5f} kg.", ""]
    c = rep["conventions"]
    L += ["Conventions: body frame = export frame (z up), forward = `" + c["forward"] + "`, leg names " + c["leg_names"]
          + "; positive coxa_yaw " + c["coxa_yaw_positive"] + "; zero pose = " + c["zero_pose"] + ".", ""]
    L += ["## Leg mounts (yaw axes in the body frame)", "",
          "| leg | x (m) | y (m) | z (m) | azimuth | radius (m) | tilt from z | heading at URDF zero yaw | CAD mount heading vs radial | chassis fasteners within 60 mm |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for leg in legs:
        r = rep["legs"][leg]
        p = r["yaw_axis_point_m"]
        L.append(f"| {leg} | {p[0]:+.4f} | {p[1]:+.4f} | {p[2]:+.4f} | {r['azimuth_deg']:+.1f} deg | {r['radius_m']:.4f} | "
                 f"{r['yaw_axis_tilt_deg']:.3f} deg | {r['heading_deg_zero_yaw']:+.1f} deg | {r['cad_heading_minus_radial_deg']:+.1f} deg | {r['chassis_fasteners_within_60mm']} |")
    L += ["", f"Yaw zero convention: `{c['yaw_zero']}`. The yaw mates are locked in the CAD, so the CAD mount headings "
          "(last column but one) are where the CAD happens to hold each leg, not a designed stance; with `radial` every "
          "leg points straight out from the body centre at coxa_yaw = 0 and the CAD placement appears as the coxa_yaw "
          "CAD-pose angle below. Heading = direction from the yaw axis to the hip axis, projected on the body xy plane.", ""]
    L += ["", "## Registration of the leg record onto the export", "",
          "| leg | body | method | fit | parts |", "|---|---|---|---|---|"]
    for leg in legs:
        for b, info in rep["legs"][leg]["registration"].items():
            seed = f"seed rms {info['seed_rms_mm']:.3f} mm" if "seed_rms_mm" in info else f"seed fit {info['seed_fit_mm']:.3f} mm ({info['dtheta_deg']:+.2f} deg)"
            fit = f"{seed}; consensus rms {info.get('rms_mm', float('nan')):.4f} mm on {info.get('inliers', 0)} parts"
            moved = len(rep["legs"][leg]["moved_parts_registration"][b])
            L.append(f"| {leg} | {b} | {info['method']} | {fit} | {info['parts']} ({moved} moved) |")
    moved = Counter()
    for leg in legs:
        for b, f, d in rep["legs"][leg]["moved_parts_assignment"]:
            moved[(b, f)] = max(moved[(b, f)], d)
    if moved:
        L += ["", "Parts whose CAD position differs from the leg record by more than 0.5 mm (assigned to the nearest expected "
              "position; these are free-spinning motor internals and re-solved fasteners, not structure):", ""]
        cnt = Counter((b, f) for leg in legs for b, f, _ in rep["legs"][leg]["moved_parts_assignment"])
        L += [f"- {b}: `{f[:60]}` x{cnt[(b, f)]} (up to {d:.2f} mm)" for (b, f), d in sorted(moved.items())]
    L += ["", "## Hinge axes carried through parent and child registrations", "",
          "| leg | joint | dir mismatch | line gap | CAD-pose angle |", "|---|---|---|---|---|"]
    for leg in legs:
        r = rep["legs"][leg]
        for j, chk in r["axis_checks"].items():
            L.append(f"| {leg} | {j} | {chk['dir_mismatch_deg']:.3f} deg | {chk['line_gap_mm']:.3f} mm | "
                     f"{r['cad_pose_joint_angles_rad'][j]:+.4f} rad ({math.degrees(r['cad_pose_joint_angles_rad'][j]):+.2f} deg) |")
    L += ["", "## Part assignment", "",
          "| leg | yaw output side (body) | coxa | femur | tibia | push lever | pushrod | worst residual (excl. free motor internals) | free motor internals |", "|---|---|---|---|---|---|---|---|---|"]
    for leg in legs:
        r = rep["legs"][leg]
        p = r["parts"]
        L.append(f"| {leg} | {p['leg_base']} | {p['coxa']} | {p['femur']} | {p['tibia']} | {p['tibia_push_lever']} | "
                 f"{p['tibia_pushrod']} | {r['worst_part_residual_mm']:.3f} mm | {r['internal_dof_parts']} |")
    a = rep["assignment"]
    L += ["", f"Matched {a['matched']} instances ({a['swapped_fasteners']} fasteners matched by size class because the catalogue "
          f"part changed), {a['chassis_parts']} chassis parts, {len(a['new_leg_parts'])} new leg parts adopted by keyword, "
          f"{len(a['missing_leg_parts'])} leg-record parts missing from the assembly, {len(a['review'])} chassis parts within 5 mm of a leg part (review)."]
    if a["new_leg_parts"]:
        cnt = Counter((f, b) for f, _, b, _ in a["new_leg_parts"])
        L += ["", "New leg parts (not in the leg record): " + ", ".join(f"{f} -> {b} x{n}" for (f, b), n in sorted(cnt.items()))]
    if a["missing_leg_parts"]:
        L += ["", "Missing: " + ", ".join(f"{leg}/{b}/{f}" for leg, b, f in a["missing_leg_parts"])]
    if a["review"]:
        L += ["", "Review (chassis part close to a leg part): " + "; ".join(f"{q} {d} mm from {leg}/{b}/{f}" for q, leg, b, f, d in a["review"])]
    mm = rep["mass_model"]
    L += ["", "## Mass model", "", f"Source: {mm['source']}. Chassis structure alone {mm['chassis_only_kg']:.4f} kg, plus the six yaw-motor output "
          f"sides (flange + hub, bolted through the frame) {mm['mounts_kg']:.4f} kg in the body link; the motor bodies, their "
          f"top plates and the screws on them rotate with the coxae. Legs "
          + ", ".join(f"{leg} {m:.4f}" for leg, m in mm["per_leg_kg"].items()) + " kg (each including its yaw output side).", "",
          "| structural part | count | mass each (g) | density (kg/m3) |", "|---|---|---|---|"]
    for row in mm["structural_parts"]:
        L.append(f"| {row['mesh']} | {row['count']} | {row['mass_g']:.2f} | {row['density_kg_m3']:.0f} |")
    L += ["", "Leg record (v3, derived from the older v2 per-part export) versus this assembly's per-part properties, per body:", "",
          "| body | record mass (kg) | assembly mass (kg) | delta (g) | COM shift (mm) | inertia trace ratio |", "|---|---|---|---|---|---|"]
    for b, v in mm["leg_record_comparison"].items():
        L.append(f"| {b} | {v['leg_record_mass_kg']:.4f} | {v['assembly_mass_kg']:.4f} | {v['mass_delta_g']:+.1f} | {v['com_shift_mm_max']:.2f} | {v['inertia_trace_ratio']:.2f} |")
    t = rep["totals"]
    act = mm.get("actuators", {})
    if act.get("motors"):
        L += ["", f"Actuators: the RS05 vendor CAD is a hollow shell weighing {np.mean(act['cad_mass_g']):.1f} g "
              f"(min {min(act['cad_mass_g']):.1f}, max {max(act['cad_mass_g']):.1f}); each of the {act['motors']} motors is overridden to "
              f"{act['motor_mass_kg']*1000:.0f} g, the difference placed on the link that carries its housing as a solid cylinder the size of "
              f"the housing (+{act['topup_kg']:.3f} kg in total). Everything else keeps its CAD mass, so the assembled mass exceeds "
              f"Onshape's fused value by exactly that amount; `--motor-mass 0` reproduces the CAD masses."]
    L += ["", f"Assembled: {t['mass_kg']:.5f} kg vs Onshape fused {s['fused_mass_kg']:.5f} kg; COM error {t['com_error_mm']:.3f} mm; "
          f"worst inertia term error {t['inertia_error_frac_of_max']*100:.3f} % of the largest term.", "",
          "| link | mass (kg) |", "|---|---|"]
    for k, v in rep["link_masses_kg"].items():
        L.append(f"| {k} | {v:.5f} |")
    L += ["", "## Leg-internal joint transforms (measured from the assembly, mean of six legs)", "",
          "| joint | origin in parent frame (m) | rpy | shift vs leg record (mm) | worst leg deviation from mean |", "|---|---|---|---|---|"]
    for j, v in rep["leg_internal_joints"].items():
        dev = max(v["leg_deviation_from_mean_mm_deg"].values(), key=lambda x: x[0])
        L.append(f"| {j} | {' '.join(f'{x:+.5f}' for x in v['origin_xyz_m'])} | {' '.join(f'{x:+.4f}' for x in v['rpy'])} | "
                 f"{' '.join(f'{x:+.3f}' for x in v['assembly_vs_leg_record_shift_mm'])} ({v['assembly_vs_leg_record_deg']:.3f} deg) | "
                 f"{dev[0]:.4f} mm / {dev[1]:.4f} deg |")
    md = rep["mount_cross_check_mm_deg"]
    L += ["", "Mount cross-check (coxa zero-yaw pose from the mount-plate registration vs from the coxa registration): worst "
          f"{max(v[0] for v in md.values()):.3f} mm / {max(v[1] for v in md.values()):.3f} deg."]
    f = rep["round_trip_fk"]
    L += ["", "## Round trip", "", f"URDF forward kinematics at the measured CAD-pose joint angles reproduces every registered "
          f"link pose to {f['worst_origin_error_mm']:.4f} mm / {f['worst_rotation_error_deg']:.4f} deg.", ""]
    L += ["Foot pad lowest point in the CAD pose (m, body frame): " + ", ".join(f"{leg} {rep['legs'][leg]['foot_lowest_z_cad_pose_m']:+.4f}" for leg in legs), ""]
    path.write_text("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/Users/andreboufama/Downloads/export (1)")
    ap.add_argument("--leg-dir", default="robot/hexapod_leg_v3")
    ap.add_argument("--limits", default="robot/hexapod_mkii_assy/joint_limits.json",
                    help="joint limits about the CAD zero (the leg v3 file holds the mock-inherited placeholders)")
    ap.add_argument("--out", default="robot/hexapod_mkii_assy")
    ap.add_argument("--forward", default="-y", choices=["+x", "-x", "+y", "-y"],
                    help="body-frame direction treated as forward for leg naming (pipeline convention: -y)")
    ap.add_argument("--yaw-output-side", default=r"^(motor_flange\.stl|0001755650)",
                    help="regex of the yaw motor parts bolted to the frame (output flange + hub); the rest of the "
                         "leg record's leg_base group rotates with the coxa")
    ap.add_argument("--overrides", default="robot/hexapod_mkii_assy/part_overrides.json",
                    help="user-verified attachment rules for parts the leg record cannot place")
    ap.add_argument("--motor-mass", type=float, default=0.191,
                    help="mass of one RS05 actuator (hard override: every motor totals this; the difference to the CAD "
                         "shell mass sits on the housing link); 0 keeps the CAD masses")
    ap.add_argument("--motor-housing", default="1_1_06_eb463_507",
                    help="mesh file prefix of the actuator housing part (one per motor)")
    ap.add_argument("--yaw-zero", default="radial", choices=["radial", "cad"],
                    help="coxa_yaw = 0 pose: 'radial' points every leg straight out from the body centre "
                         "(the yaw mates are locked at arbitrary angles in the CAD); 'cad' keeps the CAD placement")
    sys.exit(build(ap.parse_args()))
