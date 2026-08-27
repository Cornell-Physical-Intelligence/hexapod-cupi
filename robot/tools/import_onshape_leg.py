#!/usr/bin/env python3
"""Import the onshape-to-robot leg export by registering the proven v2 bodies.

The onshape-to-robot export merged the whole leg into ONE link (no ``dof_*``
mate names), so it carries exact fused dynamics and per-part colored geometry
but no structure.  This importer rebuilds the six-body structure by rigidly
registering each v2 body onto the new geometry:

1. match parts between exports by normalized mesh-name stems;
2. per body, fit a rigid transform (Kabsch over tessellation-independent
   volume centroids of parts with unique stems), trimming moved-part outliers;
3. carry every hinge axis through its parent body's transform and verify the
   child body maps it to the same line (a revolute axis is preserved by both
   sides even if the export pose changed the joint angle);
4. assign every new visual instance (screws, new parts) to the nearest
   registered body;
5. rebuild per-body dynamics: v2 per-body inertials transformed rigidly, the
   export's extra mass distributed over new-only parts by mesh volume, and the
   assembled totals validated against Onshape's exact fused mass/COM.

Outputs ``leg_v3_serial.urdf`` and ``leg_v3_linkage.urdf`` (parallelogram
mimic closure) with per-part CAD colors and primitive collisions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from consolidate_leg_urdf import (  # noqa: E402
    LIMITS,
    RS05_EFFORT_NM,
    RS05_VELOCITY_RAD_S,
    Model,
    fmt,
    make_T,
    matrix_to_rpy,
    parse_floats,
)
import xml.etree.ElementTree as ET  # noqa: E402


def norm_stem(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", Path(name).stem.lower())


def stl_mesh(path: Path):
    with open(path, "rb") as f:
        header = f.read(84)
        (n,) = struct.unpack("<I", header[80:84])
        if path.stat().st_size < 84 + n * 50:
            raise ValueError(f"ascii/truncated stl: {path.name}")
        data = np.fromfile(f, dtype=np.uint8, count=n * 50)
    return data.reshape(n, 50)[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(float)


def volume_centroid(tris: np.ndarray):
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    det = np.einsum("ij,ij->i", v0, np.cross(v1, v2))
    vol = det.sum() / 6.0
    if abs(vol) < 1e-12:
        verts = tris.reshape(-1, 3)
        return 0.0, verts.mean(axis=0), verts
    cen = ((v0 + v1 + v2 + 0.0) / 4.0 * det[:, None]).sum(axis=0) / (6.0 * vol)
    return abs(vol), cen, tris.reshape(-1, 3)


def surface_features(tris: np.ndarray):
    """Registration features: area-weighted centroid + skew-signed axis tips.

    Computed from surface moments so they are tessellation-independent, and
    each clearly asymmetric principal axis contributes one signed endpoint,
    letting a single structural part orient a whole body.
    """
    if len(tris) == 0:
        return []
    g = tris.mean(axis=1)
    w = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    w = np.where(np.isfinite(w), w, 0.0)
    W = w.sum()
    if not np.isfinite(W) or W < 1e-12:
        return [tris.reshape(-1, 3).mean(axis=0)]
    m = (w[:, None] * g).sum(axis=0) / W
    d = g - m
    cov = (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(axis=0) / W
    if not np.isfinite(cov).all():
        return [m]
    lam, vec = np.linalg.eigh(cov)
    pts = [m]
    for k in range(3):
        sig = math.sqrt(max(lam[k], 0.0))
        if sig < 1e-6:
            continue
        proj = d @ vec[:, k]
        skew = float((w * proj**3).sum() / (W * sig**3))
        if abs(skew) > 0.08:
            pts.append(m + math.copysign(1.0, skew) * sig * vec[:, k])
    return pts


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rigid transform mapping P -> Q (rows are corresponding points)."""
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cq - R @ cp
    return R, t


def build(args):
    old_src = Path(args.old_source)
    new_src = Path(args.new_source)
    out = Path(args.out)
    body_map = json.loads(Path(args.body_map).read_text())
    limits = dict(LIMITS)
    inverted = set()
    if args.limits and Path(args.limits).exists():
        for k, v in json.loads(Path(args.limits).read_text()).items():
            if not isinstance(v, dict):
                continue
            limits[k] = (float(v["lower"]), float(v["upper"]))
            if v.get("invert"):
                inverted.add(k)
    old = Model(old_src / "urdf" / "leg_subassy.urdf")

    # ---- old side: per-part global centroid + mesh stem + inertial ----
    old_parts = {}
    for name, link in old.links.items():
        vis = link.find("visual")
        if vis is None:
            continue
        mesh = vis.find("geometry/mesh")
        if mesh is None:
            continue
        fname = Path(mesh.get("filename").split("/")[-1])
        try:
            tris = stl_mesh(old_src / "meshes" / fname)
        except (OSError, ValueError):
            continue
        vol, cen_local, _ = volume_centroid(tris)
        o = vis.find("origin")
        xyz = parse_floats(o.get("xyz") if o is not None else None, [0, 0, 0])
        rpy = parse_floats(o.get("rpy") if o is not None else None, [0, 0, 0])
        Tv = old.T[name] @ make_T(xyz, rpy)
        feats = np.array(surface_features(tris))
        old_parts[name] = {
            "stem": norm_stem(fname.name),
            "centroid": Tv[:3, :3] @ cen_local + Tv[:3, 3],
            "features": feats @ Tv[:3, :3].T + Tv[:3, 3],
            "volume": vol,
        }

    # ---- new side: every visual instance ----
    new_root = ET.parse(new_src / "robot.urdf").getroot()
    link = new_root.find("link")
    fused = link.find("inertial")
    fused_mass = float(fused.find("mass").get("value"))
    fused_com = parse_floats(fused.find("origin").get("xyz"), [0, 0, 0])

    mesh_cache = {}
    instances = []  # dicts: stem, file, T, centroid, volume, rgba, verts(sample)
    for vis in link.findall("visual"):
        mesh = vis.find("geometry/mesh")
        if mesh is None:
            continue
        fname = Path(mesh.get("filename").split("/")[-1])
        if fname.name not in mesh_cache:
            tris = stl_mesh(new_src / "assets" / fname)
            vol, cen, verts = volume_centroid(tris)
            step = max(1, len(verts) // 150)
            mesh_cache[fname.name] = (vol, cen, verts[::step], np.array(surface_features(tris)))
        vol, cen, verts, feats = mesh_cache[fname.name]
        o = vis.find("origin")
        xyz = parse_floats(o.get("xyz") if o is not None else None, [0, 0, 0])
        rpy = parse_floats(o.get("rpy") if o is not None else None, [0, 0, 0])
        Tv = make_T(xyz, rpy)
        mat = vis.find("material")
        rgba = None
        if mat is not None and mat.find("color") is not None:
            rgba = mat.find("color").get("rgba")
        instances.append(
            {
                "stem": norm_stem(fname.name),
                "file": fname.name,
                "xyz": xyz,
                "rpy": rpy,
                "centroid": Tv[:3, :3] @ cen + Tv[:3, 3],
                "volume": vol,
                "rgba": rgba,
                "verts": (Tv[:3, :3] @ verts.T).T + Tv[:3, 3],
                "features": feats @ Tv[:3, :3].T + Tv[:3, 3],
            }
        )
    print(f"new export: {len(instances)} visual instances, fused mass {fused_mass:.4f} kg")

    stem_count_old = defaultdict(int)
    for p in old_parts.values():
        stem_count_old[p["stem"]] += 1
    stem_count_new = defaultdict(int)
    for i in instances:
        stem_count_new[i["stem"]] += 1

    # ---- per-body rigid registration ----
    # Bodies rich in uniquely named structural parts register directly by
    # Kabsch.  Motor-internals bodies (all duplicated stems) are then solved
    # through the mechanism itself: between exports only joint angles changed,
    # so an under-constrained body's transform is its registered neighbor's
    # transform composed with a 1-DOF rotation about their shared hinge axis.
    registrations = {}
    reg_report = {}
    for body, members in body_map["bodies"].items():
        P_list, Q_list, n_parts = [], [], 0
        for name in members:
            p = old_parts.get(name)
            if p is None:
                continue
            if stem_count_old[p["stem"]] != 1 or stem_count_new.get(p["stem"], 0) != 1:
                continue
            inst = next(i for i in instances if i["stem"] == p["stem"])
            k = min(len(p["features"]), len(inst["features"]))
            if k < 1:
                continue
            P_list.append(p["features"][:k])
            Q_list.append(inst["features"][:k])
            n_parts += 1
        P = np.concatenate(P_list) if P_list else np.zeros((0, 3))
        Q = np.concatenate(Q_list) if Q_list else np.zeros((0, 3))
        rank = 0
        if len(P) >= 3:
            rank = np.linalg.matrix_rank(P - P.mean(axis=0), tol=1e-4)
        if len(P) < 3 or rank < 2:
            reg_report[body] = (
                f"deferred ({n_parts} unique parts, {len(P)} pts, rank {rank})"
            )
            continue
        R, t = kabsch(P, Q)
        res = np.linalg.norm((P @ R.T + t) - Q, axis=1)
        keep = res < max(0.005, np.median(res) * 4)
        if keep.sum() >= 3 and keep.sum() < len(P):
            R, t = kabsch(P[keep], Q[keep])
            res = np.linalg.norm((P @ R.T + t) - Q, axis=1)
        registrations[body] = (R, t)
        reg_report[body] = (
            f"{n_parts} parts / {len(P)} feature pts, "
            f"rms {np.sqrt((res[keep] ** 2).mean()) * 1000:.2f} mm, {int((~keep).sum())} outliers"
        )

    hinge_of = {
        "leg_base": ("coxa", "coxa_yaw", -1.0),
        "tibia_push_lever": ("femur", "tibia_lever_pivot", +1.0),
        "tibia_pushrod": ("tibia_push_lever", "tibia_rod_pivot", +1.0),
    }

    def rot_about(axis_point, axis_dir, theta):
        d = axis_dir / np.linalg.norm(axis_dir)
        K = np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])
        R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = axis_point - R @ axis_point
        return T

    def solve_deferred(body):
        neighbor, jname, _sign = hinge_of[body]
        if neighbor not in registrations:
            raise SystemExit(f"{body}: neighbor {neighbor} not registered")
        Rn, tn = registrations[neighbor]
        Tn = np.eye(4)
        Tn[:3, :3], Tn[:3, 3] = Rn, tn
        spec = body_map["axes"][jname]
        ap = np.array(spec["point"])
        ad = np.array(spec["dir"])
        olds = np.array(
            [old_parts[n]["centroid"] for n in body_map["bodies"][body] if n in old_parts]
        )
        stems_here = {old_parts[n]["stem"] for n in body_map["bodies"][body] if n in old_parts}
        cands = np.array([i["centroid"] for i in instances if i["stem"] in stems_here])

        def score(theta):
            T = Tn @ rot_about(ap, ad, theta)
            moved = olds @ T[:3, :3].T + T[:3, 3]
            d = np.linalg.norm(moved[:, None, :] - cands[None, :, :], axis=2).min(axis=1)
            return np.clip(d, None, 0.02).mean()

        thetas = np.linspace(-math.pi, math.pi, 721)
        vals = [score(x) for x in thetas]
        best = thetas[int(np.argmin(vals))]
        for span in (0.02, 0.002):
            fine = np.linspace(best - span * 10, best + span * 10, 41)
            best = fine[int(np.argmin([score(x) for x in fine]))]
        T = Tn @ rot_about(ap, ad, best)
        registrations[body] = (T[:3, :3], T[:3, 3])
        reg_report[body] = f"hinge-solved via {jname}: dtheta {math.degrees(best):+.2f} deg, fit {score(best)*1000:.2f} mm"

    for body in ("leg_base", "tibia_push_lever", "tibia_pushrod"):
        if body not in registrations:
            solve_deferred(body)
    for body in body_map["bodies"]:
        print(f"  {body}: {reg_report[body]}")

    # ---- carry axes through parent registration, verify with child ----
    topology = {
        "coxa_yaw": ("leg_base", "coxa"),
        "femur_pitch": ("coxa", "femur"),
        "tibia_pitch": ("femur", "tibia"),
        "tibia_lever_pivot": ("femur", "tibia_push_lever"),
        "tibia_rod_pivot": ("tibia_push_lever", "tibia_pushrod"),
        "loop_cut": ("tibia_pushrod", "tibia"),
    }
    new_axes = {}
    for jname, spec in body_map["axes"].items():
        pb, cb = topology[jname]
        p0 = np.array(spec["point"])
        d0 = np.array(spec["dir"])
        d0 /= np.linalg.norm(d0)
        Rp, tp = registrations[pb]
        Rc, tc = registrations[cb]
        pp, dp = Rp @ p0 + tp, Rp @ d0
        pc, dc = Rc @ p0 + tc, Rc @ d0
        ang = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(np.dot(dp, dc)))))))
        gap = np.linalg.norm((pc - pp) - np.dot(pc - pp, dp) * dp) * 1000
        print(f"  axis {jname}: parent/child dir mismatch {ang:.2f} deg, line gap {gap:.2f} mm")
        if ang > 2.0 or gap > 3.0:
            print(f"    WARNING: {jname} axis disagreement exceeds tolerance")
        new_axes[jname] = (pp, dp)

    # ---- assign every instance to a body ----
    body_clouds = {}
    for body, members in body_map["bodies"].items():
        R, t = registrations[body]
        pts = [old_parts[n]["centroid"] for n in members if n in old_parts]
        body_clouds[body] = np.array(pts) @ R.T + t
    assignment = defaultdict(list)
    new_only = []
    for inst in instances:
        best, best_d = None, np.inf
        for body, cloud in body_clouds.items():
            d = np.linalg.norm(cloud - inst["centroid"], axis=1).min()
            if d < best_d:
                best, best_d = body, d
        assignment[best].append(inst)
        if stem_count_old.get(inst["stem"], 0) == 0:
            new_only.append((inst, best, best_d))
    print(f"  assigned {len(instances)} instances; {len(new_only)} are new-only parts")

    # Joint-bore audit: list every instance sitting near the knee or lever
    # axis with its current home, then apply explicit user-confirmed overrides
    # for parts whose rigid parent the geometry cannot decide (bolt-circle
    # pattern fragments live inside the bearing bore, equidistant to both
    # sides).
    def audit(jname, radius=0.016):
        ap, ad = new_axes_pre[jname]
        ad = ad / np.linalg.norm(ad)
        rows = []
        for body, insts in assignment.items():
            for inst in insts:
                v = inst["centroid"] - ap
                if np.linalg.norm(v - np.dot(v, ad) * ad) < radius:
                    rows.append((f"{inst['file'][:44]:46s} c={np.round(inst['centroid']*1000,1)}", body))
        return rows

    # axes are needed for the audit before the full axis pass below
    new_axes_pre = {}
    for jname in ("femur_pitch", "tibia_pitch", "tibia_lever_pivot", "tibia_rod_pivot"):
        pb, _cb = topology[jname]
        spec = body_map["axes"][jname]
        Rp, tp = registrations[pb]
        new_axes_pre[jname] = (Rp @ np.array(spec["point"]) + tp, Rp @ np.array(spec["dir"]))
    for jname in ("femur_pitch", "tibia_pitch", "tibia_lever_pivot"):
        print(f"  -- parts near {jname} axis --")
        for f, b in sorted(audit(jname)):
            print(f"     {f[:56]:58s} -> {b}")

    # user-verified rigid attachments the geometry cannot decide.  Each rule:
    # {"pattern": regex on filename, "target": body, "from": optional source
    #  body, "near_axis": optional joint name, "radius": meters (default 16mm)}
    OVERRIDES = (
        json.loads(Path(args.overrides).read_text())
        if args.overrides and Path(args.overrides).exists()
        else []
    )
    moved = 0
    for rule in OVERRIDES:
        rx = re.compile(rule["pattern"])
        target = rule["target"]
        axis = None
        if rule.get("near_axis"):
            axis = new_axes_pre[rule["near_axis"]]
        ref_pts = None
        if rule.get("near_part"):
            prx = re.compile(rule["near_part"])
            ref_pts = np.array(
                [i["centroid"] for b in assignment for i in assignment[b] if prx.search(i["file"])]
            )
        for body in list(assignment):
            if body == target or (rule.get("from") and body != rule["from"]):
                continue
            keep, move = [], []
            for inst in assignment[body]:
                ok = bool(rx.search(inst["file"]))
                if ok and axis is not None:
                    ap, ad = axis
                    adn = ad / np.linalg.norm(ad)
                    v = inst["centroid"] - ap
                    ok = np.linalg.norm(v - np.dot(v, adn) * adn) < rule.get("radius", 0.016)
                if ok and ref_pts is not None and len(ref_pts):
                    ok = np.linalg.norm(ref_pts - inst["centroid"], axis=1).min() < rule.get(
                        "part_radius", 0.011
                    )
                (move if ok else keep).append(inst)
            if move:
                assignment[body] = keep
                assignment[target].extend(move)
                moved += len(move)
                for inst in move:
                    print(f"  override: {inst['file'][:52]} {body} -> {target}")
    if moved:
        print(f"  applied {moved} explicit overrides")

    # ---- body frames on the new axes ----
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

    frames = {"leg_base": np.eye(4)}
    for jname, (pb, cb) in topology.items():
        if jname == "loop_cut":
            continue
        pnt, dr = new_axes[jname]
        if jname in inverted:
            dr = -dr
        frames[cb] = frame_on_axis(pnt, dr)

    # ---- dynamics: transform v2 body inertials, add new-part shares ----
    v2 = ET.parse(Path(args.v2_urdf)).getroot()
    v2_inertials = {}
    for l2 in v2.findall("link"):
        inr = l2.find("inertial")
        m = float(inr.find("mass").get("value"))
        com = parse_floats(inr.find("origin").get("xyz"), [0, 0, 0])
        e = inr.find("inertia")
        I = np.array(
            [
                [float(e.get("ixx")), float(e.get("ixy")), float(e.get("ixz"))],
                [float(e.get("ixy")), float(e.get("iyy")), float(e.get("iyz"))],
                [float(e.get("ixz")), float(e.get("iyz")), float(e.get("izz"))],
            ]
        )
        v2_inertials[l2.get("name")] = (m, com, I)
    v2_frames = json.loads(Path(args.v2_frames).read_text()) if args.v2_frames else None

    residual = fused_mass - sum(m for m, _, _ in v2_inertials.values())
    new_vol = sum(i["volume"] for i, _, _ in new_only) or 1.0
    print(f"  residual mass to distribute over new parts: {residual*1000:.1f} g")

    body_dyn = {}
    for body in body_map["bodies"]:
        R, t = registrations[body]
        m, com_local, I_local = v2_inertials[body]
        # v2 inertial was expressed in the v2 body frame; map: local -> old
        # global (v2 frame) -> new global (registration) -> new body frame.
        T_v2 = np.array(json.loads(Path(args.v2_frames).read_text())[body])
        T_reg = np.eye(4)
        T_reg[:3, :3], T_reg[:3, 3] = R, t
        T_new_inv = np.linalg.inv(frames[body])
        M = T_new_inv @ T_reg @ T_v2
        com = M[:3, :3] @ com_local + M[:3, 3]
        I = M[:3, :3] @ I_local @ M[:3, :3].T
        mass = m
        for inst, b, _ in new_only:
            if b != body:
                continue
            dm = residual * inst["volume"] / new_vol
            c_new = T_new_inv[:3, :3] @ inst["centroid"] + T_new_inv[:3, 3]
            # parallel-axis merge of a point mass
            com = (com * mass + c_new * dm) / (mass + dm)
            mass += dm
        # second pass for inertia about updated COM
        for inst, b, _ in new_only:
            if b != body:
                continue
            dm = residual * inst["volume"] / new_vol
            c_new = T_new_inv[:3, :3] @ inst["centroid"] + T_new_inv[:3, 3]
            r = c_new - com
            I = I + dm * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        body_dyn[body] = (mass, com, I)

    total_mass = sum(m for m, _, _ in body_dyn.values())
    com_global = sum(
        (frames[b][:3, :3] @ c + frames[b][:3, 3]) * m for b, (m, c, _) in body_dyn.items()
    ) / total_mass
    print(
        f"  assembled mass {total_mass:.4f} kg (Onshape {fused_mass:.4f}); "
        f"COM error vs Onshape {np.linalg.norm(com_global - fused_com)*1000:.2f} mm"
    )

    # ---- emit ----
    (out / "urdf").mkdir(parents=True, exist_ok=True)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    for f in (new_src / "assets").glob("*.stl"):
        shutil.copy2(f, out / "meshes" / f.name)

    joint_specs = [
        ("femur_pitch", "femur_pitch", None),
        ("tibia_pitch", "tibia_pitch", None),
        ("tibia_lever_pivot", None, ("tibia_pitch", 1.0)),
        ("tibia_rod_pivot", None, ("tibia_pitch", -1.0)),
    ]

    def emit(path, include_linkage):
        lines = ['<?xml version="1.0" ?>', f'<robot name="{path.stem}">']
        order = ["coxa", "femur", "tibia"] + (
            ["tibia_push_lever", "tibia_pushrod"] if include_linkage else []
        )
        merged_into = {"coxa": ["leg_base"]}
        if not include_linkage:
            merged_into["femur"] = ["tibia_push_lever", "tibia_pushrod"]
        for body in order:
            insts = list(assignment[body])
            for extra in merged_into.get(body, []):
                insts += assignment[extra]
            mass, com, I = body_dyn[body]
            if body in merged_into:
                for extra in merged_into[body]:
                    me, ce, Ie = body_dyn[extra]
                    Te = np.linalg.inv(frames[body]) @ frames[extra]
                    ce2 = Te[:3, :3] @ ce + Te[:3, 3]
                    Ie2 = Te[:3, :3] @ Ie @ Te[:3, :3].T
                    new_com = (com * mass + ce2 * me) / (mass + me)
                    for mm, cc, II in ((mass, com, I), (me, ce2, Ie2)):
                        r = cc - new_com
                        II += mm * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
                    I = I + Ie2
                    com, mass = new_com, mass + me
            Tinv = np.linalg.inv(frames[body])
            lines.append(f'  <link name="{body}">')
            lines.append("    <inertial>")
            lines.append(f'      <origin xyz="{fmt(com)}" rpy="0 0 0" />')
            lines.append(f'      <mass value="{mass:.8g}" />')
            lines.append(
                f'      <inertia ixx="{I[0,0]:.8g}" ixy="{I[0,1]:.8g}" ixz="{I[0,2]:.8g}" '
                f'iyy="{I[1,1]:.8g}" iyz="{I[1,2]:.8g}" izz="{I[2,2]:.8g}" />'
            )
            lines.append("    </inertial>")
            lo = np.full(3, np.inf)
            hi = np.full(3, -np.inf)
            for k, inst in enumerate(insts):
                Tv = Tinv @ make_T(inst["xyz"], inst["rpy"])
                lines.append("    <visual>")
                lines.append(
                    f'      <origin xyz="{fmt(Tv[:3,3])}" rpy="{fmt(matrix_to_rpy(Tv[:3,:3]))}" />'
                )
                lines.append(
                    f'      <geometry><mesh filename="package://hexapod_leg_v3/meshes/{inst["file"]}" /></geometry>'
                )
                if inst["rgba"]:
                    lines.append(
                        f'      <material name="{body}_m{k}"><color rgba="{inst["rgba"]}" /></material>'
                    )
                lines.append("    </visual>")
                pts = inst["verts"] @ Tinv[:3, :3].T + Tinv[:3, 3]
                lo = np.minimum(lo, pts.min(axis=0))
                hi = np.maximum(hi, pts.max(axis=0))
            if np.isfinite(lo).all():
                size = np.maximum(hi - lo, 1e-3)
                lines.append("    <collision>")
                lines.append(f'      <origin xyz="{fmt((hi+lo)/2)}" rpy="0 0 0" />')
                lines.append(f'      <geometry><box size="{fmt(size)}" /></geometry>')
                lines.append("    </collision>")
            lines.append("  </link>")

        for jname, limkey, mimic in joint_specs:
            if not include_linkage and mimic is not None:
                continue
            pb, cb = topology[jname]
            Tj = np.linalg.inv(frames[pb]) @ frames[cb]
            lines.append(f'  <joint name="{jname}" type="revolute">')
            lines.append(
                f'    <origin xyz="{fmt(Tj[:3,3])}" rpy="{fmt(matrix_to_rpy(Tj[:3,:3]))}" />'
            )
            lines.append('    <axis xyz="0 0 1" />')
            lines.append(f'    <parent link="{pb}" />')
            lines.append(f'    <child link="{cb}" />')
            if limkey:
                lo_, hi_ = limits[limkey]
                lines.append(
                    f'    <limit effort="{RS05_EFFORT_NM}" velocity="{RS05_VELOCITY_RAD_S}" '
                    f'lower="{lo_}" upper="{hi_}" />'
                )
            else:
                span = max(abs(x) for x in limits["tibia_pitch"])
                lines.append(
                    f'    <limit effort="0.5" velocity="{RS05_VELOCITY_RAD_S}" '
                    f'lower="{-span}" upper="{span}" />'
                )
                lines.append(
                    f'    <mimic joint="{mimic[0]}" multiplier="{mimic[1]}" offset="0" />'
                )
            lines.append('    <dynamics damping="0.01" friction="0.01" />')
            lines.append("  </joint>")
        if include_linkage:
            p_cut, d_cut = new_axes["loop_cut"]
            Tr = np.linalg.inv(frames["tibia_pushrod"])
            Tt = np.linalg.inv(frames["tibia"])
            lines.append("  <!-- LOOP CLOSURE (cut joint): pushrod point "
                         f"[{fmt(Tr[:3,:3] @ p_cut + Tr[:3,3])}] pivots on tibia point "
                         f"[{fmt(Tt[:3,:3] @ p_cut + Tt[:3,3])}]; parallelogram closed by mimics. -->")
        lines.append("</robot>")
        path.write_text("\n".join(lines) + "\n")

    emit(out / "urdf" / "leg_v3_serial.urdf", include_linkage=False)
    emit(out / "urdf" / "leg_v3_linkage.urdf", include_linkage=True)

    # canonical single-leg reference for the future six-leg assembly import
    reference = {
        "source_document": "https://cad.onshape.com/documents/881b01051a1ec5c958bf7768",
        "frame": "leg_base (identity = onshape-to-robot export frame of this leg)",
        "body_interface": {
            "yaw_axis_point_m": [round(float(x), 8) for x in new_axes["coxa_yaw"][0]],
            "yaw_axis_dir": [round(float(x), 8) for x in (-new_axes["coxa_yaw"][1] if "coxa_yaw" in inverted else new_axes["coxa_yaw"][1])],
            "note": "yaw DOF exists between chassis and leg; fixed inside the single-leg reference",
        },
        "joints": {
            j: {
                "parent": topology[j][0],
                "child": topology[j][1],
                "axis_point_m": [round(float(x), 8) for x in new_axes[j][0]],
                "axis_dir": [round(float(x), 8) for x in new_axes[j][1]],
                "limits": (list(limits[j]) if j in limits else None),
                "mimic": ({"joint": "tibia_pitch", "multiplier": 1.0} if j == "tibia_lever_pivot"
                          else {"joint": "tibia_pitch", "multiplier": -1.0} if j == "tibia_rod_pivot"
                          else None),
            }
            for j in ("femur_pitch", "tibia_pitch", "tibia_lever_pivot", "tibia_rod_pivot")
        },
        "loop_closure": {
            "cut_point_m": [round(float(x), 8) for x in new_axes["loop_cut"][0]],
            "cut_axis_dir": [round(float(x), 8) for x in new_axes["loop_cut"][1]],
            "note": "parallelogram four-bar; mimics close it exactly in kinematics",
        },
        "bodies": {b: {"mass_kg": round(float(body_dyn[b][0]), 6)} for b in body_dyn},
        "total_mass_kg": round(float(total_mass), 6),
        "rs05": {"effort_nm": RS05_EFFORT_NM, "velocity_rad_s": RS05_VELOCITY_RAD_S},
    }
    (out / "leg_reference.json").write_text(json.dumps(reference, indent=1))

    # parallelogram check in the NEW geometry
    A = new_axes["tibia_lever_pivot"][0]
    B = new_axes["tibia_rod_pivot"][0]
    C = new_axes["loop_cut"][0]
    D = new_axes["tibia_pitch"][0]
    print(
        "  four-bar (new) mm: lever %.2f rod %.2f crank %.2f ground %.2f"
        % (
            np.linalg.norm(B - A) * 1000,
            np.linalg.norm(C - B) * 1000,
            np.linalg.norm(C - D) * 1000,
            np.linalg.norm(A - D) * 1000,
        )
    )
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-source", default="/Users/andreboufama/Downloads/leg_subassy")
    ap.add_argument("--new-source", default="/Users/andreboufama/Downloads/export")
    ap.add_argument("--body-map", default="robot/hexapod_leg_v2/body_map.json")
    ap.add_argument("--v2-urdf", default="robot/hexapod_leg_v2/urdf/leg_v2_linkage.urdf")
    ap.add_argument("--v2-frames", default="robot/hexapod_leg_v2/frames.json")
    ap.add_argument("--overrides", default="robot/hexapod_leg_v3/part_overrides.json")
    ap.add_argument("--limits", default="robot/hexapod_leg_v3/joint_limits.json")
    ap.add_argument("--out", default="robot/hexapod_leg_v3")
    sys.exit(build(ap.parse_args()))
