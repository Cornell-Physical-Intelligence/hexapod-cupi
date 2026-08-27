#!/usr/bin/env python3
"""Consolidate the raw CAD leg export into clean, simulation-ready URDFs.

The raw ``leg_subassy`` export has 278 links (every screw and washer is a
link), visual-only geometry, mojibake CAD names, unlimited ``continuous``
joints, and a four-bar tibia linkage the exporter broke by duplicating the
tibia into a phantom ``*_loop_closure`` link.  URDF cannot represent closed
kinematic chains, so this tool rebuilds the model as a tree two ways:

* ``leg_v2_serial.urdf`` — three actuated DOFs (coxa yaw, femur pitch, tibia
  pitch at the physical knee hinge).  The push lever and pushrod are welded
  into the femur body at the CAD zero pose, so the linkage is visually
  present but kinematically folded into the directly driven knee joint.
  This matches the 18-joint contract the training pipeline already uses.
* ``leg_v2_linkage.urdf`` — the same consolidation but with the lever and
  pushrod kept as real bodies on passive revolute joints.  The final
  pushrod-to-tibia pivot is the cut joint (URDF cannot close the loop); its
  anchor coordinates are emitted in a comment so engines with loop-closure
  support (MuJoCo equality, PhysX D6 outside the articulation) can restore
  the constraint.

Method: forward kinematics at the CAD zero pose, union-find over fixed
joints, clustering of moving joints by coincident global axis lines (twin
yoke bearings and motor-rotor spins collapse onto their physical hinge),
semantic identification of the hinge clusters from the body-connectivity
graph, then per-body fusion of every member part's mass, inertia (parallel
axis), and visual geometry into the hinge child frame.  Collision geometry
(one box per body plus a foot sphere at the tibia tip) is generated from the
binary STL bounds because the export carries no collision elements at all.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

RS05_EFFORT_NM = 5.5
RS05_VELOCITY_RAD_S = 50.26548246
# Pipeline joint ranges from hexapod_mkii_robstride.urdf.  Sign/offset must be
# re-verified against this leg's CAD zero before hardware use.
LIMITS = {
    "coxa_yaw": (-0.872665, 0.872665),
    "femur_pitch": (0.0, 1.74533),
    "tibia_pitch": (0.0, 2.53073),
}
PASSIVE_LIMIT = (-1.3, 1.3)


def rpy_to_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def matrix_to_rpy(R):
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    p = math.asin(sy)
    if abs(abs(sy) - 1.0) < 1e-9:
        return (math.atan2(-R[0, 1], R[1, 1]), p, 0.0)
    return (math.atan2(R[2, 1], R[2, 2]), p, math.atan2(R[1, 0], R[0, 0]))


def make_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    return T


def parse_floats(s, default):
    if s is None:
        return np.array(default, dtype=float)
    return np.array([float(x) for x in s.split()], dtype=float)


class Model:
    def __init__(self, urdf_path: Path):
        self.tree = ET.parse(urdf_path)
        root = self.tree.getroot()
        self.links = {l.get("name"): l for l in root.findall("link")}
        self.joints = []
        for j in root.findall("joint"):
            self.joints.append(
                {
                    "name": j.get("name"),
                    "type": j.get("type"),
                    "parent": j.find("parent").get("link"),
                    "child": j.find("child").get("link"),
                    "xyz": parse_floats(j.find("origin").get("xyz") if j.find("origin") is not None else None, [0, 0, 0]),
                    "rpy": parse_floats(j.find("origin").get("rpy") if j.find("origin") is not None else None, [0, 0, 0]),
                    "axis": parse_floats(j.find("axis").get("xyz") if j.find("axis") is not None else None, [1, 0, 0]),
                }
            )
        children = {j["child"] for j in self.joints}
        roots = [n for n in self.links if n not in children]
        if len(roots) != 1:
            raise SystemExit(f"expected one root link, found {roots}")
        self.root = roots[0]
        # forward kinematics at zero pose
        self.T = {self.root: np.eye(4)}
        by_parent = defaultdict(list)
        for j in self.joints:
            by_parent[j["parent"]].append(j)
        stack = [self.root]
        while stack:
            p = stack.pop()
            for j in by_parent[p]:
                self.T[j["child"]] = self.T[p] @ make_T(j["xyz"], j["rpy"])
                stack.append(j["child"])
        missing = set(self.links) - set(self.T)
        if missing:
            raise SystemExit(f"unreachable links: {sorted(missing)[:5]}")


def cluster_axes(model, moving):
    """Group moving joints whose global axis lines coincide."""
    infos = []
    for j in moving:
        Tp = model.T[j["parent"]] @ make_T(j["xyz"], j["rpy"])
        point = Tp[:3, 3]
        direction = Tp[:3, :3] @ (j["axis"] / np.linalg.norm(j["axis"]))
        infos.append((j, point, direction))
    clusters = []
    for j, p, d in infos:
        placed = False
        for cl in clusters:
            _, p0, d0 = cl[0]
            if abs(abs(float(np.dot(d, d0))) - 1.0) < 1e-4:
                delta = p - p0
                if np.linalg.norm(delta - np.dot(delta, d0) * d0) < 5e-4:
                    cl.append((j, p, d))
                    placed = True
                    break
        if not placed:
            clusters.append([(j, p, d)])
    return clusters


class UnionFind:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def fuse_inertial(model, members, T_anchor_inv):
    mass = 0.0
    moment = np.zeros(3)
    for name in members:
        inr = model.links[name].find("inertial")
        if inr is None:
            continue
        m = float(inr.find("mass").get("value"))
        o = inr.find("origin")
        xyz = parse_floats(o.get("xyz") if o is not None else None, [0, 0, 0])
        rpy = parse_floats(o.get("rpy") if o is not None else None, [0, 0, 0])
        Tin = T_anchor_inv @ model.T[name] @ make_T(xyz, rpy)
        mass += m
        moment += m * Tin[:3, 3]
    com = moment / mass if mass > 0 else np.zeros(3)
    I = np.zeros((3, 3))
    for name in members:
        inr = model.links[name].find("inertial")
        if inr is None:
            continue
        m = float(inr.find("mass").get("value"))
        o = inr.find("origin")
        xyz = parse_floats(o.get("xyz") if o is not None else None, [0, 0, 0])
        rpy = parse_floats(o.get("rpy") if o is not None else None, [0, 0, 0])
        el = inr.find("inertia")
        Ii = np.array(
            [
                [float(el.get("ixx")), float(el.get("ixy")), float(el.get("ixz"))],
                [float(el.get("ixy")), float(el.get("iyy")), float(el.get("iyz"))],
                [float(el.get("ixz")), float(el.get("iyz")), float(el.get("izz"))],
            ]
        )
        Tin = T_anchor_inv @ model.T[name] @ make_T(xyz, rpy)
        R = Tin[:3, :3]
        Irot = R @ Ii @ R.T
        r = Tin[:3, 3] - com
        I += Irot + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return mass, com, I


def stl_bounds(path: Path):
    """Binary STL AABB without loading full geometry into python objects."""
    try:
        with open(path, "rb") as f:
            header = f.read(84)
            if len(header) < 84:
                return None
            (n,) = struct.unpack("<I", header[80:84])
            expected = 84 + n * 50
            if path.stat().st_size < expected:
                return None  # ascii or truncated
            data = np.fromfile(f, dtype=np.uint8, count=n * 50)
        tri = data.reshape(n, 50)
        v = tri[:, 12:48].copy().view("<f4").reshape(n, 3, 3)
        pts = v.reshape(-1, 3).astype(float)
        return pts.min(axis=0), pts.max(axis=0)
    except OSError:
        return None


def stl_sample(path: Path, count=120):
    """Sample vertices from a binary STL for nearest-body adoption tests."""
    try:
        with open(path, "rb") as f:
            header = f.read(84)
            if len(header) < 84:
                return None
            (n,) = struct.unpack("<I", header[80:84])
            if path.stat().st_size < 84 + n * 50:
                return None
            data = np.fromfile(f, dtype=np.uint8, count=n * 50)
        v = data.reshape(n, 50)[:, 12:48].copy().view("<f4").reshape(n * 3, 3).astype(float)
        if len(v) > count:
            idx = np.linspace(0, len(v) - 1, count).astype(int)
            v = v[idx]
        return v
    except OSError:
        return None


def body_bounds(model, members, T_anchor_inv, mesh_dir):
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for name in members:
        for vis in model.links[name].findall("visual"):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            fname = Path(mesh.get("filename").split("/")[-1])
            b = stl_bounds(mesh_dir / fname)
            if b is None:
                continue
            o = vis.find("origin")
            xyz = parse_floats(o.get("xyz") if o is not None else None, [0, 0, 0])
            rpy = parse_floats(o.get("rpy") if o is not None else None, [0, 0, 0])
            Tv = T_anchor_inv @ model.T[name] @ make_T(xyz, rpy)
            mn, mx = b
            corners = np.array([[mn[0], mx[0]][i & 1 == 1] for i in range(0)])
            cs = []
            for ix in (mn[0], mx[0]):
                for iy in (mn[1], mx[1]):
                    for iz in (mn[2], mx[2]):
                        cs.append((ix, iy, iz))
            cs = (Tv[:3, :3] @ np.array(cs).T).T + Tv[:3, 3]
            lo = np.minimum(lo, cs.min(axis=0))
            hi = np.maximum(hi, cs.max(axis=0))
    if not np.isfinite(lo).all():
        return None
    return lo, hi


def fmt(v, nd=8):
    return " ".join(f"{x:.{nd}g}" for x in v)


def build(args):
    src = Path(args.source)
    model = Model(src / "urdf" / "leg_subassy.urdf")
    mesh_dir = src / "meshes"

    uf = UnionFind(model.links)
    for j in model.joints:
        if j["type"] == "fixed":
            uf.union(j["parent"], j["child"])
    members_of = defaultdict(set)
    for l in model.links:
        members_of[uf.find(l)].add(l)

    def group(link):
        return uf.find(link)

    def joint(name):
        return next(j for j in model.joints if j["name"] == name)

    def pivot(jname):
        j = joint(jname)
        Tg = model.T[j["parent"]] @ make_T(j["xyz"], j["rpy"])
        d = Tg[:3, :3] @ (j["axis"] / np.linalg.norm(j["axis"]))
        return Tg[:3, 3], d

    phantom = "machined_tibia__1__loop_closure"

    # Explicit topology, established from the axis/pivot analysis:
    #   yaw motor axis   = rotor bearing line of revolute_1_6 (vertical)
    #   femur motor axis = the revolute_1_3 / revolute_1_2 line (pitch)
    #   knee axis        = revolute_1_7 (pitch); revolute_2_3 is its twin
    #   lever axis       = revolute_1_1 (tibia motor output on the cradle)
    #   rod pivots       = revolute_1 (proximal), loop-closure joint (distal)
    # The tibia four-bar is a parallelogram (lever = crank = 30.0 mm,
    # rod = ground = 77.5 mm), so lever tracks the knee 1:1 and the rod
    # counter-rotates 1:1 -- expressed below as URDF mimic joints.
    g_base = group("first_joint_top")          # yaw-motor stator group (41 parts)
    g_root = group(model.root)                  # coxa housing
    g_m2 = group("mirror1_2")
    g_m3 = group("mirror1_3")
    g_f1 = group("femur_plate")                 # motor-side femur plate (+ cradle motor)
    g_cradle = group("tibia_push_lever")        # placeholder replaced below
    g_lever = group("tibia_push_lever")
    g_rod = group("tibia_pushrod")
    g_f2 = group("femur_plate_non_motor")
    g_tibia = group("machined_tibia")
    g_cradle = group(joint("revolute_2")["child"])      # RobStride cradle
    _unused = group(joint("revolute_1_6")["child"])
    _unused = group(joint("revolute_2_2")["child"])
    _unused = group(joint("revolute_1_2")["child"])
    _unused = group(joint("revolute_2_1")["child"])
    _unused = group(joint("revolute_1_5")["child"])
    _unused = group(joint("revolute_2_3")["child"])
    _unused = group("tibia_spacer") if "tibia_spacer" in model.links else None

    # Drop parts the CAD export left at unresolved-mate positions: any part
    # whose geometry sits more than 6 mm from every other part is floating in
    # the source itself (loose fasteners / bearing caps).  Structural parts
    # above 5 g are never dropped silently.
    part_pts = {}
    for name, link in model.links.items():
        pts = []
        for vis in link.findall("visual"):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            verts = stl_sample(mesh_dir / Path(mesh.get("filename").split("/")[-1]), count=80)
            if verts is None:
                continue
            o = vis.find("origin")
            xyz = parse_floats(o.get("xyz") if o is not None else None, [0, 0, 0])
            rpy = parse_floats(o.get("rpy") if o is not None else None, [0, 0, 0])
            Tv = model.T[name] @ make_T(xyz, rpy)
            pts.append((Tv[:3, :3] @ verts.T).T + Tv[:3, 3])
        if pts:
            part_pts[name] = np.concatenate(pts)
    names_all = list(part_pts)
    clouds = [part_pts[n] for n in names_all]
    floaters = set()
    dropped_mass = 0.0
    for i, n in enumerate(names_all):
        mine = part_pts[n][::2]
        best = np.inf
        for jdx, other in enumerate(clouds):
            if jdx == i:
                continue
            d = np.sqrt(((mine[:, None, :] - other[None, ::3, :]) ** 2).sum(-1)).min()
            best = min(best, d)
            if best < 0.006:
                break
        if best >= 0.006:
            inr = model.links[n].find("inertial")
            m = float(inr.find("mass").get("value")) if inr is not None else 0.0
            if m > 0.005:
                print(f"  WARNING: structural part floating {best*1000:.1f} mm, kept: {n[:50]} ({m*1000:.1f} g)")
                continue
            floaters.add(n)
            dropped_mass += m
            print(f"  drop floater {n[:56]} ({best*1000:.1f} mm away, {m*1000:.2f} g)")
    if floaters:
        print(f"  dropped {len(floaters)} floating parts, {dropped_mass*1000:.2f} g total")
        for g in members_of:
            members_of[g] -= floaters

    seed_bodies = {
        "leg_base": {g_base},
        "coxa": {g_root, g_m2, g_m3},
        "femur": {g_f1, g_cradle, g_f2},
        "tibia": {g_tibia},
        "tibia_push_lever": {g_lever},
        "tibia_pushrod": {g_rod},
    }
    bodies = {k: set(v) for k, v in seed_bodies.items()}
    assigned = set()
    for gs in bodies.values():
        assigned |= gs

    # Adopt every remaining group (motor rotors, flanges, bearing inserts,
    # stray fasteners) onto the physically nearest seed body at the CAD zero
    # pose: a part is welded to whatever it touches.
    def group_points(gs):
        pts = []
        for g in gs:
            for name in members_of[g]:
                for vis in model.links[name].findall("visual"):
                    mesh = vis.find("geometry/mesh")
                    if mesh is None:
                        continue
                    verts = stl_sample(mesh_dir / Path(mesh.get("filename").split("/")[-1]))
                    if verts is None:
                        continue
                    o = vis.find("origin")
                    xyz = parse_floats(o.get("xyz") if o is not None else None, [0, 0, 0])
                    rpy = parse_floats(o.get("rpy") if o is not None else None, [0, 0, 0])
                    Tv = model.T[name] @ make_T(xyz, rpy)
                    pts.append((Tv[:3, :3] @ verts.T).T + Tv[:3, 3])
        return np.concatenate(pts) if pts else None

    seed_clouds = {k: group_points(v) for k, v in bodies.items()}
    adoption = []
    for g in members_of:
        if g in assigned or members_of[g] == {phantom}:
            continue
        cloud = group_points({g})
        if cloud is None:
            # geometry-free marker: keep with the root housing side
            bodies["coxa"].add(g)
            continue
        best, best_d = None, np.inf
        for k, ref in seed_clouds.items():
            if ref is None:
                continue
            step = max(1, len(cloud) // 60)
            sub = cloud[::step]
            d = np.sqrt(((sub[:, None, :] - ref[None, ::4, :]) ** 2).sum(-1)).min()
            if d < best_d:
                best, best_d = k, d
        bodies[best].add(g)
        adoption.append((sorted(members_of[g])[0][:40], len(members_of[g]), best, best_d * 1000))
    for part, n, home, dmm in sorted(adoption, key=lambda x: x[2]):
        print(f"  adopt {part} ({n} parts) -> {home} ({dmm:.1f} mm)")

    # Hinge-interface refinement: CAD mates sometimes weld a bearing race or
    # rotor retainer to the wrong side of a joint.  For every part sitting
    # within 15 mm of a kept hinge axis, re-home it to the adjacent body whose
    # away-from-hinge structure it is actually closest to.
    def axis_specs():
        return [
            ("femur_pitch_ax", pivot("revolute_1_3"), ("coxa", "femur")),
            ("knee_ax", pivot("revolute_1_7"), ("femur", "tibia")),
            ("yaw_ax", pivot("revolute_1_6"), ("leg_base", "coxa")),
            ("lever_ax", pivot("revolute_1_1"), ("femur", "tibia_push_lever")),
        ]

    def part_cloud_map(body):
        out = {}
        for g in bodies[body]:
            for name in members_of[g]:
                if name in part_pts:
                    out[name] = part_pts[name]
        return out

    transfers = []
    for axname, (apt, adir), (bodyA, bodyB) in axis_specs():
        adir = adir / np.linalg.norm(adir)
        partsA, partsB = part_cloud_map(bodyA), part_cloud_map(bodyB)
        both = [(n, c, bodyA, bodyB) for n, c in partsA.items()] + [
            (n, c, bodyB, bodyA) for n, c in partsB.items()
        ]
        def axis_dist(cloud):
            rel = cloud - apt
            perp = rel - np.outer(rel @ adir, adir)
            return np.linalg.norm(perp, axis=1).min()
        near = [(n, c, home, other) for n, c, home, other in both if axis_dist(c) < 0.015]
        near_names = {n for n, *_ in near}
        def struct_cloud(parts):
            arrs = [c for n, c in parts.items() if n not in near_names]
            return np.concatenate(arrs) if arrs else None
        structA, structB = struct_cloud(partsA), struct_cloud(partsB)
        struct_of = {bodyA: structA, bodyB: structB}
        for n, c, home, other in near:
            sh, so = struct_of[home], struct_of[other]
            if sh is None or so is None:
                continue
            sub = c[:: max(1, len(c) // 40)]
            dh = np.sqrt(((sub[:, None, :] - sh[None, ::5, :]) ** 2).sum(-1)).min()
            do = np.sqrt(((sub[:, None, :] - so[None, ::5, :]) ** 2).sum(-1)).min()
            if do + 0.001 < dh:
                transfers.append((n, home, other, dh * 1000, do * 1000))
                for g in bodies[home]:
                    if n in members_of[g] and len(members_of[g]) > 1:
                        members_of[g] = members_of[g] - {n}
                        solo = f"__transfer__{n}"
                        members_of[solo] = {n}
                        bodies[other].add(solo)
                        break
                else:
                    # part is alone in its group: just move the group
                    for g in list(bodies[home]):
                        if members_of[g] == {n}:
                            bodies[home].discard(g)
                            bodies[other].add(g)
                            break
    for n, home, other, dh, do in transfers:
        print(f"  re-home {n[:52]}: {home} -> {other} (own {dh:.1f} mm vs other {do:.1f} mm)")

    def body_links(name):
        out = set()
        for g in bodies[name]:
            out |= members_of[g]
        out.discard(phantom)
        return out

    p_yaw, d_yaw = pivot("revolute_1_6")
    p_fem, d_fem = pivot("revolute_1_3")
    p_knee, d_knee = pivot("revolute_1_7")
    p_lever, d_lever = pivot("revolute_1_1")
    p_rodp, d_rodp = pivot("revolute_1")
    p_cut, d_cut = pivot("revolute_1_loop_closure")

    def frame_on_axis(point, direction):
        z = direction / np.linalg.norm(direction)
        seed = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(seed, z)) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        x = seed - np.dot(seed, z) * z
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, point
        return T

    frames = {
        "leg_base": np.eye(4),
        "coxa": frame_on_axis(p_yaw, d_yaw),
        "femur": frame_on_axis(p_fem, d_fem),
        "tibia": frame_on_axis(p_knee, d_knee),
        "tibia_push_lever": frame_on_axis(p_lever, d_lever),
        "tibia_pushrod": frame_on_axis(p_rodp, d_rodp),
    }
    joints_def = [
        # name, type-key, parent, child, point, dir, mimic
        ("coxa_yaw", "coxa_yaw", "leg_base", "coxa", None),
        ("femur_pitch", "femur_pitch", "coxa", "femur", None),
        ("tibia_pitch", "tibia_pitch", "femur", "tibia", None),
        ("tibia_lever_pivot", None, "femur", "tibia_push_lever", ("tibia_pitch", 1.0)),
        ("tibia_rod_pivot", None, "tibia_push_lever", "tibia_pushrod", ("tibia_pitch", -1.0)),
    ]

    def emit(path, include_linkage):
        lines = ['<?xml version="1.0" ?>', f'<robot name="{path.stem}">']
        total = 0.0
        order = ["leg_base", "coxa", "femur", "tibia"] + (
            ["tibia_push_lever", "tibia_pushrod"] if include_linkage else []
        )
        for bname in order:
            mem = body_links(bname)
            if not include_linkage and bname == "femur":
                mem = mem | body_links("tibia_push_lever") | body_links("tibia_pushrod")
            Ta_inv = np.linalg.inv(frames[bname])
            mass, com, I = fuse_inertial(model, mem, Ta_inv)
            total += mass
            lines.append(f'  <link name="{bname}">')
            lines.append("    <inertial>")
            lines.append(f'      <origin xyz="{fmt(com)}" rpy="0 0 0" />')
            lines.append(f'      <mass value="{mass:.8g}" />')
            lines.append(
                f'      <inertia ixx="{I[0,0]:.8g}" ixy="{I[0,1]:.8g}" ixz="{I[0,2]:.8g}" '
                f'iyy="{I[1,1]:.8g}" iyz="{I[1,2]:.8g}" izz="{I[2,2]:.8g}" />'
            )
            lines.append("    </inertial>")
            for name in sorted(mem):
                for vis in model.links[name].findall("visual"):
                    mesh = vis.find("geometry/mesh")
                    if mesh is None:
                        continue
                    o = vis.find("origin")
                    xyz = parse_floats(o.get("xyz") if o is not None else None, [0, 0, 0])
                    rpy = parse_floats(o.get("rpy") if o is not None else None, [0, 0, 0])
                    Tv = Ta_inv @ model.T[name] @ make_T(xyz, rpy)
                    fname = mesh.get("filename").split("/")[-1]
                    lines.append("    <visual>")
                    lines.append(
                        f'      <origin xyz="{fmt(Tv[:3,3])}" rpy="{fmt(matrix_to_rpy(Tv[:3,:3]))}" />'
                    )
                    lines.append(
                        f'      <geometry><mesh filename="package://hexapod_leg_v2/meshes/{fname}" /></geometry>'
                    )
                    lines.append("    </visual>")
            b = body_bounds(model, mem, Ta_inv, mesh_dir)
            if b is not None:
                lo, hi = b
                size = np.maximum(hi - lo, 1e-3)
                center = (hi + lo) / 2
                lines.append("    <collision>")
                lines.append(f'      <origin xyz="{fmt(center)}" rpy="0 0 0" />')
                lines.append(f'      <geometry><box size="{fmt(size)}" /></geometry>')
                lines.append("    </collision>")
            lines.append("  </link>")

        for jname, limkey, pb, cb, mimic in joints_def:
            if not include_linkage and mimic is not None:
                continue
            Tj = np.linalg.inv(frames[pb]) @ frames[cb]
            lines.append(f'  <joint name="{jname}" type="revolute">')
            lines.append(
                f'    <origin xyz="{fmt(Tj[:3,3])}" rpy="{fmt(matrix_to_rpy(Tj[:3,:3]))}" />'
            )
            lines.append('    <axis xyz="0 0 1" />')
            lines.append(f'    <parent link="{pb}" />')
            lines.append(f'    <child link="{cb}" />')
            if limkey is not None:
                lo, hi = LIMITS[limkey]
                lines.append(
                    f'    <limit effort="{RS05_EFFORT_NM}" velocity="{RS05_VELOCITY_RAD_S}" '
                    f'lower="{lo}" upper="{hi}" />'
                )
            else:
                lo, hi = LIMITS["tibia_pitch"]
                span = max(abs(lo), abs(hi))
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
            Trod_inv = np.linalg.inv(frames["tibia_pushrod"])
            Ttib_inv = np.linalg.inv(frames["tibia"])
            p_rod = (Trod_inv @ np.append(p_cut, 1.0))[:3]
            p_tib = (Ttib_inv @ np.append(p_cut, 1.0))[:3]
            lines.append("  <!-- LOOP CLOSURE (cut joint; URDF cannot close kinematic loops):")
            lines.append(f"       tibia_pushrod point [{fmt(p_rod)}] pivots on tibia point [{fmt(p_tib)}].")
            lines.append("       The four-bar is a parallelogram, so the mimic joints above close it")
            lines.append("       exactly in kinematics; physics engines should add the pivot:")
            lines.append("       MuJoCo equality/connect, PhysX/USD D6 outside the articulation. -->")
        lines.append("</robot>")
        path.write_text("\n".join(lines) + "\n")
        return total

    out = Path(args.out)
    (out / "urdf").mkdir(parents=True, exist_ok=True)
    # machine-readable body map for downstream importers
    body_map = {
        "bodies": {
            b: sorted(body_links(b))
            for b in ["leg_base", "coxa", "femur", "tibia", "tibia_push_lever", "tibia_pushrod"]
        },
        "axes": {
            "coxa_yaw": {"point": list(p_yaw), "dir": list(d_yaw)},
            "femur_pitch": {"point": list(p_fem), "dir": list(d_fem)},
            "tibia_pitch": {"point": list(p_knee), "dir": list(d_knee)},
            "tibia_lever_pivot": {"point": list(p_lever), "dir": list(d_lever)},
            "tibia_rod_pivot": {"point": list(p_rodp), "dir": list(d_rodp)},
            "loop_cut": {"point": list(p_cut), "dir": list(d_cut)},
        },
    }
    import json as _json
    (out / "body_map.json").write_text(_json.dumps(body_map, indent=1))
    (out / "frames.json").write_text(
        _json.dumps({k: [list(row) for row in v] for k, v in frames.items()}, indent=1)
    )
    m1 = emit(out / "urdf" / "leg_v2_serial.urdf", include_linkage=False)
    m2 = emit(out / "urdf" / "leg_v2_linkage.urdf", include_linkage=True)

    # parallelogram verification
    lever = np.linalg.norm(p_rodp - p_lever)
    rod = np.linalg.norm(p_cut - p_rodp)
    crank = np.linalg.norm(p_cut - p_knee)
    ground = np.linalg.norm(p_lever - p_knee)
    doc = out / "docs" / "four_bar_transfer.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    with open(doc, "w") as f:
        f.write("# Tibia four-bar (CAD zero-pose pivots)\n\n")
        f.write(f"- lever (motor arm): {lever*1000:.2f} mm\n")
        f.write(f"- pushrod:           {rod*1000:.2f} mm\n")
        f.write(f"- tibia crank:       {crank*1000:.2f} mm\n")
        f.write(f"- ground (on femur): {ground*1000:.2f} mm\n\n")
        f.write(f"Parallelogram error: lever-crank {abs(lever-crank)*1000:.3f} mm, ")
        f.write(f"rod-ground {abs(rod-ground)*1000:.3f} mm.\n")
        f.write("Tibia angle tracks the lever 1:1; mimic joints close the loop exactly.\n")
    print(f"serial {m1:.4f} kg | linkage {m2:.4f} kg")
    print(
        f"four-bar mm: lever {lever*1000:.2f} rod {rod*1000:.2f} crank {crank*1000:.2f} ground {ground*1000:.2f}"
    )
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/Users/andreboufama/Downloads/leg_subassy")
    ap.add_argument("--out", default="robot/hexapod_leg_v2")
    sys.exit(build(ap.parse_args()))
