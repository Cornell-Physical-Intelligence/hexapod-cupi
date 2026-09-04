#!/usr/bin/env python3
"""Pack a URDF package into one self-contained HTML viewer.

Meshes are welded, quantised to 16-bit and embedded as base64 next to the
link/joint tree, so the page opens anywhere without a server.  Vendor motor
internals that sit inside the housings can be left out with ``--drop``.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import struct
import sys
from pathlib import Path

import numpy as np
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent))
from consolidate_leg_urdf import parse_floats, rpy_to_matrix  # noqa: E402


def quat_xyzw(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    if i == 1:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]


def origin(el):
    o = el.find("origin")
    xyz = parse_floats(o.get("xyz") if o is not None else None, [0, 0, 0])
    rpy = parse_floats(o.get("rpy") if o is not None else None, [0, 0, 0])
    return [round(float(x), 7) for x in xyz], [round(float(x), 7) for x in quat_xyzw(rpy_to_matrix(rpy))]


def load_stl(path: Path):
    with open(path, "rb") as f:
        header = f.read(84)
        (n,) = struct.unpack("<I", header[80:84])
        data = np.fromfile(f, dtype=np.uint8, count=n * 50)
    return data.reshape(n, 50)[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(np.float64)


def pack_mesh(tris, blob: bytearray):
    verts = tris.reshape(-1, 3)
    uniq, inv = np.unique(np.round(verts, 6), axis=0, return_inverse=True)
    idx = inv.reshape(-1, 3)
    ok = (idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & (idx[:, 0] != idx[:, 2])
    idx = idx[ok]
    lo, hi = uniq.min(axis=0), uniq.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    q = np.round((uniq - lo) / span * 65535.0).astype(np.int64) - 32768
    q = np.clip(q, -32768, 32767).astype("<i2")
    while len(blob) % 4:
        blob.append(0)
    off_pos = len(blob)
    blob.extend(q.tobytes())
    while len(blob) % 4:
        blob.append(0)
    off_idx = len(blob)
    i32 = len(uniq) > 65535
    blob.extend(idx.astype("<u4" if i32 else "<u2").tobytes())
    edges = feature_edges(uniq, idx, math.radians(EDGE_ANGLE_DEG))
    while len(blob) % 4:
        blob.append(0)
    off_edges = len(blob)
    blob.extend(edges.astype("<u4" if i32 else "<u2").tobytes())
    return {"nv": int(len(uniq)), "nt": int(len(idx)), "min": [round(float(x), 7) for x in lo],
            "span": [round(float(x), 9) for x in span], "op": off_pos, "oi": off_idx, "i32": i32,
            "oe": off_edges, "ne": int(len(edges))}


EDGE_ANGLE_DEG = 22.0


def feature_edges(verts, idx, thresh):
    """Vertex pairs of the solid's hard edges: boundary edges, non-manifold edges,
    and edges whose two faces meet at more than ``thresh`` radians (the STEP
    face boundaries; the tessellation of smooth surfaces stays below it)."""
    nv = len(verts)
    v0, v1, v2 = verts[idx[:, 0]], verts[idx[:, 1]], verts[idx[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-18)
    e = np.concatenate([idx[:, [0, 1]], idx[:, [1, 2]], idx[:, [2, 0]]])
    face = np.tile(np.arange(len(idx)), 3)
    key = np.sort(e, axis=1)
    k = key[:, 0].astype(np.int64) * nv + key[:, 1]
    order = np.argsort(k, kind="stable")
    ks, fs = k[order], face[order]
    uk, start, counts = np.unique(ks, return_index=True, return_counts=True)
    keep = counts != 2
    pair = counts == 2
    f1, f2 = fs[start[pair]], fs[start[pair] + 1]
    ang = np.arccos(np.clip(np.einsum("ij,ij->i", n[f1], n[f2]), -1.0, 1.0))
    keep[np.nonzero(pair)[0][ang > thresh]] = True
    sel = uk[keep]
    return np.stack([sel // nv, sel % nv], axis=1)


def build(args):
    urdf = Path(args.urdf)
    pkg_dir = urdf.parent.parent
    root = ET.parse(urdf).getroot()
    drop = re.compile(args.drop) if args.drop else None
    colors = {m.get("name"): m.find("color").get("rgba") for m in root.findall("material") if m.find("color") is not None}
    color_list, color_index = [], {}

    def color_id(name):
        rgba = colors.get(name, "0.6 0.64 0.7 1")
        if rgba not in color_index:
            color_index[rgba] = len(color_list)
            r, g, b = [float(x) for x in rgba.split()[:3]]
            color_list.append("#%02x%02x%02x" % (int(r * 255 + 0.5), int(g * 255 + 0.5), int(b * 255 + 0.5)))
        return color_index[rgba]

    blob = bytearray()
    meshes, mesh_index = [], {}
    links = {}
    dropped = 0
    for link in root.findall("link"):
        vis, colls = [], []
        for v in link.findall("visual"):
            m = v.find("geometry/mesh")
            if m is None:
                continue
            fname = m.get("filename").split("/")[-1]
            if drop and drop.search(fname):
                dropped += 1
                continue
            if fname not in mesh_index:
                rec = pack_mesh(load_stl(pkg_dir / "meshes" / fname), blob)
                rec["n"] = fname[:-4]
                mesh_index[fname] = len(meshes)
                meshes.append(rec)
            p, q = origin(v)
            mat = v.find("material")
            vis.append([mesh_index[fname], p, q, color_id(mat.get("name")) if mat is not None else 0])
        for c in link.findall("collision"):
            g = c.find("geometry")
            p, q = origin(c)
            if g.find("box") is not None:
                colls.append(["box", p, q, [float(x) for x in g.find("box").get("size").split()]])
            elif g.find("cylinder") is not None:
                e = g.find("cylinder")
                colls.append(["cyl", p, q, [float(e.get("radius")), float(e.get("length"))]])
            elif g.find("sphere") is not None:
                colls.append(["sph", p, q, [float(g.find("sphere").get("radius"))]])
        mass = link.find("inertial/mass")
        links[link.get("name")] = {"v": vis, "c": colls, "m": round(float(mass.get("value")), 5) if mass is not None else 0.0}
    joints = []
    for j in root.findall("joint"):
        p, q = origin(j)
        lim = j.find("limit")
        mim = j.find("mimic")
        joints.append({
            "n": j.get("name"), "t": j.get("type"), "p": p, "q": q,
            "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
            "axis": [float(x) for x in (j.find("axis").get("xyz").split() if j.find("axis") is not None else "1 0 0".split())],
            "lo": float(lim.get("lower")) if lim is not None else 0.0, "hi": float(lim.get("upper")) if lim is not None else 0.0,
            "mimic": ({"j": mim.get("joint"), "k": float(mim.get("multiplier", 1)), "o": float(mim.get("offset", 0))} if mim is not None else None),
        })
    extra = {}
    if args.report and Path(args.report).exists():
        rep = json.loads(Path(args.report).read_text())
        extra["cad_pose"] = {f"{leg}_{j}": round(v, 5) for leg, r in rep["legs"].items() for j, v in r["cad_pose_joint_angles_rad"].items()}
        extra["mounts"] = {leg: r["yaw_axis_point_m"] for leg, r in rep["legs"].items()}
        extra["source"] = rep["source"]["robot_name"]
    stance_file = pkg_dir / "stance.json"
    if stance_file.exists():
        st = json.loads(stance_file.read_text())
        extra["stance"] = {}
        for J in joints:
            for key in ("coxa_yaw", "femur_pitch", "tibia_pitch"):
                if J["n"].endswith(key):
                    extra["stance"][J["n"]] = st[f"{key}_rad"]
        extra["stance_root_height_m"] = st.get("root_height_m")
    model = {"name": root.get("name"), "links": links, "joints": joints, "colors": color_list, "meshes": meshes,
             "dropped_visuals": dropped, **extra}
    b64 = base64.b64encode(bytes(blob)).decode("ascii")
    html = Path(args.template).read_text()
    html = html.replace("__MODEL__", json.dumps(model, separators=(",", ":"))).replace("__BLOB__", b64)
    Path(args.out).write_text(html)
    print(f"{len(meshes)} meshes, {sum(m['nt'] for m in meshes)} triangles, {sum(len(l['v']) for l in links.values())} visuals "
          f"({dropped} dropped), blob {len(blob)/1e6:.2f} MB, html {len(html)/1e6:.2f} MB -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default="robot/hexapod_mkii_assy/urdf/hexapod_mkii_linkage.urdf")
    ap.add_argument("--template", default="robot/hexapod_mkii_assy/preview/standalone_template.html")
    ap.add_argument("--report", default="robot/hexapod_mkii_assy/assembly_report.json")
    ap.add_argument("--drop", default=r"^motor_(1_2_05_|1_21_|6706|0001755652)",
                    help="regex of mesh files to leave out (motor internals hidden inside the housings)")
    ap.add_argument("--out", default="tmp/hexapod_mkii_viewer.html")
    sys.exit(build(ap.parse_args()))
