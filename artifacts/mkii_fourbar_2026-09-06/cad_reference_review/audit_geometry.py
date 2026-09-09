"""Read-only CAD-description measurements; never modifies a robot or simulation.

Run with the repository's .venv/bin/python. The foot coverage statistic counts
unique tessellation vertices, not surface area, volume, or contact probability.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "robot/tools"))
sys.path.insert(0, str(ROOT / "tools"))
from import_onshape_leg import stl_mesh
from audit_mkii_stance import _origin


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def description_metrics(path):
    root = ET.parse(path).getroot()
    tensors = []
    for link in root.findall("link"):
        element = link.find("inertial/inertia")
        if element is None:
            continue
        a = {k: float(v) for k, v in element.attrib.items()}
        tensor = np.array([[a['ixx'], a['ixy'], a['ixz']],
                           [a['ixy'], a['iyy'], a['iyz']],
                           [a['ixz'], a['iyz'], a['izz']]])
        moments = np.linalg.eigvalsh(tensor)
        tensors.append({"link": link.get("name"),
                        "mass_kg": float(link.find("inertial/mass").get("value")),
                        "principal_moments_kg_m2": moments.tolist(),
                        "positive_definite": bool(moments[0] > 0),
                        "triangle_inequality": bool(moments[0] + moments[1] >= moments[2] - 1e-10)})
    return {
        "sha256": sha256(path),
        "links": len(root.findall("link")),
        "joints": len(root.findall("joint")),
        "joint_types": dict(Counter(j.get("type") for j in root.findall("joint"))),
        "visual_instances": len(root.findall("./link/visual")),
        "unique_visual_meshes": len({m.get("filename") for m in root.findall("./link/visual/geometry/mesh")}),
        "collision_instances": len(root.findall("./link/collision")),
        "collision_types": dict(Counter(list(g)[0].tag for g in root.findall("./link/collision/geometry"))),
        "total_mass_kg": sum(t["mass_kg"] for t in tensors),
        "inertias": tensors,
        "joint_limits": {j.get("name"): j.find("limit").attrib for j in root.findall("joint") if j.find("limit") is not None},
    }


def foot_metrics(root):
    mesh = ROOT / "robot/hexapod_mkii_assy/meshes/silicone_foot.stl"
    vertices = stl_mesh(mesh).reshape(-1, 3)
    rows = []
    for leg in ("lf", "lm", "lr", "rf", "rm", "rr"):
        link = root.find(f"link[@name='{leg}_tibia']")
        visual = next(v for v in link.findall("visual")
                      if v.find("geometry/mesh").get("filename").endswith("silicone_foot.stl"))
        scale = np.array([float(v) for v in visual.find("geometry/mesh").get("scale", "1 1 1").split()])
        transform = _origin(visual)
        points = np.unique((vertices * scale) @ transform[:3, :3].T + transform[:3, 3], axis=0)
        colliders = [c for c in link.findall("collision") if c.find("geometry/sphere") is not None]
        if len(colliders) != 2:
            raise ValueError(f"Expected source two-sphere foot for {leg}")
        centers = np.array([_origin(c)[:3, 3] for c in colliders])
        radii = np.array([float(c.find("geometry/sphere").get("radius")) for c in colliders])
        if not np.allclose(radii, radii[0], atol=0, rtol=0):
            raise ValueError("Midplane formula expects equal sphere radii")
        separation = float(np.linalg.norm(centers[1] - centers[0]))
        outside_distance = np.min(np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2) - radii, axis=1)
        rows.append({
            "link": leg + "_tibia",
            "mesh_sha256": sha256(mesh),
            "unique_mesh_vertices": len(points),
            "sphere_radius_m": float(radii[0]),
            "sphere_center_separation_m": separation,
            "sphere_union_midplane_radius_m": math.sqrt(max(0, float(radii[0] ** 2 - (separation / 2) ** 2))),
            "vertices_outside_sphere_union_by_over_1um": int(np.sum(outside_distance > 1e-6)),
            "max_vertex_outside_distance_m": float(np.max(outside_distance)),
        })
    return rows


if __name__ == "__main__":
    path = ROOT / "robot/hexapod_mkii_assy/urdf/hexapod_mkii_linkage.urdf"
    result = {"schema": "hexapod.cad_reference_review.geometry.v1",
              "scope": "Source URDF and its exact local STL bytes; no physics run, measured hardware fit, or continuous-surface guarantee",
              "source": str(path.relative_to(ROOT)),
              "mkii": description_metrics(path),
              "foot_envelope": foot_metrics(ET.parse(path).getroot())}
    if len(sys.argv) == 2:
        reference = Path(sys.argv[1])
        result["so101_reference"] = description_metrics(reference)
    print(json.dumps(result, indent=2, allow_nan=False))
