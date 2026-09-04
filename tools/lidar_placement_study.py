#!/usr/bin/env python3
"""Score candidate Livox Mid-360 mounts on the Hexapod MKII by self-occlusion.

The Mid-360 is a 360 x 59 degree sensor whose vertical field of view runs from
-7 to +52 degrees, so almost all of it points above the horizon. On a robot
whose deck sits about 0.19 m off the ground that geometry, not the point rate,
decides what the sensor can see: the -7 degree lower edge alone puts the
nearest ground return metres away, and the six legs eat into exactly the band
that matters.

This tool answers "where should it go" without a simulator. It runs forward
kinematics on the URDF at a fixed stance, transforms the visual meshes into
world space, and rasterizes them into a spherical occupancy grid seen from each
candidate optical centre. From that grid it reports how much of the field of
view the robot blocks and how far away the nearest visible ground is, per
azimuth.

What is modelled: the deck, the coxa, femur and tibia meshes at one stance, and
(optionally) a cylindrical mast under the sensor. What is not modelled: the
sensor's own housing, brackets, cabling, the swept envelope over a gait cycle,
and any terrain other than a flat plane at z = 0. Treat the numbers as the
geometric ceiling for each mount, not as a prediction of a live point cloud.

Frames follow ``packages/hexapod_core/hexapod_core/frames.py``: anatomical
forward is body -Y, lateral is body +X, up is body +Z. Reported azimuths are in
that navigation frame, 0 degrees forward, +90 degrees toward body +X.

Examples
--------
Score every mount in the shipped candidate list at the Stage2C stance::

    python3 tools/lidar_placement_study.py

Sweep an upright mast from flush to 300 mm::

    python3 tools/lidar_placement_study.py --sweep mast --sweep-range 0 0.30 0.02

Sweep forward pitch on a 60 mm mast::

    python3 tools/lidar_placement_study.py --sweep pitch --sweep-range 0 60 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_URDF = REPO_ROOT / "robot/hexapod_mkii_mock_assy/urdf/hexapod_mkii_robstride.urdf"
DEFAULT_MESH_DIR = REPO_ROOT / "robot/hexapod_mkii_mock_assy/meshes"
DEFAULT_SENSOR = REPO_ROOT / "robot/sensors/livox_mid360/config/mid360_sensor.json"
DEFAULT_MOUNTS = REPO_ROOT / "robot/sensors/livox_mid360/config/mounts.json"

# Stage2C reset stance, from packages/hexapod_core/hexapod_core/joints.py
# (STAGE2C_DEFAULT_JOINT_POSITIONS_RAD) and its recorded root height.
STANCES: dict[str, dict[str, float]] = {
    "stage2c": {"coxa": 0.0, "femur": 0.60, "tibia": 2.2335, "root_height": 0.185},
    "base": {"coxa": 0.0, "femur": 0.40, "tibia": 2.10, "root_height": 0.205},
}

# Joint name sets, mirroring packages/hexapod_core/hexapod_core/joints.py.
COXA_JOINTS = frozenset(
    ("revolute_1_1", "revolute_1_7", "revolute_2_5", "revolute_3", "revolute_4", "revolute_5")
)
FEMUR_JOINTS = frozenset(
    ("revolute_1", "revolute_1_2", "revolute_1_3", "revolute_1_4", "revolute_1_5", "revolute_1_6")
)
TIBIA_JOINTS = frozenset(
    ("revolute_2", "revolute_2_1", "revolute_2_2", "revolute_2_3", "revolute_2_4", "revolute_2_6")
)


# --------------------------------------------------------------------------
# URDF and mesh loading
# --------------------------------------------------------------------------


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


@dataclass
class Link:
    name: str
    mesh: str | None = None
    mesh_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mesh_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))


def rotation_from_rpy(rpy: Sequence[float]) -> np.ndarray:
    """Extrinsic roll-pitch-yaw about X, Y then Z, matching the URDF convention."""
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about an arbitrary unit axis."""
    a = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(a)
    if norm == 0.0:
        return np.eye(3)
    a = a / norm
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _vec(text: str | None, default: Sequence[float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.fromstring(text, sep=" ", dtype=np.float64)


def parse_urdf(path: Path) -> tuple[dict[str, Link], list[Joint]]:
    root = ET.parse(path).getroot()
    links: dict[str, Link] = {}
    for element in root.findall("link"):
        link = Link(name=element.get("name", ""))
        mesh = element.find("visual/geometry/mesh")
        if mesh is not None:
            link.mesh = os.path.basename(mesh.get("filename", ""))
            origin = element.find("visual/origin")
            if origin is not None:
                link.mesh_xyz = _vec(origin.get("xyz"), (0, 0, 0))
                link.mesh_rpy = _vec(origin.get("rpy"), (0, 0, 0))
        links[link.name] = link

    joints: list[Joint] = []
    for element in root.findall("joint"):
        origin = element.find("origin")
        axis = element.find("axis")
        joints.append(
            Joint(
                name=element.get("name", ""),
                kind=element.get("type", "fixed"),
                parent=element.find("parent").get("link", ""),
                child=element.find("child").get("link", ""),
                xyz=_vec(origin.get("xyz") if origin is not None else None, (0, 0, 0)),
                rpy=_vec(origin.get("rpy") if origin is not None else None, (0, 0, 0)),
                axis=_vec(axis.get("xyz") if axis is not None else None, (0, 0, 1)),
            )
        )
    return links, joints


def joint_angle(name: str, stance: dict[str, float]) -> float:
    if name in COXA_JOINTS:
        return stance["coxa"]
    if name in FEMUR_JOINTS:
        return stance["femur"]
    if name in TIBIA_JOINTS:
        return stance["tibia"]
    return 0.0


def forward_kinematics(
    links: dict[str, Link], joints: list[Joint], stance: dict[str, float]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return link name -> (rotation, translation) in the base link frame."""
    children: dict[str, list[Joint]] = {}
    has_parent = set()
    for joint in joints:
        children.setdefault(joint.parent, []).append(joint)
        has_parent.add(joint.child)
    roots = [name for name in links if name not in has_parent]
    if len(roots) != 1:
        raise ValueError(f"expected exactly one root link, found {roots}")

    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {roots[0]: (np.eye(3), np.zeros(3))}
    stack = [roots[0]]
    while stack:
        parent = stack.pop()
        parent_rot, parent_pos = poses[parent]
        for joint in children.get(parent, ()):
            rot = parent_rot @ rotation_from_rpy(joint.rpy)
            pos = parent_pos + parent_rot @ joint.xyz
            if joint.kind in ("revolute", "continuous"):
                rot = rot @ rotation_about_axis(joint.axis, joint_angle(joint.name, stance))
            poses[joint.child] = (rot, pos)
            stack.append(joint.child)
    return poses


def load_stl(path: Path) -> np.ndarray:
    """Load a binary STL as an (N, 3, 3) array of triangle vertices in metres."""
    data = path.read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    expected = 84 + count * 50
    if len(data) < expected:
        raise ValueError(f"{path.name}: truncated STL ({len(data)} < {expected} bytes)")
    raw = np.frombuffer(data[84:expected], dtype=np.uint8).reshape(count, 50)
    floats = raw[:, :48].copy().view(np.float32).reshape(count, 4, 3)
    return floats[:, 1:, :].astype(np.float64)


def build_world_triangles(
    urdf: Path, mesh_dir: Path, stance: dict[str, float]
) -> tuple[np.ndarray, dict[str, int]]:
    """Triangle soup of the robot at ``stance``, with the deck top at z = root_height."""
    links, joints = parse_urdf(urdf)
    poses = forward_kinematics(links, joints, stance)
    base_lift = np.array([0.0, 0.0, stance["root_height"]])

    cache: dict[str, np.ndarray] = {}
    chunks: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for name, link in links.items():
        if not link.mesh:
            continue
        if link.mesh not in cache:
            cache[link.mesh] = load_stl(mesh_dir / link.mesh)
        tris = cache[link.mesh]
        rot, pos = poses[name]
        mesh_rot = rot @ rotation_from_rpy(link.mesh_rpy)
        mesh_pos = pos + rot @ link.mesh_xyz + base_lift
        chunks.append(tris @ mesh_rot.T + mesh_pos)
        counts[name] = len(tris)
    return np.concatenate(chunks, axis=0), counts


def mast_triangles(
    top_z: float, bottom_z: float, centre_xy: Sequence[float], radius: float, segments: int = 48
) -> np.ndarray:
    """A closed cylinder standing between two heights, as a triangle soup."""
    if radius <= 0.0 or top_z <= bottom_z:
        return np.zeros((0, 3, 3))
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    ring_x = centre_xy[0] + radius * np.cos(angles)
    ring_y = centre_xy[1] + radius * np.sin(angles)
    lower = np.stack([ring_x, ring_y, np.full(segments, bottom_z)], axis=1)
    upper = np.stack([ring_x, ring_y, np.full(segments, top_z)], axis=1)
    nxt = np.roll(np.arange(segments), -1)
    side_a = np.stack([lower, upper, upper[nxt]], axis=1)
    side_b = np.stack([lower, upper[nxt], lower[nxt]], axis=1)
    cap_centre = np.array([centre_xy[0], centre_xy[1], top_z])
    cap = np.stack([np.repeat(cap_centre[None], segments, axis=0), upper, upper[nxt]], axis=1)
    return np.concatenate([side_a, side_b, cap], axis=0)


# --------------------------------------------------------------------------
# Spherical occupancy
# --------------------------------------------------------------------------


# Barycentric lattices used to sample a triangle. Index by lattice order.
def _lattice(order: int) -> np.ndarray:
    rows = []
    denom = max(order - 1, 1)
    for i in range(order):
        for j in range(order - i):
            rows.append((i / denom, j / denom, (denom - i - j) / denom))
    return np.asarray(rows, dtype=np.float64)


_LATTICES = {order: _lattice(order) for order in (2, 4, 8, 16, 32, 64)}
_LATTICE_ORDERS = sorted(_LATTICES)


@dataclass
class Grid:
    az_edges: np.ndarray
    el_edges: np.ndarray
    blocked: np.ndarray

    @property
    def az_centres(self) -> np.ndarray:
        return 0.5 * (self.az_edges[:-1] + self.az_edges[1:])

    @property
    def el_centres(self) -> np.ndarray:
        return 0.5 * (self.el_edges[:-1] + self.el_edges[1:])


def occupancy_grid(
    triangles: np.ndarray,
    origin: np.ndarray,
    rotation: np.ndarray,
    el_range: tuple[float, float],
    az_res_deg: float,
    el_res_deg: float,
    sample_gain: float = 6.0,
    chunk: int = 4096,
    max_subdivisions: int = 12,
) -> Grid:
    """Rasterize ``triangles`` into a sensor-frame azimuth/elevation blocked mask.

    ``rotation`` maps sensor-frame vectors into world; its transpose is applied
    here to bring world geometry into the sensor frame.
    """
    az_bins = int(round(360.0 / az_res_deg))
    el_bins = int(round((el_range[1] - el_range[0]) / el_res_deg))
    az_edges = np.linspace(-180.0, 180.0, az_bins + 1)
    el_edges = np.linspace(el_range[0], el_range[1], el_bins + 1)
    blocked = np.zeros((az_bins, el_bins), dtype=bool)

    local = (triangles - origin) @ rotation
    cell = math.radians(az_res_deg) * math.radians(el_res_deg)
    largest_lattice = len(_LATTICES[_LATTICE_ORDERS[-1]])

    def _solid_angle(tris: np.ndarray) -> np.ndarray:
        centroid = tris.mean(axis=1)
        dist2 = np.maximum((centroid**2).sum(axis=1), 1e-12)
        area = 0.5 * np.linalg.norm(
            np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1
        )
        return area / dist2

    # A triangle wider than the largest lattice can cover would be sampled too
    # sparsely and would silently under-report occlusion, so split it into four
    # until every piece fits. Robot meshes never need this; a hand-built ground
    # plane or a very close surface does.
    solid_angle = _solid_angle(local)
    for _ in range(max_subdivisions):
        oversized = np.ceil(sample_gain * solid_angle / cell) > largest_lattice
        if not oversized.any():
            break
        big = local[oversized]
        m01 = 0.5 * (big[:, 0] + big[:, 1])
        m12 = 0.5 * (big[:, 1] + big[:, 2])
        m20 = 0.5 * (big[:, 2] + big[:, 0])
        split = np.concatenate(
            [
                np.stack([big[:, 0], m01, m20], axis=1),
                np.stack([m01, big[:, 1], m12], axis=1),
                np.stack([m20, m12, big[:, 2]], axis=1),
                np.stack([m01, m12, m20], axis=1),
            ],
            axis=0,
        )
        local = np.concatenate([local[~oversized], split], axis=0)
        solid_angle = _solid_angle(local)

    wanted = np.ceil(sample_gain * solid_angle / cell).astype(np.int64)

    # Reject triangles that cannot reach the field of view, with a margin equal
    # to the triangle's own angular size so a bulging spherical edge cannot be
    # dropped.
    norms = np.linalg.norm(local, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        vertex_el = np.degrees(np.arcsin(np.clip(local[..., 2] / np.maximum(norms, 1e-12), -1, 1)))
    extent = np.degrees(np.sqrt(np.maximum(solid_angle, 0.0)))
    margin = np.maximum(extent, 2.0)
    keep = (vertex_el.max(axis=1) + margin >= el_range[0]) & (
        vertex_el.min(axis=1) - margin <= el_range[1]
    )

    for order in _LATTICE_ORDERS:
        points = len(_LATTICES[order])
        lower = 0 if order == _LATTICE_ORDERS[0] else len(_LATTICES[_LATTICE_ORDERS[
            _LATTICE_ORDERS.index(order) - 1]])
        if order == _LATTICE_ORDERS[-1]:
            selection = keep & (wanted > lower)
        else:
            selection = keep & (wanted > lower) & (wanted <= points)
        if not selection.any():
            continue
        bary = _LATTICES[order]
        subset = local[selection]
        for start in range(0, len(subset), chunk):
            block = subset[start : start + chunk]
            # (T, P, 3) = sum_k bary[P, k] * vertex[T, k, :]
            samples = np.einsum("pk,tkc->tpc", bary, block).reshape(-1, 3)
            _mark(samples, blocked, az_edges, el_edges, el_range)
    return Grid(az_edges=az_edges, el_edges=el_edges, blocked=blocked)


def _mark(
    samples: np.ndarray,
    blocked: np.ndarray,
    az_edges: np.ndarray,
    el_edges: np.ndarray,
    el_range: tuple[float, float],
) -> None:
    norm = np.linalg.norm(samples, axis=1)
    valid = norm > 1e-9
    if not valid.any():
        return
    samples = samples[valid]
    norm = norm[valid]
    el = np.degrees(np.arcsin(np.clip(samples[:, 2] / norm, -1.0, 1.0)))
    inside = (el >= el_range[0]) & (el <= el_range[1])
    if not inside.any():
        return
    samples = samples[inside]
    el = el[inside]
    az = np.degrees(np.arctan2(samples[:, 1], samples[:, 0]))
    ia = np.clip(np.searchsorted(az_edges, az, side="right") - 1, 0, blocked.shape[0] - 1)
    ie = np.clip(np.searchsorted(el_edges, el, side="right") - 1, 0, blocked.shape[1] - 1)
    blocked[ia, ie] = True


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def evaluate_mount(
    triangles: np.ndarray,
    plate_xyz: Sequence[float],
    plate_rpy_deg: Sequence[float],
    optical_offset: float,
    root_height: float,
    el_range: tuple[float, float],
    az_res_deg: float,
    el_res_deg: float,
    mast_radius: float,
    nav_az_bins: int = 360,
) -> dict:
    """Score one mount. Positions are plate poses in the body frame."""
    rotation = rotation_from_rpy(np.radians(np.asarray(plate_rpy_deg, dtype=np.float64)))
    plate = np.asarray(plate_xyz, dtype=np.float64) + np.array([0.0, 0.0, root_height])
    origin = plate + rotation @ np.array([0.0, 0.0, optical_offset])

    scene = triangles
    upright = abs(float(plate_rpy_deg[0])) < 90.0 and abs(float(plate_rpy_deg[1])) < 90.0
    if mast_radius > 0.0 and plate[2] > root_height + 1e-6 and upright:
        scene = np.concatenate(
            [triangles, mast_triangles(plate[2], root_height, plate[:2], mast_radius)], axis=0
        )

    grid = occupancy_grid(
        scene, origin, rotation, el_range, az_res_deg, el_res_deg
    )

    el_c = grid.el_centres
    az_c = grid.az_centres
    weight = np.cos(np.radians(el_c))[None, :] * np.ones((len(az_c), 1))
    total = weight.sum()
    blocked_total = float((weight * grid.blocked).sum() / total)

    # World direction of every cell centre.
    az_r = np.radians(az_c)[:, None]
    el_r = np.radians(el_c)[None, :]
    dirs = np.stack(
        [
            np.broadcast_to(np.cos(el_r) * np.cos(az_r), (len(az_c), len(el_c))),
            np.broadcast_to(np.cos(el_r) * np.sin(az_r), (len(az_c), len(el_c))),
            np.broadcast_to(np.sin(el_r) * np.ones_like(az_r), (len(az_c), len(el_c))),
        ],
        axis=-1,
    )
    world = dirs @ rotation.T
    world_el = np.degrees(np.arcsin(np.clip(world[..., 2], -1.0, 1.0)))
    # Navigation frame: forward = -y, lateral = +x.
    nav_az = np.degrees(np.arctan2(world[..., 0], -world[..., 1]))

    # The band that matters for terrain and obstacles is defined in the world,
    # not in the sensor frame: a tilted or inverted mount rotates the whole FOV.
    below = world_el < 0.0
    below_weight = weight * below
    blocked_lower = float(
        (below_weight * grid.blocked).sum() / max(below_weight.sum(), 1e-12)
    )

    height = float(origin[2])
    downward = (world_el < -1e-6) & (~grid.blocked)
    ranges = np.full(world_el.shape, np.inf)
    np.divide(
        height,
        np.tan(np.radians(-world_el)),
        out=ranges,
        where=downward,
    )
    ranges[~downward] = np.inf

    bin_index = np.clip(((nav_az + 180.0) / 360.0 * nav_az_bins).astype(int), 0, nav_az_bins - 1)
    nearest = np.full(nav_az_bins, np.inf)
    np.minimum.at(nearest, bin_index.ravel(), ranges.ravel())

    visible = np.isfinite(nearest)
    forward_bin = nav_az_bins // 2  # nav azimuth 0 degrees
    ideal = height / math.tan(math.radians(-el_range[0])) if el_range[0] < 0 else math.inf

    def _stat(fn) -> float | None:
        return float(fn(nearest[visible])) if visible.any() else None

    return {
        "optical_centre_body_xyz_m": [round(float(v), 5) for v in (origin - np.array([0, 0, root_height]))],
        "optical_centre_height_m": round(height, 4),
        "blocked_fraction_fov": round(blocked_total, 4),
        "blocked_fraction_below_horizon": round(blocked_lower, 4),
        "ground_visible_azimuth_fraction": round(float(visible.mean()), 4),
        "ground_blind_radius_m": {
            "min": None if not visible.any() else round(_stat(np.min), 3),
            "median": None if not visible.any() else round(_stat(np.median), 3),
            "p90": None
            if not visible.any()
            else round(float(np.percentile(nearest[visible], 90)), 3),
            "max": None if not visible.any() else round(_stat(np.max), 3),
        },
        "forward_ground_blind_radius_m": (
            round(float(nearest[forward_bin]), 3) if np.isfinite(nearest[forward_bin]) else None
        ),
        "unobstructed_blind_radius_m": round(ideal, 3),
        "mast_modelled": bool(scene is not triangles),
        # Nearest visible ground per navigation azimuth bin, index 0 = -180 deg
        # (directly behind), index nav_az_bins/2 = 0 deg (forward). null where
        # no ground is visible in that direction.
        "ground_range_profile_m": [
            None if not math.isfinite(v) else round(float(v), 3) for v in nearest
        ],
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def format_table(rows: Iterable[tuple[str, dict]]) -> str:
    header = (
        f"{'mount':<26} {'h(m)':>6} {'blocked':>8} {'blk<0':>7} "
        f"{'az_ok':>6} {'ground fwd':>11} {'ground med':>11} {'ideal':>7}"
    )
    lines = [header, "-" * len(header)]
    for name, r in rows:
        blind = r["ground_blind_radius_m"]["median"]
        fwd = r["forward_ground_blind_radius_m"]
        lines.append(
            f"{name:<26} {r['optical_centre_height_m']:>6.3f} "
            f"{r['blocked_fraction_fov'] * 100:>7.1f}% {r['blocked_fraction_below_horizon'] * 100:>6.1f}% "
            f"{r['ground_visible_azimuth_fraction'] * 100:>5.0f}% "
            f"{'-' if fwd is None else f'{fwd:>8.2f} m'} "
            f"{'-' if blind is None else f'{blind:>8.2f} m'} "
            f"{r['unobstructed_blind_radius_m']:>6.2f}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--sensor", type=Path, default=DEFAULT_SENSOR)
    parser.add_argument("--mounts", type=Path, default=DEFAULT_MOUNTS)
    parser.add_argument("--stance", choices=sorted(STANCES), default="stage2c")
    parser.add_argument("--root-height", type=float, default=None, help="Override the deck height in metres.")
    parser.add_argument("--az-res", type=float, default=0.5, help="Azimuth cell size in degrees.")
    parser.add_argument("--el-res", type=float, default=0.25, help="Elevation cell size in degrees.")
    parser.add_argument(
        "--mast-radius",
        type=float,
        default=0.025,
        help="Radius of the cylindrical mast modelled under raised upright mounts. 0 disables it.",
    )
    parser.add_argument("--sweep", choices=("mast", "pitch"), default=None)
    parser.add_argument("--sweep-range", type=float, nargs=3, metavar=("START", "STOP", "STEP"))
    parser.add_argument("--sweep-mast-height", type=float, default=0.060, help="Mast height used by the pitch sweep.")
    parser.add_argument("--out", type=Path, default=None, help="Write the full result as JSON here.")
    args = parser.parse_args(argv)

    sensor = json.loads(args.sensor.read_text())
    el_range = tuple(float(v) for v in sensor["fov"]["elevation_deg"])
    optical_offset = float(sensor["geometry"]["optical_centre_above_mount_plate_m"])

    stance = dict(STANCES[args.stance])
    if args.root_height is not None:
        stance["root_height"] = args.root_height

    triangles, counts = build_world_triangles(args.urdf, args.mesh_dir, stance)
    print(
        f"stance={args.stance} deck_height={stance['root_height']:.3f} m  "
        f"links={len(counts)} triangles={len(triangles)}  "
        f"fov_elevation={el_range[0]:+.1f}..{el_range[1]:+.1f} deg  "
        f"optical_centre=+{optical_offset * 1000:.1f} mm above plate",
        file=sys.stderr,
    )

    if args.sweep:
        start, stop, step = args.sweep_range or (0.0, 0.30, 0.02)
        values = np.arange(start, stop + 0.5 * step, step)
        candidates = []
        for value in values:
            if args.sweep == "mast":
                candidates.append(
                    (f"mast_{value * 1000:.0f}mm", [0.0, 0.0, float(value)], [0.0, 0.0, 0.0])
                )
            else:
                candidates.append(
                    (
                        f"pitch_{value:.0f}deg",
                        [0.0, 0.0, args.sweep_mast_height],
                        [float(value), 0.0, 0.0],
                    )
                )
    else:
        mounts = json.loads(args.mounts.read_text())["mounts"]
        candidates = [(m["name"], m["xyz"], m["rpy_deg"]) for m in mounts]

    results: list[tuple[str, dict]] = []
    for name, xyz, rpy in candidates:
        results.append(
            (
                name,
                evaluate_mount(
                    triangles,
                    xyz,
                    rpy,
                    optical_offset,
                    stance["root_height"],
                    el_range,
                    args.az_res,
                    args.el_res,
                    args.mast_radius,
                ),
            )
        )
        print(".", end="", flush=True, file=sys.stderr)
    print(file=sys.stderr)

    print(format_table(results))
    print()
    print(
        "blocked = fraction of the 360x59 deg FOV solid angle blocked by the robot; "
        "blk<0 = same for the band below the horizon;\n"
        "az_ok  = fraction of navigation azimuths with any visible ground; "
        "ground fwd/med = nearest visible ground, forward and median over azimuth;\n"
        "ideal  = nearest ground for an unobstructed sensor at the same height, h/tan(7 deg)."
    )

    if args.out:
        payload = {
            "stance": args.stance,
            "root_height_m": stance["root_height"],
            "joint_positions_rad": {k: v for k, v in stance.items() if k != "root_height"},
            "fov_elevation_deg": list(el_range),
            "optical_centre_above_plate_m": optical_offset,
            "grid_resolution_deg": {"azimuth": args.az_res, "elevation": args.el_res},
            "mast_radius_m": args.mast_radius,
            "triangle_count": int(len(triangles)),
            "results": {name: value for name, value in results},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
