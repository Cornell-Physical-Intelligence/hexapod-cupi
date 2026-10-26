#!/usr/bin/env python3
"""Write a run file for the viewer's Runs page from a native Spark or surrogate evaluation folder.

Both evaluators write ``evaluation/batch_*/report.json`` and ``control_trace.npz``. Each batch with a trace
becomes one case: its leg points in the start frame's navigation axes (forward, left, up; millimetres,
every second control), foot contacts, heading, command, the destination ("ghost") and the evaluator's own
pass/fail and failed checks. The ghost starts at the reset pose (the origin, heading zero) and moves exactly
along the command once per control; a new command restarts it at the body, as the destination experiments
train it. Leg points come from the recorded root pose and joint positions through the robot model's joint
frames; the toes are the recorded toe positions.

Usage: uv run python viewer/run_view.py <attempt or evaluation folder> --output run.json
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hexapod_run_view_v1"
LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")
PARTS = ("coxa_yaw", "femur_pitch", "tibia_pitch")
NAMES = [f"{leg}_{part}" for leg in LEGS for part in PARTS]
CONTROL_SECONDS = 0.02
STEP = 2


def quaternion_multiply(a, b):
    """Hamilton product of xyzw quaternions, broadcast over leading axes."""
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        -1,
    )


def rotate(q, v):
    """Rotate vectors ``v`` by xyzw quaternions ``q``."""
    u, w = q[..., :3], q[..., 3:]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def axis_angle(axis, angle):
    half = np.asarray(angle)[..., None] / 2
    return np.concatenate((np.asarray(axis) * np.sin(half), np.cos(half)), -1)


def yaw(pose):
    x, y, z, w = pose[..., 3], pose[..., 4], pose[..., 5], pose[..., 6]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def leg_points(pose, q, toe, joints):
    """Coxa, femur and tibia joint positions and the toe of each leg: (controls, 6, 4, 3) in world metres."""
    points = np.zeros((len(pose), len(LEGS), 4, 3))
    for i, leg in enumerate(LEGS):
        rotation, position = pose[:, 3:7].copy(), pose[:, :3].copy()
        for k, part in enumerate(PARTS):
            joint = joints[f"{leg}_{part}"]
            position = position + rotate(rotation, np.asarray(joint["xyz"], float))
            points[:, i, k] = position
            rotation = quaternion_multiply(rotation, np.asarray(joint["quaternion_xyzw"], float))
            rotation = quaternion_multiply(
                rotation, axis_angle(joint["axis"], q[:, NAMES.index(f"{leg}_{part}")])
            )
        points[:, i, 3] = toe[:, i]
    return points


def ghost(pose, command):
    """The destination after each control, (controls, 3) as world x, y and heading."""
    out, g = np.zeros((len(pose), 3)), np.zeros(3)
    for j in range(len(pose)):
        if j and (command[j] != command[j - 1]).any():
            g = np.array([pose[j - 1, 0], pose[j - 1, 1], yaw(pose[j - 1])])
        forward, left, turn = command[j]
        # Body forward is -Y and left is +X, so a heading-zero body walks forward along world -Y.
        bx, by = left, -forward
        middle = g[2] + turn * CONTROL_SECONDS / 2
        g = g + np.array(
            [
                (np.cos(middle) * bx - np.sin(middle) * by) * CONTROL_SECONDS,
                (np.sin(middle) * bx + np.cos(middle) * by) * CONTROL_SECONDS,
                turn * CONTROL_SECONDS,
            ]
        )
        out[j] = g
    return out


def navigation(xy, origin_yaw=0.0):
    """World planar points in the start frame's navigation axes (forward, left)."""
    c, s = np.cos(-origin_yaw), np.sin(-origin_yaw)
    x, y = c * xy[..., 0] - s * xy[..., 1], s * xy[..., 0] + c * xy[..., 1]
    return np.stack((-y, x), -1)


def source_of(evaluation):
    """Surrogate output folders hold surrogate_summary.json beside evaluation/; native attempts do not."""
    return "surrogate" if (evaluation.parent / "surrogate_summary.json").exists() else "native"


def evaluation_folder(path):
    path = Path(path)
    for candidate in (path, path / "evaluation", path / "run/standing/evaluation"):
        if any(candidate.glob("batch_*/report.json")):
            return candidate
    raise ValueError(f"No evaluation batches under {path}")


def attempt_name(evaluation):
    """A native attempt keeps its batches under <attempt>/run/standing/evaluation; a surrogate run under <output>/evaluation."""
    parents = evaluation.parents
    if (
        evaluation.name == "evaluation"
        and parents[0].name == "standing"
        and parents[1].name == "run"
    ):
        return parents[2].name
    return parents[0].name if evaluation.name == "evaluation" else evaluation.name


def case(batch, joints):
    report = json.loads((batch / "report.json").read_text())
    result, trace = report["results"][0], np.load(batch / "control_trace.npz")
    pose = trace["root_pose_xyzw"][:, 0].astype(float)
    q = trace["joint_position_rad"][:, 0].astype(float)
    toe = trace["toe_xyz_world_m"][:, 0].astype(float)
    contact = trace["distal_contact"][:, 0].astype(bool)
    command = trace["command"][:, 0].astype(float)
    points = leg_points(pose, q, toe, joints)
    nav = np.concatenate((navigation(points[..., :2]), points[..., 2:]), -1)
    destination = ghost(pose, command)
    index = np.arange(0, len(pose), STEP)
    heading = np.unwrap(yaw(pose))
    checks = result.get("checks") or {}
    failed = [
        {
            "check": name,
            "value": checks.get(name, {}).get("value"),
            "bound": checks.get(name, {}).get("bound"),
        }
        for name in result.get("failed_bounds") or []
    ]
    metrics = result.get("metrics") or {}
    return {
        "case_id": report["assigned_case_ids"][0],
        "batch": batch.name,
        "pass": bool(result["pass"]),
        "failed": failed,
        "controls": len(pose),
        "fell": bool(trace["terminated"][:, 0].any()) if "terminated" in trace.files else False,
        "mean_velocity_mps": metrics.get("mean_velocity_mps"),
        "mean_yaw_rate_rad_s": metrics.get("mean_yaw_rate_rad_s"),
        "frames": len(index),
        "command": np.round(command[index], 4).tolist(),
        "heading_deg": np.round(np.degrees(heading[index]), 2).tolist(),
        "commanded_heading_deg": np.round(np.degrees(destination[index, 2]), 2).tolist(),
        "ghost_mm": np.round(navigation(destination[index, :2]) * 1000, 1).tolist(),
        "ghost_distance_mm": np.round(
            np.hypot(*(pose[index, :2] - destination[index, :2]).T) * 1000, 2
        ).tolist(),
        # Frames keep every second control; the end gap uses the last control itself.
        "ghost_distance_end_mm": round(
            float(np.hypot(*(pose[-1, :2] - destination[-1, :2])) * 1000), 3
        ),
        "_points": np.round(nav[index] * 1000).astype("<i2"),
        "_contacts": (contact[index] * (1 << np.arange(6))).sum(-1).astype("u1"),
    }


def compress(data):
    """Gzip with a fixed header: no file time and GzipFile's own operating-system byte, the same on any host."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as stream:
        stream.write(data)
    return buffer.getvalue()


def build(folder, model=None):
    """The run file as a dict; cases without a control trace are listed but not drawn."""
    evaluation = evaluation_folder(folder)
    active = json.loads((ROOT / "robot/active_model.json").read_text())
    model = Path(model) if model else ROOT / active["model"]["path"]
    joints = {j["name"]: j for j in json.loads(model.read_text())["joints"]}
    cases, skipped, points, contacts = [], [], [], []
    for batch in sorted(evaluation.glob("batch_*")):
        if not (batch / "control_trace.npz").exists():
            if (batch / "report.json").exists():
                skipped.append(
                    json.loads((batch / "report.json").read_text())["assigned_case_ids"][0]
                )
            continue
        row = case(batch, joints)
        row["points_offset"] = sum(p.nbytes for p in points)
        row["contacts_offset"] = sum(c.nbytes for c in contacts)
        points.append(row.pop("_points"))
        contacts.append(row.pop("_contacts"))
        cases.append(row)
    if not cases:
        raise ValueError(f"No batch under {evaluation} has a control trace")
    blob = b"".join(p.tobytes() for p in points)
    summary = evaluation / "summary.json"
    return {
        "schema": SCHEMA,
        "source": source_of(evaluation),
        "name": attempt_name(evaluation),
        "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest()
        if summary.exists()
        else None,
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "frame_seconds": STEP * CONTROL_SECONDS,
        "legs": list(LEGS),
        "contacts_base": len(blob),
        "data": base64.b64encode(compress(blob + b"".join(c.tobytes() for c in contacts))).decode(),
        "cases": cases,
        "cases_without_trace": skipped,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="An evaluation attempt or its evaluation folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Robot model JSON; the active model by default")
    args = parser.parse_args(argv)
    value = build(args.folder, args.model)
    args.output.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    print(
        f"{args.output}: {len(value['cases'])} cases from a {value['source']} run, "
        f"{sum(c['pass'] for c in value['cases'])} passing"
    )


if __name__ == "__main__":
    main()
