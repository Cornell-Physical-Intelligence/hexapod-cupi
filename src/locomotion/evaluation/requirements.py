"""Judge recorded evaluation traces against ``configs/walking_requirements.json``.

The requirements file owns every number. This module turns one requirement set into the cases to
run, measures each recorded trace against the set's limits and writes one judgement per
evaluation. Nothing else in the package decides pass or fail.

Cases: the stand, each translation speed on each bearing, both turns and both arcs. Every case
holds its command for ``hold_max_s`` and then commands zero for twice ``stop.settle_s``, so every
moving case also measures its stop. A set may also name routes under ``stop.routes_m``: goals in
metres relative to the start pose. A route case runs the waypoint follower in closed loop from the
simulator's true pose, so the command at each control depends on where the body is.

Checks per case: drift per hold (the gap between the body and the path its command traces),
stopping distance (travel after the command becomes zero, read after the settle), tilt (the
largest body tilt) and applied motor torque (the largest absolute value). The drift limit is set
at ``drift_speed_mps`` and applied to every command of the set at or below that speed. A route
case adds the stop position: the body must finish within ``stop.position_tolerance_m`` of the last
goal and stay there, under a zero command, for ``stop.stay_stopped_s``. An incomplete trace fails.

Run: ``python -m locomotion.evaluation.requirements <evaluation dir> [--set stage2_flat]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from contracts.pose import Pose2D

ROOT = Path(__file__).resolve().parents[3]
REQUIREMENTS = ROOT / "configs/walking_requirements.json"
CONTROL_DT_S = 0.02
SCHEMA = "hexapod_walking_requirements_judgement_v2"


def load_requirements(path=REQUIREMENTS, name="stage2_flat"):
    """The named requirement set and the SHA-256 of the file that owns it."""
    data = json.loads(Path(path).read_text())
    return data["sets"][name], hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases(rules):
    """The cases of one requirement set: id, navigation command, controls and the stop control."""
    hold = round(rules["hold_max_s"] / CONTROL_DT_S)
    stop = round(2 * rules["stop"]["settle_s"] / CONTROL_DT_S)
    commands = rules["commands"]
    rows = [("stand", [0.0, 0.0, 0.0])] if commands["stand"] else []
    step = 2 * math.pi / commands["translate_bearings"]
    for speed in commands["translate_speeds_mps"]:
        for bearing in range(commands["translate_bearings"]):
            angle = bearing * step
            rows.append(
                (
                    f"walk_{speed:.3f}_{round(math.degrees(angle))}deg",
                    [speed * math.cos(angle), speed * math.sin(angle), 0.0],
                )
            )
    rate = commands["turn_rate_rad_s"]
    rows += [(f"turn_{sign * rate:+.2f}", [0.0, 0.0, sign * rate]) for sign in (-1, 1)]
    arc = commands["arc"]
    rows += [
        (
            f"arc_{sign * arc['turn_rate_rad_s']:+.2f}",
            [arc["speed_mps"], 0.0, sign * arc["turn_rate_rad_s"]],
        )
        for sign in (-1, 1)
    ]
    fixed = [
        {"case_id": case_id, "command": command, "controls": hold + stop, "stop_control": hold}
        for case_id, command in rows
    ]
    slowest = min(commands["translate_speeds_mps"])
    routes = []
    for name, goals in rules["stop"].get("routes_m", {}).items():
        points = [(0.0, 0.0)] + [tuple(goal) for goal in goals]
        length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        seconds = length / slowest + rules["stop"]["stay_stopped_s"] + 2 * rules["stop"]["settle_s"]
        routes.append(
            {
                "case_id": "route_" + name,
                "route_m": goals,
                "controls": round(seconds / CONTROL_DT_S),
            }
        )
    return fixed + routes


def follower_pose(root_pose_xyzw):
    """World x, y and forward heading of one root pose row; body forward is -y, so yaw - pi/2."""
    x, y, yaw = planar_pose(np.asarray(root_pose_xyzw)[None, :])[0]
    return float(x), float(y), float(yaw - math.pi / 2)


def route_goals(case, start):
    """The route's goals in the world, given relative to the start pose in its forward frame."""
    x0, y0, heading = start
    c, s = math.cos(heading), math.sin(heading)
    return [(x0 + c * gx - s * gy, y0 + s * gx + c * gy) for gx, gy in case["route_m"]]


def producer(case):
    """The command of ``case`` at each control: ``produce(control, root_pose_row)``.

    A fixed case ignores the pose. A route case builds the waypoint follower on its first call,
    at the body's start pose, and then asks it for the bank command from the current pose.
    """
    if "route_m" not in case:
        return lambda control, root: command_at(case, control)
    from navigation.waypoint import WaypointFollower

    follower = []

    def produce(control, root):
        x, y, heading = follower_pose(root)
        if not follower:
            follower.append(WaypointFollower(route_goals(case, (x, y, heading))))
        command = follower[0].update(
            Pose2D(x_m=x, y_m=y, heading_rad=heading), control * CONTROL_DT_S
        )
        return [command.vx_mps, command.vy_mps, command.wz_rad_s]

    return produce


def command_at(case, control):
    """The navigation command of ``case`` at ``control``: its command, then zero from the stop control."""
    if not 0 <= control < case["controls"]:
        raise ValueError("Control lies outside the declared case")
    return list(case["command"]) if control < case["stop_control"] else [0.0, 0.0, 0.0]


def planar_pose(root_pose_xyzw):
    """World x, y and yaw of root pose rows ``(x, y, z, qx, qy, qz, qw)``."""
    p = np.asarray(root_pose_xyzw, dtype=np.float64)
    x, y, z, w = p[:, 3], p[:, 4], p[:, 5], p[:, 6]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack((p[:, 0], p[:, 1], yaw), -1)


def tilt_degrees(root_pose_xyzw):
    """The angle between the body's up axis and world up, per row, in degrees."""
    p = np.asarray(root_pose_xyzw, dtype=np.float64)
    up = 1 - 2 * (p[:, 3] ** 2 + p[:, 4] ** 2)
    return np.degrees(np.arccos(np.clip(up, -1.0, 1.0)))


def ghost_path(start, commands, dt=CONTROL_DT_S):
    """Where the commands lead from ``start`` ``(x, y, yaw)``, one row per control, the start first.

    Commands are navigation-frame ``(forward, left, yaw rate)``; navigation (x, y) is body (-y, x),
    the arithmetic of ``omniwalk.mdp.destination.advance_destination``.
    """
    commands = np.asarray(commands, dtype=np.float64)
    out = np.empty((len(commands) + 1, 3))
    out[0] = start
    for i, (vx, vy, wz) in enumerate(commands):
        x, y, yaw = out[i]
        bx, by = vy, -vx
        middle = yaw + wz * dt / 2
        c, s = math.cos(middle), math.sin(middle)
        out[i + 1] = (x + (c * bx - s * by) * dt, y + (s * bx + c * by) * dt, yaw + wz * dt)
    return out


def hold_spans(commands):
    """``(first, last_exclusive, command)`` for each run of equal nonzero commands."""
    commands = np.asarray(commands, dtype=np.float64)
    spans, start = [], 0
    for i in range(1, len(commands) + 1):
        if i == len(commands) or np.any(commands[i] != commands[start]):
            if np.any(commands[start] != 0):
                spans.append((start, i, commands[start].copy()))
            start = i
    return spans


def drift_per_hold(trace):
    """Per hold: the largest and the final planar gap between the body and its ghost, in metres."""
    pose = planar_pose(trace["root_pose_xyzw"][:, 0, :])
    commands = np.asarray(trace["command"][:, 0, :], dtype=np.float64)
    rows = []
    for first, last, command in hold_spans(commands):
        ghost = ghost_path(pose[first], commands[first:last])
        gap = np.hypot(*(pose[first:last, :2] - ghost[: last - first, :2]).T)
        rows.append(
            {
                "first_control": int(first),
                "controls": int(last - first),
                "command": [float(v) for v in command],
                "max_gap_m": float(gap.max()),
                "end_gap_m": float(gap[-1]),
            }
        )
    return rows


def stopping_distance(trace, settle_s):
    """Planar travel after the first zero command that follows a nonzero one; None if there is no stop."""
    pose = planar_pose(trace["root_pose_xyzw"][:, 0, :])
    commands = np.asarray(trace["command"][:, 0, :], dtype=np.float64)
    moving = np.any(commands != 0, axis=1)
    stops = np.nonzero(moving[:-1] & ~moving[1:])[0] + 1
    if len(stops) == 0:
        return None
    k = int(stops[0])
    settle = min(len(pose) - 1, k + int(round(settle_s / CONTROL_DT_S)))
    travel = np.hypot(*(pose[k:, :2] - pose[k, :2]).T)
    return {
        "stop_control": k,
        "settle_distance_m": float(travel[settle - k]),
        "final_distance_m": float(travel[-1]),
        "max_distance_m": float(travel.max()),
    }


def _check(value, bound):
    return {"value": float(value), "bound": float(bound), "pass": bool(value <= bound)}


def stop_position(trace, case, stay_stopped_s):
    """Distance to the route's last goal over the final stay window, and whether the body stayed stopped."""
    poses = planar_pose(trace["root_pose_xyzw"][:, 0, :])
    goal = route_goals(case, follower_pose(trace["root_pose_xyzw"][0, 0, :]))[-1]
    distance = np.hypot(poses[:, 0] - goal[0], poses[:, 1] - goal[1])
    moving = np.any(np.asarray(trace["command"][:, 0, :]) != 0, axis=1)
    stopped = len(moving) - (int(np.nonzero(moving)[0][-1]) + 1 if moving.any() else 0)
    window = int(round(stay_stopped_s / CONTROL_DT_S))
    arrived = stopped >= window
    return {
        "goal_world_m": [float(goal[0]), float(goal[1])],
        "stopped_controls": int(stopped),
        "stay_window_controls": window,
        "arrived": bool(arrived),
        "final_distance_m": float(distance[-1]),
        "stay_distance_max_m": float(distance[-window:].max()) if arrived else None,
    }


def judge_case(trace, rules, controls=None, case=None):
    """One recorded trace against the controller, tilt, torque and, for a route, stop rows of a set."""
    n = len(trace["command"]) if "command" in trace else 0
    if n == 0 or (controls is not None and n != controls):
        check = {"value": float(n), "bound": float(controls or 0), "pass": False}
        return {
            "holds": [],
            "stop": None,
            "checks": {"complete_trace": check},
            "failed_checks": ["complete_trace"],
            "pass": False,
        }
    limits = rules["controller"]
    holds = drift_per_hold(trace)
    stop = stopping_distance(trace, rules["stop"]["settle_s"])
    checks = {}
    walks = [
        h
        for h in holds
        if math.hypot(*h["command"][:2]) <= limits["drift_speed_mps"] + 1e-9
        and h["controls"] * CONTROL_DT_S >= limits["drift_hold_s"] - 1e-9
    ]
    if walks:
        checks["drift_per_hold_m"] = _check(
            max(h["max_gap_m"] for h in walks), limits["drift_per_hold_max_m"]
        )
    if stop is not None:
        checks["stopping_distance_m"] = _check(
            stop["final_distance_m"], limits["stopping_distance_max_m"]
        )
    checks["tilt_deg"] = _check(
        tilt_degrees(trace["root_pose_xyzw"][:, 0, :]).max(), rules["tilt_max_deg"]
    )
    checks["applied_torque_nm"] = _check(
        np.abs(trace["applied_torque_abs_max_400hz"]).max(), rules["motor_torque_max_nm"]
    )
    route = None
    if case is not None and "route_m" in case:
        route = stop_position(trace, case, rules["stop"]["stay_stopped_s"])
        value = route["stay_distance_max_m"] if route["arrived"] else route["final_distance_m"]
        checks["stop_position_m"] = _check(value, rules["stop"]["position_tolerance_m"])
        checks["stop_position_m"]["pass"] &= route["arrived"]
    failed = [name for name, check in checks.items() if not check["pass"]]
    return {
        "holds": holds,
        "stop": stop,
        "route": route,
        "checks": checks,
        "failed_checks": failed,
        "pass": not failed,
    }


def judge(evaluation_dir, requirements=None, name="stage2_flat"):
    """Every recorded batch under ``evaluation_dir`` against one requirement set."""
    rules, requirements_sha = load_requirements(requirements or REQUIREMENTS, name)
    rows, checkpoints = [], set()
    for report_path in sorted(Path(evaluation_dir).glob("batch_*/report.json")):
        report = json.loads(report_path.read_text())
        checkpoints.add(report.get("checkpoint_sha256"))
        case = report["cases"][0]
        trace_path = report_path.with_name("control_trace.npz")
        trace = dict(np.load(trace_path)) if trace_path.exists() else {}
        rows.append(
            {
                "case_id": case["case_id"],
                "command": case.get("command"),
                "route_m": case.get("route_m"),
                **judge_case(trace, rules, case.get("controls"), case),
            }
        )
    return {
        "schema": SCHEMA,
        "requirements_set": name,
        "requirements_sha256": requirements_sha,
        "checkpoint_sha256": sorted(c for c in checkpoints if c),
        "cases": rows,
        "judged_cases": len(rows),
        "failed_cases": [row["case_id"] for row in rows if not row["pass"]],
        "pass": all(row["pass"] for row in rows) if rows else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("evaluation_dir")
    parser.add_argument("--set", default="stage2_flat")
    parser.add_argument("--requirements", default=str(REQUIREMENTS))
    args = parser.parse_args(argv)
    print(json.dumps(judge(args.evaluation_dir, args.requirements, args.set), indent=2))


if __name__ == "__main__":
    main()
