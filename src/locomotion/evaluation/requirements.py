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
largest body tilt while standing and after each stop, past the settle; the body may tilt while it
moves) and applied motor torque (the largest absolute value). The drift limit is set
at ``drift_speed_mps`` and applied to every command of the set at or below that speed. A route
case adds the stop position: the body must finish within ``stop.position_tolerance_m`` of the last
goal and stay there, under a zero command, for ``stop.stay_stopped_s``. Pure turns must achieve
yaw in the commanded direction during their hold. The requirement set gives no yaw-rate tracking
limit. An incomplete or terminal trace fails.

Run: ``python -m locomotion.evaluation.requirements <evaluation dir> [--set stage2_flat]``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from contracts.command import CommandSet, default_requirements_path, load_requirement_set
from contracts.pose import Pose2D
from locomotion.evaluation.requirement_cases import case_definitions

REQUIREMENTS = default_requirements_path()
CONTROL_DT_S = 0.02
SCHEMA = "hexapod_walking_requirements_judgement_v3"


def load_requirements(path=REQUIREMENTS, name="stage2_flat"):
    """The named requirement set and the SHA-256 of the file that owns it."""
    return load_requirement_set(path, name)


def cases(rules):
    """Add the acquisition timing to each case in the declared requirement set."""
    hold = round(rules["hold_max_s"] / CONTROL_DT_S)
    stop = round(2 * rules["stop"]["settle_s"] / CONTROL_DT_S)
    declared = case_definitions(rules)
    slowest = min(map(float, rules["commands"]["translate_speeds_mps"]))
    for case in declared:
        if "command" in case:
            case.update(controls=hold + stop, stop_control=hold)
        else:
            points = [(0.0, 0.0), *case["route_m"]]
            length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
            seconds = (
                length / slowest + rules["stop"]["stay_stopped_s"] + 2 * rules["stop"]["settle_s"]
            )
            case["controls"] = round(seconds / CONTROL_DT_S)
    return declared


def follower_pose(root_pose_xyzw):
    """World x, y and forward heading of one root pose row; body forward is -y, so yaw - pi/2."""
    x, y, yaw = planar_pose(np.asarray(root_pose_xyzw)[None, :])[0]
    return float(x), float(y), float(yaw - math.pi / 2)


def route_goals(case, start):
    """The route's goals in the world, given relative to the start pose in its forward frame."""
    x0, y0, heading = start
    c, s = math.cos(heading), math.sin(heading)
    return [(x0 + c * gx - s * gy, y0 + s * gx + c * gy) for gx, gy in case["route_m"]]


def producer(case, rules=None):
    """The command of ``case`` at each control: ``produce(control, root_pose_row)``.

    A fixed case ignores the pose. A route case builds the waypoint follower on its first call,
    at the body's start pose, and then asks it for the bank command from the current pose.
    """
    if "route_m" not in case:
        return lambda control, root: command_at(case, control)
    from navigation.waypoint import WaypointFollower

    rules = load_requirements()[0] if rules is None else rules
    commands = CommandSet.from_mapping(rules["commands"])
    follower = []

    def produce(control, root):
        x, y, heading = follower_pose(root)
        if not follower:
            follower.append(
                WaypointFollower(
                    route_goals(case, (x, y, heading)),
                    command_set=commands,
                    hold_min_s=rules["hold_min_s"],
                )
            )
        command = follower[0].update(
            Pose2D(x_m=x, y_m=y, heading_rad=heading), control * CONTROL_DT_S
        )
        return [command.vx_mps, command.vy_mps, command.wz_rad_s]

    return produce


def command_at(case, control):
    """The navigation command of ``case`` at ``control``: its command, then zero from the stop control."""
    if not 0 <= control < case["controls"]:
        raise ValueError("Control lies outside the declared case")
    return (
        list(case["command"])
        if control < case.get("stop_control", case["controls"])
        else [0.0, 0.0, 0.0]
    )


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
    settle = min(len(pose) - 1, k - 1 + round(settle_s / CONTROL_DT_S))
    # Each trace pose follows its control, so the switch to zero starts at row k - 1.
    travel = np.hypot(*(pose[k - 1 :, :2] - pose[k - 1, :2]).T)
    return {
        "stop_control": k,
        "settle_distance_m": float(travel[settle - k + 1]),
        "final_distance_m": float(travel[-1]),
        "max_distance_m": float(travel.max()),
    }


def stand_and_stop_rows(trace, settle_s):
    """Rows under a zero command, excluding the first ``settle_s`` after each stop; tilt is read there."""
    commands = np.asarray(trace["command"][:, 0, :], dtype=np.float64)
    moving = np.any(commands != 0, axis=1)
    still = ~moving
    settle = int(round(settle_s / CONTROL_DT_S))
    for k in np.nonzero(moving[:-1] & ~moving[1:])[0] + 1:
        still[k : k + settle] = False
    return still


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


def _case_matches(recorded, canonical):
    recorded = dict(recorded)
    if "route_m" in canonical and recorded.get("stop_control") == canonical["controls"]:
        del recorded["stop_control"]
    return recorded == canonical


def judge_case(trace, rules, controls=None, case=None):
    """One recorded trace against the controller, tilt, torque and, for a route, stop rows of a set."""
    n = len(trace["command"]) if "command" in trace else 0
    if controls is None and case is not None:
        controls = case["controls"]
    if n == 0 or (controls is not None and n != controls):
        check = {"value": float(n), "bound": float(controls or 0), "pass": False}
        return {
            "holds": [],
            "stop": None,
            "checks": {"complete_trace": check},
            "failed_checks": ["complete_trace"],
            "pass": False,
        }
    # Validate the evidence before deciding which physical measurements apply. A zero-command
    # trace must not turn a declared walk into a stand and silently omit drift and stop checks.
    shapes = {"command": (n, 1, 3), "root_pose_xyzw": (n, 1, 7)}
    valid = all(
        key in trace and np.asarray(trace[key]).shape == shape and np.isfinite(trace[key]).all()
        for key, shape in shapes.items()
    )
    torque = np.asarray(trace.get("applied_torque_abs_max_400hz", []))
    valid = valid and torque.shape in ((n, 1), (n, 1, 18)) and np.isfinite(torque).all()
    failure = None if valid else "valid_trace"
    if valid and case is not None and "command" in case:
        expected = np.asarray([command_at(case, k) for k in range(n)], dtype=np.float32)
        # Native commands are recorded as float32; compare in that representation, without
        # widening any motion limit or allowing an off-bank command through a tolerance.
        if not np.array_equal(np.asarray(trace["command"][:, 0], dtype=np.float32), expected):
            failure = "command_schedule"
    elif valid and case is not None and "route_m" in case:
        commands = CommandSet.from_mapping(rules["commands"])
        allowed = [command.as_tuple() for command in commands.nonzero()]
        allowed.extend(
            cardinal.as_tuple()
            for speed in commands.translate_speeds_mps
            for bearing in range(commands.translate_bearings)
            if (cardinal := commands.cardinal_translation(speed, bearing)) is not None
        )
        if commands.stand:
            allowed.append((0.0, 0.0, 0.0))
        recorded = np.asarray(trace["command"][:, 0], dtype=np.float32)
        membership = (recorded[:, None, :] == np.asarray(allowed, dtype=np.float32)).all(-1).any(-1)
        if not membership.all():
            failure = "command_set"
    if failure is not None:
        return {
            "holds": [],
            "stop": None,
            "route": None,
            "checks": {failure: {"value": 0.0, "bound": 1.0, "pass": False}},
            "failed_checks": [failure],
            "pass": False,
        }
    limits = rules["controller"]
    holds = drift_per_hold(trace)
    stop = stopping_distance(trace, rules["stop"]["settle_s"])
    checks = {}
    if any(np.asarray(trace.get(key, False)).any() for key in ("terminated", "truncated")):
        checks["terminal_state"] = _check(1, 0)
    if case is not None and "command" in case:
        vx, vy, wz = case["command"]
        if vx == vy == 0.0 and wz:
            yaw = np.unwrap(planar_pose(trace["root_pose_xyzw"][:, 0, :])[:, 2])
            last = case.get("stop_control", n) - 1
            signed = math.copysign(1.0, wz) * (yaw[last] - yaw[0])
            checks["turn_direction"] = {
                "value": float(signed),
                "bound": 0.0,
                "pass": bool(signed > 0),
            }
    walks = [
        h
        for h in holds
        # Recorded commands are float32: 0.05 m/s on a diagonal reads back as 0.0500000023.
        if math.hypot(*h["command"][:2]) <= limits["drift_speed_mps"] * (1 + 1e-6)
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
    still = stand_and_stop_rows(trace, rules["stop"]["settle_s"])
    if still.any():
        checks["tilt_deg"] = _check(
            tilt_degrees(trace["root_pose_xyzw"][still, 0, :]).max(), rules["tilt_max_deg"]
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
    canonical = {case["case_id"]: case for case in cases(rules)}
    directory = Path(evaluation_dir)
    allocation_path = directory / "allocation.json"
    allocation = json.loads(allocation_path.read_text()) if allocation_path.exists() else None
    selected = list(canonical)
    if allocation is not None:
        declared = allocation["cases"]
        selected = [case["case_id"] for case in declared]
        if (
            allocation.get("schema") != "hexapod_requirements_allocation_v1"
            or allocation.get("requirements_set") != name
            or not isinstance(allocation.get("requirements_sha256"), str)
            or len(allocation["requirements_sha256"]) != 64
            or not selected
            or len(selected) != len(set(selected))
            or not all(
                key in canonical and _case_matches(case, canonical[key])
                for key, case in zip(selected, declared)
            )
        ):
            raise ValueError("Evaluation allocation differs from the named requirement set")
    rows, checkpoints, seen = [], set(), set()
    for index, report_path in enumerate(sorted(directory.glob("batch_*/report.json"))):
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes)
        if len(report["cases"]) != 1:
            raise ValueError("A requirement batch must declare one case")
        declared = report["cases"][0]
        case_id = declared["case_id"]
        if (
            case_id in seen
            or case_id not in selected
            or (
                allocation is not None
                and (case_id != selected[index] or report_path.parent.name != f"batch_{index:03d}")
            )
        ):
            raise ValueError("Unexpected or duplicate requirement batch: " + case_id)
        seen.add(case_id)
        checkpoint = report.get("checkpoint_sha256")
        checkpoints.add(checkpoint)
        trace_path = report_path.with_name("control_trace.npz")
        trace, trace_sha = {}, None
        if trace_path.is_file():
            trace_bytes = trace_path.read_bytes()
            trace_sha = hashlib.sha256(trace_bytes).hexdigest()
            recorded = report.get("files", {}).get(trace_path.name)
            if trace_path.name in report.get("files", {}) and trace_sha != recorded:
                raise ValueError("Trace bytes differ from the batch report: " + case_id)
            with np.load(io.BytesIO(trace_bytes), allow_pickle=False) as archive:
                trace = dict(archive)
        case = canonical[case_id]
        result = judge_case(trace, rules, case=case)
        if not _case_matches(declared, case):
            result["checks"]["case_declaration"] = _check(1, 0)
        if allocation is not None and checkpoint != allocation.get("checkpoint_sha256"):
            result["checks"]["checkpoint_identity"] = _check(1, 0)
        if "complete_trace" not in result["checks"]:
            if report.get("failure_kind") == "native_terminal_prefix":
                result["checks"]["terminal_state"] = _check(1, 0)
            elif (
                report.get("failure") is not None
                or report.get("acquisition_complete") is False
                or report.get("native_capture_failure") is not None
                or any(
                    row.get("native_capture_complete") is False for row in report.get("results", [])
                )
            ):
                result["checks"]["capture_complete"] = _check(1, 0)
        result["failed_checks"] = [
            key for key, check in result["checks"].items() if not check["pass"]
        ]
        result["pass"] = not result["failed_checks"]
        rows.append(
            {
                "case_id": case_id,
                "batch": report_path.parent.name,
                "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "control_trace_sha256": trace_sha,
                "command": case.get("command"),
                "route_m": case.get("route_m"),
                **result,
            }
        )
    return {
        "schema": SCHEMA,
        "requirements_set": name,
        "requirements_sha256": requirements_sha,
        "recorded_requirements_sha256": None
        if allocation is None
        else allocation["requirements_sha256"],
        "checkpoint_sha256": sorted(c for c in checkpoints if c),
        "cases": rows,
        "judged_cases": len(rows),
        "failed_cases": [row["case_id"] for row in rows if not row["pass"]],
        "missing_case_ids": [key for key in selected if key not in seen],
        "pass": seen == set(selected) and all(row["pass"] for row in rows),
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
