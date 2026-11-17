"""Enumerate walking requirement cases for evaluation and evidence validation."""

import math

from contracts.command import CommandSet

CONTROL_DT_S = 0.02


def cases(rules):
    """The cases of one requirement set: id, navigation command, controls and the stop control."""
    hold = round(rules["hold_max_s"] / CONTROL_DT_S)
    stop = round(2 * rules["stop"]["settle_s"] / CONTROL_DT_S)
    commands = CommandSet.from_mapping(rules["commands"])
    rows = [("stand", [0.0, 0.0, 0.0])] if commands.stand else []
    for command in commands.nonzero():
        if command.wz_rad_s == 0.0:
            speed = math.hypot(command.vx_mps, command.vy_mps)
            bearing = round(math.degrees(math.atan2(command.vy_mps, command.vx_mps))) % 360
            case_id = f"walk_{speed:.3f}_{bearing}deg"
        else:
            kind = "arc" if command.vx_mps else "turn"
            case_id = f"{kind}_{command.wz_rad_s:+.2f}"
        rows.append((case_id, list(command.as_tuple())))
    fixed = [
        {"case_id": case_id, "command": command, "controls": hold + stop, "stop_control": hold}
        for case_id, command in rows
    ]
    slowest = min(commands.translate_speeds_mps)
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
