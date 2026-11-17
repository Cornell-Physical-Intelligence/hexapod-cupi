"""Declare the case IDs, commands and routes shared by acquisition and publication."""

import math

from contracts.command import CommandSet


def case_definitions(rules):
    """Declare each fixed command and waypoint route in evaluation order."""
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
    return [{"case_id": key, "command": command} for key, command in rows] + [
        {"case_id": "route_" + name, "route_m": goals}
        for name, goals in rules["stop"].get("routes_m", {}).items()
    ]
