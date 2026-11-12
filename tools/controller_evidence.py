"""Validate the accepted controller's published projections without importing the simulator.

The registry owns the selected evidence and its hashes. This module checks its copied values,
limits and verdicts against those records. Historical probe evidence keeps its recorded scope;
validating a projection does not qualify the controller on the current native requirement cases.
"""

import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError("Accepted controller: " + message)


def read(root, reference, digest=None):
    path = Path(reference)
    require(not path.is_absolute() and ".." not in path.parts, "unsafe evidence path")
    path = root / path
    require(
        path.is_file() and path.resolve().is_relative_to(root.resolve()),
        "missing evidence: " + reference,
    )
    data = path.read_bytes()
    if digest is not None:
        require(hashlib.sha256(data).hexdigest() == digest, "evidence hash differs: " + reference)
    return json.loads(data)


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), "nonfinite or nonnumeric " + name)
    return value


def indexed(rows):
    by_id = {row["case_id"]: row for row in rows}
    require(bool(rows) and len(by_id) == len(rows), "missing or duplicate evidence cases")
    return by_id


def judgement(value, controller, requirements_set):
    require(
        value["schema"]
        in (
            "hexapod_walking_requirements_judgement_v1",
            "hexapod_walking_requirements_judgement_v2",
        ),
        "unsupported judgement schema",
    )
    require(
        value["checkpoint_sha256"] == [controller["checkpoint_sha256"]],
        "judgement checkpoint differs",
    )
    require(value["requirements_set"] == requirements_set, "judgement requirement set differs")
    rows = indexed(value["cases"])
    # The v1 rescore counted only cases with a drift or stop measurement.
    count = (
        sum(bool(row["checks"]) for row in rows.values())
        if value["schema"].endswith("_v1")
        else len(rows)
    )
    require(value["judged_cases"] == count, "judgement case count differs")
    return rows


def check_values(rows, metric, limit):
    selected = [row["checks"][metric] for row in rows if metric in row["checks"]]
    require(bool(selected), "missing evidence metric: " + metric)
    for check in selected:
        value = finite(check["value"], metric)
        require(check["bound"] == limit, "evidence bound differs: " + metric)
        require(check["pass"] is (value <= limit), "evidence verdict differs: " + metric)
    return [check["value"] for check in selected]


def validate(root, accepted=None):
    """Reject a changed registry value, limit, verdict, evidence hash or checkpoint identity."""
    root = Path(root)
    try:
        _validate(
            root, read(root, "configs/accepted_controller.json") if accepted is None else accepted
        )
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Accepted controller: malformed evidence or registry") from error


def _validate(root, accepted):
    require(accepted["schema_version"] == 1, "unsupported registry schema")
    controller = accepted["controller"]
    requirements = read(root, "configs/walking_requirements.json", accepted["requirements_sha256"])
    rules = requirements["sets"][accepted["requirements_set"]]
    receipt_ref = accepted["compatibility_receipt"]
    receipt = read(root, receipt_ref["path"], receipt_ref["sha256"])
    for key in ("checkpoint_sha256", "declaration_sha256"):
        require(receipt[key] == controller[key], "receipt " + key + " differs")
    model = read(root, "robot/active_model.json")["model"]["sha256"]
    require(
        controller["robot_model_sha256"] == model == receipt["original_identity"]["model_sha256"],
        "controller model differs",
    )
    judged = accepted["judged"]
    limits = {
        "stopping_distance_m": rules["controller"]["stopping_distance_max_m"],
        "drift_per_hold_m": rules["controller"]["drift_per_hold_max_m"],
        "stop_position_tolerance_m": rules["stop"]["position_tolerance_m"],
        "tilt_deg": rules["tilt_max_deg"],
        "motor_torque_nm": rules["motor_torque_max_nm"],
    }
    require(set(judged) == set(limits) | {"commands"}, "judged rows differ")
    documents = {}
    for name, row in judged.items():
        paths, digests = row["evidence"], row["evidence_sha256"]
        if isinstance(paths, str):
            paths, digests = [paths], [digests]
        require(
            isinstance(paths, list)
            and isinstance(digests, list)
            and bool(paths)
            and len(paths) == len(digests)
            and len(set(paths)) == len(paths),
            "evidence declarations differ: " + name,
        )
        require(name == "drift_per_hold_m" or len(paths) == 1, "ambiguous evidence: " + name)
        documents[name] = [read(root, path, digest) for path, digest in zip(paths, digests)]
        require(
            all(isinstance(digest, str) and len(digest) == 64 for digest in digests),
            "missing evidence hash: " + name,
        )

    verification = documents["commands"][0]
    require(
        verification["schema"] == "hexapod_floor_restoration_native_v1",
        "unsupported command evidence",
    )
    require(
        verification["checkpoint"]["sha256"] == controller["checkpoint_sha256"]
        and verification["checkpoint"]["declaration_sha256"] == controller["declaration_sha256"],
        "command evidence checkpoint differs",
    )
    replay = verification["replays"]["certified_loader"]
    count = replay["probes_passed"]
    command_pass = (
        verification["pass"] is True
        and count == 13
        and replay["failed_probe_cases"] == []
        and replay["exit_code"] == 0
        and replay["cleanup_checked"] is True
    )
    require(
        judged["commands"]["value"] == f"{count} of 13 native learning probes",
        "command value differs",
    )
    require(judged["commands"]["pass"] is command_pass, "command verdict differs")

    values = {}
    for name in (
        "stopping_distance_m",
        "drift_per_hold_m",
        "tilt_deg",
        "stop_position_tolerance_m",
    ):
        evidence = documents[name][0]
        rows = judgement(evidence, controller, accepted["requirements_set"])
        # The native 13-probe rescore predates the routes and stand/stop tilt rule. Its two
        # selected bounds remain unchanged; retain that historical requirements hash.
        if name in ("tilt_deg", "stop_position_tolerance_m"):
            require(
                evidence["requirements_sha256"] == accepted["requirements_sha256"],
                "judgement requirements hash differs",
            )
        metric = "stop_position_m" if name == "stop_position_tolerance_m" else name
        selected_rows = list(rows.values())
        if name == "stop_position_tolerance_m":
            expected = {
                "route_" + route: goals for route, goals in rules["stop"]["routes_m"].items()
            }
            routes = {key: row for key, row in rows.items() if row.get("route_m")}
            require(set(routes) == set(expected), "route evidence differs")
            require(
                all(row["route_m"] == expected[key] for key, row in routes.items()),
                "route goals differ",
            )
            selected_rows = list(routes.values())
        measured = check_values(selected_rows, metric, limits[name])
        if name in ("tilt_deg", "stop_position_tolerance_m"):
            require(len(measured) == len(selected_rows), "missing evidence metric: " + metric)
        values[name] = max(measured)
        for supporting in documents[name][1:]:
            support = judgement(supporting, controller, accepted["requirements_set"])
            check_values(support.values(), metric, limits[name])

    torque_row = judged["motor_torque_nm"]
    torque = documents["motor_torque_nm"][0]
    require(torque["schema"] == "canonical_learning_probe_suite_v1", "unsupported torque evidence")
    require(
        torque["allocation"]["checkpoint_sha256"] == controller["checkpoint_sha256"],
        "torque checkpoint differs",
    )
    require(
        torque_row["evidence"] == str(Path(receipt_ref["path"]).with_name("summary.json"))
        and torque_row["evidence_sha256"] == receipt["files"]["summary.json"],
        "torque evidence differs from compatibility receipt",
    )
    case = indexed(torque["results"])[torque_row["case_id"]]
    require(
        case["lineage"]["checkpoint_sha256"] == controller["checkpoint_sha256"],
        "torque case checkpoint differs",
    )
    require(torque_row["scope"] == "native_quiet_window", "torque evidence scope differs")
    require(case["profile"] == "quiet_stand", "torque case scope differs")
    metric = case["checks"]["max_applied_torque_nm"]
    values["motor_torque_nm"] = finite(metric["value"], "motor torque")
    require(
        metric["operator"] == "<="
        and metric["status"] == ("pass" if metric["value"] <= metric["bound"] else "fail"),
        "historical torque verdict differs",
    )
    for name, limit in limits.items():
        row = judged[name]
        require(finite(row["limit"], name + " limit") == limit, "limit differs: " + name)
        digits = 4 if name == "stopping_distance_m" else 3
        require(finite(row["value"], name) == round(values[name], digits), "value differs: " + name)
        require(row["pass"] is (values[name] <= limit), "verdict differs: " + name)
