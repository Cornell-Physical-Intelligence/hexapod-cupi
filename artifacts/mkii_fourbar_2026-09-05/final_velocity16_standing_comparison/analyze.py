#!/usr/bin/env python3
"""Compare completed short standing reports without issuing training admission."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def check_velocity_arithmetic(window, substeps, num_envs):
    """Check reported population and RMS arithmetic, not unrecorded raw state."""
    checks = {}
    for prefix, members, sum_key, rms_key in (
        ("passive_velocity_relation", 12, "passive_velocity_relation_squared_sum",
         "rms_passive_velocity_relation_error_rad_s"),
        ("closure_relative_point_velocity", 6, "closure_relative_point_velocity_squared_norm_sum",
         "rms_closure_relative_point_velocity_m_s"),
    ):
        count = substeps * num_envs * members
        if window[prefix + "_samples"] != count:
            raise ValueError("Wrong velocity RMS population: " + prefix)
        total, reported = window[sum_key], window[rms_key]
        if not finite_number(total) or total < 0 or not finite_number(reported):
            raise ValueError("Invalid velocity squared sum or RMS")
        derived = math.sqrt(total / count)
        if not math.isclose(derived, reported, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("Velocity RMS differs from reported squared sum")
        checks[prefix] = {"samples": count, "rms_from_reported_squared_sum": derived,
                          "reported_rms": reported, "arithmetic_consistent": True}
    return checks


def load_report(path, multiplier):
    path = Path(path)
    report = json.loads(path.read_text())
    supervisor_path = path.with_name("supervisor.json")
    supervisor = json.loads(supervisor_path.read_text())
    manifest_path = path.with_name("source.SHA256SUMS")
    if (supervisor["cleanup"] != "removed_exact_id"
            or supervisor["source_identity_unchanged_at_finish"] is not True
            or supervisor["contract"] != report["contract"]):
        raise ValueError("Incomplete cleanup or incompatible source identity")
    source_manifest = supervisor["source_manifest"]
    if (sha256(manifest_path) != source_manifest["sha256"]
            or len(manifest_path.read_text().splitlines()) != source_manifest["files"]):
        raise ValueError("Source manifest capture differs from supervisor identity")
    captured = {}
    for line in manifest_path.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        if relative in captured:
            raise ValueError("Duplicate captured source path")
        captured[relative] = digest
    if any(captured.get(name) != digest for name, digest in report["contract"]["files"].items()):
        raise ValueError("Captured source manifest differs from functional contract")
    expected_success = report.get("pass") is True and report.get("errors") == []
    if expected_success != (supervisor.get("supervisor_exit_code") == 0
                            and supervisor.get("validator_report_status") == "passed"):
        raise ValueError("Supervisor and primary physical result disagree")
    if (report["num_envs"] != 32 or report["steps_requested"] != 600
            or report["steps_completed"] != 600 or report["driven_steps"] != 0
            or report["physics_substeps"] != 9600
            or report["solver_multiplier"] != multiplier
            or report["solver_iterations"] != [64 * multiplier, 16]):
        raise ValueError("Unexpected standing coverage or solver recipe")
    recipe = report["numerical_recipe"]
    if (recipe["physics_dt_s"] != .00125 or recipe["decimation"] != 16
            or recipe["solver_type"] != 1 or recipe["solver_velocity_iterations"] != 16
            or recipe["solver_position_iterations"] != 64 * multiplier
            or recipe["enable_external_forces_every_iteration"] is not True):
        raise ValueError("Unexpected numerical recipe")
    if report["simulation_training_admission"] is not False or report["hardware_admission"] is not False:
        raise ValueError("Short standing report unexpectedly claims admission")
    arithmetic = {}
    for name, count in (("startup", 1920), ("settled", 7680)):
        window = report["windows"][name]
        if window["substeps"] != count:
            raise ValueError("Wrong startup/settled sample window")
        if not all(finite_number(value) for value in window.values()):
            raise ValueError("Nonfinite or invalid window metric")
        arithmetic[name] = check_velocity_arithmetic(window, count, 32)
    if set(report["windows"]) != {"startup", "settled"}:
        raise ValueError("Unexpected extra measurement window")
    return report, supervisor, {
        "report_path": str(path), "report_sha256": sha256(path),
        "supervisor_sha256": sha256(supervisor_path),
        "source_commit": supervisor["source_commit"], "source_manifest": source_manifest,
        "source_unchanged_and_exact_cleanup_verified": True,
        "functional_contract_files_match_captured_source_manifest": True,
        "physical_standing_pass": expected_success, "primary_errors": report["errors"],
        "velocity_report_arithmetic": arithmetic,
    }


def compare(nominal, refined):
    reports, identities = {}, {}
    for label, path, multiplier in (("nominal", nominal, 1), ("refined", refined, 2)):
        reports[label], _, identities[label] = load_report(path, multiplier)
    a, b = reports["nominal"], reports["refined"]
    if a["contract"] != b["contract"]:
        raise ValueError("Source contracts differ")
    if identities["nominal"]["source_commit"] != identities["refined"]["source_commit"]:
        raise ValueError("Source commits differ")
    runtimes = [copy.deepcopy(r["runtime_manifest"]) for r in (a, b)]
    for runtime, count in zip(runtimes, (64, 128)):
        if runtime["resolved_simulation"].pop("solver_position_iterations") != count:
            raise ValueError("Actual runtime position iteration count differs")
    if runtimes[0] != runtimes[1]:
        raise ValueError("Runtime differs beyond solver position iterations")
    for key in ("reset_root_positions_m", "reset_first_env_joint_positions_rad",
                "joint_names", "active_motor_names", "body_names", "velocity_constraint_telemetry"):
        if a[key] != b[key]:
            raise ValueError("Recorded reset, mapping or telemetry definition differs: " + key)
    roots = a["reset_root_positions_m"]
    expected_xy = [[5. - 2 * (i // 6), -5. + 2 * (i % 6)] for i in range(32)]
    if (len(roots) != 32 or any(len(row) != 3 or not all(finite_number(v) for v in row) for row in roots)
            or [row[:2] for row in roots] != expected_xy):
        raise ValueError("Actual root placement differs from the 32-world grid")
    if any(r["reset_max_joint_error_rad"] > 5e-6 for r in (a, b)):
        raise ValueError("Initial joint state violates existing reset bound")
    descriptor = a["velocity_constraint_telemetry"]
    if (descriptor["acceptance_use"] != "observational_only_no_velocity_thresholds"
            or len(descriptor["passive_relation_names"]) != 12
            or len(descriptor["closure_pin_names"]) != 6):
        raise ValueError("Wrong velocity telemetry semantics")
    comparisons = {}
    for key, absolute, relative in (("mean_height_m", .001, 0.), ("max_applied_nm", .05, .05)):
        x, y = (r["windows"]["settled"][key] for r in (a, b))
        difference, bound = abs(x-y), max(absolute, relative*abs(y))
        comparisons[key] = {"nominal": x, "refined": y, "absolute_delta": difference,
                            "unchanged_bound": bound, "within_bound": difference <= bound}
    x, y = (r["windows"]["settled"]["max_demand_nm"] for r in (a, b))
    raw_bound = max(.05, .05 * abs(y))
    raw_comparison = {"nominal": x, "refined": y, "absolute_delta": abs(x-y),
        "same_torque_error_budget_bound": raw_bound, "within_bound": abs(x-y) <= raw_bound,
        "applicability": "Raw-demand metric added by qualifier fd34f66 using the existing delivered-torque error budget; not a claim about the c804169 producer's checks."}
    metrics = {}
    for window in ("startup", "settled"):
        metrics[window] = {}
        for key in a["windows"][window]:
            x, y = (r["windows"][window][key] for r in (a, b))
            metrics[window][key] = {"nominal": x, "refined": y, "delta": y-x,
                                    "refined_over_nominal": y/x if x else None}
    original_checks_pass = all(x["within_bound"] for x in comparisons.values()) and all(
        x["physical_standing_pass"] for x in identities.values())
    return {
        "schema": "hexapod.final_velocity16_short_standing_comparison.v1",
        "simulation_training_admission": False, "hardware_admission": False,
        "complete_full_validation": False, "reports": identities,
        "contract_sha256": a["contract"]["sha256"],
        "runtime_equivalent_except_position_iterations": True,
        "common_runtime_except_position_iterations": runtimes[0],
        "actual_reset_root_positions_match": True, "actual_reset_root_positions_m": roots,
        "joint_and_motor_name_mapping_match": True,
        "first_environment_initial_joint_coordinates_match": True,
        "all_environment_reset_error_rad": {label: r["reset_max_joint_error_rad"] for label, r in reports.items()},
        "matched_window": {"num_envs": 32, "physics_dt_s": .00125,
            "startup_samples_half_open": [0, 1920], "startup_time_s": [0., 2.4],
            "settled_samples_half_open": [1920, 9600], "settled_time_s": [2.4, 12.]},
        "standing_comparisons": comparisons,
        "raw_demand_comparison": raw_comparison,
        "standing_comparison_pass": original_checks_pass,
        "standing_comparison_scope": "c804169 delivered-torque and height comparison plus primary short physical checks",
        "current_qualifier_standing_metrics_pass": original_checks_pass and raw_comparison["within_bound"],
        "measurement_comparison": metrics,
        "limits": [
            "No driven phase, full physical qualification, PPO admission or hardware claim.",
            "Velocity metrics are observational; no new velocity acceptance bounds are assigned.",
            "Reports aggregate environments and substeps; no raw NPZ trace or per-environment velocity series exists for these validation runs.",
            "RMS verification checks the primary report's squared sums and denominators, not an independent reconstruction from raw physical state.",
            "The identical source supplies seed zero and zero standing actions; actual delivered targets were not traced in this validator.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nominal", type=Path)
    parser.add_argument("refined", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(compare(args.nominal, args.refined), indent=2, sort_keys=True) + "\n")
