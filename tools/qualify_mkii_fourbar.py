#!/usr/bin/env python3
"""Combine two live solver-resolution runs into narrow simulation admission."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/hexapod_core"))
from hexapod_core.fourbar_v1 import validate_numerical_recipe_report
from mkii_asset_binding import solver_runtime_equivalent
from mkii_training_contract import identity, read_json, write_json, digest


def reset_root_positions(result):
    """Require the recorded physical placement for every compared environment."""
    count, positions = result.get("num_envs"), result.get("reset_root_positions_m")
    if (type(count) is not int or count < 1 or not isinstance(positions, list)
            or len(positions) != count):
        raise ValueError("Missing or incomplete actual reset-root placement")
    for xyz in positions:
        if (not isinstance(xyz, list) or len(xyz) != 3
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in xyz)):
            raise ValueError("Actual reset-root placement requires finite XYZ values")
    return positions


def convergence_method(recipes):
    """Describe only recipes that have passed the actual-runtime checks."""
    if set(recipes) != {"nominal", "refined"}:
        return "Numerical recipe validation incomplete; solver comparison unavailable"
    a, b = recipes["nominal"], recipes["refined"]
    solver = "TGS" if a["solver_type"] == 1 else "PGS"
    forces = "enabled" if a["enable_external_forces_every_iteration"] else "disabled"
    return (f"{solver} {a['solver_position_iterations']}/{a['solver_velocity_iterations']} versus "
            f"{b['solver_position_iterations']}/{b['solver_velocity_iterations']} iterations at fixed "
            f"{a['physics_dt_s']*1000:g} ms; external forces every iteration {forces}")


def qualify(nominal, refined, contract):
    errors, recipes, placements = [], {}, {}
    for label, result, multiplier in (("nominal", nominal, 1), ("refined", refined, 2)):
        try:
            recipes[label] = validate_numerical_recipe_report(result, multiplier)
        except ValueError as error:
            errors.append(f"{label}: {error}")
        try:
            placements[label] = reset_root_positions(result)
        except ValueError as error:
            errors.append(f"{label}: {error}")
        if (result.get("pass") is not True or result.get("errors") != []
                or result.get("contract") != contract or result.get("solver_multiplier") != multiplier
                or result.get("task_id") != contract.get("task_id")
                or result.get("num_envs", 0) < 32 or result.get("steps_completed", 0) < 1000
                or result.get("steps_requested") != result.get("steps_completed")
                or result.get("driven_steps") != 2400 or result.get("driven_coordinate_pass") is not True):
            errors.append(f"{label}: incomplete or incompatible physical validation")
    if nominal.get("num_envs") != refined.get("num_envs") or nominal.get("steps_completed") != refined.get("steps_completed"):
        errors.append("Solver comparison requires the same environment and step counts")
    placement_match = (set(placements) == {"nominal", "refined"}
                       and placements["nominal"] == placements["refined"])
    if not placement_match:
        errors.append("Solver comparison requires exactly matching recorded reset-root positions")
    for label, result in (("nominal", nominal), ("refined", refined)):
        if result.get("asset_binding", {}).get("pass") is not True:
            errors.append(f"{label}: selected CPU/Kit/runtime asset binding did not pass")
    if not solver_runtime_equivalent(nominal.get("runtime_manifest"), refined.get("runtime_manifest")):
        errors.append("Solver comparison changed runtime identity beyond position iterations")
    comparisons = {}
    for window in ("settled", "driven"):
        a, b = nominal.get("windows", {}).get(window, {}), refined.get("windows", {}).get(window, {})
        # Applied peaks can both equal the actuator cap while the underlying PD
        # requests diverge. Compare pre-envelope demand with the same empirical
        # torque tolerance; this is not an absolute cap on requested torque.
        for key, absolute, relative in (("mean_height_m", .001, 0.), ("max_applied_nm", .05, .05),
                                        ("max_demand_nm", .05, .05)):
            if key not in a or key not in b:
                errors.append(f"{window}: missing solver convergence metric {key}")
                continue
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in (a[key], b[key])):
                errors.append(f"{window}: nonfinite or invalid solver convergence metric {key}")
                continue
            delta, bound = abs(a[key]-b[key]), max(absolute, relative*abs(b[key]))
            comparisons[f"{window}.{key}"] = {"absolute_delta": delta, "bound": bound, "pass": delta <= bound}
            if delta > bound:
                errors.append(f"{window}: solver convergence difference exceeds {key} bound")
    result = copy.deepcopy(nominal)
    result["mode"] = "physical_fourbar_simulation_admission"
    result["pass"] = result["simulation_training_admission"] = not errors
    result["errors"] = errors
    result["convergence"] = {"pass": not errors,
                             "method": convergence_method(recipes),
                             "numerical_recipes": recipes,
                             "reset_root_positions_match": placement_match,
                             "comparisons": comparisons}
    result["admission_scope"] = (
        "Standing and +/-0.04 rad driven qualification only. Provisional flat-ground scratch PPO may explore "
        "the configured +/-0.30 rad action offsets under a continuous 1.25 ms closure point/axis and motor-envelope "
        "monitor; a monitor bound violation invalidates the run. Self-collision is off, so this does not qualify "
        "the collision-free workspace or hardware. This is not full joint-range, timestep-convergence or terrain "
        "qualification. The 48 V reference and uncalibrated metal-mount overload recovery remain explicit assumptions.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nominal", type=Path)
    parser.add_argument("refined", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite qualification evidence")
    result = qualify(read_json(args.nominal), read_json(args.refined), identity())
    result["source_reports"] = {"nominal_sha256": digest(args.nominal), "refined_sha256": digest(args.refined)}
    write_json(args.output, result)
    print(f"FOURBAR_ADMISSION pass={result['pass']} errors={result['errors']}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
