"""Solver comparisons require matched actual placement and truthful recipes."""
import copy
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"tools"), str(ROOT/"packages/hexapod_core")]
from hexapod_core.fourbar_v1 import TASK_ID, numerical_recipe
from qualify_mkii_fourbar import qualify


CONTRACT = {"task_id": TASK_ID}


def report(multiplier):
    recipe = numerical_recipe(multiplier)
    return {"pass": True, "errors": [], "contract": CONTRACT, "task_id": TASK_ID,
            "num_envs": 32, "steps_completed": 1000, "steps_requested": 1000,
            "driven_steps": 2400, "driven_coordinate_pass": True, "asset_binding": {"pass": True},
            "solver_multiplier": multiplier, "numerical_recipe": recipe,
            "solver_iterations": [recipe["solver_position_iterations"], recipe["solver_velocity_iterations"]],
            "runtime_manifest": {"resolved_simulation": {key: value for key, value in recipe.items() if key != "recipe_id"}},
            "reset_root_positions_m": [[float(index % 8)*2., float(index//8)*2., .14297] for index in range(32)],
            "windows": {window: {"mean_height_m": .138, "max_applied_nm": 1.2, "max_demand_nm": 1.3}
                        for window in ("settled", "driven")}}


class QualificationPlacementTests(unittest.TestCase):
    def test_equal_clipped_applied_peaks_cannot_hide_divergent_raw_demand(self):
        for window in ("settled", "driven"):
            a, b = report(1), report(2)
            a["windows"][window].update(max_applied_nm=5.5, max_demand_nm=6.)
            b["windows"][window].update(max_applied_nm=5.5, max_demand_nm=7.)
            result = qualify(a, b, CONTRACT)
            self.assertFalse(result["pass"])
            compared = result["convergence"]["comparisons"]
            self.assertTrue(compared[f"{window}.max_applied_nm"]["pass"])
            self.assertFalse(compared[f"{window}.max_demand_nm"]["pass"])
            self.assertEqual(compared[f"{window}.max_demand_nm"]["absolute_delta"], 1.)
            self.assertAlmostEqual(compared[f"{window}.max_demand_nm"]["bound"], .35)
            # A matched request above the motor cap is not itself forbidden.
            a["windows"][window]["max_demand_nm"] = 7.
            self.assertTrue(qualify(a, b, CONTRACT)["pass"])

    def test_raw_demand_requires_finite_real_values_and_existing_absolute_floor(self):
        for window in ("settled", "driven"):
            for value in (None, float("nan"), float("inf"), True, "1.3"):
                a, b = report(1), report(2)
                if value is None: del b["windows"][window]["max_demand_nm"]
                else: b["windows"][window]["max_demand_nm"] = value
                result = qualify(a, b, CONTRACT)
                self.assertFalse(result["pass"])
                self.assertTrue(any("max_demand_nm" in error for error in result["errors"]))
            a, b = report(1), report(2)
            a["windows"][window]["max_demand_nm"] = .24
            b["windows"][window]["max_demand_nm"] = .20
            result = qualify(a, b, CONTRACT)
            self.assertTrue(result["pass"])
            self.assertEqual(result["convergence"]["comparisons"][f"{window}.max_demand_nm"]["bound"], .05)
            a["windows"][window]["max_demand_nm"] = .26
            self.assertFalse(qualify(a, b, CONTRACT)["pass"])

    def test_actual_positions_match_and_method_uses_both_validated_recipes(self):
        result = qualify(report(1), report(2), CONTRACT)
        self.assertTrue(result["pass"], result["errors"])
        self.assertTrue(result["convergence"]["reset_root_positions_match"])
        self.assertEqual(result["convergence"]["numerical_recipes"], {"nominal": numerical_recipe(1), "refined": numerical_recipe(2)})
        a, b = numerical_recipe(1), numerical_recipe(2)
        self.assertIn(f"{a['solver_position_iterations']}/{a['solver_velocity_iterations']} versus "
                      f"{b['solver_position_iterations']}/{b['solver_velocity_iterations']}", result["convergence"]["method"])
        self.assertIn(f"{a['physics_dt_s']*1000:g} ms", result["convergence"]["method"])
        self.assertIn("external forces every iteration enabled", result["convergence"]["method"])

    def test_translation_height_permutation_and_single_ulp_mismatch_fail(self):
        for change in ("translation", "height", "permutation", "last_environment", "tiny_difference"):
            with self.subTest(change=change):
                refined = report(2)
                xyz = refined["reset_root_positions_m"]
                if change == "translation":
                    for row in xyz: row[0] += 2.
                elif change == "height": xyz[0][2] += .001
                elif change == "permutation": xyz[0], xyz[1] = xyz[1], xyz[0]
                elif change == "last_environment": xyz[-1][1] += 2.
                else: xyz[0][2] = math.nextafter(xyz[0][2], math.inf)
                result = qualify(report(1), refined, CONTRACT)
                self.assertFalse(result["pass"])
                self.assertFalse(result["convergence"]["reset_root_positions_match"])
                self.assertTrue(any("exactly matching recorded reset-root" in error for error in result["errors"]))

    def test_missing_nonfinite_wrong_shape_and_boolean_placement_fail_even_if_both_match(self):
        for positions in (None, [], [[0., 0., .14297]]*31, [[0., 0.]]*32,
                          [[0., 0., float("nan")]]*32, [[0., 0., float("inf")]]*32,
                          [[False, 0., .14297]]*32, [["0", 0., .14297]]*32):
            with self.subTest(positions=positions):
                a, b = report(1), report(2)
                a["reset_root_positions_m"] = copy.deepcopy(positions)
                b["reset_root_positions_m"] = copy.deepcopy(positions)
                result = qualify(a, b, CONTRACT)
                self.assertFalse(result["pass"])
                self.assertFalse(result["convergence"]["reset_root_positions_match"])
        a, b = report(1), report(2)
        del b["reset_root_positions_m"]
        self.assertFalse(qualify(a, b, CONTRACT)["pass"])

    def test_failed_recipe_is_not_described_as_a_completed_solver_comparison(self):
        refined = report(2)
        refined["runtime_manifest"]["resolved_simulation"]["solver_velocity_iterations"] += 1
        result = qualify(report(1), refined, CONTRACT)
        self.assertFalse(result["pass"])
        self.assertIn("validation incomplete", result["convergence"]["method"])
        self.assertNotIn("refined", result["convergence"]["numerical_recipes"])

    def test_complete_matching_standing_pair_remains_nonadmitting(self):
        a, b = report(1), report(2)
        for item in (a, b):
            item.update(steps_requested=600, steps_completed=600, driven_steps=0)
            item["windows"].pop("driven")
        result = qualify(a, b, CONTRACT)
        self.assertFalse(result["pass"])
        self.assertFalse(result["simulation_training_admission"])


if __name__ == "__main__":
    unittest.main()
