"""CPU/static contract for exact action-processing checkpoint playback."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


ISAACLAB_DIR = Path(__file__).parents[1]
EVALUATOR_PATH = ISAACLAB_DIR / "evaluate_checkpoint.py"
SOURCE = EVALUATOR_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _function(name: str):
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            namespace = {"Any": object, "math": math}
            exec(compile(module, EVALUATOR_PATH, "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"Missing function {name}")


apply_slew_override = _function("_apply_processed_joint_target_slew_override")


class EvaluateCheckpointSlewOverrideTest(unittest.TestCase):
    def test_cli_contract_validates_finite_positive_value(self):
        argument_source = SOURCE[
            SOURCE.index('"--processed-joint-target-slew-limit-rad-per-20ms"') :
        ][:7000]
        self.assertIn("type=float", argument_source)
        self.assertIn("math.isfinite", argument_source)
        self.assertIn("> 0.0", argument_source)

    def test_override_is_applied_and_reports_cli_provenance(self):
        env_cfg = SimpleNamespace(
            processed_joint_target_slew_limit_rad_per_20ms=0.04
        )
        provenance = apply_slew_override(env_cfg, 0.028)
        self.assertEqual(
            env_cfg.processed_joint_target_slew_limit_rad_per_20ms, 0.028
        )
        self.assertEqual(
            provenance,
            {
                "processed_joint_target_slew_limit_rad_per_20ms": 0.028,
                "source": "cli_override",
            },
        )

    def test_default_and_disabled_values_are_resolved_without_mutation(self):
        default_cfg = SimpleNamespace(
            processed_joint_target_slew_limit_rad_per_20ms=0.04
        )
        self.assertEqual(
            apply_slew_override(default_cfg, None),
            {
                "processed_joint_target_slew_limit_rad_per_20ms": 0.04,
                "source": "task_default",
            },
        )
        disabled_cfg = SimpleNamespace()
        self.assertEqual(
            apply_slew_override(disabled_cfg, None),
            {
                "processed_joint_target_slew_limit_rad_per_20ms": None,
                "source": "task_default",
            },
        )
        self.assertFalse(
            hasattr(
                disabled_cfg, "processed_joint_target_slew_limit_rad_per_20ms"
            )
        )

    def test_helper_rejects_invalid_override_or_task_default(self):
        for bad in (0.0, -0.01, float("inf"), float("nan")):
            with self.subTest(bad=bad), self.assertRaisesRegex(
                ValueError, "finite and positive"
            ):
                apply_slew_override(SimpleNamespace(), bad)
            with self.subTest(default=bad), self.assertRaisesRegex(
                ValueError, "finite and positive"
            ):
                apply_slew_override(
                    SimpleNamespace(
                        processed_joint_target_slew_limit_rad_per_20ms=bad
                    ),
                    None,
                )

    def test_resolved_provenance_is_attached_to_every_report_and_wrapper(self):
        self.assertIn(
            'reports[-1]["action_processing"] = dict(action_processing)', SOURCE
        )
        self.assertIn('"action_processing": dict(action_processing)', SOURCE)
        self.assertLess(
            SOURCE.index("_apply_processed_joint_target_slew_override(", SOURCE.index("def main")),
            SOURCE.index("with launch_simulation(env_cfg, args):"),
        )


if __name__ == "__main__":
    unittest.main()
