"""CPU/static tests for bilateral longitudinal contact-moment shaping."""

from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
PACKAGE_ROOT = ROOT.parent / "packages" / "hexapod_env" / "hexapod_env"
ENV_PATH = PACKAGE_ROOT / "env.py"
CFG_PATH = PACKAGE_ROOT / "env_cfg.py"
REWARDS_MODULES = tuple(
    (path, ast.parse(path.read_text()))
    for path in sorted((PACKAGE_ROOT / "rewards").glob("*.py"))
    if path.name != "__init__.py"
)
ENV_SOURCE = ENV_PATH.read_text()
CFG_SOURCE = CFG_PATH.read_text()
ENV_TREE = ast.parse(ENV_SOURCE)
CFG_TREE = ast.parse(CFG_SOURCE)


def _function(name: str):
    path, node = next(
        (module_path, item)
        for module_path, tree in REWARDS_MODULES
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"math": math, "torch": torch}
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)


def _literal(class_node: ast.ClassDef, name: str):
    for node in class_node.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing {class_node.name}.{name}")


cost = _function("bounded_inactive_bilateral_longitudinal_contact_moment_cost")
reset_batch_summary = _function("reset_batch_time_mean_and_p50")


class BilateralLongitudinalContactMomentTest(unittest.TestCase):
    def inputs(self):
        offsets = torch.zeros(4, 2, 3)
        offsets[:, 0, 1] = -0.1
        offsets[:, 1, 1] = 0.1
        forces = torch.zeros_like(offsets)
        forces[:, :, 0] = 10.0
        contacts = torch.ones(4, 2, dtype=torch.bool)
        longitudinal = torch.tensor([True, True, False, True])
        yaw = torch.tensor([False, False, False, True])
        return offsets, forces, contacts, longitudinal, yaw

    def test_common_mode_forward_thrust_cancels_without_limiting_magnitude(self):
        offsets, forces, contacts, longitudinal, yaw = self.inputs()
        bounded, magnitude = cost(
            offsets, forces, contacts, longitudinal, yaw, reference_nm=1.0
        )
        torch.testing.assert_close(bounded, torch.zeros(4))
        torch.testing.assert_close(magnitude, torch.zeros(4))
        forces *= 100.0
        bounded, _ = cost(
            offsets, forces, contacts, longitudinal, yaw, reference_nm=1.0
        )
        torch.testing.assert_close(bounded, torch.zeros(4))

    def test_one_sided_reference_load_costs_half_and_command_masks_apply(self):
        offsets, forces, contacts, longitudinal, yaw = self.inputs()
        contacts[:, 1] = False
        bounded, magnitude = cost(
            offsets, forces, contacts, longitudinal, yaw, reference_nm=1.0
        )
        torch.testing.assert_close(bounded, torch.tensor([0.5, 0.5, 0.0, 0.0]))
        torch.testing.assert_close(magnitude, torch.tensor([1.0, 1.0, 0.0, 0.0]))

    def test_vertical_and_lateral_forces_are_ignored_and_invalid_data_is_finite(self):
        offsets, forces, contacts, longitudinal, yaw = self.inputs()
        forces[:, :, 0] = 0.0
        forces[:, :, 1:] = 1.0e5
        bounded, magnitude = cost(
            offsets, forces, contacts, longitudinal, yaw, reference_nm=0.5
        )
        torch.testing.assert_close(bounded, torch.zeros(4))
        torch.testing.assert_close(magnitude, torch.zeros(4))
        forces[0, 0, 0] = float("inf")
        bounded, magnitude = cost(
            offsets, forces, contacts, longitudinal, yaw, reference_nm=0.5
        )
        self.assertTrue(torch.isfinite(bounded).all())
        self.assertTrue(torch.isfinite(magnitude).all())
        self.assertGreater(float(bounded[0]), 0.99)

    def test_rejects_bad_reference_and_masks(self):
        offsets, forces, contacts, longitudinal, yaw = self.inputs()
        with self.assertRaises(ValueError):
            cost(offsets, forces, contacts, longitudinal, yaw, reference_nm=0.0)
        with self.assertRaises(ValueError):
            cost(offsets, forces, contacts.float(), longitudinal, yaw, reference_nm=1.0)


class ResetBatchTelemetryTest(unittest.TestCase):
    def test_reports_mean_and_interpolated_p50_of_episode_time_means(self):
        # Per-episode means for envs 0, 2, and 3 are 1, 5, and 3.  The
        # unrelated env 1 must not affect either reset-batch statistic.  This
        # mirrors _reset_idx selecting non-contiguous full-space env IDs before
        # passing its already-aligned elapsed-time clone to the helper.
        integrated = torch.tensor([2.0, 1000.0, 5.0, 12.0])
        elapsed = torch.tensor([2.0, 1.0, 1.0, 4.0])
        env_ids = torch.tensor([0, 2, 3])
        mean, p50 = reset_batch_summary(
            integrated[env_ids],
            elapsed[env_ids],
            min_elapsed_s=0.02,
        )
        torch.testing.assert_close(mean, torch.tensor(3.0))
        torch.testing.assert_close(p50, torch.tensor(3.0))

    def test_clamps_zero_duration_and_rejects_invalid_inputs(self):
        mean, p50 = reset_batch_summary(
            torch.tensor([0.02, 0.06]),
            torch.tensor([0.0, 0.02]),
            min_elapsed_s=0.02,
        )
        torch.testing.assert_close(mean, torch.tensor(2.0))
        torch.testing.assert_close(p50, torch.tensor(2.0))
        with self.assertRaisesRegex(ValueError, "matching 1-D"):
            reset_batch_summary(
                torch.zeros(2, 1),
                torch.zeros(2),
                min_elapsed_s=0.02,
            )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            reset_batch_summary(
                torch.tensor([]),
                torch.tensor([]),
                min_elapsed_s=0.02,
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            reset_batch_summary(
                torch.zeros(2),
                torch.zeros(2),
                min_elapsed_s=0.0,
            )


class BilateralContactStaticIntegrationTest(unittest.TestCase):
    def test_default_off_conditional_reward_and_raw_metric_are_wired(self):
        cfg = _class(CFG_TREE, "HexapodFlatEnvCfg")
        self.assertEqual(
            _literal(
                cfg,
                "inactive_bilateral_longitudinal_contact_moment_reward_scale",
            ),
            0.0,
        )
        self.assertGreater(
            _literal(
                cfg,
                "inactive_bilateral_longitudinal_contact_moment_reference_nm",
            ),
            0.0,
        )
        env = _class(ENV_TREE, "HexapodEnv")
        reward = next(
            node for node in env.body if isinstance(node, ast.FunctionDef) and node.name == "_get_rewards"
        )
        source = ast.get_source_segment(ENV_SOURCE, reward)
        assert source is not None
        self.assertIn(
            "if (\n            self.cfg.inactive_bilateral_longitudinal_contact_moment_reward_scale",
            source,
        )
        self.assertIn('rewards["inactive_bilateral_longitudinal_contact_moment"]', source)
        self.assertIn('"bilateral_longitudinal_contact_moment_nm"', source)
        self.assertIn("self._vector_in_command_frame", source)

    def test_reset_telemetry_is_enabled_only_with_the_bilateral_reward_path(self):
        env = _class(ENV_TREE, "HexapodEnv")
        reset = next(
            node
            for node in env.body
            if isinstance(node, ast.FunctionDef) and node.name == "_reset_idx"
        )
        source = ast.get_source_segment(ENV_SOURCE, reset)
        assert source is not None
        gate = "if bilateral_reward_scale != 0.0:"
        gate_index = source.index(gate)
        loop_index = source.index("for name, values in self._episode_sums.items()")
        telemetry_source = source[gate_index:loop_index]
        self.assertIn(
            '"Episode_Metric/bilateral_longitudinal_contact_moment_nm_p50"',
            telemetry_source,
        )
        self.assertIn(
            '"bilateral_longitudinal_contact_moment_cauchy_cost_mean"',
            telemetry_source,
        )
        self.assertIn(
            '"bilateral_longitudinal_contact_moment_cauchy_cost_p50"',
            telemetry_source,
        )
        self.assertIn(
            '"inactive_bilateral_longitudinal_contact_moment"', telemetry_source
        )
        self.assertIn("/ bilateral_reward_scale", telemetry_source)


if __name__ == "__main__":
    unittest.main()
