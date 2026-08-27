"""CPU/static tests for dormant Stage-2D oblique-homotopy primitives."""

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
    for path, tree in REWARDS_MODULES:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                module = ast.Module(body=[node], type_ignores=[])
                ast.fix_missing_locations(module)
                namespace = {"math": math, "torch": torch}
                exec(compile(module, path, "exec"), namespace)
                return namespace[name]
    raise AssertionError(f"Missing function {name}")


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Missing class {name}")


def _method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Missing method {class_node.name}.{name}")


def _literal(class_node: ast.ClassDef, name: str):
    for node in class_node.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing {class_node.name}.{name}")


reset_safe_ema = _function("reset_safe_exponential_moving_average")
capped_axis_error = _function("capped_normalized_axis_error")
limit_joint_target_slew = _function("limit_processed_joint_target_slew")


class ResetSafeLateralEmaTest(unittest.TestCase):
    def test_first_sample_after_reset_is_exact_and_history_is_filtered(self):
        current = torch.tensor([2.0, 2.0, -3.0])
        previous = torch.tensor([100.0, 0.0, 100.0])
        valid = torch.tensor([False, True, False])
        actual = reset_safe_ema(
            current, previous, valid, step_dt=0.020, tau_s=0.10
        )
        alpha = 1.0 - math.exp(-0.020 / 0.10)
        expected = torch.tensor([2.0, 2.0 * alpha, -3.0])
        torch.testing.assert_close(actual, expected)

    def test_filter_rejects_bad_shapes_types_and_time_constants(self):
        values = torch.zeros(2)
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            reset_safe_ema(
                values, torch.zeros(3), torch.zeros(2, dtype=torch.bool),
                step_dt=0.020, tau_s=0.10,
            )
        with self.assertRaisesRegex(ValueError, "boolean"):
            reset_safe_ema(
                values, values, torch.zeros(2), step_dt=0.020, tau_s=0.10
            )
        for bad_tau in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(bad_tau=bad_tau), self.assertRaisesRegex(
                ValueError, "tau_s must be finite and positive"
            ):
                reset_safe_ema(
                    values, values, torch.zeros(2, dtype=torch.bool),
                    step_dt=0.020, tau_s=bad_tau,
                )


class CappedNormalizedLateralErrorTest(unittest.TestCase):
    def test_inactive_is_zero_target_is_zero_and_wrong_sign_is_capped(self):
        command = torch.tensor([0.0, 0.009, 0.10, -0.10, 0.10, 0.10])
        achieved = torch.tensor([99.0, 99.0, 0.10, 0.0, -0.05, -9.0])
        actual = capped_axis_error(
            command, achieved, active_threshold=0.01, error_cap=2.0
        )
        torch.testing.assert_close(
            actual, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.5, 2.0])
        )

    def test_error_is_sign_symmetric_and_validates_cap(self):
        command = torch.tensor([0.10, -0.20])
        achieved = torch.tensor([0.04, -0.08])
        positive = capped_axis_error(
            command, achieved, active_threshold=0.01, error_cap=2.0
        )
        mirrored = capped_axis_error(
            -command, -achieved, active_threshold=0.01, error_cap=2.0
        )
        torch.testing.assert_close(positive, mirrored)
        with self.assertRaisesRegex(ValueError, "error_cap must be finite and positive"):
            capped_axis_error(
                command, achieved, active_threshold=0.01, error_cap=0.0
            )

        nonfinite = capped_axis_error(
            torch.tensor([0.10, -0.10]),
            torch.tensor([float("nan"), float("inf")]),
            active_threshold=0.01,
            error_cap=2.0,
        )
        torch.testing.assert_close(nonfinite, torch.full((2,), 2.0))


class ProcessedTargetSlewLimiterTest(unittest.TestCase):
    def test_disabled_path_is_exact_and_does_not_inspect_history(self):
        target = torch.tensor([[0.20, -0.20]])
        actual, fraction = limit_joint_target_slew(
            target,
            torch.zeros(99),
            torch.zeros(3),
            max_delta_rad_per_20ms=None,
            step_dt=float("nan"),
        )
        self.assertIs(actual, target)
        torch.testing.assert_close(fraction, torch.zeros(1))

    def test_limit_is_in_processed_radians_and_scales_from_20ms(self):
        # The current Stage-2 lineage has action_scale=0.20.  The primitive
        # receives those already-scaled radian targets and does not assume or
        # reconstruct an action amplitude.
        default = torch.tensor([[0.60, 2.17], [0.60, 2.17]])
        action = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
        target = default + 0.20 * action
        valid = torch.tensor([True, False])
        actual, fraction = limit_joint_target_slew(
            target,
            default,
            valid,
            max_delta_rad_per_20ms=0.04,
            step_dt=0.010,
        )
        torch.testing.assert_close(actual[0], default[0] + torch.tensor([0.02, -0.02]))
        torch.testing.assert_close(actual[1], target[1])
        torch.testing.assert_close(fraction, torch.tensor([1.0, 0.0]))

    def test_reset_mask_prevents_cross_episode_target_carryover(self):
        target = torch.tensor([[0.2, -0.2], [0.2, -0.2]])
        stale = torch.tensor([[9.0, 9.0], [0.0, 0.0]])
        actual, _ = limit_joint_target_slew(
            target,
            stale,
            torch.tensor([False, True]),
            max_delta_rad_per_20ms=0.05,
            step_dt=0.020,
        )
        torch.testing.assert_close(actual[0], target[0])
        torch.testing.assert_close(actual[1], torch.tensor([0.05, -0.05]))


class StaticIntegrationContractTest(unittest.TestCase):
    def test_all_new_controls_are_dormant_by_default(self):
        cfg = _class(CFG_TREE, "HexapodFlatEnvCfg")
        self.assertIsNone(_literal(cfg, "navigation_lateral_velocity_ema_tau_s"))
        self.assertEqual(_literal(cfg, "lateral_normalized_error_penalty_scale"), 0.0)
        self.assertEqual(_literal(cfg, "lateral_normalized_error_cap"), 2.0)
        self.assertIsNone(
            _literal(cfg, "processed_joint_target_slew_limit_rad_per_20ms")
        )

    def test_sampler_dispatch_and_shaping_use_the_new_primitives(self):
        env = _class(ENV_TREE, "HexapodEnv")
        sample_source = ast.get_source_segment(
            ENV_SOURCE, _method(env, "_sample_commands")
        )
        reward_source = ast.get_source_segment(
            ENV_SOURCE, _method(env, "_get_rewards")
        )
        assert sample_source is not None and reward_source is not None
        self.assertIn('sampling_mode == "stage2d_oblique_homotopy"', sample_source)
        self.assertIn("sample_stage2d_oblique_homotopy_commands", sample_source)
        self.assertIn("reset_safe_exponential_moving_average", reward_source)
        self.assertIn("lateral_velocity_for_shaping", reward_source)
        self.assertIn("capped_normalized_axis_error", reward_source)
        self.assertIn('"lateral_normalized_error_penalty"', reward_source)

    def test_limiter_is_after_action_scaling_and_soft_limits_and_is_reset_safe(self):
        env = _class(ENV_TREE, "HexapodEnv")
        pre_source = ast.get_source_segment(
            ENV_SOURCE, _method(env, "_pre_physics_step")
        )
        reset_source = ast.get_source_segment(
            ENV_SOURCE, _method(env, "_reset_idx")
        )
        assert pre_source is not None and reset_source is not None
        self.assertLess(
            pre_source.index("self.cfg.action_scale"),
            pre_source.index("clamped_targets"),
        )
        self.assertLess(
            pre_source.index("clamped_targets"),
            pre_source.index("limit_processed_joint_target_slew"),
        )
        self.assertIn("step_dt=self.step_dt", pre_source)
        self.assertIn("self._navigation_lateral_velocity_ema[env_ids] = 0.0", reset_source)
        self.assertIn("self._has_navigation_lateral_velocity_ema[env_ids] = False", reset_source)
        self.assertIn("self._previous_processed_joint_target[env_ids] = joint_pos", reset_source)
        self.assertIn("self._has_previous_processed_joint_target[env_ids] = True", reset_source)

    def test_new_reward_and_diagnostics_are_episode_logged(self):
        env = _class(ENV_TREE, "HexapodEnv")
        init_source = ast.get_source_segment(ENV_SOURCE, _method(env, "__init__"))
        reward_source = ast.get_source_segment(ENV_SOURCE, _method(env, "_get_rewards"))
        reset_source = ast.get_source_segment(ENV_SOURCE, _method(env, "_reset_idx"))
        assert init_source is not None and reward_source is not None and reset_source is not None
        for key in (
            "lateral_normalized_error_penalty",
            "navigation_lateral_velocity_ema_mps",
            "normalized_lateral_error",
            "joint_target_slew_limited_fraction",
        ):
            self.assertIn(f'"{key}"', init_source)
            self.assertIn(f'"{key}"', reward_source)
        self.assertIn("self._episode_metric_names", reset_source)


if __name__ == "__main__":
    unittest.main()
