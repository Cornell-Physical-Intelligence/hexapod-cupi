"""CPU/static tests for opt-in platform-stability reward primitives."""

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
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
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
            isinstance(target, ast.Name) and target.id == name for target in node.targets
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


normalized_deck_stability_reward = _function(
    "normalized_deck_stability_reward"
)
select_support_contact_target = _function("select_support_contact_target")
select_command_conditioned_nominal_height = _function(
    "select_command_conditioned_nominal_height"
)
applied_torque_slew_l2 = _function("applied_torque_slew_l2")
bounded_inactive_yaw_rate_slew_cost = _function(
    "bounded_inactive_yaw_rate_slew_cost"
)
bounded_inactive_ground_contact_yaw_moment_cost = _function(
    "bounded_inactive_ground_contact_yaw_moment_cost"
)
max_joint_rated_torque_excess_l2 = _function(
    "max_joint_rated_torque_excess_l2"
)
max_joint_rated_torque_excess_l1 = _function(
    "max_joint_rated_torque_excess_l1"
)


class DeckStabilityRewardTest(unittest.TestCase):
    def score(
        self,
        vertical: torch.Tensor,
        angular_xy: torch.Tensor,
        gravity_xy: torch.Tensor,
        height: torch.Tensor,
        **overrides,
    ) -> torch.Tensor:
        scales = {
            "vertical_velocity_scale_mps": 0.10,
            "roll_pitch_rate_scale_rad_s": 0.50,
            "projected_gravity_xy_scale": 0.15,
            "height_error_scale_m": 0.025,
        }
        scales.update(overrides)
        return normalized_deck_stability_reward(
            vertical, angular_xy, gravity_xy, height, **scales
        )

    def test_perfect_deck_scores_one_and_all_scale_errors_score_half(self):
        perfect = self.score(
            torch.zeros(3), torch.zeros(3, 2), torch.zeros(3, 2), torch.zeros(3)
        )
        torch.testing.assert_close(perfect, torch.ones(3))

        at_scale = self.score(
            torch.tensor([0.10]),
            torch.tensor([[0.50, -0.50]]),
            torch.tensor([[0.15, -0.15]]),
            torch.tensor([0.025]),
        )
        torch.testing.assert_close(at_scale, torch.tensor([0.5]))

    def test_components_are_equal_weighted_and_sign_symmetric(self):
        vertical_only = self.score(
            torch.tensor([0.10, -0.10]),
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            torch.zeros(2),
        )
        # One component at 0.5 plus three perfect components, divided by four.
        torch.testing.assert_close(vertical_only, torch.full((2,), 0.875))

    def test_nonfinite_and_large_state_remains_finite_and_bounded(self):
        scores = self.score(
            torch.tensor([float("nan"), float("inf"), 1.0e30]),
            torch.tensor([[float("inf"), 0.0], [1.0e30, -1.0e30], [0.0, 0.0]]),
            torch.tensor([[0.0, 0.0], [float("nan"), 0.0], [1.0e30, 1.0e30]]),
            torch.tensor([float("-inf"), 1.0e30, float("nan")]),
        )
        self.assertTrue(torch.all(torch.isfinite(scores)))
        self.assertTrue(torch.all((scores >= 0.0) & (scores <= 1.0)))

    def test_normalization_scales_must_be_finite_and_positive(self):
        zeros = (torch.zeros(1), torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1))
        for invalid in (0.0, -0.1, float("inf"), float("nan")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "finite and positive"
            ):
                self.score(*zeros, vertical_velocity_scale_mps=invalid)


class SupportTargetAndTorqueSlewTest(unittest.TestCase):
    def test_disabled_support_schedule_exactly_uses_legacy_scalar(self):
        speeds = torch.tensor([0.0, 0.1, 0.3, 10.0])
        targets = select_support_contact_target(
            speeds,
            scalar_target=3.25,
            speed_conditioning_enabled=False,
            # Deliberately invalid opt-in fields prove the disabled legacy path
            # does not inspect or alter them.
            low_speed_threshold_mps=2.0,
            high_speed_threshold_mps=1.0,
            low_speed_target=float("nan"),
            medium_speed_target=-1.0,
            high_speed_target=float("inf"),
        )
        torch.testing.assert_close(targets, torch.full_like(speeds, 3.25))

    def test_speed_conditioned_support_bins_include_threshold_boundaries(self):
        speeds = torch.tensor([0.0, 0.10, 0.1001, 0.2499, 0.25, 1.0])
        targets = select_support_contact_target(
            speeds,
            scalar_target=99.0,
            speed_conditioning_enabled=True,
            low_speed_threshold_mps=0.10,
            high_speed_threshold_mps=0.25,
            low_speed_target=5.0,
            medium_speed_target=4.0,
            high_speed_target=3.0,
        )
        torch.testing.assert_close(
            targets, torch.tensor([5.0, 5.0, 4.0, 4.0, 3.0, 3.0])
        )

    def test_speed_conditioned_support_rejects_invalid_schedule(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            select_support_contact_target(
                torch.zeros(1),
                scalar_target=3.0,
                speed_conditioning_enabled=True,
                low_speed_threshold_mps=0.25,
                high_speed_threshold_mps=0.10,
                low_speed_target=5.0,
                medium_speed_target=4.0,
                high_speed_target=3.0,
            )

    def test_torque_slew_mask_removes_reset_boundary_delta(self):
        current = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
        previous = torch.tensor([[100.0, 100.0], [1.0, -1.0]])
        valid_history = torch.tensor([False, True])
        cost = applied_torque_slew_l2(current, previous, valid_history)
        torch.testing.assert_close(cost, torch.tensor([0.0, 29.0]))


class InactiveYawRateSlewTest(unittest.TestCase):
    def test_reference_delta_scores_half_and_masks_invalid_or_active_samples(self):
        cost = bounded_inactive_yaw_rate_slew_cost(
            torch.tensor([0.10, 0.20, 0.30, 0.40]),
            torch.tensor([0.10, 0.16, 0.20, 0.00]),
            torch.tensor([True, True, False, True]),
            torch.tensor([False, False, False, True]),
            reference_rad_s_per_step=0.04,
        )
        torch.testing.assert_close(cost, torch.tensor([0.0, 0.5, 0.0, 0.0]))

    def test_cost_is_sign_symmetric_finite_and_bounded(self):
        cost = bounded_inactive_yaw_rate_slew_cost(
            torch.tensor([0.08, -0.08, float("nan"), float("inf")]),
            torch.zeros(4),
            torch.ones(4, dtype=torch.bool),
            torch.zeros(4, dtype=torch.bool),
            reference_rad_s_per_step=0.04,
        )
        torch.testing.assert_close(cost[:2], torch.full((2,), 0.8))
        self.assertTrue(torch.all(torch.isfinite(cost)))
        self.assertTrue(torch.all((cost >= 0.0) & (cost <= 1.0)))

    def test_rejects_invalid_reference_shapes_and_mask_dtypes(self):
        yaw_rate = torch.zeros(2)
        boolean_mask = torch.zeros(2, dtype=torch.bool)
        for invalid in (0.0, -0.1, float("inf"), float("nan")):
            with self.subTest(reference=invalid), self.assertRaisesRegex(
                ValueError, "finite and positive"
            ):
                bounded_inactive_yaw_rate_slew_cost(
                    yaw_rate,
                    yaw_rate,
                    boolean_mask,
                    boolean_mask,
                    reference_rad_s_per_step=invalid,
                )
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            bounded_inactive_yaw_rate_slew_cost(
                yaw_rate,
                torch.zeros(3),
                boolean_mask,
                boolean_mask,
                reference_rad_s_per_step=0.04,
            )
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            bounded_inactive_yaw_rate_slew_cost(
                yaw_rate,
                yaw_rate,
                torch.zeros(2),
                boolean_mask,
                reference_rad_s_per_step=0.04,
            )


class InactiveGroundContactYawMomentTest(unittest.TestCase):
    def cost(
        self,
        points: torch.Tensor,
        forces: torch.Tensor,
        contacts: torch.Tensor,
        yaw_active: torch.Tensor | None = None,
        moving_active: torch.Tensor | None = None,
        *,
        reference_nm: float = 0.50,
    ) -> torch.Tensor:
        if yaw_active is None:
            yaw_active = torch.zeros(points.shape[0], dtype=torch.bool)
        if moving_active is None:
            moving_active = torch.ones(points.shape[0], dtype=torch.bool)
        cost, _ = bounded_inactive_ground_contact_yaw_moment_cost(
            points,
            forces,
            torch.zeros(points.shape[0], 3),
            contacts,
            moving_active,
            yaw_active,
            reference_nm=reference_nm,
        )
        return cost

    def test_reference_moment_scores_half_and_nonstraight_samples_are_masked(self):
        # r=(0.25, 0), F=(0, 2): r x F = +0.50 N*m.
        points = torch.tensor(
            [[[0.25, 0.0, 0.0]], [[0.25, 0.0, 0.0]], [[0.25, 0.0, 0.0]]]
        )
        forces = torch.tensor(
            [[[0.0, 2.0, 10.0]], [[0.0, 2.0, 10.0]], [[0.0, 2.0, 10.0]]]
        )
        cost = self.cost(
            points,
            forces,
            torch.ones(3, 1, dtype=torch.bool),
            torch.tensor([False, True, False]),
            torch.tensor([True, True, False]),
        )
        torch.testing.assert_close(cost, torch.tensor([0.5, 0.0, 0.0]))

        _, magnitude_nm = bounded_inactive_ground_contact_yaw_moment_cost(
            points,
            forces,
            torch.zeros(3, 3),
            torch.ones(3, 1, dtype=torch.bool),
            torch.tensor([True, True, False]),
            torch.tensor([False, True, False]),
            reference_nm=0.50,
        )
        torch.testing.assert_close(magnitude_nm, torch.tensor([0.5, 0.0, 0.0]))

    def test_balanced_left_right_tangential_loads_cancel_net_yaw_wrench(self):
        points = torch.tensor([[[-0.20, 0.0, 0.0], [0.20, 0.0, 0.0]]])
        balanced_forces = torch.tensor([[[0.0, 3.0, 10.0], [0.0, 3.0, 10.0]]])
        imbalanced_forces = torch.tensor([[[0.0, 1.0, 10.0], [0.0, 3.0, 10.0]]])
        contacts = torch.ones(1, 2, dtype=torch.bool)
        torch.testing.assert_close(
            self.cost(points, balanced_forces, contacts), torch.zeros(1)
        )
        # (-0.2*1) + (0.2*3) = +0.4 N*m.
        torch.testing.assert_close(
            self.cost(points, imbalanced_forces, contacts), torch.tensor([0.3902439])
        )

    def test_vertical_load_alternation_and_noncontacts_do_not_create_yaw_cost(self):
        points = torch.tensor([[[-0.20, 0.10, 0.0], [0.20, -0.10, 0.0]]])
        vertical_forces = torch.tensor([[[0.0, 0.0, 5.0], [0.0, 0.0, 50.0]]])
        no_contacts = torch.zeros(1, 2, dtype=torch.bool)
        contacts = torch.ones(1, 2, dtype=torch.bool)
        torch.testing.assert_close(
            self.cost(points, vertical_forces, contacts), torch.zeros(1)
        )
        arbitrary_forces = torch.full_like(vertical_forces, 1.0e4)
        torch.testing.assert_close(
            self.cost(points, arbitrary_forces, no_contacts), torch.zeros(1)
        )

    def test_nonfinite_inputs_remain_finite_bounded_and_sign_symmetric(self):
        points = torch.tensor(
            [[[0.25, 0.0, 0.0]], [[0.25, 0.0, 0.0]], [[float("nan"), 0.0, 0.0]]]
        )
        forces = torch.tensor(
            [[[0.0, 2.0, 0.0]], [[0.0, -2.0, 0.0]], [[0.0, float("inf"), 0.0]]]
        )
        cost = self.cost(points, forces, torch.ones(3, 1, dtype=torch.bool))
        torch.testing.assert_close(cost[:2], torch.full((2,), 0.5))
        self.assertTrue(torch.all(torch.isfinite(cost)))
        self.assertTrue(torch.all((cost >= 0.0) & (cost <= 1.0)))

    def test_rejects_invalid_reference_shapes_and_mask_dtypes(self):
        points = torch.zeros(2, 3, 3)
        forces = torch.zeros_like(points)
        contacts = torch.zeros(2, 3, dtype=torch.bool)
        yaw_active = torch.zeros(2, dtype=torch.bool)
        for invalid in (0.0, -0.1, float("inf"), float("nan")):
            with self.subTest(reference=invalid), self.assertRaisesRegex(
                ValueError, "finite and positive"
            ):
                bounded_inactive_ground_contact_yaw_moment_cost(
                    points,
                    forces,
                    torch.zeros(2, 3),
                    contacts,
                    torch.ones(2, dtype=torch.bool),
                    yaw_active,
                    reference_nm=invalid,
                )
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            bounded_inactive_ground_contact_yaw_moment_cost(
                points,
                torch.zeros(2, 2, 3),
                torch.zeros(2, 3),
                contacts,
                torch.ones(2, dtype=torch.bool),
                yaw_active,
                reference_nm=0.5,
            )
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            bounded_inactive_ground_contact_yaw_moment_cost(
                points,
                forces,
                torch.zeros(2, 3),
                contacts.float(),
                torch.ones(2, dtype=torch.bool),
                yaw_active,
                reference_nm=0.5,
            )


class MaxJointRatedTorqueExcessTest(unittest.TestCase):
    def test_uses_worst_single_joint_squared_excess_per_environment(self):
        computed = torch.tensor(
            [
                [0.0, 1.6, -1.7, 2.1],
                [2.0, -3.0, 1.6, 0.2],
                [-1.5, 0.0, 1.5, -1.6],
            ]
        )
        cost = max_joint_rated_torque_excess_l2(
            computed, rated_torque_nm=1.6
        )
        torch.testing.assert_close(cost, torch.tensor([0.25, 1.96, 0.0]))

    def test_linear_variant_preserves_near_threshold_excess(self):
        computed = torch.tensor(
            [[1.61, -1.65, 1.20], [2.10, -1.70, 0.0], [1.60, -1.60, 0.0]]
        )
        cost = max_joint_rated_torque_excess_l1(
            computed, rated_torque_nm=1.6
        )
        torch.testing.assert_close(cost, torch.tensor([0.05, 0.50, 0.0]))

    def test_non_overloaded_joints_do_not_dilute_one_overloaded_joint(self):
        one_joint = torch.tensor([[2.1]])
        eighteen_joints = torch.cat((one_joint, torch.zeros(1, 17)), dim=1)
        one_cost = max_joint_rated_torque_excess_l2(
            one_joint, rated_torque_nm=1.6
        )
        fleet_cost = max_joint_rated_torque_excess_l2(
            eighteen_joints, rated_torque_nm=1.6
        )
        torch.testing.assert_close(fleet_cost, one_cost)
        linear_one = max_joint_rated_torque_excess_l1(
            one_joint, rated_torque_nm=1.6
        )
        linear_fleet = max_joint_rated_torque_excess_l1(
            eighteen_joints, rated_torque_nm=1.6
        )
        torch.testing.assert_close(linear_fleet, linear_one)

    def test_rejects_empty_joint_axis_and_invalid_rating(self):
        with self.assertRaisesRegex(ValueError, "non-empty joint dimension"):
            max_joint_rated_torque_excess_l2(
                torch.empty(2, 0), rated_torque_nm=1.6
            )
        with self.assertRaisesRegex(ValueError, "non-empty joint dimension"):
            max_joint_rated_torque_excess_l1(
                torch.empty(2, 0), rated_torque_nm=1.6
            )
        for invalid in (-0.1, float("inf"), float("nan")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "finite and non-negative"
            ):
                max_joint_rated_torque_excess_l2(
                    torch.zeros(1, 18), rated_torque_nm=invalid
                )
            with self.subTest(linear_invalid=invalid), self.assertRaisesRegex(
                ValueError, "finite and non-negative"
            ):
                max_joint_rated_torque_excess_l1(
                    torch.zeros(1, 18), rated_torque_nm=invalid
                )


class CommandConditionedNominalHeightTest(unittest.TestCase):
    def test_none_exactly_preserves_single_legacy_height(self):
        speeds = torch.tensor([0.0, 0.05, 0.0501, 1.0])
        target = select_command_conditioned_nominal_height(
            speeds,
            moving_nominal_height_m=0.205,
            stand_nominal_height_m=None,
            # Deliberately invalid: the dormant option must not inspect this.
            moving_command_threshold_mps=float("nan"),
        )
        torch.testing.assert_close(target, torch.full_like(speeds, 0.205))

    def test_stand_height_includes_moving_threshold_boundary(self):
        speeds = torch.tensor([0.0, 0.0499, 0.05, 0.0501, 0.30])
        target = select_command_conditioned_nominal_height(
            speeds,
            moving_nominal_height_m=0.181,
            stand_nominal_height_m=0.177,
            moving_command_threshold_mps=0.05,
        )
        torch.testing.assert_close(
            target, torch.tensor([0.177, 0.177, 0.177, 0.181, 0.181])
        )

    def test_enabled_selector_rejects_invalid_heights_and_threshold(self):
        speeds = torch.zeros(1)
        invalid_cases = (
            {"moving_nominal_height_m": 0.0},
            {"stand_nominal_height_m": float("nan")},
            {"moving_command_threshold_mps": -0.01},
        )
        defaults = {
            "moving_nominal_height_m": 0.181,
            "stand_nominal_height_m": 0.177,
            "moving_command_threshold_mps": 0.05,
        }
        for override in invalid_cases:
            with self.subTest(override=override), self.assertRaises(ValueError):
                select_command_conditioned_nominal_height(
                    speeds, **(defaults | override)
                )


class StaticIntegrationContractTest(unittest.TestCase):
    def test_defaults_are_opt_in_and_legacy_compatible(self):
        cfg = _class(CFG_TREE, "HexapodFlatEnvCfg")
        self.assertEqual(_literal(cfg, "deck_stability_reward_scale"), 0.0)
        self.assertIsNone(_literal(cfg, "stand_nominal_height_m"))
        self.assertEqual(_literal(cfg, "joint_torque_slew_reward_scale"), 0.0)
        self.assertEqual(
            _literal(cfg, "inactive_yaw_rate_slew_reward_scale"), 0.0
        )
        self.assertGreater(
            _literal(cfg, "inactive_yaw_rate_slew_reference_rad_s_per_step"),
            0.0,
        )
        self.assertEqual(
            _literal(cfg, "inactive_ground_contact_yaw_moment_reward_scale"),
            0.0,
        )
        self.assertGreater(
            _literal(cfg, "inactive_ground_contact_yaw_moment_reference_nm"),
            0.0,
        )
        self.assertEqual(
            _literal(cfg, "max_joint_rated_torque_excess_reward_scale"), 0.0
        )
        self.assertEqual(
            _literal(cfg, "max_joint_rated_torque_excess_l1_reward_scale"), 0.0
        )
        self.assertEqual(_literal(cfg, "foot_slip_reward_scale"), -0.10)
        self.assertIs(_literal(cfg, "speed_conditioned_support_targets"), False)
        self.assertIs(
            _literal(cfg, "gate_longitudinal_reward_by_command"), False
        )
        self.assertEqual(_literal(cfg, "support_contact_target"), 3.0)
        for name in (
            "deck_stability_vertical_velocity_scale_mps",
            "deck_stability_roll_pitch_rate_scale_rad_s",
            "deck_stability_projected_gravity_xy_scale",
            "deck_stability_height_error_scale_m",
        ):
            value = _literal(cfg, name)
            self.assertTrue(math.isfinite(value))
            self.assertGreater(value, 0.0)

    def test_foot_sensors_track_filtered_ground_friction(self):
        self.assertIn("track_friction_forces: bool = False", CFG_SOURCE)
        self.assertIn(
            "track_friction_forces=track_friction_forces", CFG_SOURCE
        )
        self.assertIn("track_friction_forces=True", CFG_SOURCE)

    def test_reward_terms_are_logged_and_foot_slip_is_configurable(self):
        env = _class(ENV_TREE, "HexapodEnv")
        init_source = ast.get_source_segment(ENV_SOURCE, _method(env, "__init__"))
        reward_source = ast.get_source_segment(
            ENV_SOURCE, _method(env, "_get_rewards")
        )
        assert init_source is not None and reward_source is not None
        for key in (
            "deck_stability",
            "joint_torque_slew_l2",
            "inactive_yaw_rate_slew",
            "inactive_ground_contact_yaw_moment",
            "max_joint_rated_torque_excess_l2",
            "max_joint_rated_torque_excess_l1",
        ):
            self.assertIn(f'"{key}"', init_source)
            self.assertIn(f'"{key}"', reward_source)
        self.assertIn("self.cfg.foot_slip_reward_scale", reward_source)
        self.assertNotIn("foot_slip * -0.10", reward_source)

    def test_max_joint_torque_cost_is_strictly_opt_in(self):
        env = _class(ENV_TREE, "HexapodEnv")
        reward_source = ast.get_source_segment(
            ENV_SOURCE, _method(env, "_get_rewards")
        )
        assert reward_source is not None
        self.assertIn(
            "if self.cfg.max_joint_rated_torque_excess_reward_scale != 0.0:",
            reward_source,
        )
        self.assertIn(
            "if self.cfg.max_joint_rated_torque_excess_l1_reward_scale != 0.0:",
            reward_source,
        )
        self.assertIn(
            "rated_torque_nm=self.cfg.rated_torque_nm", reward_source
        )

    def test_inactive_yaw_slew_is_strictly_opt_in_and_command_masked(self):
        env = _class(ENV_TREE, "HexapodEnv")
        reward_source = ast.get_source_segment(
            ENV_SOURCE, _method(env, "_get_rewards")
        )
        assert reward_source is not None
        self.assertIn(
            "if self.cfg.inactive_yaw_rate_slew_reward_scale != 0.0:",
            reward_source,
        )
        self.assertIn("yaw_command_active,", reward_source)
        self.assertIn(
            "self._has_previous_command_frame_yaw_rate.copy_(~yaw_command_active)",
            reward_source,
        )

    def test_ground_contact_yaw_moment_is_opt_in_and_has_no_reset_history(self):
        env = _class(ENV_TREE, "HexapodEnv")
        init_source = ast.get_source_segment(ENV_SOURCE, _method(env, "__init__"))
        reward_source = ast.get_source_segment(
            ENV_SOURCE, _method(env, "_get_rewards")
        )
        reset_source = ast.get_source_segment(ENV_SOURCE, _method(env, "_reset_idx"))
        contact_source = ast.get_source_segment(
            ENV_SOURCE, _method(env, "_get_foot_contact_state")
        )
        assert init_source is not None and reward_source is not None
        assert reset_source is not None and contact_source is not None
        self.assertIn(
            "if self.cfg.inactive_ground_contact_yaw_moment_reward_scale != 0.0:",
            reward_source,
        )
        self.assertIn("foot_contact,", reward_source)
        self.assertIn(
            "commanded_planar_speed > self.cfg.moving_command_threshold_mps",
            reward_source,
        )
        self.assertIn("yaw_command_active,", reward_source)
        self.assertIn(
            "self._get_foot_contact_state(include_ground_wrench=True)",
            reward_source,
        )
        self.assertIn("self._get_foot_contact_state()", reward_source)
        self.assertIn("data.friction_forces_w", contact_source)
        self.assertIn(
            "ground_reaction_forces_w = normal_forces_w + friction_forces_w",
            contact_source,
        )
        self.assertIn(
            '"Ground-wrench rewards require track_friction_forces=True "',
            init_source,
        )
        self.assertIn('"ground_contact_yaw_moment_abs_nm"', init_source)
        self.assertIn('"distal_friction_xy_force_n"', init_source)
        self.assertIn('self._episode_sums["ground_contact_yaw_moment_abs_nm"]', reward_source)
        self.assertIn('self._episode_sums["distal_friction_xy_force_n"]', reward_source)
        self.assertNotIn("previous_ground_contact_yaw_moment", init_source)
        self.assertNotIn("previous_ground_contact_yaw_moment", reset_source)

    def test_longitudinal_gaussian_gate_is_independent_and_opt_in(self):
        env = _class(ENV_TREE, "HexapodEnv")
        reward_source = ast.get_source_segment(
            ENV_SOURCE, _method(env, "_get_rewards")
        )
        assert reward_source is not None
        self.assertIn(
            "if self.cfg.gate_longitudinal_reward_by_command:", reward_source
        )
        self.assertIn(
            "active_axis_gaussian_tracking_reward(\n"
            "                    self._commands[:, 0],\n"
            "                    root_lin_vel[:, 0],",
            reward_source,
        )
        self.assertIn(
            "if self.cfg.gate_axis_rewards_by_command:", reward_source
        )

    def test_deck_uses_world_vertical_velocity_but_legacy_z_cost_does_not(self):
        env = _class(ENV_TREE, "HexapodEnv")
        reward_source = ast.get_source_segment(
            ENV_SOURCE, _method(env, "_get_rewards")
        )
        assert reward_source is not None
        self.assertIn(
            "root_vertical_velocity_w = self._robot.data.root_lin_vel_w.torch[:, 2]",
            reward_source,
        )
        self.assertIn(
            "normalized_deck_stability_reward(\n            root_vertical_velocity_w,",
            reward_source,
        )
        self.assertIn(
            '"lin_vel_z_l2": torch.square(root_lin_vel[:, 2])', reward_source
        )
        self.assertIn(
            "commanded_planar_speed = torch.linalg.norm(self._commands[:, :2], dim=1)",
            reward_source,
        )
        self.assertIn("select_command_conditioned_nominal_height", reward_source)

    def test_reset_clears_torque_history_and_invalidates_first_delta(self):
        env = _class(ENV_TREE, "HexapodEnv")
        reset_source = ast.get_source_segment(ENV_SOURCE, _method(env, "_reset_idx"))
        assert reset_source is not None
        self.assertIn("self._previous_applied_torque[env_ids] = 0.0", reset_source)
        self.assertIn(
            "self._has_previous_applied_torque[env_ids] = False", reset_source
        )
        self.assertIn(
            "self._previous_command_frame_yaw_rate[env_ids] = 0.0", reset_source
        )
        self.assertIn(
            "self._has_previous_command_frame_yaw_rate[env_ids] = False",
            reset_source,
        )


if __name__ == "__main__":
    unittest.main()
