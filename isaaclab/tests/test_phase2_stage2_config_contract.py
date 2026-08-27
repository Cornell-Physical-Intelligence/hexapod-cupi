"""Static CPU contract tests for the Isaac-only Stage 2 recovery configs."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PACKAGE_ROOT = ROOT.parent / "packages" / "hexapod_env" / "hexapod_env"
PHASE2_CFG_PATH = PACKAGE_ROOT / "phase2_cfg.py"
REGISTER_PATH = PACKAGE_ROOT / "register.py"
EVALUATOR_PATH = ROOT / "evaluate_checkpoint.py"
PHASE2_TREE = ast.parse(PHASE2_CFG_PATH.read_text())


def _class(name: str) -> ast.ClassDef:
    for node in PHASE2_TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Missing class {name}")


def _assignment(class_node: ast.ClassDef, name: str) -> ast.expr:
    for node in class_node.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                assert node.value is not None
                return node.value
    raise AssertionError(f"Missing {class_node.name}.{name}")


def _literal(class_node: ast.ClassDef, name: str):
    return ast.literal_eval(_assignment(class_node, name))


class Phase2Stage2ConfigContractTest(unittest.TestCase):
    def test_command_buckets_and_reward_forcing_contract(self):
        command = _class("HexapodStage2RecoveryCommandCfg")
        self.assertEqual(_literal(command, "sampling_mode"), "stage2_recovery")
        self.assertEqual(
            (
                _literal(command, "longitudinal_only_probability"),
                _literal(command, "yaw_only_probability"),
                _literal(command, "lateral_only_probability"),
                _literal(command, "combined_probability"),
            ),
            (0.40, 0.30, 0.30, 0.0),
        )
        self.assertEqual(_literal(command, "forward_range"), (0.22, 0.34))
        self.assertEqual(_literal(command, "moving_forward_range"), (0.20, 0.30))
        self.assertEqual(_literal(command, "lateral_abs_range"), (0.08, 0.14))
        self.assertEqual(_literal(command, "yaw_abs_range"), (0.18, 0.30))
        self.assertEqual(_literal(command, "resampling_time_range_s"), (30.0, 30.0))

        env = _class("HexapodPhase2RecoveryStage2EnvCfg")
        self.assertEqual(
            [base.id for base in env.bases if isinstance(base, ast.Name)],
            ["HexapodPhase2RecoveryStage1EnvCfg"],
        )
        self.assertIs(_literal(env, "gate_axis_rewards_by_command"), True)
        self.assertEqual(_literal(env, "lin_vel_y_reward_scale"), 3.0)
        self.assertEqual(_literal(env, "yaw_rate_reward_scale"), 3.0)
        self.assertEqual(_literal(env, "lateral_signed_progress_reward_scale"), 2.0)
        self.assertEqual(_literal(env, "yaw_signed_progress_reward_scale"), 2.0)
        self.assertLess(_literal(env, "inactive_lateral_velocity_reward_scale"), 0.0)
        self.assertLess(_literal(env, "inactive_yaw_rate_reward_scale"), 0.0)

        # Stage 2 tightens stability while inheriting the tested Stage 1 RS05
        # excess/saturation and fall limits rather than silently relaxing them.
        stage1 = _class("HexapodPhase2RecoveryStage1EnvCfg")
        self.assertEqual(_literal(stage1, "rated_torque_excess_reward_scale"), -0.06)
        self.assertEqual(_literal(stage1, "torque_saturation_reward_scale"), -0.20)
        self.assertEqual(_literal(stage1, "fall_penalty"), -5.0)
        self.assertEqual(_literal(env, "z_vel_reward_scale"), -6.0)
        self.assertEqual(_literal(env, "ang_vel_reward_scale"), -0.75)
        self.assertEqual(_literal(env, "flat_orientation_reward_scale"), -6.0)
        self.assertEqual(_literal(env, "base_height_reward_scale"), -60.0)
        self.assertEqual(_literal(env, "nominal_height_m"), 0.189)

        robot = _assignment(env, "robot")
        self.assertIsInstance(robot, ast.Call)
        replace_calls = [
            node
            for node in ast.walk(robot)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "replace"
        ]
        self.assertEqual(len(replace_calls), 2)
        stance_source = ast.get_source_segment(PHASE2_CFG_PATH.read_text(), robot)
        assert stance_source is not None
        self.assertIn("pos=(0.0, 0.0, 0.198)", stance_source)
        self.assertIn("{name: 0.50 for name in FEMUR_JOINTS}", stance_source)
        self.assertIn("{name: 2.170 for name in TIBIA_JOINTS}", stance_source)
        self.assertNotIn(".update(", stance_source)

    def test_ppo_uses_fixed_noncollapsing_learning_rate_and_bounded_update(self):
        runner = _class("HexapodPhase2RecoveryStage2PPORunnerCfg")
        self.assertEqual(_literal(runner, "max_iterations"), 150)
        self.assertEqual(_literal(runner, "save_interval"), 25)
        self.assertEqual(
            _literal(runner, "experiment_name"),
            "hexapod_robstride_phase2_recovery_stage2_direct",
        )
        algorithm = _assignment(runner, "algorithm")
        self.assertIsInstance(algorithm, ast.Call)
        keywords = {keyword.arg: ast.literal_eval(keyword.value) for keyword in algorithm.keywords}
        self.assertEqual(keywords["schedule"], "fixed")
        self.assertEqual(keywords["learning_rate"], 1.0e-4)
        self.assertEqual(keywords["clip_param"], 0.15)
        self.assertEqual(keywords["entropy_coef"], 3.0e-4)
        self.assertEqual(keywords["num_learning_epochs"], 4)
        self.assertEqual(keywords["num_mini_batches"], 8)
        self.assertEqual(keywords["max_grad_norm"], 0.75)

    def test_task_is_registered_and_evaluator_compatible(self):
        register_source = REGISTER_PATH.read_text()
        evaluator_source = EVALUATOR_PATH.read_text()
        task_id = "Isaac-Velocity-Omni-Recovery-Stage2-Hexapod-RobStride-Direct-v0"
        self.assertIn(task_id, register_source)
        self.assertIn("HexapodPhase2RecoveryStage2EnvCfg", register_source)
        self.assertIn("HexapodPhase2RecoveryStage2PPORunnerCfg", register_source)
        self.assertIn("PHASE2_RECOVERY_STAGE2_TASK_ID", evaluator_source)
        self.assertIn("HexapodPhase2RecoveryStage2EnvCfg", evaluator_source)
        self.assertIn("HexapodPhase2RecoveryStage2PPORunnerCfg", evaluator_source)

    def test_stage2b_lateral_branch_is_conservative_and_registered(self):
        command = _class("HexapodStage2BLateralAcquisitionCommandCfg")
        self.assertEqual(
            _literal(command, "sampling_mode"), "stage2b_lateral_acquisition"
        )
        self.assertEqual(
            (
                _literal(command, "longitudinal_only_probability"),
                _literal(command, "lateral_only_probability"),
                _literal(command, "combined_probability"),
                _literal(command, "yaw_only_probability"),
            ),
            (0.40, 0.40, 0.10, 0.10),
        )
        self.assertEqual(_literal(command, "lateral_only_abs_range"), (0.08, 0.12))
        self.assertEqual(_literal(command, "combined_lateral_abs_range"), (0.06, 0.10))

        env = _class("HexapodPhase2RecoveryStage2BLateralEnvCfg")
        self.assertEqual(
            [base.id for base in env.bases if isinstance(base, ast.Name)],
            ["HexapodPhase2RecoveryStage2EnvCfg"],
        )
        self.assertEqual(_literal(env, "lin_vel_y_reward_scale"), 6.0)
        self.assertEqual(_literal(env, "lateral_signed_progress_reward_scale"), 5.0)
        self.assertEqual(_literal(env, "lin_vel_y_tracking_std_mps"), 0.18)
        self.assertEqual(_literal(env, "terminate_on_computed_torque_demand_nm"), 5.5)
        self.assertEqual(
            _literal(env, "terminate_on_computed_torque_demand_duration_s"), 0.10
        )
        self.assertEqual(_literal(env, "torque_demand_termination_grace_s"), 0.50)
        self.assertLessEqual(_literal(env, "rated_torque_excess_reward_scale"), -0.12)
        self.assertLessEqual(_literal(env, "torque_saturation_reward_scale"), -0.35)
        self.assertLessEqual(_literal(env, "fall_penalty"), -8.0)

        runner = _class("HexapodPhase2RecoveryStage2BLateralPPORunnerCfg")
        self.assertEqual(_literal(runner, "max_iterations"), 100)
        self.assertEqual(_literal(runner, "save_interval"), 10)
        algorithm = _assignment(runner, "algorithm")
        keywords = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in algorithm.keywords
        }
        self.assertEqual(keywords["schedule"], "fixed")
        self.assertEqual(keywords["learning_rate"], 4.0e-5)
        self.assertEqual(keywords["clip_param"], 0.08)
        self.assertEqual(keywords["max_grad_norm"], 0.5)

        register_source = REGISTER_PATH.read_text()
        self.assertIn(
            "Isaac-Velocity-Omni-Recovery-Stage2B-Lateral-"
            "Hexapod-RobStride-Direct-v0",
            register_source,
        )
        self.assertIn("HexapodPhase2RecoveryStage2BLateralEnvCfg", register_source)
        self.assertIn(
            "HexapodPhase2RecoveryStage2BLateralPPORunnerCfg", register_source
        )
        evaluator_source = EVALUATOR_PATH.read_text()
        self.assertIn("PHASE2_RECOVERY_STAGE2B_LATERAL_TASK_ID", evaluator_source)
        self.assertIn("HexapodPhase2RecoveryStage2BLateralEnvCfg", evaluator_source)
        self.assertIn(
            "HexapodPhase2RecoveryStage2BLateralPPORunnerCfg", evaluator_source
        )

    def test_stage2c_lower_stance_stability_branch_is_bounded_and_registered(self):
        command = _class("HexapodStage2CStabilizedForwardCommandCfg")
        self.assertEqual(_literal(command, "sampling_mode"), "mixture")
        self.assertEqual(_literal(command, "lin_vel_x_range"), (0.16, 0.32))
        self.assertEqual(_literal(command, "lin_vel_y_range"), (0.0, 0.0))
        self.assertEqual(_literal(command, "ang_vel_z_range"), (0.0, 0.0))
        self.assertEqual(_literal(command, "standing_probability"), 0.20)
        self.assertEqual(_literal(command, "longitudinal_only_probability"), 0.80)

        env = _class("HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg")
        self.assertEqual(
            [base.id for base in env.bases if isinstance(base, ast.Name)],
            ["HexapodPhase2RecoveryStage2EnvCfg"],
        )
        self.assertEqual(_literal(env, "nominal_height_m"), 0.181)
        self.assertEqual(_literal(env, "stand_nominal_height_m"), 0.177)
        self.assertEqual(
            _literal(env, "processed_joint_target_slew_limit_rad_per_20ms"),
            0.04,
        )
        self.assertIs(
            _literal(env, "gate_longitudinal_reward_by_command"), True
        )
        self.assertGreater(_literal(env, "deck_stability_reward_scale"), 0.0)
        for scale_name in (
            "deck_stability_vertical_velocity_scale_mps",
            "deck_stability_roll_pitch_rate_scale_rad_s",
            "deck_stability_projected_gravity_xy_scale",
            "deck_stability_height_error_scale_m",
        ):
            self.assertGreater(_literal(env, scale_name), 0.0)
        self.assertIs(_literal(env, "speed_conditioned_support_targets"), True)
        self.assertEqual(_literal(env, "support_contact_target_low_speed"), 5.0)
        self.assertEqual(_literal(env, "support_contact_target_medium_speed"), 4.0)
        self.assertEqual(_literal(env, "support_contact_target_high_speed"), 3.0)
        self.assertLess(_literal(env, "joint_torque_slew_reward_scale"), 0.0)
        self.assertLess(_literal(env, "foot_slip_reward_scale"), -0.10)
        self.assertEqual(_literal(env, "terminate_on_computed_torque_demand_nm"), 5.5)

        robot = _assignment(env, "robot")
        stance_source = ast.get_source_segment(PHASE2_CFG_PATH.read_text(), robot)
        assert stance_source is not None
        self.assertIn("pos=(0.0, 0.0, 0.185)", stance_source)
        self.assertIn("{name: 0.60 for name in FEMUR_JOINTS}", stance_source)
        self.assertIn("{name: 2.2335 for name in TIBIA_JOINTS}", stance_source)

        runner = _class(
            "HexapodPhase2RecoveryStage2CStabilizedForwardPPORunnerCfg"
        )
        self.assertEqual(_literal(runner, "max_iterations"), 120)
        self.assertEqual(_literal(runner, "save_interval"), 10)
        algorithm = _assignment(runner, "algorithm")
        keywords = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in algorithm.keywords
        }
        self.assertEqual(keywords["schedule"], "fixed")
        self.assertEqual(keywords["learning_rate"], 4.0e-5)
        self.assertEqual(keywords["clip_param"], 0.08)
        self.assertEqual(keywords["max_grad_norm"], 0.5)

        register_source = REGISTER_PATH.read_text()
        self.assertIn(
            "Isaac-Velocity-Omni-Recovery-Stage2C-Stabilized-Forward-",
            register_source,
        )
        self.assertIn(
            "HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg",
            register_source,
        )
        evaluator_source = EVALUATOR_PATH.read_text()
        self.assertIn(
            "PHASE2_RECOVERY_STAGE2C_STABILIZED_FORWARD_TASK_ID",
            evaluator_source,
        )


if __name__ == "__main__":
    unittest.main()
