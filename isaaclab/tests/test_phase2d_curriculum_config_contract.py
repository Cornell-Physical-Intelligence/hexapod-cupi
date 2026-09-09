"""Static CPU contracts for the six-stage Stage2D lateral homotopy."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PACKAGE_ROOT = ROOT.parent / "packages" / "hexapod_env" / "hexapod_env"
PHASE2D_PATH = PACKAGE_ROOT / "phase2d_cfg.py"
PHASE2_PATH = PACKAGE_ROOT / "phase2_cfg.py"
REGISTER_PATH = PACKAGE_ROOT / "register.py"
INIT_PATH = PACKAGE_ROOT / "__init__.py"
EVALUATOR_PATH = ROOT / "evaluate_checkpoint.py"

PHASE2D_SOURCE = PHASE2D_PATH.read_text()
PHASE2_SOURCE = PHASE2_PATH.read_text()
REGISTER_SOURCE = REGISTER_PATH.read_text()
INIT_SOURCE = INIT_PATH.read_text()
EVALUATOR_SOURCE = EVALUATOR_PATH.read_text()
PHASE2D_TREE = ast.parse(PHASE2D_SOURCE)
PHASE2_TREE = ast.parse(PHASE2_SOURCE)
REGISTER_TREE = ast.parse(REGISTER_SOURCE)
EVALUATOR_TREE = ast.parse(EVALUATOR_SOURCE)


def _classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }


PHASE2D_CLASSES = _classes(PHASE2D_TREE)
PHASE2_CLASSES = _classes(PHASE2_TREE)


def _assignment(class_node: ast.ClassDef, name: str) -> ast.expr | None:
    for node in class_node.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return node.value
    return None


def _resolved_assignment(class_name: str, name: str) -> ast.expr:
    class_node = PHASE2D_CLASSES[class_name]
    value = _assignment(class_node, name)
    if value is not None:
        return value
    for base in class_node.bases:
        if isinstance(base, ast.Name) and base.id in PHASE2D_CLASSES:
            return _resolved_assignment(base.id, name)
    raise AssertionError(f"Missing inherited {class_name}.{name}")


def _resolved_literal(class_name: str, name: str):
    return ast.literal_eval(_resolved_assignment(class_name, name))


def _base_names(class_node: ast.ClassDef) -> list[str]:
    return [base.id for base in class_node.bases if isinstance(base, ast.Name)]


COMMAND_CLASSES = [f"HexapodStage2DC{stage}CommandCfg" for stage in range(6)]
ENV_CLASSES = [f"HexapodPhase2RecoveryStage2DC{stage}EnvCfg" for stage in range(6)]
RUNNER_CLASSES = [
    f"HexapodPhase2RecoveryStage2DC{stage}PPORunnerCfg" for stage in range(6)
]
TASK_CONSTANTS = [f"PHASE2_RECOVERY_STAGE2D_C{stage}_TASK_ID" for stage in range(6)]


class Stage2DCommandHomotopyContractTest(unittest.TestCase):
    def test_c0_to_c5_heading_and_threshold_schedule_is_exact(self):
        expected_forward = (
            (0.22, 0.26),
            (0.20, 0.24),
            (0.14, 0.18),
            (0.08, 0.12),
            (0.03, 0.06),
            (0.0, 0.0),
        )
        expected_lateral = (
            (0.02, 0.04),
            (0.04, 0.06),
            (0.06, 0.08),
            (0.08, 0.10),
            (0.08, 0.10),
            (0.08, 0.10),
        )
        expected_thresholds = (0.010, 0.018, 0.030, 0.040, 0.045, 0.050)
        for stage, (command_name, env_name) in enumerate(
            zip(COMMAND_CLASSES, ENV_CLASSES)
        ):
            with self.subTest(stage=stage):
                self.assertEqual(
                    _resolved_literal(command_name, "oblique_forward_range"),
                    expected_forward[stage],
                )
                self.assertEqual(
                    _resolved_literal(command_name, "oblique_lateral_abs_range"),
                    expected_lateral[stage],
                )
                self.assertEqual(
                    _resolved_literal(env_name, "axis_command_active_threshold"),
                    expected_thresholds[stage],
                )

    def test_every_stage_uses_full_episode_fixed_balanced_bucket_contract(self):
        for stage, command_name in enumerate(COMMAND_CLASSES):
            with self.subTest(stage=stage):
                self.assertEqual(
                    _resolved_literal(command_name, "sampling_mode"),
                    "stage2d_oblique_homotopy",
                )
                self.assertEqual(
                    _resolved_literal(command_name, "resampling_time_range_s"),
                    (30.0, 30.0),
                )
                self.assertEqual(
                    (
                        _resolved_literal(
                            command_name, "combined_probability"
                        ),
                        _resolved_literal(
                            command_name, "longitudinal_only_probability"
                        ),
                        _resolved_literal(command_name, "yaw_only_probability"),
                    ),
                    (0.70, 0.20, 0.10),
                )
                self.assertEqual(
                    _resolved_literal(command_name, "standing_probability"), 0.0
                )
                self.assertEqual(
                    _resolved_literal(command_name, "lateral_only_probability"),
                    0.0,
                )

    def test_forward_and_forward_yaw_anchors_are_safe_and_constant(self):
        for stage, command_name in enumerate(COMMAND_CLASSES):
            with self.subTest(stage=stage):
                self.assertEqual(
                    _resolved_literal(command_name, "forward_anchor_range"),
                    (0.20, 0.30),
                )
                self.assertEqual(
                    _resolved_literal(command_name, "yaw_anchor_forward_range"),
                    (0.20, 0.26),
                )
                self.assertEqual(
                    _resolved_literal(command_name, "yaw_anchor_abs_range"),
                    (0.20, 0.28),
                )


class Stage2DEnvironmentInheritanceContractTest(unittest.TestCase):
    def test_c0_directly_inherits_stage2c_then_stages_form_a_chain(self):
        self.assertEqual(
            _base_names(PHASE2D_CLASSES[ENV_CLASSES[0]]),
            ["HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg"],
        )
        for stage in range(1, 6):
            self.assertEqual(
                _base_names(PHASE2D_CLASSES[ENV_CLASSES[stage]]),
                [ENV_CLASSES[stage - 1]],
            )

    def test_acquisition_objective_and_processed_target_limit_are_exact(self):
        expected = {
            "gate_axis_rewards_by_command": True,
            "lin_vel_x_reward_scale": 4.0,
            "lin_vel_y_reward_scale": 0.0,
            "lateral_signed_progress_reward_scale": 8.0,
            "navigation_lateral_velocity_ema_tau_s": 0.22,
            "lateral_normalized_error_penalty_scale": 2.0,
            "lateral_normalized_error_cap": 2.0,
            "inactive_lateral_velocity_reward_scale": -0.50,
            "processed_joint_target_slew_limit_rad_per_20ms": 0.04,
        }
        for stage, env_name in enumerate(ENV_CLASSES):
            with self.subTest(stage=stage):
                for field, value in expected.items():
                    self.assertEqual(_resolved_literal(env_name, field), value)
                velocity_command = _assignment(
                    PHASE2D_CLASSES[env_name], "velocity_command"
                )
                self.assertIsInstance(velocity_command, ast.Call)
                assert isinstance(velocity_command, ast.Call)
                self.assertIsInstance(velocity_command.func, ast.Name)
                assert isinstance(velocity_command.func, ast.Name)
                self.assertEqual(
                    velocity_command.func.id,
                    f"HexapodStage2DC{stage}CommandCfg",
                )

    def test_stage2d_cannot_shadow_stage2c_stance_stability_or_rs05_contract(self):
        protected_fields = {
            "action_scale",
            "robot",
            "nominal_height_m",
            "deck_stability_reward_scale",
            "deck_stability_vertical_velocity_scale_mps",
            "deck_stability_roll_pitch_rate_scale_rad_s",
            "deck_stability_projected_gravity_xy_scale",
            "deck_stability_height_error_scale_m",
            "z_vel_reward_scale",
            "ang_vel_reward_scale",
            "flat_orientation_reward_scale",
            "base_height_reward_scale",
            "speed_conditioned_support_targets",
            "support_contact_target_low_speed",
            "support_contact_target_medium_speed",
            "support_contact_target_high_speed",
            "support_shortfall_reward_scale",
            "foot_slip_reward_scale",
            "joint_torque_slew_reward_scale",
            "rated_torque_excess_reward_scale",
            "torque_saturation_reward_scale",
            "fall_penalty",
            "terminate_on_computed_torque_demand_nm",
            "terminate_on_computed_torque_demand_duration_s",
            "torque_demand_termination_grace_s",
        }
        for env_name in ENV_CLASSES:
            assigned = {
                target.id
                for node in PHASE2D_CLASSES[env_name].body
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                for target in (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                )
                if isinstance(target, ast.Name)
            }
            self.assertFalse(assigned & protected_fields)

        stage2c = PHASE2_CLASSES[
            "HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg"
        ]
        self.assertEqual(ast.literal_eval(_assignment(stage2c, "nominal_height_m")), 0.181)
        self.assertEqual(
            ast.literal_eval(_assignment(stage2c, "deck_stability_reward_scale")),
            3.0,
        )
        self.assertEqual(
            ast.literal_eval(
                _assignment(stage2c, "terminate_on_computed_torque_demand_nm")
            ),
            5.5,
        )
        stance = _assignment(stage2c, "robot")
        assert stance is not None
        stance_source = ast.get_source_segment(PHASE2_SOURCE, stance)
        assert stance_source is not None
        self.assertIn("pos=(0.0, 0.0, 0.185)", stance_source)
        self.assertIn("{name: 0.60 for name in FEMUR_JOINTS}", stance_source)
        self.assertIn("{name: 2.2335 for name in TIBIA_JOINTS}", stance_source)


class Stage2DPPOContractTest(unittest.TestCase):
    def test_each_runner_has_distinct_experiment_and_bounded_schedule(self):
        experiments = []
        for stage, runner_name in enumerate(RUNNER_CLASSES):
            with self.subTest(stage=stage):
                self.assertEqual(
                    _resolved_literal(runner_name, "max_iterations"),
                    40 if stage == 5 else 25,
                )
                self.assertEqual(_resolved_literal(runner_name, "save_interval"), 5)
                experiments.append(
                    _resolved_literal(runner_name, "experiment_name")
                )
        self.assertEqual(len(set(experiments)), 6)

        actor = _resolved_assignment(RUNNER_CLASSES[0], "actor")
        self.assertIsInstance(actor, ast.Call)
        assert isinstance(actor, ast.Call)
        distribution = next(
            keyword.value
            for keyword in actor.keywords
            if keyword.arg == "distribution_cfg"
        )
        self.assertIsInstance(distribution, ast.Call)
        assert isinstance(distribution, ast.Call)
        distribution_keywords = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in distribution.keywords
        }
        self.assertEqual(distribution_keywords["init_std"], 0.14)

        algorithm = _resolved_assignment(RUNNER_CLASSES[0], "algorithm")
        self.assertIsInstance(algorithm, ast.Call)
        assert isinstance(algorithm, ast.Call)
        algorithm_keywords = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in algorithm.keywords
        }
        self.assertEqual(algorithm_keywords["schedule"], "fixed")
        self.assertEqual(algorithm_keywords["learning_rate"], 6.0e-5)
        self.assertEqual(algorithm_keywords["clip_param"], 0.10)
        self.assertEqual(algorithm_keywords["entropy_coef"], 3.0e-4)
        self.assertEqual(algorithm_keywords["max_grad_norm"], 0.5)


class Stage2DRegistrationAndEvaluatorContractTest(unittest.TestCase):
    def test_six_distinct_task_ids_register_phase2d_config_pairs(self):
        constants = {
            target.id: ast.literal_eval(node.value)
            for node in REGISTER_TREE.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id in TASK_CONSTANTS
        }
        self.assertEqual(set(constants), set(TASK_CONSTANTS))
        self.assertEqual(len(set(constants.values())), 6)
        for stage, constant in enumerate(TASK_CONSTANTS):
            self.assertIn(f"Stage2D-Homotopy-C{stage}", constants[constant])

        register_envs = next(
            node
            for node in REGISTER_TREE.body
            if isinstance(node, ast.FunctionDef) and node.name == "register_envs"
        )
        registrations = next(
            node.value
            for node in register_envs.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "stage2d_registrations"
                for target in node.targets
            )
        )
        self.assertIsInstance(registrations, ast.Tuple)
        assert isinstance(registrations, ast.Tuple)
        self.assertEqual(len(registrations.elts), 6)
        for stage, row in enumerate(registrations.elts):
            self.assertIsInstance(row, ast.Tuple)
            assert isinstance(row, ast.Tuple)
            self.assertEqual(ast.unparse(row.elts[0]), TASK_CONSTANTS[stage])
            self.assertEqual(
                ast.literal_eval(row.elts[1]),
                f"HexapodPhase2RecoveryStage2DC{stage}EnvCfg",
            )
            self.assertEqual(
                ast.literal_eval(row.elts[2]),
                f"HexapodPhase2RecoveryStage2DC{stage}PPORunnerCfg",
            )
        self.assertIn("hexapod_rl.phase2d_cfg", REGISTER_SOURCE)

    def test_package_exports_and_evaluator_map_every_task(self):
        for constant in TASK_CONSTANTS:
            self.assertIn(constant, INIT_SOURCE)

        task_configs = next(
            node.value
            for node in EVALUATOR_TREE.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "TASK_CONFIGS"
                for target in node.targets
            )
        )
        self.assertIsInstance(task_configs, ast.Dict)
        assert isinstance(task_configs, ast.Dict)
        mapping = {
            ast.unparse(key): tuple(ast.unparse(element) for element in value.elts)
            for key, value in zip(task_configs.keys, task_configs.values)
            if key is not None and isinstance(value, ast.Tuple)
        }
        for stage, constant in enumerate(TASK_CONSTANTS):
            self.assertEqual(
                mapping[constant],
                (
                    f"HexapodPhase2RecoveryStage2DC{stage}EnvCfg",
                    f"HexapodPhase2RecoveryStage2DC{stage}PPORunnerCfg",
                ),
            )
        self.assertIn("from hexapod_rl.phase2d_cfg import", EVALUATOR_SOURCE)


if __name__ == "__main__":
    unittest.main()
