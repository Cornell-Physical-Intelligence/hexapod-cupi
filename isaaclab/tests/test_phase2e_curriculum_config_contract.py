"""AST contracts for the three-stage Stage2E joystick curriculum."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
PHASE2E_PATH = ROOT / "hexapod_rl" / "phase2e_cfg.py"
PHASE2D_PATH = ROOT / "hexapod_rl" / "phase2d_cfg.py"
REGISTER_PATH = ROOT / "hexapod_rl" / "register.py"
INIT_PATH = ROOT / "hexapod_rl" / "__init__.py"
EVALUATOR_PATH = ROOT / "evaluate_checkpoint.py"

PHASE2E_SOURCE = PHASE2E_PATH.read_text(encoding="utf-8")
PHASE2D_SOURCE = PHASE2D_PATH.read_text(encoding="utf-8")
REGISTER_SOURCE = REGISTER_PATH.read_text(encoding="utf-8")
INIT_SOURCE = INIT_PATH.read_text(encoding="utf-8")
EVALUATOR_SOURCE = EVALUATOR_PATH.read_text(encoding="utf-8")

PHASE2E_TREE = ast.parse(PHASE2E_SOURCE)
REGISTER_TREE = ast.parse(REGISTER_SOURCE)

CLASSES = {
    node.name: node
    for node in PHASE2E_TREE.body
    if isinstance(node, ast.ClassDef)
}
COMMAND_CLASSES = tuple(f"HexapodStage2EE{stage}CommandCfg" for stage in range(3))
ENV_CLASSES = tuple(
    f"HexapodPhase2RecoveryStage2EE{stage}EnvCfg" for stage in range(3)
)
RUNNER_CLASSES = tuple(
    f"HexapodPhase2RecoveryStage2EE{stage}PPORunnerCfg" for stage in range(3)
)
TASK_CONSTANTS = tuple(
    f"PHASE2_RECOVERY_STAGE2E_E{stage}_TASK_ID" for stage in range(3)
)


def _bases(name: str) -> list[str]:
    return [ast.unparse(base) for base in CLASSES[name].bases]


def _assignment(class_name: str, field: str):
    for node in CLASSES[class_name].body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == field
            for target in node.targets
        ):
            return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == field
        ):
            return node.value
    return None


def _resolved_assignment(class_name: str, field: str):
    current = class_name
    seen: set[str] = set()
    while current in CLASSES and current not in seen:
        seen.add(current)
        value = _assignment(current, field)
        if value is not None:
            return value
        bases = _bases(current)
        current = bases[0] if bases else ""
    raise AssertionError(f"missing {class_name}.{field}")


def _literal(class_name: str, field: str):
    return ast.literal_eval(_resolved_assignment(class_name, field))


class Stage2ECommandContractTest(unittest.TestCase):
    def test_shortest_safe_reverse_and_transition_schedule_is_exact(self):
        expected = (
            {
                "counts": (4, 8, 4, 4, 4, 12, 4),
                "hold": (8.0, 8.0),
                "reverse": (0.02, 0.05),
                "yaw_forward": (0.14, 0.20),
                "x": (-0.05, 0.30),
            },
            {
                "counts": (4, 8, 4, 4, 4, 8, 8),
                "hold": (5.0, 5.0),
                "reverse": (0.04, 0.10),
                "yaw_forward": (0.05, 0.10),
                "x": (-0.10, 0.30),
            },
            {
                "counts": (4, 8, 4, 4, 4, 8, 8),
                "hold": (3.0, 3.0),
                "reverse": (0.08, 0.20),
                "yaw_forward": (0.0, 0.0),
                "x": (-0.20, 0.30),
            },
        )
        for stage, values in enumerate(expected):
            name = COMMAND_CLASSES[stage]
            with self.subTest(stage=stage):
                self.assertEqual(_literal(name, "sampling_mode"), "stage2e_joystick_transitions")
                self.assertEqual(_literal(name, "bucket_stride"), 13)
                self.assertEqual(_literal(name, "bucket_counts"), values["counts"])
                self.assertEqual(sum(values["counts"]), 40)
                self.assertEqual(_literal(name, "resampling_time_range_s"), values["hold"])
                self.assertEqual(_literal(name, "joystick_reverse_abs_range"), values["reverse"])
                self.assertEqual(_literal(name, "yaw_anchor_forward_range"), values["yaw_forward"])
                self.assertEqual(_literal(name, "lin_vel_x_range"), values["x"])
                self.assertEqual(_literal(name, "lin_vel_y_range"), (-0.10, 0.10))
                self.assertEqual(_literal(name, "ang_vel_z_range"), (-0.28, 0.28))

    def test_c5_translation_and_yaw_magnitude_anchors_survive_every_stage(self):
        anchor_contract = {
            "lateral_anchor_abs_range": (0.08, 0.10),
            "forward_anchor_range": (0.20, 0.30),
            "yaw_anchor_abs_range": (0.20, 0.28),
        }
        for stage, name in enumerate(COMMAND_CLASSES):
            with self.subTest(stage=stage):
                for field, value in anchor_contract.items():
                    self.assertEqual(_literal(name, field), value)
                probabilities = tuple(
                    _literal(name, field)
                    for field in (
                        "standing_probability",
                        "longitudinal_only_probability",
                        "lateral_only_probability",
                        "yaw_only_probability",
                        "combined_probability",
                    )
                )
                self.assertAlmostEqual(sum(probabilities), 1.0)
                self.assertEqual(probabilities[0], 0.10)


class Stage2EEnvironmentContractTest(unittest.TestCase):
    def test_e0_directly_inherits_c5_and_then_forms_a_three_stage_chain(self):
        self.assertEqual(
            _bases(ENV_CLASSES[0]),
            ["HexapodPhase2RecoveryStage2DC5EnvCfg"],
        )
        self.assertEqual(_bases(ENV_CLASSES[1]), [ENV_CLASSES[0]])
        self.assertEqual(_bases(ENV_CLASSES[2]), [ENV_CLASSES[1]])
        for stage, name in enumerate(ENV_CLASSES):
            command = _resolved_assignment(name, "velocity_command")
            self.assertIsInstance(command, ast.Call)
            assert isinstance(command, ast.Call)
            self.assertEqual(ast.unparse(command.func), COMMAND_CLASSES[stage])

    def test_reverse_shaping_is_explicit_and_small_commands_are_active(self):
        expected = {
            "axis_command_active_threshold": 0.015,
            "longitudinal_signed_progress_reward_scale": 6.0,
            "longitudinal_normalized_error_penalty_scale": 2.0,
            "longitudinal_normalized_error_cap": 2.0,
            "yaw_signed_progress_reward_scale": 4.0,
        }
        for stage, name in enumerate(ENV_CLASSES):
            with self.subTest(stage=stage):
                for field, value in expected.items():
                    self.assertEqual(_literal(name, field), value)

    def test_stage2e_does_not_shadow_lower_stance_stability_or_rs05_contract(self):
        protected = {
            "action_scale",
            "robot",
            "nominal_height_m",
            "stand_nominal_height_m",
            "processed_joint_target_slew_limit_rad_per_20ms",
            "deck_stability_reward_scale",
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
            "terminate_on_computed_torque_demand_nm",
            "terminate_on_computed_torque_demand_duration_s",
            "torque_demand_termination_grace_s",
        }
        for name in ENV_CLASSES:
            assigned = {
                target.id
                for node in CLASSES[name].body
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if isinstance(target, ast.Name)
            }
            self.assertFalse(assigned & protected)
        for required in (
            "pos=(0.0, 0.0, 0.185)",
            "nominal_height_m = 0.181",
            "deck_stability_reward_scale = 3.0",
            "terminate_on_computed_torque_demand_nm = 5.5",
            "processed_joint_target_slew_limit_rad_per_20ms = 0.04",
        ):
            self.assertTrue(
                required in PHASE2D_SOURCE
                or required
                in (ROOT / "hexapod_rl" / "phase2_cfg.py").read_text(encoding="utf-8")
            )


class Stage2ERunnerAndRegistrationContractTest(unittest.TestCase):
    def test_ppo_is_conservative_dense_and_stage_bounded(self):
        expected_iterations = (30, 35, 45)
        experiments = []
        for stage, name in enumerate(RUNNER_CLASSES):
            with self.subTest(stage=stage):
                self.assertEqual(_literal(name, "max_iterations"), expected_iterations[stage])
                self.assertEqual(_literal(name, "save_interval"), 5)
                experiment = _literal(name, "experiment_name")
                self.assertIn(f"stage2e_e{stage}", experiment)
                experiments.append(experiment)
        self.assertEqual(len(set(experiments)), 3)

        actor = _resolved_assignment(RUNNER_CLASSES[0], "actor")
        assert isinstance(actor, ast.Call)
        distribution = next(
            keyword.value for keyword in actor.keywords if keyword.arg == "distribution_cfg"
        )
        assert isinstance(distribution, ast.Call)
        self.assertEqual(
            next(
                ast.literal_eval(keyword.value)
                for keyword in distribution.keywords
                if keyword.arg == "init_std"
            ),
            0.12,
        )
        algorithm = _resolved_assignment(RUNNER_CLASSES[0], "algorithm")
        assert isinstance(algorithm, ast.Call)
        values = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in algorithm.keywords
        }
        self.assertEqual(values["schedule"], "fixed")
        self.assertEqual(values["learning_rate"], 4.0e-5)
        self.assertEqual(values["clip_param"], 0.08)
        self.assertEqual(values["entropy_coef"], 1.0e-4)
        self.assertEqual(values["max_grad_norm"], 0.5)

    def test_three_tasks_are_registered_exported_and_evaluator_compatible(self):
        constants = {
            target.id: ast.literal_eval(node.value)
            for node in REGISTER_TREE.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id in TASK_CONSTANTS
        }
        self.assertEqual(set(constants), set(TASK_CONSTANTS))
        self.assertEqual(len(set(constants.values())), 3)
        for stage, constant in enumerate(TASK_CONSTANTS):
            self.assertIn(f"Stage2E-Joystick-E{stage}", constants[constant])
            self.assertIn(constant, INIT_SOURCE)
            self.assertIn(constant, EVALUATOR_SOURCE)
            self.assertIn(ENV_CLASSES[stage], EVALUATOR_SOURCE)
            self.assertIn(RUNNER_CLASSES[stage], EVALUATOR_SOURCE)

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
                isinstance(target, ast.Name) and target.id == "stage2e_registrations"
                for target in node.targets
            )
        )
        assert isinstance(registrations, ast.Tuple)
        self.assertEqual(len(registrations.elts), 3)


if __name__ == "__main__":
    unittest.main()
