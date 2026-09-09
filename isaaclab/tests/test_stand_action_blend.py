"""CPU/static tests for command-conditioned stand action blending."""

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
PHASE2_CFG_PATH = PACKAGE_ROOT / "phase2_cfg.py"
REWARDS_MODULES = tuple(
    (path, ast.parse(path.read_text(encoding="utf-8")))
    for path in sorted((PACKAGE_ROOT / "rewards").glob("*.py"))
    if path.name != "__init__.py"
)
ENV_SOURCE = ENV_PATH.read_text(encoding="utf-8")
CFG_SOURCE = CFG_PATH.read_text(encoding="utf-8")
PHASE2_CFG_SOURCE = PHASE2_CFG_PATH.read_text(encoding="utf-8")
ENV_TREE = ast.parse(ENV_SOURCE)
CFG_TREE = ast.parse(CFG_SOURCE)
PHASE2_CFG_TREE = ast.parse(PHASE2_CFG_SOURCE)


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


apply_stand_scale = _function("apply_command_conditioned_stand_action_scale")


class StandActionBlendPrimitiveTest(unittest.TestCase):
    def test_zero_command_at_scale_zero_requests_exact_nominal_action(self):
        actions = torch.tensor([[1.0, -0.4, 0.25], [-0.2, 0.7, -1.0]])
        actual = apply_stand_scale(
            actions,
            torch.zeros(2, 3),
            stand_action_scale=0.0,
            command_active_threshold=0.01,
        )
        torch.testing.assert_close(actual, torch.zeros_like(actions))

    def test_active_x_y_and_yaw_rows_are_bit_exact_pass_throughs(self):
        actions = torch.tensor(
            [[0.1, -0.2], [0.3, -0.4], [0.5, -0.6]], dtype=torch.float32
        )
        commands = torch.tensor(
            [[0.0101, 0.0, 0.0], [0.0, -0.0101, 0.0], [0.0, 0.0, 0.0101]],
            dtype=torch.float32,
        )
        actual = apply_stand_scale(
            actions,
            commands,
            stand_action_scale=0.0,
            command_active_threshold=0.01,
        )
        self.assertTrue(torch.equal(actual, actions))

    def test_threshold_boundary_is_inactive_but_just_above_is_active(self):
        actions = torch.tensor([[0.8, -0.4], [0.8, -0.4]])
        commands = torch.tensor(
            [[0.01, -0.01, 0.01], [0.01001, 0.0, 0.0]]
        )
        actual = apply_stand_scale(
            actions,
            commands,
            stand_action_scale=0.0,
            command_active_threshold=0.01,
        )
        torch.testing.assert_close(actual[0], torch.zeros(2))
        self.assertTrue(torch.equal(actual[1], actions[1]))

    def test_fractional_scale_interpolates_only_inactive_rows(self):
        actions = torch.tensor([[0.8, -0.4], [0.8, -0.4]])
        commands = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.02, 0.0]])
        actual = apply_stand_scale(
            actions,
            commands,
            stand_action_scale=0.25,
            command_active_threshold=0.01,
        )
        torch.testing.assert_close(actual[0], torch.tensor([0.2, -0.1]))
        self.assertTrue(torch.equal(actual[1], actions[1]))

    def test_legacy_scale_one_is_the_exact_input_object_and_dormant(self):
        actions = torch.tensor([[float("nan"), 0.25]])
        malformed_dormant_commands = torch.zeros(7)
        actual = apply_stand_scale(
            actions,
            malformed_dormant_commands,
            stand_action_scale=1.0,
            command_active_threshold=float("nan"),
        )
        self.assertIs(actual, actions)

    def test_enabled_blend_rejects_invalid_scale_threshold_and_shapes(self):
        actions = torch.zeros(2, 4)
        commands = torch.zeros(2, 3)
        for invalid in (-0.01, 1.01, float("inf"), float("nan")):
            with self.subTest(scale=invalid), self.assertRaisesRegex(
                ValueError, "stand_action_scale"
            ):
                apply_stand_scale(
                    actions,
                    commands,
                    stand_action_scale=invalid,
                    command_active_threshold=0.01,
                )
        for invalid in (-0.01, float("inf"), float("nan")):
            with self.subTest(threshold=invalid), self.assertRaisesRegex(
                ValueError, "command_active_threshold"
            ):
                apply_stand_scale(
                    actions,
                    commands,
                    stand_action_scale=0.0,
                    command_active_threshold=invalid,
                )
        with self.assertRaisesRegex(ValueError, "three x/y/yaw"):
            apply_stand_scale(
                actions,
                torch.zeros(2, 2),
                stand_action_scale=0.0,
                command_active_threshold=0.01,
            )
        with self.assertRaisesRegex(ValueError, "matching leading"):
            apply_stand_scale(
                actions,
                torch.zeros(3, 3),
                stand_action_scale=0.0,
                command_active_threshold=0.01,
            )


class StandActionBlendIntegrationContractTest(unittest.TestCase):
    def test_base_is_legacy_noop_and_stage2c_opts_into_nominal_stand(self):
        base = _class(CFG_TREE, "HexapodFlatEnvCfg")
        stage2c = _class(
            PHASE2_CFG_TREE,
            "HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg",
        )
        self.assertEqual(_literal(base, "stand_action_scale"), 1.0)
        self.assertEqual(_literal(stage2c, "stand_action_scale"), 0.0)

    def test_effective_action_precedes_limits_and_drives_history(self):
        env = _class(ENV_TREE, "HexapodEnv")
        pre_source = ast.get_source_segment(
            ENV_SOURCE, _method(env, "_pre_physics_step")
        )
        observation_source = ast.get_source_segment(
            ENV_SOURCE, _method(env, "_get_observations")
        )
        reward_source = ast.get_source_segment(
            ENV_SOURCE, _method(env, "_get_rewards")
        )
        assert pre_source is not None
        assert observation_source is not None
        assert reward_source is not None
        self.assertLess(
            pre_source.index("apply_command_conditioned_stand_action_scale"),
            pre_source.index("self.cfg.action_scale"),
        )
        self.assertLess(
            pre_source.index("self.cfg.action_scale"),
            pre_source.index("clamped_targets"),
        )
        self.assertLess(
            pre_source.index("clamped_targets"),
            pre_source.index("limit_processed_joint_target_slew"),
        )
        self.assertIn("self._actions,", observation_source)
        self.assertIn("self._previous_actions = self._actions.clone()", observation_source)
        self.assertIn("self._actions - self._previous_actions", reward_source)

    def test_environment_validates_configured_scale(self):
        env = _class(ENV_TREE, "HexapodEnv")
        init_source = ast.get_source_segment(ENV_SOURCE, _method(env, "__init__"))
        assert init_source is not None
        self.assertIn("self._stand_action_scale = float(self.cfg.stand_action_scale)", init_source)
        self.assertIn("not 0.0 <= self._stand_action_scale <= 1.0", init_source)


if __name__ == "__main__":
    unittest.main()
