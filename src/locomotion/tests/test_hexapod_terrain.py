"""Check the terrain v2.1 hexapod task: registration, trained defaults and the CPU curriculum arithmetic."""

import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src/locomotion/hexapod/terrain"


def load_curriculum():
    spec = importlib.util.spec_from_file_location(
        "hexapod_terrain_curriculum", PACKAGE / "curriculum.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assignments(name, class_name=None):
    body = ast.parse((PACKAGE / name).read_text()).body
    if class_name is not None:
        body = next(
            node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name
        ).body
    return {
        item.targets[0].id if isinstance(item, ast.Assign) else item.target.id: item.value
        for item in body
        if isinstance(item, (ast.Assign, ast.AnnAssign))
        and isinstance(getattr(item, "target", None) or item.targets[0], ast.Name)
    }


class TerrainTaskSourceTests(unittest.TestCase):
    def test_registration_adds_one_id_beside_the_original_tasks(self):
        calls = []
        tree = ast.parse((PACKAGE / "__init__.py").read_text())
        tree.body = [
            node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        scope = {
            "__name__": "locomotion.hexapod.terrain",
            "gym": SimpleNamespace(register=lambda **kwargs: calls.append(kwargs)),
        }
        exec(compile(tree, str(PACKAGE / "__init__.py"), "exec"), scope)
        self.assertEqual([entry["id"] for entry in calls], ["Hexapod-Rough-v21-Direct-v0"])
        entry = calls[0]
        self.assertEqual(entry["entry_point"], "locomotion.hexapod.terrain.env:HexapodTerrainEnv")
        self.assertEqual(
            entry["kwargs"]["env_cfg_entry_point"],
            "locomotion.hexapod.terrain.env_cfg:HexapodTerrainEnvCfg",
        )
        self.assertEqual(
            entry["kwargs"]["rsl_rl_cfg_entry_point"],
            "locomotion.hexapod.terrain.agents:HexapodTerrainPPORunnerCfg",
        )
        self.assertTrue(entry["disable_env_checker"])
        self.assertIn("register", scope)

    def test_terrain_constants_are_the_trained_v21_values(self):
        values = assignments("terrains_cfg.py")
        for name, value in {
            "STEP_CAP": 0.052,
            "STAIR_TREAD": 0.25,
            "STAIR_PLATFORM": 0.8,
            "BOX_GRID_WIDTH": 0.12,
            "TILE_SIZE": 2.0,
            "NUM_LEVELS": 10,
        }.items():
            self.assertEqual(ast.literal_eval(values[name]), value, name)

    def test_task_defaults_are_the_trained_settings(self):
        values = assignments("env_cfg.py", "HexapodTerrainEnvCfg")
        for name, value in {
            "gait_clock_enabled": True,
            "gait_period_s": 0.4,
            "command_forward_only": True,
            "command_lin_vel_min": 0.1,
            "command_lin_vel_max": 0.35,
            "command_yaw_rate_max": 0.1,
            "feet_air_time_threshold": 0.0,
            "flat_orientation_reward_scale": 0.0,
            "swing_clearance_reward_scale": 0.05,
            "swing_clearance_target": 0.06,
            "success_speed_tol_ratio": 0.25,
            "terrain_curriculum": True,
            "curriculum_move_down_speed_ratio": 0.5,
            "command_curriculum": False,
        }.items():
            self.assertEqual(ast.literal_eval(values[name]), value, name)

    def test_task_imports_no_package_outside_hexapod(self):
        for path in PACKAGE.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    with self.subTest(file=path.name, module=node.module):
                        self.assertFalse(node.module.startswith("locomotion."))
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(alias.name.startswith("locomotion."))


class TerrainCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.c = load_curriculum()

    def test_success_needs_a_timeout_without_fall_and_speed_within_tolerance(self):
        steps = torch.tensor([100.0, 100.0, 100.0, 100.0, 0.0])
        cmd = torch.tensor([0.28, 0.28, 0.28, 0.28, 0.28])
        v_sum = torch.tensor([0.28, 0.20, 0.28, 0.10, 0.0]) * steps
        fell = torch.tensor([False, False, True, False, False])
        timed_out = torch.tensor([True, True, True, True, True])
        success, slow, err, mean_v = self.c.episode_outcome(
            v_sum, steps, cmd, fell, timed_out, 0.35
        )
        # 0.20 m/s is 0.08 below 0.28, outside 25% (0.07); 0.10 m/s is below half the command
        torch.testing.assert_close(success, torch.tensor([True, False, False, False, False]))
        torch.testing.assert_close(slow, torch.tensor([False, False, False, True, True]))
        torch.testing.assert_close(mean_v, torch.tensor([0.28, 0.20, 0.28, 0.10, 0.0]))
        torch.testing.assert_close(err, torch.tensor([0.0, 0.08, 0.0, 0.18, 0.28]))

    def test_small_commands_use_the_tolerance_floor(self):
        steps = torch.tensor([10.0])
        success, *_ = self.c.episode_outcome(
            torch.tensor([0.02]) * steps,
            steps,
            torch.tensor([0.0]),
            torch.tensor([False]),
            torch.tensor([True]),
            0.35,
        )
        # floor 0.1 * 0.35 = 0.035, tolerance 25% of it = 0.00875 < 0.02
        self.assertFalse(bool(success))

    def test_terrain_moves_up_on_success_and_down_on_fall_or_slow(self):
        success = torch.tensor([True, False, False, False, False])
        fell = torch.tensor([False, True, False, False, True])
        slow = torch.tensor([False, False, True, False, False])
        ran = torch.tensor([True, True, True, True, False])
        up, down = self.c.terrain_moves(success, fell, slow, ran)
        torch.testing.assert_close(up, torch.tensor([True, False, False, False, False]))
        torch.testing.assert_close(down, torch.tensor([False, True, True, False, False]))

    def test_swing_clearance_counts_lifted_swing_feet_up_to_the_target(self):
        toe_z = torch.tensor(
            [[0.0, 0.0, 0.0, 0.03, 0.10, 0.0], [0.02, 0.05, 0.08, 0.02, 0.05, 0.08]]
        )
        in_contact = torch.tensor([[True, True, True, False, False, True], [False] * 6])
        value = self.c.swing_clearance(toe_z, in_contact, 0.06)
        # env 0: stance mean 0, swing feet at 0.03 (half) and 0.10 (capped at 1); env 1: lowest toe 0.02 is the base
        torch.testing.assert_close(value, torch.tensor([1.5, (0.0 + 0.5 + 1.0) * 2]))

    def test_command_curriculum_rises_after_a_successful_top_bucket_window(self):
        cc = self.c.CommandCurriculum(
            start=0.15, step=0.05, cap=0.25, success=0.8, window_episodes=4
        )
        ran = torch.ones(4, dtype=torch.bool)
        top = torch.full((4,), 0.14)
        low = torch.full((4,), 0.05)
        self.assertEqual(cc.update(low, torch.ones(4, dtype=torch.bool), ran), 0.15)
        self.assertEqual(cc.update(top, torch.tensor([True, True, True, False]), ran), 0.15)
        self.assertAlmostEqual(cc.update(top, torch.ones(4, dtype=torch.bool), ran), 0.20)
        self.assertAlmostEqual(
            cc.update(torch.full((4,), 0.19), torch.ones(4, dtype=torch.bool), ran), 0.25
        )
        self.assertAlmostEqual(
            cc.update(torch.full((4,), 0.24), torch.ones(4, dtype=torch.bool), ran), 0.25
        )


if __name__ == "__main__":
    unittest.main()
