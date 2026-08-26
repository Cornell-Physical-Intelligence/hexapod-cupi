"""CPU/static tests for the Stage2G insect-gait phase and clearance shaping."""

from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
ENV_PATH = ROOT / "hexapod_rl" / "env.py"
CFG_PATH = ROOT / "hexapod_rl" / "phase2g_cfg.py"
BASE_CFG_PATH = ROOT / "hexapod_rl" / "env_cfg.py"
ENV_TREE = ast.parse(ENV_PATH.read_text())
CFG_TREE = ast.parse(CFG_PATH.read_text())
BASE_CFG_TREE = ast.parse(BASE_CFG_PATH.read_text())


def _function(name: str):
    node = next(
        item
        for item in ENV_TREE.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"math": math, "torch": torch}
    exec(compile(module, ENV_PATH, "exec"), namespace)
    return namespace[name]


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name
    )


def _literal(class_node: ast.ClassDef, name: str):
    for node in class_node.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing {class_node.name}.{name}")


assign_tripods = _function("assign_tripod_pairs_from_foot_offsets")
expected_stance_fn = _function("tripod_expected_stance")
phase_contact_reward = _function("gait_phase_contact_reward")
clearance_reward = _function("swing_clearance_reward")
advance_phase = _function("advance_gait_phase")


def _hexagon_offsets() -> torch.Tensor:
    """Idealized stance: forward is -Y, left is +X, three feet per side."""

    offsets = torch.tensor(
        [
            [0.20, -0.25, -0.15],  # left front
            [0.25, 0.00, -0.15],  # left mid
            [0.20, 0.25, -0.15],  # left hind
            [-0.20, -0.25, -0.15],  # right front
            [-0.25, 0.00, -0.15],  # right mid
            [-0.20, 0.25, -0.15],  # right hind
        ]
    )
    return offsets


class TripodAssignmentTest(unittest.TestCase):
    def test_alternating_tripods_from_geometry(self):
        tripod_a = assign_tripods(_hexagon_offsets())
        # Left front, left hind, right mid.
        self.assertEqual(tripod_a.tolist(), [True, False, True, False, True, False])

    def test_assignment_is_order_invariant(self):
        offsets = _hexagon_offsets()
        permutation = torch.tensor([3, 0, 5, 2, 4, 1])
        permuted = assign_tripods(offsets[permutation])
        base = assign_tripods(offsets)
        self.assertEqual(permuted.tolist(), base[permutation].tolist())

    def test_rejects_unbalanced_sides(self):
        offsets = _hexagon_offsets()
        offsets[:, 0] = 0.1  # all feet on one side
        with self.assertRaises(ValueError):
            assign_tripods(offsets)

    def test_rejects_bad_shapes_and_nonfinite(self):
        with self.assertRaises(ValueError):
            assign_tripods(torch.zeros(5, 3))
        offsets = _hexagon_offsets()
        offsets[0, 0] = math.nan
        with self.assertRaises(ValueError):
            assign_tripods(offsets)


class ExpectedStanceTest(unittest.TestCase):
    def setUp(self):
        self.tripod_a = torch.tensor([True, False, True, False, True, False])

    def test_antiphase_windows(self):
        phase = torch.tensor([0.25, 0.75])
        stance = expected_stance_fn(
            phase, self.tripod_a, duty_factor=0.5, sharpness=30.0
        )
        # Mid-window of tripod A at phase 0.25: A in stance, B in swing.
        self.assertTrue(torch.all(stance[0][self.tripod_a] > 0.99))
        self.assertTrue(torch.all(stance[0][~self.tripod_a] < 0.01))
        # And exactly swapped half a cycle later.
        self.assertTrue(torch.all(stance[1][~self.tripod_a] > 0.99))
        self.assertTrue(torch.all(stance[1][self.tripod_a] < 0.01))

    def test_duty_factor_widens_stance(self):
        phase = torch.full((1,), 0.45)
        narrow = expected_stance_fn(
            phase, self.tripod_a, duty_factor=0.5, sharpness=30.0
        )
        wide = expected_stance_fn(
            phase, self.tripod_a, duty_factor=0.8, sharpness=30.0
        )
        self.assertTrue(torch.all(wide[0][self.tripod_a] >= narrow[0][self.tripod_a]))

    def test_validation(self):
        phase = torch.zeros(2)
        with self.assertRaises(ValueError):
            expected_stance_fn(phase, self.tripod_a, duty_factor=0.0, sharpness=30.0)
        with self.assertRaises(ValueError):
            expected_stance_fn(phase, self.tripod_a, duty_factor=0.5, sharpness=0.0)
        with self.assertRaises(ValueError):
            expected_stance_fn(
                phase, self.tripod_a.float(), duty_factor=0.5, sharpness=30.0
            )


class PhaseContactRewardTest(unittest.TestCase):
    def setUp(self):
        self.tripod_a = torch.tensor([True, False, True, False, True, False])
        self.stance = expected_stance_fn(
            torch.tensor([0.25]), self.tripod_a, duty_factor=0.5, sharpness=30.0
        )

    def test_perfect_tripod_near_one(self):
        contact = self.tripod_a.unsqueeze(0)
        reward = phase_contact_reward(contact, self.stance)
        self.assertGreater(float(reward[0]), 0.99)

    def test_antiphase_tripod_near_zero(self):
        contact = (~self.tripod_a).unsqueeze(0)
        reward = phase_contact_reward(contact, self.stance)
        self.assertLess(float(reward[0]), 0.01)

    def test_all_feet_planted_is_half(self):
        contact = torch.ones(1, 6, dtype=torch.bool)
        reward = phase_contact_reward(contact, self.stance)
        self.assertAlmostEqual(float(reward[0]), 0.5, places=2)

    def test_validation(self):
        with self.assertRaises(ValueError):
            phase_contact_reward(torch.ones(1, 6), self.stance)
        with self.assertRaises(ValueError):
            phase_contact_reward(torch.ones(1, 5, dtype=torch.bool), self.stance)


class SwingClearanceRewardTest(unittest.TestCase):
    def setUp(self):
        self.tripod_a = torch.tensor([True, False, True, False, True, False])
        self.stance = expected_stance_fn(
            torch.tensor([0.25]), self.tripod_a, duty_factor=0.5, sharpness=30.0
        )

    def test_apex_at_target_scores_high(self):
        heights = torch.full((1, 6), 0.005)
        heights[0, ~self.tripod_a] = 0.030
        reward = clearance_reward(
            heights, self.stance, target_m=0.030, tolerance_m=0.015
        )
        self.assertGreater(float(reward[0]), 0.95)

    def test_dragging_swing_scores_low(self):
        heights = torch.full((1, 6), 0.002)
        reward = clearance_reward(
            heights, self.stance, target_m=0.030, tolerance_m=0.015
        )
        self.assertLess(float(reward[0]), 0.05)

    def test_stance_feet_do_not_contribute(self):
        low = torch.full((1, 6), 0.002)
        low[0, ~self.tripod_a] = 0.030
        lifted_stance = low.clone()
        lifted_stance[0, self.tripod_a] = 0.030
        r_low = clearance_reward(low, self.stance, target_m=0.030, tolerance_m=0.015)
        r_lifted = clearance_reward(
            lifted_stance, self.stance, target_m=0.030, tolerance_m=0.015
        )
        self.assertAlmostEqual(float(r_low[0]), float(r_lifted[0]), places=2)

    def test_validation(self):
        heights = torch.zeros(1, 6)
        with self.assertRaises(ValueError):
            clearance_reward(heights, self.stance, target_m=0.0, tolerance_m=0.015)
        with self.assertRaises(ValueError):
            clearance_reward(heights, self.stance, target_m=0.03, tolerance_m=0.0)
        with self.assertRaises(ValueError):
            clearance_reward(torch.zeros(1, 5), self.stance, target_m=0.03, tolerance_m=0.015)


class AdvanceGaitPhaseTest(unittest.TestCase):
    def test_speed_proportional_frequency_with_band(self):
        phase = torch.zeros(3)
        speed = torch.tensor([0.16, 0.30, 0.60])
        moving = torch.ones(3, dtype=torch.bool)
        out = advance_phase(
            phase,
            speed,
            moving,
            step_dt=0.02,
            cycles_per_meter=8.0,
            min_frequency_hz=1.2,
            max_frequency_hz=3.0,
        )
        self.assertAlmostEqual(float(out[0]), 0.02 * 1.28, places=6)
        self.assertAlmostEqual(float(out[1]), 0.02 * 2.4, places=6)
        # 0.60 m/s would be 4.8 Hz; the band caps it at 3.0 Hz.
        self.assertAlmostEqual(float(out[2]), 0.02 * 3.0, places=6)

    def test_standing_holds_phase_and_wraps(self):
        phase = torch.tensor([0.4, 0.999])
        speed = torch.tensor([0.0, 0.30])
        moving = torch.tensor([False, True])
        out = advance_phase(
            phase,
            speed,
            moving,
            step_dt=0.02,
            cycles_per_meter=8.0,
            min_frequency_hz=1.2,
            max_frequency_hz=3.0,
        )
        self.assertAlmostEqual(float(out[0]), 0.4, places=6)
        self.assertLess(float(out[1]), 0.5)  # wrapped past 1.0

    def test_slow_moving_command_uses_frequency_floor(self):
        phase = torch.zeros(1)
        out = advance_phase(
            phase,
            torch.tensor([0.06]),
            torch.ones(1, dtype=torch.bool),
            step_dt=0.02,
            cycles_per_meter=8.0,
            min_frequency_hz=1.2,
            max_frequency_hz=3.0,
        )
        self.assertAlmostEqual(float(out[0]), 0.02 * 1.2, places=6)

    def test_validation(self):
        phase = torch.zeros(2)
        speed = torch.zeros(2)
        moving = torch.zeros(2, dtype=torch.bool)
        with self.assertRaises(ValueError):
            advance_phase(
                phase, speed, moving.float(), step_dt=0.02,
                cycles_per_meter=8.0, min_frequency_hz=1.2, max_frequency_hz=3.0,
            )
        with self.assertRaises(ValueError):
            advance_phase(
                phase, speed, moving, step_dt=0.0,
                cycles_per_meter=8.0, min_frequency_hz=1.2, max_frequency_hz=3.0,
            )
        with self.assertRaises(ValueError):
            advance_phase(
                phase, speed, moving, step_dt=0.02,
                cycles_per_meter=8.0, min_frequency_hz=3.0, max_frequency_hz=1.2,
            )


class Stage2GConfigContractTest(unittest.TestCase):
    def test_base_defaults_stay_dormant(self):
        base = _class(BASE_CFG_TREE, "HexapodFlatEnvCfg")
        self.assertEqual(_literal(base, "gait_phase_contact_reward_scale"), 0.0)
        self.assertEqual(_literal(base, "swing_clearance_reward_scale"), 0.0)
        self.assertIs(_literal(base, "include_gait_phase_observation"), False)

    def test_scratch_arm_widens_observation_for_phase(self):
        scratch = _class(CFG_TREE, "HexapodStage2GInsectGaitEnvCfg")
        self.assertEqual(_literal(scratch, "observation_space"), 68)
        self.assertIs(_literal(scratch, "include_gait_phase_observation"), True)
        self.assertGreater(_literal(scratch, "gait_phase_contact_reward_scale"), 0.0)
        self.assertGreater(_literal(scratch, "swing_clearance_reward_scale"), 0.0)
        self.assertEqual(_literal(scratch, "support_contact_target"), 3.0)
        self.assertIs(_literal(scratch, "speed_conditioned_support_targets"), False)

    def test_adapt_arm_keeps_deployed_interface(self):
        adapt = _class(CFG_TREE, "HexapodStage2GInsectGaitAdaptEnvCfg")
        self.assertEqual(_literal(adapt, "observation_space"), 66)
        self.assertIs(_literal(adapt, "include_gait_phase_observation"), False)
        self.assertEqual(_literal(adapt, "action_scale"), 0.20)


if __name__ == "__main__":
    unittest.main()
