"""CPU/static tests for rotating Stage2E joystick primitives."""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).parents[1]
SAMPLER_PATH = ROOT / "hexapod_rl" / "command_sampling.py"
SAMPLER_SPEC = importlib.util.spec_from_file_location(
    "hexapod_stage2e_command_sampling", SAMPLER_PATH
)
assert SAMPLER_SPEC is not None and SAMPLER_SPEC.loader is not None
command_sampling = importlib.util.module_from_spec(SAMPLER_SPEC)
SAMPLER_SPEC.loader.exec_module(command_sampling)

NUM_STAGE2E_CATEGORIES = command_sampling.NUM_STAGE2E_CATEGORIES
STAGE2E_FORWARD = command_sampling.STAGE2E_FORWARD
STAGE2E_FORWARD_JOYSTICK = command_sampling.STAGE2E_FORWARD_JOYSTICK
STAGE2E_LATERAL = command_sampling.STAGE2E_LATERAL
STAGE2E_REVERSE = command_sampling.STAGE2E_REVERSE
STAGE2E_REVERSE_JOYSTICK = command_sampling.STAGE2E_REVERSE_JOYSTICK
STAGE2E_STAND = command_sampling.STAGE2E_STAND
STAGE2E_YAW = command_sampling.STAGE2E_YAW
normalized_signed_axis_progress = command_sampling.normalized_signed_axis_progress
sample_stage2e_joystick_transition_commands = (
    command_sampling.sample_stage2e_joystick_transition_commands
)


ENV_PATH = ROOT / "hexapod_rl" / "env.py"
CFG_PATH = ROOT / "hexapod_rl" / "env_cfg.py"
ENV_SOURCE = ENV_PATH.read_text(encoding="utf-8")
CFG_SOURCE = CFG_PATH.read_text(encoding="utf-8")
ENV_TREE = ast.parse(ENV_SOURCE)
CFG_TREE = ast.parse(CFG_SOURCE)


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


def _method(name: str) -> ast.FunctionDef:
    env = next(
        item
        for item in ENV_TREE.body
        if isinstance(item, ast.ClassDef) and item.name == "HexapodEnv"
    )
    return next(
        item
        for item in env.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )


def _base_cfg_literal(name: str):
    cfg = next(
        item
        for item in CFG_TREE.body
        if isinstance(item, ast.ClassDef) and item.name == "HexapodFlatEnvCfg"
    )
    for node in cfg.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        ) or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            value = node.value
            assert value is not None
            return ast.literal_eval(value)
    raise AssertionError(f"missing HexapodFlatEnvCfg.{name}")


capped_axis_error = _function("capped_normalized_axis_error")


def _sample(
    counts: tuple[int, ...],
    epoch: int,
    yaw_forward_range: tuple[float, float] = (0.20, 0.26),
):
    indices = torch.arange(40, dtype=torch.long)
    epochs = torch.full_like(indices, epoch)
    generator = torch.Generator().manual_seed(100 + epoch)
    return sample_stage2e_joystick_transition_commands(
        indices,
        resample_epochs=epochs,
        bucket_counts=counts,
        bucket_stride=13,
        lateral_anchor_abs_range=(0.08, 0.10),
        forward_anchor_range=(0.20, 0.30),
        yaw_anchor_forward_range=yaw_forward_range,
        yaw_anchor_abs_range=(0.20, 0.28),
        joystick_forward_range=(0.03, 0.30),
        joystick_reverse_abs_range=(0.02, 0.20),
        joystick_lateral_abs_range=(0.03, 0.10),
        joystick_yaw_abs_range=(0.06, 0.28),
        generator=generator,
    )


class Stage2ERotatingSamplerTest(unittest.TestCase):
    def test_each_stage_and_epoch_has_exact_categories_and_sign_balance(self):
        stage_counts = (
            (4, 8, 4, 4, 4, 12, 4),
            (4, 8, 4, 4, 4, 8, 8),
            (4, 8, 4, 4, 4, 8, 8),
        )
        for counts in stage_counts:
            for epoch in (0, 1, 7, 39):
                with self.subTest(counts=counts, epoch=epoch):
                    commands, categories = _sample(counts, epoch)
                    actual_counts = torch.bincount(
                        categories, minlength=NUM_STAGE2E_CATEGORIES
                    )
                    torch.testing.assert_close(actual_counts, torch.tensor(counts))
                    torch.testing.assert_close(
                        commands[categories == STAGE2E_STAND],
                        torch.zeros((counts[STAGE2E_STAND], 3)),
                    )
                    self.assertTrue(
                        torch.all(commands[categories == STAGE2E_FORWARD, 0] > 0.0)
                    )
                    self.assertTrue(
                        torch.all(
                            commands[categories == STAGE2E_FORWARD_JOYSTICK, 0]
                            > 0.0
                        )
                    )
                    self.assertTrue(
                        torch.all(
                            commands[categories == STAGE2E_REVERSE_JOYSTICK, 0]
                            < 0.0
                        )
                    )
                    reverse_anchor = commands[categories == STAGE2E_REVERSE]
                    self.assertTrue(torch.all(reverse_anchor[:, 0] < 0.0))
                    torch.testing.assert_close(
                        reverse_anchor[:, 1:], torch.zeros_like(reverse_anchor[:, 1:])
                    )

                    lateral = commands[categories == STAGE2E_LATERAL, 1]
                    self.assertEqual(torch.count_nonzero(lateral > 0).item(), len(lateral) // 2)
                    self.assertEqual(torch.count_nonzero(lateral < 0).item(), len(lateral) // 2)
                    yaw = commands[categories == STAGE2E_YAW, 2]
                    self.assertEqual(torch.count_nonzero(yaw > 0).item(), len(yaw) // 2)
                    self.assertEqual(torch.count_nonzero(yaw < 0).item(), len(yaw) // 2)

                    for category in (
                        STAGE2E_FORWARD_JOYSTICK,
                        STAGE2E_REVERSE_JOYSTICK,
                    ):
                        rows = commands[categories == category]
                        quadrants = (
                            (rows[:, 1] < 0).long()
                            + 2 * (rows[:, 2] < 0).long()
                        )
                        expected = torch.full((4,), len(rows) // 4, dtype=torch.long)
                        torch.testing.assert_close(
                            torch.bincount(quadrants, minlength=4), expected
                        )

    def test_coprime_epoch_stride_changes_categories_and_visits_all_slots(self):
        counts = (4, 8, 4, 4, 4, 12, 4)
        _, initial = _sample(counts, 0)
        _, next_epoch = _sample(counts, 1)
        self.assertGreater(torch.count_nonzero(initial != next_epoch).item(), 0)
        _, wrapped = _sample(counts, 40)
        torch.testing.assert_close(initial, wrapped)

        visited = []
        for epoch in range(40):
            _, categories = _sample(counts, epoch)
            visited.append(int(categories[0]))
        self.assertEqual(set(visited), set(range(NUM_STAGE2E_CATEGORIES)))

    def test_terminal_yaw_bucket_is_true_in_place_pivot(self):
        commands, categories = _sample(
            (4, 8, 4, 4, 4, 8, 8), 0, yaw_forward_range=(0.0, 0.0)
        )
        yaw_rows = commands[categories == STAGE2E_YAW]
        torch.testing.assert_close(yaw_rows[:, :2], torch.zeros_like(yaw_rows[:, :2]))
        self.assertTrue(torch.all(torch.abs(yaw_rows[:, 2]) > 0.0))

    def test_sampler_rejects_invalid_epoch_balance_stride_and_ranges(self):
        indices = torch.arange(40, dtype=torch.long)
        base = dict(
            bucket_indices=indices,
            resample_epochs=torch.zeros_like(indices),
            bucket_counts=(4, 8, 4, 4, 4, 12, 4),
            bucket_stride=13,
            lateral_anchor_abs_range=(0.08, 0.10),
            forward_anchor_range=(0.20, 0.30),
            yaw_anchor_forward_range=(0.20, 0.26),
            yaw_anchor_abs_range=(0.20, 0.28),
            joystick_forward_range=(0.03, 0.30),
            joystick_reverse_abs_range=(0.02, 0.20),
            joystick_lateral_abs_range=(0.03, 0.10),
            joystick_yaw_abs_range=(0.06, 0.28),
        )
        mutations = (
            ("resample_epochs", torch.zeros(39, dtype=torch.long), "must match"),
            ("bucket_counts", (4, 7, 4, 4, 5, 12, 4), "must be even"),
            ("bucket_counts", (4, 8, 4, 4, 4, 10, 6), "divisible by four"),
            ("bucket_stride", 10, "coprime"),
            ("joystick_reverse_abs_range", (0.0, 0.1), "strictly positive"),
        )
        for key, value, message in mutations:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                args = dict(base)
                args[key] = value
                sample_stage2e_joystick_transition_commands(**args)


class Stage2ELongitudinalAcquisitionTest(unittest.TestCase):
    def test_small_reverse_cannot_profit_by_standing_still(self):
        command = torch.tensor([-0.03, 0.03, -0.03])
        achieved = torch.tensor([0.0, 0.0, 0.03])
        progress = normalized_signed_axis_progress(
            command, achieved, active_threshold=0.015
        )
        error = capped_axis_error(
            command,
            achieved,
            active_threshold=0.015,
            error_cap=2.0,
        )
        torch.testing.assert_close(progress, torch.tensor([0.0, 0.0, -1.0]))
        torch.testing.assert_close(error, torch.tensor([1.0, 1.0, 2.0]))

        correct = torch.tensor([-0.03, 0.03])
        torch.testing.assert_close(
            normalized_signed_axis_progress(
                command[:2], correct, active_threshold=0.015
            ),
            torch.ones(2),
        )
        torch.testing.assert_close(
            capped_axis_error(
                command[:2], correct, active_threshold=0.015, error_cap=2.0
            ),
            torch.zeros(2),
        )

    def test_new_reward_controls_are_dormant_by_default_and_logged(self):
        self.assertEqual(_base_cfg_literal("longitudinal_signed_progress_reward_scale"), 0.0)
        self.assertEqual(
            _base_cfg_literal("longitudinal_normalized_error_penalty_scale"), 0.0
        )
        self.assertEqual(_base_cfg_literal("longitudinal_normalized_error_cap"), 2.0)
        init_source = ast.get_source_segment(ENV_SOURCE, _method("__init__"))
        reward_source = ast.get_source_segment(ENV_SOURCE, _method("_get_rewards"))
        assert init_source is not None and reward_source is not None
        for key in (
            "longitudinal_signed_progress",
            "longitudinal_normalized_error_penalty",
            "normalized_longitudinal_error",
        ):
            self.assertIn(f'"{key}"', init_source)
            self.assertIn(f'"{key}"', reward_source)

    def test_resample_epoch_rotates_before_sampling_and_resets_before_initial_sample(self):
        sample_source = ast.get_source_segment(ENV_SOURCE, _method("_sample_commands"))
        resample_source = ast.get_source_segment(
            ENV_SOURCE, _method("_resample_commands_if_due")
        )
        reset_source = ast.get_source_segment(ENV_SOURCE, _method("_reset_idx"))
        assert sample_source is not None and resample_source is not None and reset_source is not None
        self.assertIn('sampling_mode == "stage2e_joystick_transitions"', sample_source)
        self.assertIn("resample_epochs=self._command_resample_epoch[env_ids]", sample_source)
        self.assertLess(
            resample_source.index("self._command_resample_epoch[due_env_ids] += 1"),
            resample_source.index("self._sample_commands(due_env_ids)"),
        )
        self.assertLess(
            reset_source.index("self._command_resample_epoch[env_ids] = 0"),
            reset_source.index("self._sample_commands(env_ids)"),
        )


if __name__ == "__main__":
    unittest.main()
