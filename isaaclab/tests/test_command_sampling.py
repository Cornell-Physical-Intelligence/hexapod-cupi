"""Focused CPU tests for the Phase 2 command curriculum utilities."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


PACKAGE_ROOT = (
    Path(__file__).parents[2] / "packages" / "hexapod_env" / "hexapod_env"
)
MODULE_PATH = PACKAGE_ROOT / "command_sampling.py"
SPEC = importlib.util.spec_from_file_location("hexapod_command_sampling", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
command_sampling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(command_sampling)


class CommandSamplingTest(unittest.TestCase):
    def test_navigation_frame_matches_anatomical_axes(self):
        body_vectors = torch.tensor(
            [[0.0, -0.30, 0.0], [0.20, 0.0, 0.10], [1.0, 2.0, 3.0]]
        )
        actual = command_sampling.body_to_navigation_frame(body_vectors)
        expected = torch.tensor(
            [[0.30, 0.0, 0.0], [0.0, 0.20, 0.10], [-2.0, 1.0, 3.0]]
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(body_vectors[2], torch.tensor([1.0, 2.0, 3.0]))

    def test_each_category_is_mutually_exclusive(self):
        category_count = command_sampling.NUM_COMMAND_CATEGORIES
        for expected_category in range(category_count):
            probabilities = [0.0] * category_count
            probabilities[expected_category] = 1.0
            generator = torch.Generator().manual_seed(100 + expected_category)
            commands, categories = command_sampling.sample_velocity_command_mixture(
                512,
                lin_vel_x_range=(1.0, 2.0),
                lin_vel_y_range=(3.0, 4.0),
                ang_vel_z_range=(5.0, 6.0),
                category_probabilities=probabilities,
                device="cpu",
                generator=generator,
            )
            self.assertTrue(torch.all(categories == expected_category))
            if expected_category == command_sampling.STANDING:
                self.assertTrue(torch.all(commands == 0.0))
            elif expected_category == command_sampling.LONGITUDINAL_ONLY:
                self.assertTrue(torch.all((commands[:, 0] >= 1.0) & (commands[:, 0] <= 2.0)))
                self.assertTrue(torch.all(commands[:, 1:] == 0.0))
            elif expected_category == command_sampling.LATERAL_ONLY:
                self.assertTrue(torch.all(commands[:, 0] == 0.0))
                self.assertTrue(torch.all((commands[:, 1] >= 3.0) & (commands[:, 1] <= 4.0)))
                self.assertTrue(torch.all(commands[:, 2] == 0.0))
            elif expected_category == command_sampling.YAW_ONLY:
                self.assertTrue(torch.all(commands[:, :2] == 0.0))
                self.assertTrue(torch.all((commands[:, 2] >= 5.0) & (commands[:, 2] <= 6.0)))
            else:
                self.assertTrue(torch.all((commands[:, 0] >= 1.0) & (commands[:, 0] <= 2.0)))
                self.assertTrue(torch.all((commands[:, 1] >= 3.0) & (commands[:, 1] <= 4.0)))
                self.assertTrue(torch.all((commands[:, 2] >= 5.0) & (commands[:, 2] <= 6.0)))

    def test_empirical_mixture_matches_declared_probabilities(self):
        probabilities = torch.tensor([0.10, 0.25, 0.15, 0.15, 0.35])
        _, categories = command_sampling.sample_velocity_command_mixture(
            100_000,
            lin_vel_x_range=(-0.2, 0.55),
            lin_vel_y_range=(-0.2, 0.2),
            ang_vel_z_range=(-0.5, 0.5),
            category_probabilities=probabilities.tolist(),
            device="cpu",
            generator=torch.Generator().manual_seed(42),
        )
        empirical = torch.bincount(categories, minlength=len(probabilities)).float()
        empirical /= categories.numel()
        torch.testing.assert_close(empirical, probabilities, atol=0.005, rtol=0.0)

    def test_intervals_and_active_timer_mask(self):
        intervals = command_sampling.sample_uniform_intervals(
            10_000,
            (6.0, 10.0),
            device="cpu",
            generator=torch.Generator().manual_seed(7),
        )
        self.assertGreaterEqual(float(intervals.min()), 6.0)
        self.assertLessEqual(float(intervals.max()), 10.0)

        time_left = torch.tensor([0.020, 0.005, 0.001])
        active = torch.tensor([True, True, False])
        updated, due = command_sampling.advance_resampling_timers(
            time_left, active, step_dt=0.010
        )
        torch.testing.assert_close(updated, torch.tensor([0.010, -0.005, 0.001]))
        torch.testing.assert_close(due, torch.tensor([False, True, False]))
        # The helper is functional: callers choose when to commit the update.
        torch.testing.assert_close(time_left, torch.tensor([0.020, 0.005, 0.001]))

    def test_recovery_stage1_uses_exact_fixed_buckets_and_balanced_signs(self):
        bucket_indices = torch.arange(400)
        commands, categories = command_sampling.sample_stage1_recovery_commands(
            bucket_indices,
            generator=torch.Generator().manual_seed(2026),
        )
        counts = torch.bincount(
            categories, minlength=command_sampling.NUM_RECOVERY_CATEGORIES
        )
        torch.testing.assert_close(counts, torch.tensor([220, 80, 80, 20]))

        forward = categories == command_sampling.RECOVERY_FORWARD
        yaw = categories == command_sampling.RECOVERY_FORWARD_YAW
        lateral = categories == command_sampling.RECOVERY_FORWARD_LATERAL
        combined = categories == command_sampling.RECOVERY_COMBINED
        self.assertTrue(torch.all((commands[forward, 0] >= 0.20) & (commands[forward, 0] <= 0.35)))
        self.assertTrue(torch.all(commands[forward, 1:] == 0.0))
        self.assertTrue(torch.all((commands[yaw, 0] >= 0.20) & (commands[yaw, 0] <= 0.32)))
        self.assertTrue(torch.all(commands[yaw, 1] == 0.0))
        self.assertTrue(torch.all((torch.abs(commands[yaw, 2]) >= 0.06) & (torch.abs(commands[yaw, 2]) <= 0.15)))
        self.assertEqual(int(torch.count_nonzero(commands[yaw, 2] > 0.0)), 40)
        self.assertEqual(int(torch.count_nonzero(commands[yaw, 2] < 0.0)), 40)
        self.assertTrue(torch.all((commands[lateral, 0] >= 0.20) & (commands[lateral, 0] <= 0.32)))
        self.assertTrue(torch.all(commands[lateral, 2] == 0.0))
        self.assertTrue(torch.all((torch.abs(commands[lateral, 1]) >= 0.03) & (torch.abs(commands[lateral, 1]) <= 0.07)))
        self.assertEqual(int(torch.count_nonzero(commands[lateral, 1] > 0.0)), 40)
        self.assertEqual(int(torch.count_nonzero(commands[lateral, 1] < 0.0)), 40)
        self.assertTrue(torch.all((commands[combined, 0] >= 0.22) & (commands[combined, 0] <= 0.30)))
        self.assertTrue(torch.all((torch.abs(commands[combined, 1]) >= 0.03) & (torch.abs(commands[combined, 1]) <= 0.06)))
        self.assertTrue(torch.all((torch.abs(commands[combined, 2]) >= 0.06) & (torch.abs(commands[combined, 2]) <= 0.12)))
        sign_pairs = set(
            zip(
                torch.sign(commands[combined, 1]).tolist(),
                torch.sign(commands[combined, 2]).tolist(),
            )
        )
        self.assertEqual(sign_pairs, {(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)})

    def test_recovery_category_is_stable_across_subset_resets(self):
        all_indices = torch.arange(127)
        _, all_categories = command_sampling.sample_stage1_recovery_commands(
            all_indices,
            generator=torch.Generator().manual_seed(1),
        )
        subset = torch.tensor([1, 11, 19, 20, 39, 88, 126])
        first_commands, first_categories = command_sampling.sample_stage1_recovery_commands(
            subset,
            generator=torch.Generator().manual_seed(2),
        )
        second_commands, second_categories = command_sampling.sample_stage1_recovery_commands(
            subset,
            generator=torch.Generator().manual_seed(3),
        )
        torch.testing.assert_close(first_categories, all_categories[subset])
        torch.testing.assert_close(second_categories, all_categories[subset])
        self.assertFalse(torch.equal(first_commands, second_commands))

    def test_recovery_stage2_uses_exact_isolated_buckets_and_balanced_signs(self):
        bucket_indices = torch.arange(400)
        commands, categories = command_sampling.sample_stage2_recovery_commands(
            bucket_indices,
            generator=torch.Generator().manual_seed(2027),
        )
        counts = torch.bincount(
            categories, minlength=command_sampling.NUM_STAGE2_CATEGORIES
        )
        torch.testing.assert_close(counts, torch.tensor([160, 120, 120]))

        forward = categories == command_sampling.STAGE2_FORWARD
        yaw = categories == command_sampling.STAGE2_FORWARD_YAW
        lateral = categories == command_sampling.STAGE2_FORWARD_LATERAL
        self.assertTrue(
            torch.all(
                (commands[forward, 0] >= 0.22)
                & (commands[forward, 0] <= 0.34)
            )
        )
        self.assertTrue(torch.all(commands[forward, 1:] == 0.0))
        self.assertTrue(
            torch.all((commands[yaw, 0] >= 0.20) & (commands[yaw, 0] <= 0.30))
        )
        self.assertTrue(torch.all(commands[yaw, 1] == 0.0))
        self.assertTrue(
            torch.all(
                (torch.abs(commands[yaw, 2]) >= 0.18)
                & (torch.abs(commands[yaw, 2]) <= 0.30)
            )
        )
        self.assertEqual(int(torch.count_nonzero(commands[yaw, 2] > 0.0)), 60)
        self.assertEqual(int(torch.count_nonzero(commands[yaw, 2] < 0.0)), 60)
        self.assertTrue(
            torch.all(
                (commands[lateral, 0] >= 0.20)
                & (commands[lateral, 0] <= 0.30)
            )
        )
        self.assertTrue(torch.all(commands[lateral, 2] == 0.0))
        self.assertTrue(
            torch.all(
                (torch.abs(commands[lateral, 1]) >= 0.08)
                & (torch.abs(commands[lateral, 1]) <= 0.14)
            )
        )
        self.assertEqual(int(torch.count_nonzero(commands[lateral, 1] > 0.0)), 60)
        self.assertEqual(int(torch.count_nonzero(commands[lateral, 1] < 0.0)), 60)
        self.assertTrue(torch.all(torch.count_nonzero(commands[:, 1:], dim=1) <= 1))

        subset = torch.tensor([0, 7, 8, 13, 14, 19, 20, 39, 126])
        _, subset_categories = command_sampling.sample_stage2_recovery_commands(
            subset,
            generator=torch.Generator().manual_seed(2028),
        )
        torch.testing.assert_close(subset_categories, categories[subset])

    def test_signed_axis_progress_is_antisymmetric_bounded_and_inactive_at_zero(self):
        commands = torch.tensor([0.0, 0.009, 0.08, -0.10, 0.10, -0.20, 0.10])
        achieved = torch.tensor([99.0, 99.0, 0.04, -0.05, -0.20, 0.10, 0.20])
        actual = command_sampling.normalized_signed_axis_progress(
            commands, achieved, active_threshold=0.01
        )
        expected = torch.tensor([0.0, 0.0, 0.5, 0.5, -1.0, -0.5, 1.0])
        torch.testing.assert_close(actual, expected)

        mirrored = command_sampling.normalized_signed_axis_progress(
            -commands, -achieved, active_threshold=0.01
        )
        torch.testing.assert_close(mirrored, actual)

    def test_stage2b_removes_forward_shortcut_and_balances_lateral_signs(self):
        bucket_indices = torch.arange(400)
        commands, categories = (
            command_sampling.sample_stage2b_lateral_acquisition_commands(
                bucket_indices,
                generator=torch.Generator().manual_seed(2029),
            )
        )
        counts = torch.bincount(
            categories, minlength=command_sampling.NUM_STAGE2B_CATEGORIES
        )
        torch.testing.assert_close(counts, torch.tensor([160, 160, 40, 40]))

        forward = categories == command_sampling.STAGE2B_FORWARD
        lateral_only = categories == command_sampling.STAGE2B_LATERAL_ONLY
        forward_lateral = categories == command_sampling.STAGE2B_FORWARD_LATERAL
        forward_yaw = categories == command_sampling.STAGE2B_FORWARD_YAW
        self.assertTrue(torch.all(commands[forward, 0] >= 0.22))
        self.assertTrue(torch.all(commands[forward, 0] <= 0.32))
        self.assertTrue(torch.all(commands[forward, 1:] == 0.0))

        self.assertTrue(torch.all(commands[lateral_only, 0] == 0.0))
        self.assertTrue(torch.all(commands[lateral_only, 2] == 0.0))
        self.assertTrue(torch.all(torch.abs(commands[lateral_only, 1]) >= 0.08))
        self.assertTrue(torch.all(torch.abs(commands[lateral_only, 1]) <= 0.12))
        self.assertEqual(int(torch.count_nonzero(commands[lateral_only, 1] > 0.0)), 80)
        self.assertEqual(int(torch.count_nonzero(commands[lateral_only, 1] < 0.0)), 80)

        self.assertTrue(torch.all(commands[forward_lateral, 0] >= 0.18))
        self.assertTrue(torch.all(commands[forward_lateral, 0] <= 0.24))
        self.assertTrue(torch.all(torch.abs(commands[forward_lateral, 1]) >= 0.06))
        self.assertTrue(torch.all(torch.abs(commands[forward_lateral, 1]) <= 0.10))
        self.assertTrue(torch.all(commands[forward_lateral, 2] == 0.0))
        self.assertEqual(int(torch.count_nonzero(commands[forward_lateral, 1] > 0.0)), 20)
        self.assertEqual(int(torch.count_nonzero(commands[forward_lateral, 1] < 0.0)), 20)

        self.assertTrue(torch.all(commands[forward_yaw, 0] >= 0.20))
        self.assertTrue(torch.all(commands[forward_yaw, 0] <= 0.26))
        self.assertTrue(torch.all(commands[forward_yaw, 1] == 0.0))
        self.assertTrue(torch.all(torch.abs(commands[forward_yaw, 2]) >= 0.20))
        self.assertTrue(torch.all(torch.abs(commands[forward_yaw, 2]) <= 0.28))
        self.assertEqual(int(torch.count_nonzero(commands[forward_yaw, 2] > 0.0)), 20)
        self.assertEqual(int(torch.count_nonzero(commands[forward_yaw, 2] < 0.0)), 20)

        subset = torch.tensor([0, 7, 8, 15, 16, 17, 18, 19, 20, 39, 126])
        _, subset_categories = (
            command_sampling.sample_stage2b_lateral_acquisition_commands(
                subset,
                generator=torch.Generator().manual_seed(2030),
            )
        )
        torch.testing.assert_close(subset_categories, categories[subset])

    def test_stage2d_oblique_homotopy_has_exact_mix_ranges_and_sign_pairs(self):
        bucket_indices = torch.arange(400)
        ranges = {
            "oblique_forward_range": (0.16, 0.22),
            "oblique_lateral_abs_range": (0.02, 0.05),
            "forward_anchor_range": (0.20, 0.30),
            "yaw_anchor_forward_range": (0.18, 0.24),
            "yaw_anchor_abs_range": (0.10, 0.20),
        }
        commands, categories = (
            command_sampling.sample_stage2d_oblique_homotopy_commands(
                bucket_indices,
                **ranges,
                generator=torch.Generator().manual_seed(2031),
            )
        )
        counts = torch.bincount(
            categories, minlength=command_sampling.NUM_STAGE2D_CATEGORIES
        )
        # Category IDs are forward, oblique, forward+yaw.
        torch.testing.assert_close(counts, torch.tensor([80, 280, 40]))

        oblique = categories == command_sampling.STAGE2D_OBLIQUE
        forward = categories == command_sampling.STAGE2D_FORWARD
        yaw = categories == command_sampling.STAGE2D_FORWARD_YAW
        self.assertTrue(torch.all(commands[oblique, 0] >= 0.16))
        self.assertTrue(torch.all(commands[oblique, 0] <= 0.22))
        self.assertTrue(torch.all(torch.abs(commands[oblique, 1]) >= 0.02))
        self.assertTrue(torch.all(torch.abs(commands[oblique, 1]) <= 0.05))
        self.assertTrue(torch.all(commands[oblique, 2] == 0.0))
        self.assertEqual(int(torch.count_nonzero(commands[oblique, 1] > 0.0)), 140)
        self.assertEqual(int(torch.count_nonzero(commands[oblique, 1] < 0.0)), 140)

        self.assertTrue(torch.all(commands[forward, 0] >= 0.20))
        self.assertTrue(torch.all(commands[forward, 0] <= 0.30))
        self.assertTrue(torch.all(commands[forward, 1:] == 0.0))

        self.assertTrue(torch.all(commands[yaw, 0] >= 0.18))
        self.assertTrue(torch.all(commands[yaw, 0] <= 0.24))
        self.assertTrue(torch.all(commands[yaw, 1] == 0.0))
        self.assertTrue(torch.all(torch.abs(commands[yaw, 2]) >= 0.10))
        self.assertTrue(torch.all(torch.abs(commands[yaw, 2]) <= 0.20))
        self.assertEqual(int(torch.count_nonzero(commands[yaw, 2] > 0.0)), 20)
        self.assertEqual(int(torch.count_nonzero(commands[yaw, 2] < 0.0)), 20)

        # Pairing is exact inside every complete twenty-environment period,
        # not merely over the aggregate fleet.
        for start in range(0, 400, command_sampling.STAGE2D_BUCKET_PERIOD):
            period_commands = commands[start : start + command_sampling.STAGE2D_BUCKET_PERIOD]
            self.assertEqual(int(torch.count_nonzero(period_commands[:, 1] > 0.0)), 7)
            self.assertEqual(int(torch.count_nonzero(period_commands[:, 1] < 0.0)), 7)
            self.assertEqual(int(torch.count_nonzero(period_commands[:, 2] > 0.0)), 1)
            self.assertEqual(int(torch.count_nonzero(period_commands[:, 2] < 0.0)), 1)

    def test_stage2d_assignment_and_sign_survive_asynchronous_episode_resets(self):
        ranges = {
            "oblique_forward_range": (0.16, 0.22),
            "oblique_lateral_abs_range": (0.02, 0.05),
            "forward_anchor_range": (0.20, 0.30),
            "yaw_anchor_forward_range": (0.18, 0.24),
            "yaw_anchor_abs_range": (0.10, 0.20),
        }
        all_commands, all_categories = (
            command_sampling.sample_stage2d_oblique_homotopy_commands(
                torch.arange(160),
                **ranges,
                generator=torch.Generator().manual_seed(2032),
            )
        )
        subset = torch.tensor([0, 1, 13, 14, 17, 18, 19, 20, 39, 126, 159])
        reset_commands, reset_categories = (
            command_sampling.sample_stage2d_oblique_homotopy_commands(
                subset,
                **ranges,
                generator=torch.Generator().manual_seed(2033),
            )
        )
        torch.testing.assert_close(reset_categories, all_categories[subset])
        torch.testing.assert_close(
            torch.sign(reset_commands[:, 1:]),
            torch.sign(all_commands[subset, 1:]),
        )
        # Physical magnitudes are still resampled at the episode boundary.
        self.assertFalse(torch.equal(reset_commands[:, 0], all_commands[subset, 0]))

    def test_stage2d_allows_pure_y_endpoint_but_keeps_other_ranges_positive(self):
        valid = {
            "oblique_forward_range": (0.16, 0.22),
            "oblique_lateral_abs_range": (0.02, 0.05),
            "forward_anchor_range": (0.20, 0.30),
            "yaw_anchor_forward_range": (0.18, 0.24),
            "yaw_anchor_abs_range": (0.10, 0.20),
        }
        pure_y = dict(valid)
        pure_y["oblique_forward_range"] = (0.0, 0.0)
        commands, categories = (
            command_sampling.sample_stage2d_oblique_homotopy_commands(
                torch.arange(20),
                **pure_y,
                generator=torch.Generator().manual_seed(2034),
            )
        )
        oblique = categories == command_sampling.STAGE2D_OBLIQUE
        self.assertTrue(torch.all(commands[oblique, 0] == 0.0))
        self.assertTrue(torch.all(torch.abs(commands[oblique, 1]) >= 0.02))
        self.assertTrue(torch.all(torch.abs(commands[oblique, 1]) <= 0.05))

        invalid_oblique = dict(valid)
        invalid_oblique["oblique_forward_range"] = (-0.01, 0.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            command_sampling.sample_stage2d_oblique_homotopy_commands(
                torch.arange(20), **invalid_oblique
            )

        for name in set(valid) - {"oblique_forward_range"}:
            invalid = dict(valid)
            invalid[name] = (0.0, valid[name][1])
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "strictly positive"
            ):
                command_sampling.sample_stage2d_oblique_homotopy_commands(
                    torch.arange(20), **invalid
                )

    def test_active_axis_gaussian_reward_masks_inactive_commands(self):
        commands = torch.tensor([0.0, 0.009, 0.010, 0.011, 0.10, -0.10])
        achieved = torch.tensor([0.0, 99.0, 99.0, 0.011, 0.10, 0.10])
        actual = command_sampling.active_axis_gaussian_tracking_reward(
            commands,
            achieved,
            tracking_std=0.20,
            reward_scale=3.0,
            active_threshold=0.01,
        )
        expected = torch.tensor(
            [
                0.0,
                0.0,
                0.0,
                3.0,
                3.0,
                3.0 * torch.exp(torch.tensor(-1.0)),
            ]
        )
        torch.testing.assert_close(actual, expected)

    def test_empty_batches_keep_expected_shapes(self):
        commands, categories = command_sampling.sample_velocity_command_mixture(
            0,
            lin_vel_x_range=(-1.0, 1.0),
            lin_vel_y_range=(-1.0, 1.0),
            ang_vel_z_range=(-1.0, 1.0),
            category_probabilities=(0.1, 0.2, 0.2, 0.2, 0.3),
            device="cpu",
        )
        self.assertEqual(commands.shape, (0, 3))
        self.assertEqual(categories.shape, (0,))
        self.assertEqual(
            command_sampling.sample_uniform_intervals(0, (6.0, 10.0), device="cpu").shape,
            (0,),
        )
        recovery_commands, recovery_categories = (
            command_sampling.sample_stage1_recovery_commands(
                torch.empty(0, dtype=torch.long)
            )
        )
        self.assertEqual(recovery_commands.shape, (0, 3))
        self.assertEqual(recovery_categories.shape, (0,))
        stage2_commands, stage2_categories = (
            command_sampling.sample_stage2_recovery_commands(
                torch.empty(0, dtype=torch.long)
            )
        )
        self.assertEqual(stage2_commands.shape, (0, 3))
        self.assertEqual(stage2_categories.shape, (0,))
        stage2b_commands, stage2b_categories = (
            command_sampling.sample_stage2b_lateral_acquisition_commands(
                torch.empty(0, dtype=torch.long)
            )
        )
        self.assertEqual(stage2b_commands.shape, (0, 3))
        self.assertEqual(stage2b_categories.shape, (0,))
        stage2d_commands, stage2d_categories = (
            command_sampling.sample_stage2d_oblique_homotopy_commands(
                torch.empty(0, dtype=torch.long),
                oblique_forward_range=(0.16, 0.22),
                oblique_lateral_abs_range=(0.02, 0.05),
                forward_anchor_range=(0.20, 0.30),
                yaw_anchor_forward_range=(0.18, 0.24),
                yaw_anchor_abs_range=(0.10, 0.20),
            )
        )
        self.assertEqual(stage2d_commands.shape, (0, 3))
        self.assertEqual(stage2d_categories.shape, (0,))


if __name__ == "__main__":
    unittest.main()
