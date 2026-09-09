"""CPU checks of the actual physics-rate setpoint scheduler, without Isaac."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
for package in ("hexapod_core", "hexapod_env"):
    sys.path.insert(0, str(ROOT / "packages" / package))
from hexapod_core import fourbar_v1 as contract
from hexapod_env.tasks.mkii_fourbar_v1.math import MotorCoordinates
from hexapod_env.tasks.mkii_fourbar_v1.target_schedule import MotorTargetRamp


class MotorTargetRampTests(unittest.TestCase):
    def test_1600hz_ramp_preserves_800hz_targets_at_common_physical_times(self):
        initial = torch.linspace(-1., 1., 54, dtype=torch.float64).reshape(3, 18)
        endpoint = initial + torch.linspace(-.04, .04, 18, dtype=torch.float64)
        coarse, fine = MotorTargetRamp(initial, substeps=16), MotorTargetRamp(initial, substeps=32)
        coarse.begin(endpoint)
        fine.begin(endpoint)
        previous = initial
        for index in range(1, 33):
            current = fine.step()
            self.assertLessEqual(float((current-previous).abs().max()), .00125 + 1e-15)
            if index % 2 == 0:
                self.assertTrue(torch.equal(current, coarse.step()))
            previous = current
        self.assertTrue(torch.equal(current, endpoint))
        with self.assertRaises(RuntimeError):
            fine.step()

    def test_32_substep_partial_reset_preserves_other_rows_and_phase(self):
        initial = torch.zeros(3, 18, dtype=torch.float64)
        endpoint = torch.tensor([.016, .032, -.04], dtype=torch.float64)[:, None].repeat(1, 18)
        ramp = MotorTargetRamp(initial, substeps=32)
        ramp.begin(endpoint)
        for _ in range(11):
            ramp.step()
        ramp.reset(torch.full((1, 18), .5, dtype=torch.float64), env_ids=[1])
        for index in range(12, 33):
            delivered = ramp.step()
            self.assertTrue(torch.equal(delivered[1], torch.full((18,), .5, dtype=torch.float64)))
            torch.testing.assert_close(delivered[[0, 2]], endpoint[[0, 2]] * index/32, rtol=0, atol=1e-17)
        self.assertTrue(torch.equal(delivered[[0, 2]], endpoint[[0, 2]]))

    def test_all16_fractions_reach_exact_endpoint_without_initial_jump(self):
        initial = torch.linspace(-1., 1., 54, dtype=torch.float64).reshape(3, 18)
        endpoint = initial + torch.linspace(-.04, .04, 18, dtype=torch.float64)
        ramp = MotorTargetRamp(initial)
        ramp.begin(endpoint)
        for index in range(1, 17):
            actual = ramp.step()
            expected = initial + (endpoint-initial) * index/16
            torch.testing.assert_close(actual, expected, rtol=0, atol=2e-16)
        self.assertTrue(torch.equal(actual, endpoint))

    def test_sign_reversal_starts_at_last_delivered_target_and_never_overshoots(self):
        initial = torch.zeros(2, 18, dtype=torch.float64)
        ramp = MotorTargetRamp(initial)
        previous = initial
        for endpoint in (torch.full_like(initial, .04), torch.zeros_like(initial),
                         torch.full_like(initial, -.04), torch.zeros_like(initial)):
            start = previous.clone()
            ramp.begin(endpoint)
            for index in range(16):
                current = ramp.step()
                self.assertTrue(bool((current >= torch.minimum(start, endpoint)).all()))
                self.assertTrue(bool((current <= torch.maximum(start, endpoint)).all()))
                self.assertLessEqual(float((current-previous).abs().max()), .0025 + 1e-15)
                previous = current
            self.assertTrue(torch.equal(previous, endpoint))

    def test_processed_action_endpoints_and_actual_joint_limits_are_preserved(self):
        kin = contract.load_kinematics(ROOT / contract.KINEMATICS_PATH)
        coordinates = MotorCoordinates(contract.TREE_JOINT_NAMES, kin, dtype=torch.float64)
        target = coordinates.default.repeat(3, 1)
        ramp = MotorTargetRamp(target, substeps=contract.DECIMATION)
        generator = torch.Generator().manual_seed(34)
        previous = target.clone()
        for _ in range(30):
            action = torch.randn((3, 18), generator=generator, dtype=torch.float64) * 3
            _, target, _ = coordinates.process_action(action, target, step_dt=contract.POLICY_DT_S)
            ramp.begin(target)
            for _ in range(contract.DECIMATION):
                delivered = ramp.step()
                self.assertTrue(bool((delivered >= coordinates.soft_limits[:, 0]).all()))
                self.assertTrue(bool((delivered <= coordinates.soft_limits[:, 1]).all()))
                self.assertLessEqual(float((delivered-previous).abs().max()),
                                    contract.SLEW_RAD_PER_20MS/contract.DECIMATION + 1e-15)
                previous = delivered
            self.assertTrue(torch.equal(delivered, target))

    def test_selected_reset_holds_only_selected_rows_without_restarting_other_phases(self):
        initial = torch.zeros(3, 18, dtype=torch.float64)
        endpoint = torch.tensor([.016, .032, -.048], dtype=torch.float64)[:, None].repeat(1, 18)
        ramp = MotorTargetRamp(initial)
        ramp.begin(endpoint)
        for _ in range(5):
            ramp.step()
        reset = torch.tensor([.5, -.7], dtype=torch.float64)[:, None].repeat(1, 18)
        ramp.reset(reset, env_ids=torch.tensor([2, 0], dtype=torch.int32))
        reset.fill_(99.)  # The reset buffer belongs to the caller.
        for index in range(6, 17):
            delivered = ramp.step()
            self.assertTrue(torch.equal(delivered[0], torch.full((18,), -.7, dtype=torch.float64)))
            self.assertTrue(torch.equal(delivered[2], torch.full((18,), .5, dtype=torch.float64)))
            torch.testing.assert_close(delivered[1], endpoint[1]*index/16, rtol=0, atol=1e-17)
        next_endpoint = delivered + .016
        ramp.begin(next_endpoint)
        torch.testing.assert_close(ramp.step(), delivered + .001, rtol=0, atol=2e-16)

    def test_full_reset_cancels_interval_and_requires_new_endpoint(self):
        ramp = MotorTargetRamp(torch.zeros(2, 18))
        ramp.begin(torch.ones(2, 18))
        ramp.step()
        reset = torch.full((2, 18), .2)
        ramp.reset(reset)
        with self.assertRaises(RuntimeError):
            ramp.step()
        ramp.begin(torch.full((2, 18), .216))
        torch.testing.assert_close(ramp.step(), torch.full((2, 18), .201))

    def test_fraction_count_is_strict_and_failed_early_begin_does_not_change_ramp(self):
        for count in (1, 4, 16, 32):
            with self.subTest(count=count):
                ramp = MotorTargetRamp(torch.zeros(1, 18), substeps=count)
                with self.assertRaises(RuntimeError):
                    ramp.step()
                ramp.begin(torch.ones(1, 18))
                with self.assertRaises(RuntimeError):
                    ramp.begin(torch.full((1, 18), -1.))
                for index in range(1, count+1):
                    self.assertTrue(torch.equal(ramp.step(), torch.full((1, 18), index/count)))
                with self.assertRaises(RuntimeError):
                    ramp.step()
        for count in (0, -1, True, 16., math.nan):
            with self.subTest(count=count), self.assertRaises(ValueError):
                MotorTargetRamp(torch.zeros(1, 18), substeps=count)

    def test_initial_endpoint_and_returned_buffers_cannot_mutate_scheduler(self):
        initial = torch.zeros(1, 18)
        endpoint = torch.full((1, 18), .016)
        ramp = MotorTargetRamp(initial)
        initial.fill_(99.)
        ramp.begin(endpoint)
        endpoint.fill_(99.)
        first = ramp.step()
        first.fill_(99.)
        torch.testing.assert_close(ramp.step(), torch.full((1, 18), .002))
        for _ in range(14):
            final = ramp.step()
        self.assertTrue(torch.equal(final, torch.full((1, 18), .016)))
        final.fill_(99.)
        ramp.begin(torch.full((1, 18), .032))
        torch.testing.assert_close(ramp.step(), torch.full((1, 18), .017))

    def test_invalid_target_inputs_fail_without_corrupting_existing_state(self):
        wrong = [torch.zeros(1, 30), torch.zeros(18), torch.zeros(1, 18, dtype=torch.long),
                 torch.full((1, 18), math.nan), torch.full((1, 18), math.inf), [[0.]*18]]
        for value in wrong:
            with self.subTest(value=str(type(value))), self.assertRaises(ValueError):
                MotorTargetRamp(value)
        ramp = MotorTargetRamp(torch.zeros(1, 18))
        for value in [*wrong, torch.zeros(2, 18), torch.zeros(1, 18, dtype=torch.float64)]:
            with self.assertRaises(ValueError):
                ramp.begin(value)
        ramp.begin(torch.full((1, 18), .016))
        with self.assertRaises(ValueError):
            ramp.reset(torch.full((1, 18), math.nan))
        torch.testing.assert_close(ramp.step(), torch.full((1, 18), .001))

    def test_invalid_selected_resets_are_atomic_and_empty_selection_is_noop(self):
        ramp = MotorTargetRamp(torch.zeros(3, 18))
        ramp.begin(torch.full((3, 18), .016))
        ramp.step()
        for indices in ([0, 0], [-1], [3], [1.5], [True], [[0]]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                ramp.reset(torch.ones(len(indices), 18), env_ids=indices)
        with self.assertRaises(ValueError):
            ramp.reset(torch.ones(3, 18), env_ids=[1])
        ramp.reset(torch.empty(0, 18), env_ids=[])
        torch.testing.assert_close(ramp.step(), torch.full((3, 18), .002))


if __name__ == "__main__":
    unittest.main()
