"""Check startup without moving six planted feet to stride endpoints."""
from dataclasses import replace
import json
import unittest

import numpy as np

from locomotion.tripod import PHASE
from locomotion.tripod_config import SWEEPS
from locomotion.tripod_evaluate import cases_for
from locomotion.tests.test_tripod import measured_fixture
from locomotion.tests.test_tripod_geometry import make_geometry


class TripodStartupTests(unittest.TestCase):
    def test_native_snapshot_serializes_after_startup_activation(self):
        c = make_geometry(SWEEPS['startup'][0])
        for _ in range(5):c.step(np.array([.1, 0., 0.]), np.full(6, 12.))
        recorded = json.loads(json.dumps(c.snapshot(), allow_nan=False))
        self.assertIs(recorded['ramping_stride'], True)
        self.assertEqual(recorded['state'], 'start')

    def test_hold_standing_then_grow_stride_over_one_cycle_and_reset(self):
        c = make_geometry(SWEEPS['startup'][0]);endpoints = {}
        for _ in range(250):
            before = c.state
            c.step([.1, 0., 0.], measured_fixture(c))
            if before in ('idle', 'start'):
                np.testing.assert_array_equal(c.target, c.neutral)
            if c.state == 'walk' and c.progress == 0. and c.half in (1, 2):
                endpoints[c.half] = c.target[:, 0].copy()
            self.assertIsNone(c.fault)
        coefficient = c._sweep()[:, 0]
        np.testing.assert_allclose(endpoints[1], .5*coefficient*np.sin(np.pi/2+PHASE), atol=1e-12)
        np.testing.assert_allclose(endpoints[2], coefficient*np.sin(3*np.pi/2+PHASE), atol=1e-12)
        c.reset();self.assertEqual(c.stride_elapsed, 0.)
        np.testing.assert_array_equal(c.step([.1, 0., 0.], np.full(6, 12.)), c.neutral.reshape(18))
        with self.assertRaises(ValueError):
            replace(SWEEPS['pd_filtered'][0], startup_stride_ramp=True)

    def test_startup_stop_and_command_change_keep_feet_standing(self):
        for requested in ([0., 0., 0.], [.05, 0., 0.]):
            c = make_geometry(SWEEPS['startup'][0])
            for _ in range(10):c.step([.1, 0., 0.], np.full(6, 12.))
            for _ in range(60):
                before = c.state;c.step(requested, measured_fixture(c))
                if not any(requested):
                    self.assertNotEqual(c.state, 'walk')
                    np.testing.assert_array_equal(c.target, c.neutral)
                elif before == 'start':
                    np.testing.assert_array_equal(c.command, requested)
            self.assertIsNone(c.fault)
            if not any(requested):self.assertTrue(c.stopped)

    def test_full_commands_and_clearance_preserve_targets_and_canonical_case(self):
        from locomotion.evaluate import case_manifest
        expected = next(c for c in case_manifest() if c['case_id'] == 'static:translate_0.10_0deg')
        self.assertEqual(cases_for('forward_high'), [expected])
        for touchdown in (.51, .7, .95):
            c = make_geometry(SWEEPS['startup'][0]);targets = [c.target.reshape(18).copy()]
            for command, mode in [([.1, 0., 0.], 'low'), ([0., 0., .2], 'raised'),
                                  ([0., 0., -.2], 'raised'), ([.1, 0., 0.], 'raised'),
                                  ([0., 0., 0.], 'low')]:
                for _ in range(300):
                    q = c.step(command, measured_fixture(c, touchdown=touchdown), mode)
                    c._validate_target(q.reshape(6, 3));targets.append(q)
                self.assertIsNone(c.fault)
            self.assertTrue(c.stopped)
            self.assertLessEqual(np.max(abs(np.diff(targets, axis=0))), .040000001)

    def test_early_yaw_touchdown_returns_lift_before_the_next_half_cycle(self):
        c = make_geometry(SWEEPS['startup'][0]);held = [];observed_landing = False
        for _ in range(250):
            half = c.half
            c.step([0., 0., .2], measured_fixture(c, touchdown=.51), 'raised')
            if c.state == 'walk' and c.landed.any():
                held.append(float(np.linalg.norm(c.pitch_hold[c.landed])))
                observed_landing = True
            if c.half > half:
                np.testing.assert_array_equal(c.pitch_hold, np.zeros((6, 2)))
                if held:
                    self.assertTrue(np.all(np.diff(held) <= 0.))
                    held = []
        self.assertTrue(observed_landing)


if __name__ == '__main__':
    unittest.main()
