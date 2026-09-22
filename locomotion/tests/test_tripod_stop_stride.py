"""Check standing return through alternating support with retained gates."""
from dataclasses import replace
import json
import unittest

import numpy as np

from locomotion.evaluate import case_manifest
from locomotion.tests.test_tripod import measured_fixture
from locomotion.tests.test_tripod_geometry import make_geometry
from locomotion.tripod_config import SWEEPS
from locomotion.tripod_evaluate import cases_for


class StopStrideTests(unittest.TestCase):
    def test_held_walking_targets_match_the_accepted_controller(self):
        for mode in ('low', 'raised'):
            for command in ([.05, 0., 0.], [.10, 0., 0.], [0., 0., -.2], [0., 0., .2]):
                old = make_geometry(SWEEPS['speed_lift'][0])
                new = make_geometry(SWEEPS['stop_stride'][0])
                for _ in range(250):
                    force = measured_fixture(old)
                    np.testing.assert_array_equal(old.step(command, force, mode), new.step(command, force, mode))
                self.assertIsNone(new.fault)

    def test_stop_places_both_tripods_before_the_standing_blend(self):
        for mode in ('low', 'raised'):
            for command in ([.05, 0., 0.], [.10, 0., 0.], [0., 0., -.2], [0., 0., .2]):
                for stop_control in (175, 190, 205):
                    c = make_geometry(SWEEPS['stop_stride'][0]); targets = []; halves = set()
                    end = None
                    for control in range(stop_control+160):
                        before = c.state
                        q = c.step(command if control < stop_control else [0., 0., 0.],
                                   measured_fixture(c, touchdown=.51), mode)
                        targets.append(q); c._validate_target(q.reshape(6, 3))
                        if c.ramping_stop and c.state == 'walk':
                            halves.add(c.half)
                        if before == 'walk' and c.state == 'settle':
                            end = control
                            np.testing.assert_allclose(q.reshape(6, 3), c.stance(mode), atol=1e-12)
                    self.assertIsNone(c.fault); self.assertTrue(c.stopped)
                    self.assertEqual(len(halves), 2)
                    self.assertIsNotNone(end); self.assertLessEqual(end-stop_control, 90)
                    self.assertLessEqual(np.max(abs(np.diff(targets, axis=0))), .040000001)
                    self.assertFalse(c.ramping_stop)
                    c.reset(); self.assertEqual(c.stop_stride_elapsed, 0.)
                    self.assertFalse(json.loads(json.dumps(c.snapshot()))['ramping_stop'])

    def test_command_changes_and_startup_stop_retain_bounds(self):
        c = make_geometry(SWEEPS['stop_stride'][0]); previous = c.target.reshape(18).copy()
        for command, mode, controls in (([.1, 0., 0.], 'low', 70),
                ([0., 0., 0.], 'low', 200), ([.05, 0., 0.], 'low', 250),
                ([0., 0., .2], 'raised', 400), ([0., 0., -.2], 'raised', 300),
                ([0., 0., 0.], 'low', 350)):
            for _ in range(controls):
                q = c.step(command, measured_fixture(c), mode)
                self.assertLessEqual(np.max(abs(q-previous)), .040000001); previous = q
                c._validate_target(q.reshape(6, 3))
            self.assertIsNone(c.fault)
        self.assertTrue(c.stopped); self.assertEqual(c.mode, 'low')
        with self.assertRaises(ValueError):
            replace(SWEEPS['forward_overlap'][0], stop_stride_ramp=True)

    def test_stop_suite_retains_the_canonical_cases(self):
        expected = [c for c in case_manifest() if c['case_id'] in
                    ('stop:forward', 'stop:turn_left', 'stop:turn_right')]
        self.assertEqual(cases_for('stops'), expected)
        self.assertEqual(len(expected), 3)


if __name__ == '__main__':
    unittest.main()
