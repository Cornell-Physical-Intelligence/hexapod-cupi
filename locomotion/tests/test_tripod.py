"""CPU evidence for the paper waveform and approved-geometry adaptation."""
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from locomotion.env_config import JOINT_NAMES, LEGS, MODEL_SHA256, sha
from locomotion.tripod import PHASE, TRIPOD_A, TripodController, joint_order, supported, wave
from locomotion.tripod_config import SWEEP, TripodConfig
from locomotion.tripod_evaluate import NativeController, cases_for, clearance_at

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT/'robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json'
MODEL = json.loads(MODEL_PATH.read_text())
STANCE = json.loads((MODEL_PATH.parent/'stance.json').read_text())
JOINTS = {j['name']: j for j in MODEL['joints']}


def make(config=None):
    return TripodController(STANCE['nominal_joint_position_rad'],
        [JOINTS[n]['lower'] for n in JOINT_NAMES], [JOINTS[n]['upper'] for n in JOINT_NAMES], config)


def measured_fixture(controller, *, touchdown=.88):
    """A synthetic contact sequence exercises logic and supplies no native proof."""
    force = np.full(6, 12.)
    if controller.state == 'walk' and .08 < controller.progress < touchdown:
        force[controller.swing] = 0.
    return force


def rotation(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def feet(q):
    """Compose approved joint transforms, independent of the controller equations."""
    poses = {'body': (np.zeros(3), np.eye(3))}
    for name, angle in zip(JOINT_NAMES, np.asarray(q).reshape(18)):
        j = JOINTS[name]; p, r = poses[j['parent']]
        x, y, z = j['axis']; k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        local = np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)
        poses[j['child']] = (p+r@j['xyz'], r@rotation(j['quaternion_xyzw'])@local)
    return np.array([poses[leg+'_tibia'][0]+poses[leg+'_tibia'][1]@point
                     for leg, point in zip(LEGS, STANCE['toe_local_points_m'])])


class TripodTests(unittest.TestCase):
    def test_named_joint_mapping_and_model_identity(self):
        self.assertEqual(sha(MODEL_PATH), MODEL_SHA256)
        names = list(reversed(JOINT_NAMES))
        self.assertEqual([names[i] for i in joint_order(names)], list(JOINT_NAMES))
        with self.assertRaises(ValueError): joint_order([JOINT_NAMES[0]]*18)
        np.testing.assert_allclose(feet(STANCE['nominal_joint_position_rad']), STANCE['neutral_feet_body_m'], atol=1e-10)

    def test_paper_values_and_tripod_phases(self):
        np.testing.assert_allclose(wave([0, math.pi/2, math.pi, 3*math.pi/2])[0], [0, 1, 0, -1], atol=1e-14)
        np.testing.assert_allclose(wave([0, math.pi/2, math.pi, 3*math.pi/2])[1], [1, 0, 0, 0], atol=1e-14)
        self.assertEqual([LEGS[i] for i in np.flatnonzero(TRIPOD_A)], ['lf', 'lr', 'rm'])
        for phase in np.linspace(-1.5, 1.5, 50):
            _, lift = wave(phase+PHASE)
            self.assertTrue(np.array_equal(lift > 1e-10, TRIPOD_A))
            np.testing.assert_allclose(wave(phase+PHASE)[0][TRIPOD_A], -wave(phase+PHASE)[0][~TRIPOD_A])

    def test_wave_and_derivative_are_continuous_at_touchdown(self):
        epsilon = 1e-6
        for boundary in (-math.pi/2, math.pi/2, 3*math.pi/2):
            for channel in (0, 1):
                before = float(wave(boundary-epsilon)[channel]); at = float(wave(boundary)[channel])
                after = float(wave(boundary+epsilon)[channel])
                self.assertLess(abs(after-before), 3e-6)
                self.assertLess(abs((after-at)/epsilon-(at-before)/epsilon), 3e-6)

    def test_forward_and_turn_signs_follow_approved_axes(self):
        c = make(); neutral = c.neutral.copy()
        c.command = np.array([.05, 0, 0]); q = neutral.copy(); q[:, 0] += c._hip()*.1
        delta = feet(q)-feet(neutral)
        self.assertTrue(np.all(delta[:, 1] < 0))
        c.command = np.array([0., 0., .2]); q = neutral.copy(); q[:, 0] += c._hip()*.1
        delta = feet(q)-feet(neutral)
        positions = feet(neutral)
        self.assertTrue(np.all(positions[:, 0]*delta[:, 1]-positions[:, 1]*delta[:, 0] > 0))

    def test_geometry_targets_and_sweep_joint_bounds(self):
        for cfg in SWEEP:
            c = make(cfg)
            low, raised = c.stance('low'), c.stance('raised')
            increase = feet(low)[:, 2]-feet(raised)[:, 2]
            self.assertTrue(np.all(increase > .0105))
            for mode, minimum in [('low', .012), ('raised', .016)]:
                base = c.stance(mode); lifted = base.copy(); lifted[:, 1:] += c.lift(mode)
                self.assertTrue(np.all(feet(lifted)[:, 2]-feet(base)[:, 2] >= minimum))
                for phase in np.linspace(0, 2*math.pi, 301):
                    hip, lift = wave(phase+PHASE)
                    target = base.copy(); target[:, 0] += .30*hip
                    target[:, 1:] += lift[:, None]*c.lift(mode)
                    c._validate_target(target)

    def test_support_debounce_and_hysteresis(self):
        c = make()
        c.step([0, 0, 0], np.full(6, 3.)); self.assertFalse(c.contacts.any())
        c.step([0, 0, 0], np.full(6, 3.)); self.assertTrue(c.stopped)
        for _ in range(5): c.step([0, 0, 0], np.full(6, 1.5))
        self.assertTrue(c.contacts.all())
        c.step([0, 0, 0], np.zeros(6)); self.assertTrue(c.contacts.all())
        c.step([0, 0, 0], np.zeros(6)); self.assertFalse(c.contacts.any())

    def test_reset_replays_exact_startup(self):
        c = make()
        def sequence():
            return np.array([c.step([.05, 0, 0], measured_fixture(c)) for _ in range(240)])
        first = sequence(); self.assertIsNone(c.fault)
        c.reset(); second = sequence()
        np.testing.assert_array_equal(first, second)
        self.assertLessEqual(np.max(abs(np.diff(first, axis=0))), .040000001)

    def test_early_touchdown_holds_pitch_and_requires_release(self):
        c = make()
        for _ in range(100):
            c.step([.05, 0, 0], measured_fixture(c, touchdown=.6))
            if c.landed[c.swing].all() and c.state == 'walk': break
        self.assertTrue(c.landed[c.swing].all()); self.assertLess(c.progress, .9)
        held = c.target[c.swing, 1:].copy()
        c.step([.05, 0, 0], measured_fixture(c, touchdown=.6))
        np.testing.assert_array_equal(c.target[c.swing, 1:], held)
        stuck = make()
        for _ in range(100): stuck.step([.05, 0, 0], np.full(6, 12.))
        self.assertEqual(stuck.fault, 'swing_failed_to_lift')

    def test_missing_touchdown_recovery_is_bounded_and_fault_latches(self):
        c = make(); lowest = 0.
        for _ in range(150):
            force = measured_fixture(c, touchdown=2.)
            q = c.step([.05, 0, 0], force).reshape(6, 3)
            lowest = min(lowest, float((q-c.neutral)[:, 1].min()))
        self.assertEqual(c.fault, 'touchdown_timeout'); self.assertGreaterEqual(lowest, -.040000001)
        held = c.target.copy()
        c.step([0, 0, 0], np.full(6, 12.))
        np.testing.assert_array_equal(c.target, held)
        c.reset(); self.assertIsNone(c.fault)

    def test_touchdown_offset_returns_to_stance_before_next_lift(self):
        c = make(); offsets = []
        for _ in range(145):
            half = c.half
            c.step([.05, 0, 0], measured_fixture(c, touchdown=.6))
            if half == 1:
                offsets.append((c.target-c.neutral)[TRIPOD_A, 1:].copy())
        self.assertIsNone(c.fault)
        self.assertGreater(len(offsets), 10)
        values = np.asarray(offsets)
        self.assertGreater(abs(values[0]).max(), .05)
        self.assertTrue(np.all(np.diff(abs(values), axis=0) <= 1e-12))
        np.testing.assert_allclose(values[-1], 0., atol=1e-12)

    def test_stance_loss_and_absent_initial_contact_fault(self):
        c = make()
        for _ in range(110): c.step([.05, 0, 0], np.zeros(6))
        self.assertEqual(c.fault, 'initial_support_timeout')
        c.reset()
        while c.state != 'walk': c.step([.05, 0, 0], np.full(6, 12.))
        for _ in range(25): c.step([.05, 0, 0], np.zeros(6))
        self.assertEqual(c.fault, 'stance_support_loss')

    def test_stop_and_command_clearance_transitions_preserve_continuity(self):
        c = make(); targets = [c.target.reshape(18).copy()]
        for command, mode in [([.05, 0, 0], 'low'), ([0, 0, .2], 'low'),
                              ([0, 0, -.2], 'raised'), ([.05, 0, 0], 'low'), ([0, 0, 0], 'low')]:
            for _ in range(300): targets.append(c.step(command, measured_fixture(c), mode))
            self.assertIsNone(c.fault)
            self.assertEqual(c.mode, mode)
            if any(command): np.testing.assert_array_equal(c.command, command)
        self.assertTrue(c.stopped)
        np.testing.assert_allclose(c.target, c.neutral, atol=1e-12)
        self.assertLessEqual(np.max(abs(np.diff(targets, axis=0))), .040000001)

    def test_reject_unsupported_commands_and_invalid_feedback(self):
        for command in ([-.01, 0, 0], [.05, .01, 0], [.05, 0, .1], [0, 0, .21], [np.nan, 0, 0]):
            self.assertFalse(supported(command))
            with self.assertRaises(ValueError): make().step(command, np.ones(6))
        for force in (np.zeros(5), np.full(6, np.nan), -np.ones(6)):
            with self.assertRaises(ValueError): make().step([0, 0, 0], force)
        with self.assertRaises(ValueError): TripodConfig(period_s=.5)
        with self.assertRaises(ValueError): make(replace(TripodConfig(), raised_offset_rad=(-.5, .5)))

    def test_native_adapter_uses_distal_samples_and_records_exact_input(self):
        c = make(); env = SimpleNamespace(neutral=torch.tensor(c.neutral.reshape(18)),
            lower=torch.tensor(c.lower.reshape(18)), upper=torch.tensor(c.upper.reshape(18)),
            commands=torch.tensor([[.1, 0, 0]]), device='cpu', cfg=SimpleNamespace(action_scale_rad=.35),
            capture=SimpleNamespace(last_control=[]))
        stream = io.StringIO(); policy = NativeController(env, {}, SWEEP[0], stream)
        np.testing.assert_array_equal(policy(None).numpy(), np.zeros((1, 18)))
        env.capture.last_control = [{'distal_force_world_n': np.tile([0., 0., i], (1, 6, 1))} for i in range(8)]
        policy(None); policy(None)
        row = json.loads(stream.getvalue().splitlines()[-1])
        np.testing.assert_array_equal(row['measured_force_n'], np.full(6, 3.5))
        self.assertEqual(row['state'], 'start')
        env.capture.last_control.pop()
        with self.assertRaises(ValueError): policy(None)

    def test_matrix_preserves_existing_cases_and_clearance_schedule(self):
        from locomotion.evaluate import case_manifest
        full = {c['case_id']: c for c in case_manifest()}
        for case in cases_for('qualification'):
            self.assertEqual(case, full[case['case_id']]); self.assertTrue(supported(case['command']))
        self.assertEqual(len(cases_for('screen')), 3)
        switch = next(c for c in cases_for('clearance') if c['clearance'] == 'switch')
        self.assertEqual([clearance_at(switch, i) for i in (0, 300, 1049, 1050)], ['low', 'raised', 'raised', 'low'])
        with self.assertRaises(ValueError): cases_for('terrain')

    def test_prepare_freezes_tripod_source_and_requires_single_robot(self):
        from locomotion.prepare import prepare
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'pack'
            binding = prepare(output, '/home/orionh/HEXAPOD_runs/restart_20260914/tripod_cpu_fixture',
                              mode='tripod', candidate=3, suite='clearance')
            self.assertEqual(binding['module'], 'locomotion.tripod_evaluate')
            self.assertIn('--standing-admission', binding['command_args'])
            self.assertNotIn('--updates', binding['command_args'])
            frozen = json.loads((output/'source/FREEZE_SHA256.json').read_text())
            self.assertEqual(frozen['locomotion/tripod.py'], sha(ROOT/'locomotion/tripod.py'))
            self.assertEqual(frozen['locomotion/tripod_config.py'], sha(ROOT/'locomotion/tripod_config.py'))
            with self.assertRaises(ValueError):
                prepare(Path(temp)/'bad', '/home/orionh/HEXAPOD_runs/restart_20260914/tripod_cpu_fixture',
                        mode='tripod', num_envs=32)

    def test_guard_rejects_mixed_controller_and_training_entries(self):
        from locomotion.tests.test_launch import OwnedLaunchTests
        fixture = OwnedLaunchTests()
        binding = fixture.binding(); binding['mode'] = 'tripod'
        binding['command_args'][1] = 'tripod'
        with self.assertRaises(ValueError): fixture.verify_without_filesystem(binding)
        binding['module'] = 'locomotion.tripod_evaluate'
        fixture.verify_without_filesystem(binding)
        binding['mode'] = 'train'; binding['command_args'][1] = 'train'
        with self.assertRaises(ValueError): fixture.verify_without_filesystem(binding)

    def test_stops_from_each_supported_motion_and_mode(self):
        for command in ([.1, 0, 0], [0, 0, .2], [0, 0, -.2]):
            for mode in ('low', 'raised'):
                c = make()
                for _ in range(220): c.step(command, measured_fixture(c), mode)
                self.assertIsNone(c.fault)
                for _ in range(130): c.step([0, 0, 0], measured_fixture(c), mode)
                self.assertTrue(c.stopped, (command, mode, c.snapshot()))
                np.testing.assert_allclose(c.target, c.stance(mode), atol=1e-12)


class ExtendedClearanceScoringTests(unittest.TestCase):
    def score(self, data, windows=None):
        from locomotion.evaluate import score_case
        return score_case(data, {'profile': 'omni_static', 'command': [.05, 0., 0.],
            'contact_classification': 'exact_distal_points'},
            {'profile': 'omni_static', 'controls': 1800,
             'score_control_windows': [[0, 1000], [800, 1800]] if windows is None else windows})

    def test_full_switch_capture_uses_original_static_thresholds(self):
        from locomotion.tests.test_evaluation import fixture, score
        data = fixture(1800, (.05, 0., 0.))
        result = self.score(data)
        self.assertTrue(result['pass'])
        for row, (start, end) in zip(result['window_reports'], ((0, 1000), (800, 1800))):
            expected = score({k: v[start:end] for k, v in data.items()}, 'omni_static', command=[.05, 0., 0.])
            self.assertEqual(row, expected)

    def test_prefix_or_failure_in_either_window_cannot_pass(self):
        from locomotion.tests.test_evaluation import fixture
        for index in (20, 850, 1799):
            data = fixture(1800, (.05, 0., 0.)); data['terminated'][index] = 1
            self.assertFalse(self.score(data)['pass'])
        data = fixture(1500, (.05, 0., 0.))
        self.assertFalse(self.score(data)['pass'])
        data = fixture(1800, (.05, 0., 0.)); data['velocity_navigation_mps'][1100:, 0] = -.1
        self.assertFalse(self.score(data)['pass'])

    def test_window_gap_and_shortened_duration_are_rejected(self):
        from locomotion.tests.test_evaluation import fixture
        data = fixture(1800, (.05, 0., 0.))
        for windows in ([[0, 1000]], [[100, 1100], [800, 1800]], [[0, 900], [800, 1800]],
                        [[0, 1000], [950, 1950]], [[0, 1000], [800., 1800]], []):
            with self.assertRaises(ValueError):
                self.score(data, windows)


if __name__ == '__main__':
    unittest.main()
