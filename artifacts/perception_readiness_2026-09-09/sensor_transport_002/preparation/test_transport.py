"""CPU regressions, including actual old-fail/new-pass counterexamples."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import torch

HERE = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module

old = load('_transport_original', HERE/'original/sensor_model.py')
new = load('_transport_successor', HERE/'sensor_model.py')

def stream(module=new, *, delayed=False):
    return module._DelayedStream(module.TransportModelCfg(.02, .04 if delayed else 0., .04 if delayed else 0., 0., .2), 1)

def enqueue(s, value, t, valid=None):
    v = torch.tensor(value, dtype=torch.float32).reshape(-1, 1)
    s.enqueue({'range': v}, t, torch.ones(len(v), dtype=torch.bool) if valid is None else torch.tensor(valid))

def capture_all(m, t):
    m.capture_camera(torch.full((3, 2, 3, 3), 128, dtype=torch.uint8), torch.ones(3, 2, 3), t)
    m.capture_lidar(torch.tensor([[[1., 0., 0.], [0., 2., 0.]]] * 3), torch.zeros(3, 3), t)
    m.capture_imu(torch.zeros(3, 3), torch.tensor([[0., 0., 9.81]] * 3), t)

class TransportTest(unittest.TestCase):
    def test_old_failure_and_new_fix_late_old_arrival_per_row(self):
        outcomes = []
        for module in (old, new):
            s = stream(module, delayed=True); s._jitter_rng.uniform = Mock(side_effect=[.04, -.04])
            enqueue(s, [10., 11., 12.], 0., [True, True, False])
            enqueue(s, [20., 21., 22.], .02, [True, False, True])
            first = s.read(.025)
            self.assertEqual(first.valid.tolist(), [True, False, True])
            last = s.read(.09); outcomes.append(last.values['range'].flatten().tolist())
        self.assertEqual(outcomes[0], [10., 11., 22.])
        self.assertEqual(outcomes[1], [20., 11., 22.])

    def test_equal_capture_time_tie_is_last_enqueued_valid_per_row(self):
        s = stream(delayed=True); s._jitter_rng.uniform = Mock(side_effect=[.04, -.04])
        enqueue(s, [10., 11.], 0., [True, True])
        enqueue(s, [20., 21.], 0., [True, False])
        s.read(.01); frame = s.read(.09)
        self.assertEqual(frame.values['range'].flatten().tolist(), [20., 11.])
        self.assertEqual(s._order_sequence.tolist(), [1, 0])

    def test_one_read_drains_reordered_frames_without_regressing(self):
        s = stream(delayed=True); s._jitter_rng.uniform = Mock(side_effect=[.04, -.04])
        enqueue(s, [10.], 0.); enqueue(s, [20.], .02)
        self.assertEqual(s.read(.1).values['range'].item(), 20.)

    def test_order_does_not_use_rounded_float32_capture_times(self):
        s = stream(delayed=True); s._jitter_rng.uniform = Mock(side_effect=[.04, -.04])
        enqueue(s, [10.], 10000.); enqueue(s, [20.], 10000.0001)
        self.assertEqual(s.read(10000.1).values['range'].item(), 20.)
        self.assertAlmostEqual(s._order_time_s.item(), 10000.0001, places=8)

    def test_old_inference_failure_and_new_pending_delivered_reset(self):
        s = stream(old)
        with torch.inference_mode():
            enqueue(s, [10., 20.], 0.); s.read(.01)
        with self.assertRaisesRegex(RuntimeError, 'inference tensor'):
            s.reset([0])
        s = stream(delayed=True)
        with torch.inference_mode():
            enqueue(s, [10., 20., 30.], 0.)
        self.assertFalse(torch.is_inference(s._pending[0].valid))
        self.assertFalse(torch.is_inference(s._pending[0].values['range']))
        s.reset([1])  # Before the first delivery.
        with torch.inference_mode():
            f = s.read(.1)
            enqueue(s, [40., 50., 60.], .1)
        self.assertEqual(f.valid.tolist(), [True, False, True])
        keep = {k: v[[0, 2]].clone() for k, v in f.values.items()}
        s.reset([1])  # Delivered and pending rows coexist.
        for k, value in keep.items(): self.assertTrue(torch.equal(s._latest_values[k][[0, 2]], value))
        self.assertEqual(s._pending[0].valid.tolist(), [True, False, True])
        self.assertFalse(torch.is_inference(s._order_sequence))
        self.assertEqual(s.read(.2).valid.tolist(), [True, False, True])
        enqueue(s, [70., 80., 90.], .2)
        self.assertEqual(s.read(.3).values['range'].flatten().tolist(), [70., 80., 90.])

    def test_full_sensor_model_inference_reset_including_imu_biases(self):
        a, b = new.Phase3SensorModel(seed=13), new.Phase3SensorModel(seed=13)
        with torch.inference_mode():
            capture_all(a, 0.); a.read(.15)
        capture_all(b, 0.); b.read(.15)
        unchanged_bias = a._gyro_bias[[0, 2]].clone()
        for name in ('_gyro_bias', '_accel_bias', '_imu_bias_initialized'):
            self.assertFalse(torch.is_inference(getattr(a, name)))
        a.reset([1]); b.reset([1])
        self.assertTrue(torch.equal(a._gyro_bias[[0, 2]], unchanged_bias))
        self.assertEqual(a._imu_bias_initialized.tolist(), [True, False, True])
        with torch.inference_mode(): capture_all(a, .2)
        capture_all(b, .2)
        self.assert_frames_equal(a.read(.4), b.read(.4))
        a.reset(); capture_all(a, 0.); a.read(.1)  # New explicit full epoch.

    def test_old_rewound_read_failure_and_new_atomic_rejection(self):
        s = stream(old); enqueue(s, [1.], 1.); s.read(1.01)
        self.assertLess(s.read(.5).age_s.item(), 0.)
        s = stream(); enqueue(s, [1.], 1.); s.read(1.01)
        before = s._capture_time_s.clone()
        for value in (.5, float('nan'), float('inf'), -float('inf'), -.1):
            with self.subTest(value=value), self.assertRaises(ValueError): s.read(value)
            self.assertTrue(torch.equal(before, s._capture_time_s))
            self.assertEqual(s._last_read_s, 1.01)

    def test_clock_validation_precedes_capture_rng_or_bias_mutation(self):
        model = new.Phase3SensorModel(seed=29)
        for value in (float('nan'), float('inf'), -float('inf'), -.1):
            with self.subTest(value=value), self.assertRaises(ValueError): capture_all(model, value)
        self.assertFalse(model._random._generators)
        self.assertIsNone(model._gyro_bias)
        capture_all(model, 1.)
        before = model._gyro_bias.clone()
        rng_states = {k: g.get_state().clone() for k, g in model._random._generators.items()}
        for dt in (float('nan'), float('inf'), 0., -1.):
            with self.assertRaises(ValueError): model.capture_imu(torch.zeros(3, 3), torch.zeros(3, 3), 1.1, dt_s=dt)
        with self.assertRaises(ValueError): model.capture_imu(torch.zeros(3, 3), torch.zeros(3, 3), .9)
        self.assertTrue(torch.equal(before, model._gyro_bias))
        for key, state in rng_states.items(): self.assertTrue(torch.equal(state, model._random._generators[key].get_state()))

    def test_full_reset_restarts_epoch_partial_reset_cannot_rewind(self):
        s = stream(); enqueue(s, [1., 2.], 1.); s.read(1.1); s.reset([0])
        with self.assertRaises(ValueError): s.read(0.)
        with self.assertRaises(ValueError): enqueue(s, [3., 4.], .5)
        s.reset(); enqueue(s, [3., 4.], 0.)
        self.assertEqual(s.read(0.).values['range'].flatten().tolist(), [3., 4.])

    def test_model_read_prevalidates_all_streams_without_partial_drain(self):
        model = new.Phase3SensorModel(); capture_all(model, 0.)
        model._imu_stream.read(.2)
        pending = len(model._camera_stream._pending)
        with self.assertRaises(ValueError): model.read(.1)
        self.assertEqual(len(model._camera_stream._pending), pending)
        self.assertIsNone(model._camera_stream._last_read_s)

    def test_near_future_capture_not_released_by_epsilon(self):
        s = stream(); enqueue(s, [1.], 5e-10)
        self.assertIsNone(s.read(0.)); self.assertEqual(len(s._pending), 1)
        self.assertTrue(s.read(5e-10).valid.item())

    def test_invalid_new_sample_holds_prior_until_original_lease_expires(self):
        s = stream(); enqueue(s, [1.], 0.); s.read(0.)
        enqueue(s, [2.], .1, [False])
        self.assertEqual(s.read(.15).values['range'].item(), 1.)
        self.assertFalse(s.read(.21).valid.item())
        self.assertEqual(s._capture_time_s.item(), 0.)

    def test_transport_rejects_nonfinite_parameters(self):
        for key in ('sample_period_s', 'latency_s', 'latency_jitter_s', 'stale_after_s', 'frame_dropout_probability'):
            for value in (float('nan'), float('inf'), -float('inf')):
                fields = dict(sample_period_s=.02, latency_s=.02, latency_jitter_s=.01, frame_dropout_probability=0., stale_after_s=.2)
                fields[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError): new.TransportModelCfg(**fields)

    def test_schema_mismatch_is_rejected_before_enqueue(self):
        s = stream(); enqueue(s, [1., 2.], 0.)
        sequence = s._sequence; rng = s._jitter_rng.getstate()
        with self.assertRaises(ValueError): enqueue(s, [3.], .02)
        self.assertEqual(s._sequence, sequence); self.assertEqual(s._jitter_rng.getstate(), rng)
        self.assertEqual(s._last_capture_s, 0.)

    def assert_frames_equal(self, a, b):
        self.assertEqual(set(a), set(b))
        for stream_name in a:
            x, y = a[stream_name], b[stream_name]
            for field in ('valid', 'age_s', 'capture_time_s'):
                self.assertTrue(torch.equal(getattr(x, field), getattr(y, field)), (stream_name, field))
            self.assertEqual(set(x.values), set(y.values))
            for field in x.values: self.assertTrue(torch.equal(x.values[field], y.values[field]), (stream_name, field))

    def test_nominal_cadence_dropout_noise_seeded_parity(self):
        # Normal configured cadences/jitter never reorder; original numerical
        # outputs, timestamps, staleness and random draws remain bitwise equal.
        a, b = old.Phase3SensorModel(seed=81), new.Phase3SensorModel(seed=81)
        rgb = torch.full((3, 2, 3, 3), 180, dtype=torch.uint8); depth = torch.ones(3, 2, 3)
        hits = torch.tensor([[[1., 0., 0.], [0., 2., 0.]]] * 3); pos = torch.zeros(3, 3)
        next_capture = {'camera': 0, 'lidar': 0, 'imu': 0}
        periods = {'camera': 1/30, 'lidar': .1, 'imu': .005}
        for step in range(401):
            now = step * .005
            for sensor, period in periods.items():
                while next_capture[sensor] * period <= now:
                    t = next_capture[sensor] * period
                    for model in (a, b):
                        if sensor == 'camera': model.capture_camera(rgb, depth, t)
                        elif sensor == 'lidar': model.capture_lidar(hits, pos, t)
                        else: model.capture_imu(torch.zeros(3, 3), torch.ones(3, 3), t)
                    next_capture[sensor] += 1
            self.assert_frames_equal(a.read(now), b.read(now))
            if step == 151:
                a.reset([1]); b.reset([1])
        self.assert_frames_equal(a.read(2.5), b.read(2.5))

if __name__ == '__main__': unittest.main()
