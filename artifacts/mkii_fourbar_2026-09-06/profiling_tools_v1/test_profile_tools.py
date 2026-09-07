"""CPU lifecycle and exact-call/coverage tests; no Isaac app or CUDA work."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profile = load('external_profile_tools_test', HERE/'profile_validation.py')
validator = load('profiling_original_validator', ROOT/'isaaclab/validate_mkii_fourbar.py')


class FakeProfiler:
    def __init__(self, **options):
        self.options, self.started, self.stopped = options, False, False
    def start(self):
        assert not self.started
        self.started = True
    def stop(self):
        assert self.started and not self.stopped
        self.stopped = True
    def export_chrome_trace(self, path):
        assert self.stopped
        Path(path).write_text('{}')
    def key_averages(self):
        return SimpleNamespace(table=lambda **kwargs: 'synthetic CPU fixture')
    def events(self):
        return [SimpleNamespace(device_type='DeviceType.CPU')]


FAKE_TORCH = SimpleNamespace(profiler=SimpleNamespace(profile=FakeProfiler,
    ProfilerActivity=SimpleNamespace(CPU='cpu', CUDA='cuda'), record_function=lambda name: nullcontext()))


class MethodWrapperTests(unittest.TestCase):
    def test_instance_and_classmethod_wrappers_delegate_once_and_restore(self):
        class Subject:
            @classmethod
            def compute(cls, value):
                return value + 1
        original = vars(Subject)['compute']
        with tempfile.TemporaryDirectory() as directory:
            recorder = profile.Recorder(FAKE_TORCH, Path(directory))
            recorder.method(Subject, 'compute', 'native/classmethod')
            recorder.index, recorder.active = 50, True
            self.assertEqual(Subject.compute(4), 5)
            self.assertEqual(recorder.regions['native/classmethod']['calls'], 1)
            recorder.active = False
            recorder.close()
            self.assertIs(vars(Subject)['compute'], original)

    def test_readonly_native_method_is_reported_without_replacing_original(self):
        class Native:
            __slots__ = ()
            def call(self):
                return 3
        with tempfile.TemporaryDirectory() as directory:
            recorder = profile.Recorder(FAKE_TORCH, Path(directory))
            native = Native()
            recorder.method(native, 'call', 'native/readonly')
            self.assertEqual(native.call(), 3)
            self.assertIn('native/readonly: AttributeError', recorder.unavailable)

    def test_property_does_not_add_a_lazy_acquisition_and_restores_descriptor(self):
        class Data:
            calls = 0
            @property
            def pose(self):
                self.calls += 1
                return self.calls
        original = vars(Data)['pose']
        with tempfile.TemporaryDirectory() as directory:
            recorder = profile.Recorder(FAKE_TORCH, Path(directory))
            data = Data()
            recorder.properties(data, ('pose',), 'state')
            self.assertEqual(data.calls, 0)
            recorder.index, recorder.active = 50, True
            self.assertEqual(data.pose, 1)
            recorder.active = False
            self.assertEqual(data.pose, 2)
            recorder.close()
            self.assertIs(vars(Data)['pose'], original)


class CoverageTests(unittest.TestCase):
    def test_actual_substep_hook_keeps_every_capture_drain_and_original_scene_update(self):
        for decimation in (16, 32):
            with self.subTest(decimation=decimation), tempfile.TemporaryDirectory() as directory:
                class Scene:
                    def __init__(self):
                        self.calls = 0
                    def update(self, dt):
                        self.calls += 1
                class Metrics:
                    def __init__(self):
                        self.pending, self.total = [], 0
                    def capture(self):
                        self.pending.append(self.total)
                        self.total += 1
                    def drain(self):
                        assert len(self.pending) == decimation
                        result = tuple(self.pending)
                        self.pending.clear()
                        return result
                raw = SimpleNamespace(scene=Scene(), _physics_handles_decimation=False,
                    cfg=SimpleNamespace(sim=SimpleNamespace(dt=.02/decimation), decimation=decimation),
                    _body_contact_sensors={})
                class Env:
                    def step(self, actions):
                        for _ in range(decimation):
                            raw.scene.update(.02/decimation)
                        return actions
                api = SimpleNamespace(SubstepHook=validator.SubstepHook, PhysicalMetrics=Metrics)
                recorder = profile.Recorder(FAKE_TORCH, Path(directory))
                recorder.install = lambda current: recorder.method(current.scene, 'update', 'scene/update_original')
                restore = profile.instrument(api, recorder)
                try:
                    with patch.object(validator, 'DECIMATION', decimation), patch.object(validator, 'PHYSICS_DT_S', .02/decimation):
                        metrics = Metrics()
                        with api.SubstepHook(raw, metrics.capture) as hook:
                            for index in range(100):
                                self.assertEqual(hook.step(Env(), index), index)
                                samples = metrics.drain()
                                self.assertEqual(samples, tuple(range(index*decimation, (index+1)*decimation)))
                    self.assertEqual(raw.scene.calls, 100*decimation)
                    self.assertEqual(hook.total, 100*decimation)
                    self.assertEqual(len(recorder.control_rows), 100)
                    self.assertTrue(all(row['physics_captures'] == decimation and row['drains'] == 1
                                        for row in recorder.control_rows))
                    self.assertEqual(recorder.regions['scene/update_original']['calls'], 2*decimation)
                    self.assertEqual(recorder.regions['metrics/capture_including_acquisition']['calls'], 2*decimation)
                    self.assertEqual(recorder.regions['metrics/drain_including_gpu_wait']['calls'], 2)
                    self.assertEqual(set(recorder.regions['metrics/drain_including_gpu_wait']['calls_by_control']), {'50', '51'})
                    self.assertNotIn('update', vars(raw.scene))
                    result = recorder.export()
                    self.assertEqual(result['baseline_windows']['before']['count'], 30)
                    self.assertEqual(result['baseline_windows']['traced']['count'], 2)
                    self.assertEqual(result['baseline_windows']['after']['count'], 48)
                    self.assertFalse(result['cuda_activity_observed'])
                finally:
                    recorder.close()
                    restore()


if __name__ == '__main__':
    unittest.main()
