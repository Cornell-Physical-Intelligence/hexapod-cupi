#!/usr/bin/env python3
"""External 100-control standing profiler; never grants training admission."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import functools
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import statistics
import sys
import time

STEPS, TRACE_START, TRACE_STOP = 100, 50, 52


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class Recorder:
    """Inclusive/exclusive host ranges only during the two traced controls."""
    def __init__(self, torch, output):
        self.torch, self.output = torch, output
        self.active, self.index, self.profiler = False, -1, None
        self.start_ns = None
        self.control_rows, self.stack, self.undo = [], [], []
        self.regions, self.sources, self.unavailable = {}, {}, []
        self.captures, self.drains = Counter(), Counter()
        self.env_step_ns, self.boundary_overhead_ns = {}, {}
        self.runtime = {}

    def source(self, obj):
        try:
            path = inspect.getsourcefile(obj) or getattr(inspect.getmodule(obj), '__file__', None)
            if path and Path(path).is_file() and path not in self.sources:
                self.sources[path] = digest(path)
        except (TypeError, OSError):
            pass

    @contextmanager
    def region(self, name):
        if not self.active:
            yield
            return
        frame = [time.perf_counter_ns(), 0]
        self.stack.append(frame)
        try:
            with self.torch.profiler.record_function('hexapod/' + name):
                yield
        finally:
            elapsed = time.perf_counter_ns() - frame[0]
            self.stack.pop()
            if self.stack:
                self.stack[-1][1] += elapsed
            row = self.regions.setdefault(name, {'calls': 0, 'inclusive_ns': 0,
                'exclusive_ns': 0, 'max_ns': 0, 'calls_by_control': {}})
            row['calls'] += 1
            row['inclusive_ns'] += elapsed
            row['exclusive_ns'] += elapsed - frame[1]
            row['max_ns'] = max(row['max_ns'], elapsed)
            key = str(self.index)
            row['calls_by_control'][key] = row['calls_by_control'].get(key, 0) + 1

    def method(self, obj, attribute, label):
        if obj is None:
            self.unavailable.append(label + ': object unavailable')
            return
        original = getattr(obj, attribute, None)
        if not callable(original):
            self.unavailable.append(label + ': method unavailable')
            return
        self.source(type(obj))
        self.source(original)
        local = vars(obj) if hasattr(obj, '__dict__') else {}
        existed, previous = attribute in local, local.get(attribute)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if not self.active:
                return original(*args, **kwargs)
            with self.region(label):
                return original(*args, **kwargs)
        try:
            setattr(obj, attribute, wrapped)
        except (AttributeError, TypeError) as error:
            self.unavailable.append(label + ': ' + type(error).__name__)
            return
        self.undo.append(lambda: setattr(obj, attribute, previous) if existed else delattr(obj, attribute))

    def properties(self, obj, names, prefix):
        """Class property replacement delegates once to the exact original getter."""
        cls = type(obj)
        self.source(cls)
        for name in names:
            original = inspect.getattr_static(cls, name, None)
            if not isinstance(original, property) or original.fget is None:
                self.unavailable.append(prefix + '/' + name + ': no Python property')
                continue
            getter = original.fget
            self.source(getter)
            @functools.wraps(getter)
            def get(instance, _getter=getter, _name=name):
                if instance is not obj or not self.active:
                    return _getter(instance)
                with self.region(prefix + '/' + _name):
                    return _getter(instance)
            existed = name in vars(cls)
            previous = vars(cls).get(name)
            setattr(cls, name, property(get, original.fset, original.fdel, original.__doc__))
            self.undo.append(lambda cls=cls, name=name, existed=existed, previous=previous:
                             setattr(cls, name, previous) if existed else delattr(cls, name))

    def install(self, raw):
        self.runtime = {'device': str(raw.device), 'num_envs': raw.num_envs,
            'physics_dt_s': raw.cfg.sim.dt, 'decimation': raw.cfg.decimation,
            'policy_dt_s': raw.step_dt, 'lazy_sensor_update': raw.cfg.scene.lazy_sensor_update,
            'body_names': list(raw._robot.body_names), 'joint_names': list(raw._robot.joint_names),
            'runtime_manifest': raw.runtime_manifest, 'sensor_cfg': {name: {
                'update_period': sensor.cfg.update_period,
                'type': type(sensor).__module__ + '.' + type(sensor).__qualname__}
                for name, sensor in raw._body_contact_sensors.items()}}
        for obj, name, label in (
            (raw, '_apply_action', 'target_schedule'), (raw, '_get_dones', 'post_control/dones'),
            (raw, '_get_rewards', 'post_control/rewards'), (raw, '_get_observations', 'post_control/observations'),
            (raw.scene, 'write_data_to_sim', 'scene/write_data_to_sim'),
            (raw.scene, 'update', 'scene/update_original'), (raw.sim, 'step', 'sim/step'),
            (raw.sim, 'render', 'sim/render'), (raw.sim, 'wait_for_playing', 'sim/wait_for_playing'),
            (raw._robot, '_apply_actuator_model', 'actuator/apply_model'),
            (raw._motor_model, 'compute', 'actuator/compute'),
            (getattr(raw._motor_model, '_budget', None), 'step', 'actuator/budget'),
        ):
            self.method(obj, name, label)
        manager = getattr(raw.sim, 'physics_manager', None)
        self.method(manager, 'step', 'physx/step')
        self.method(manager, 'wait_for_playing', 'physx/wait_for_playing')
        native_sim = getattr(manager, '_physx_sim', None)
        for name in ('simulate', 'fetch_results'):
            self.method(native_sim, name, 'physx/native/' + name)
        data = raw._robot.data  # Retrieves the existing data object, not a lazy tensor property.
        self.method(data, 'update', 'articulation_data/update')
        self.properties(data, ('body_link_pose_w', 'body_link_vel_w', 'body_com_vel_w',
            'body_link_pos_w', 'body_link_quat_w', 'body_link_lin_vel_w',
            'body_link_ang_vel_w', 'joint_pos', 'joint_vel', 'joint_acc', 'root_pos_w'), 'state/get')
        for attribute, methods in (('_root_view', ('get_link_transforms', 'get_link_velocities',
                'get_dof_positions', 'get_dof_velocities', 'get_dof_accelerations')),
                ('_physics_sim_view', ('update_articulations_kinematic',))):
            view = getattr(data, attribute, None)
            for name in methods:
                self.method(view, name, 'state/native/' + name)
        for body, sensor in raw._body_contact_sensors.items():
            self.method(sensor, 'update', 'sensor/' + body + '/timestamp_update')
            self.method(sensor, '_update_buffers_impl', 'sensor/' + body + '/acquire')
            for attribute, methods in (('_contact_view', ('get_net_contact_forces',
                    'get_contact_force_matrix', 'get_contact_data')), ('_body_physx_view', ('get_transforms',))):
                view = getattr(sensor, attribute, None)
                for name in methods:
                    self.method(view, name, 'sensor/' + body + '/native/' + name)

    def finish_control(self):
        if self.start_ns is not None:
            elapsed = time.perf_counter_ns() - self.start_ns
            self.control_rows.append({'control': self.index, 'wall_ns': elapsed,
                'env_step_ns': self.env_step_ns.get(self.index),
                'physics_captures': self.captures[self.index], 'drains': self.drains[self.index],
                'traced': TRACE_START <= self.index < TRACE_STOP})
            self.start_ns = None

    def stop_trace(self):
        if self.profiler is None or not self.active:
            return
        self.active = False
        start = time.perf_counter_ns()
        self.profiler.stop()  # Whole-window finalization only; no per-region synchronize.
        self.boundary_overhead_ns['profiler_stop'] = time.perf_counter_ns() - start

    def next_control(self):
        self.finish_control()
        self.index += 1
        if self.index == TRACE_STOP:
            self.stop_trace()
        if self.index == TRACE_START:
            start = time.perf_counter_ns()
            self.profiler = self.torch.profiler.profile(activities=[self.torch.profiler.ProfilerActivity.CPU,
                self.torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=False, with_stack=False)
            self.profiler.start()
            self.active = True
            self.boundary_overhead_ns['profiler_start'] = time.perf_counter_ns() - start
        self.start_ns = time.perf_counter_ns()

    def close(self):
        self.finish_control()
        self.stop_trace()
        while self.undo:
            self.undo.pop()()

    def export(self):
        result = {'runtime': self.runtime, 'sdk_source_sha256': self.sources,
            'unavailable_instrumentation': self.unavailable, 'control_timings': self.control_rows,
            'profile_host_regions': self.regions, 'profiler_boundary_overhead_ns': self.boundary_overhead_ns,
            'baseline_windows': {}, 'trace_controls_half_open': [TRACE_START, TRACE_STOP],
            'warmup_controls_half_open': [0, 20]}
        for label, lo, hi in (('before', 20, TRACE_START), ('traced', TRACE_START, TRACE_STOP), ('after', TRACE_STOP, STEPS)):
            values = sorted(row['wall_ns'] for row in self.control_rows if lo <= row['control'] < hi)
            if values:
                result['baseline_windows'][label] = {'controls_half_open': [lo, hi], 'count': len(values),
                    'median_ns': statistics.median(values), 'p90_ns': values[min(len(values)-1, int(.9*len(values)))],
                    'p99_ns': values[min(len(values)-1, int(.99*len(values)))], 'max_ns': max(values), 'sum_ns': sum(values)}
        if self.profiler is not None:
            start = time.perf_counter_ns()
            self.profiler.export_chrome_trace(str(self.output/'trace.json'))
            (self.output/'torch_summary.txt').write_text(self.profiler.key_averages().table(
                sort_by='self_cpu_time_total', row_limit=100))
            result['profiler_export_ns'] = time.perf_counter_ns() - start
            events = self.profiler.events()
            kinds = Counter(str(event.device_type) for event in events)
            result['profiler_device_event_counts'] = dict(kinds)
            result['cuda_activity_observed'] = any('CUDA' in name and count > 0 for name, count in kinds.items())
            result['trace_sha256'] = digest(self.output/'trace.json')
        return result


def instrument(validator, recorder):
    """Wrap originals once; retain the existing substep hook and full metric rows."""
    originals = {(cls, name): getattr(cls, name) for cls, name in (
        (validator.SubstepHook, '__enter__'), (validator.SubstepHook, 'step'),
        (validator.SubstepHook, '__exit__'), (validator.PhysicalMetrics, 'capture'), (validator.PhysicalMetrics, 'drain'))}
    def enter(hook):
        recorder.install(hook.raw)  # Wrap original scene.update before SubstepHook captures it.
        return originals[(validator.SubstepHook, '__enter__')](hook)
    def step(hook, *args, **kwargs):
        recorder.next_control()
        start = time.perf_counter_ns()
        try:
            with recorder.region('env/step'):
                return originals[(validator.SubstepHook, 'step')](hook, *args, **kwargs)
        finally:
            recorder.env_step_ns[recorder.index] = time.perf_counter_ns() - start
    def leave(hook, *args):
        try:
            return originals[(validator.SubstepHook, '__exit__')](hook, *args)
        finally:
            recorder.close()
    def capture(metrics, *args, **kwargs):
        with recorder.region('metrics/capture_including_acquisition'):
            result = originals[(validator.PhysicalMetrics, 'capture')](metrics, *args, **kwargs)
        recorder.captures[recorder.index] += 1
        return result
    def drain(metrics, *args, **kwargs):
        with recorder.region('metrics/drain_including_gpu_wait'):
            result = originals[(validator.PhysicalMetrics, 'drain')](metrics, *args, **kwargs)
        recorder.drains[recorder.index] += 1
        return result
    for cls, name, function in ((validator.SubstepHook, '__enter__', enter),
            (validator.SubstepHook, 'step', step), (validator.SubstepHook, '__exit__', leave),
            (validator.PhysicalMetrics, 'capture', capture), (validator.PhysicalMetrics, 'drain', drain)):
        setattr(cls, name, function)
    return lambda: [setattr(cls, name, original) for (cls, name), original in originals.items()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args, validator_args = parser.parse_known_args(argv)
    source, out = args.source_dir.resolve(strict=True), args.output_dir.resolve(strict=True)
    if any((out/name).exists() for name in ('profile.json', 'trace.json', 'validation_report.json')):
        raise ValueError('Profiling output must be fresh')
    spec = importlib.util.spec_from_file_location('profiled_fourbar_validator', source/'isaaclab/validate_mkii_fourbar.py')
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    early, _ = validator.parser().parse_known_args(validator_args)
    if early.num_envs != 32 or early.steps != STEPS or early.report.resolve() != out/'validation_report.json':
        raise ValueError('Profiler requires 32 environments, 100 standing controls and its diagnostic report path')
    contract = validator.identity(source)
    if contract['sha256'] != args.source_sha256:
        raise ValueError('Profiling source differs from expected frozen identity')
    import torch
    recorder = Recorder(torch, out)
    report = {'schema': 'hexapod.physical_substep_profile.v1', 'pass': False,
        'simulation_training_admission': False, 'hardware_admission': False,
        'purpose': 'Diagnostic 100-control standing profile, not physical qualification or PPO',
        'source_identity': contract, 'wrapper_sha256': digest(__file__), 'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda, 'steps_requested': STEPS,
        'limitations': ['Baseline windows keep pass-through wrappers and coverage counters; no detailed timers or profiler.',
            'Host control timings include the normal drain and validation checks plus loop overhead.',
            'Trace initialization/finalization/export are excluded from control timing and reported separately.',
            'Nested host timings are inclusive/exclusive submission or blocking time, not synchronized kernel durations.',
            'Torch CUDA tracing may omit native PhysX/Warp streams; inspect the trace before attributing GPU idle time.',
            'Standing state and motor budget evolve between windows; this is not a repeated-state A/B benchmark.',
            'No detailed metrics body rewrite: capture arithmetic remains combined; drain includes queued GPU waits.']}
    restore = instrument(validator, recorder)
    try:
        report['validator_exit_code'] = validator.main(validator_args)
        if validator.identity(source) != contract:
            raise ValueError('Frozen source changed during profiling')
        report['source_unchanged'] = True
        validation = json.loads((out/'validation_report.json').read_text())
        report['validation_report_sha256'] = digest(out/'validation_report.json')
        report['physical_report_pass'] = validation.get('pass') is True
        report['physics_substeps'] = validation.get('physics_substeps')
        report.update(recorder.export())
        expected = STEPS * validator.DECIMATION
        report['coverage_pass'] = (len(recorder.control_rows) == STEPS
            and all(row['physics_captures'] == validator.DECIMATION and row['drains'] == 1 for row in recorder.control_rows)
            and validation.get('physics_substeps') == expected and validation.get('steps_completed') == STEPS)
        report['pass'] = (report['validator_exit_code'] == 0 and report['physical_report_pass']
            and report['coverage_pass'] and report.get('cuda_activity_observed') is True)
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        recorder.close()
        restore()
        if 'control_timings' not in report:
            try:
                report.update(recorder.export())
            except BaseException as error:
                report['export_error'] = type(error).__name__ + ': ' + str(error)
        write_json(out/'profile.json', report)
    print('FOURBAR_PROFILE_RESULT ' + json.dumps({'pass': report['pass'], 'profile': str(out/'profile.json')}), flush=True)
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
