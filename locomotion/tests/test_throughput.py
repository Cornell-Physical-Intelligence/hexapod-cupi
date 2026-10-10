"""CPU checks of the throughput profiler with stock PPO and a stand-in simulator; no native timing."""
from contextlib import ExitStack, redirect_stdout
import copy
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from locomotion import launch, reservation, throughput
from locomotion import task as task_module
from locomotion.env import LocomotionEnv
from locomotion.env_config import BODY_NAMES, EnvConfig, JOINT_NAMES, sha
from locomotion.ppo import TrainingLoads, VanillaVecEnv, ppo_config
from locomotion.prepare import ROOT, prepare
from locomotion.task import TaskConfig, TrainingTask

MODEL = ROOT/'robot/hexapod_mkii_updated_v1'
REMOTE = '/home/orionh/HEXAPOD_runs/restart_20260914/throughput_fixture'


class Sim:
    def __init__(self):
        self.count = 0

    def step(self, render=False):
        self.count += 1

    def get_physics_step_count(self):
        return self.count


class Buffer:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


class Contact:
    """Six toe sensors per robot report two patches; one patch carries no force."""

    def __init__(self, num_envs):
        self.num_envs = num_envs

    def get_contact_force_matrix(self, dt):
        forces = torch.zeros(19*self.num_envs, 1, 3)
        forces[:, 0, 2] = 2.
        return forces

    def get_contact_data(self, dt):
        sensors, capacity = 19*self.num_envs, 1024*self.num_envs
        force, counts, starts = np.zeros((capacity, 1)), np.zeros((sensors, 1), int), np.zeros((sensors, 1), int)
        cursor = 0
        for sensor in range(sensors):
            if BODY_NAMES[sensor % 19].endswith('_tibia'):
                counts[sensor], starts[sensor] = 2, cursor
                force[cursor] = 3.
                cursor += 2
        return [Buffer(value) for value in (force, np.zeros((capacity, 3)), np.zeros((capacity, 3)),
                                            np.zeros((capacity, 1)), counts, starts)]


class StandIn:
    """Follow LocomotionEnv's call structure with CPU tensors and no physics."""

    def __init__(self, num_envs=2, fail_at=None):
        n = self.num_envs = num_envs
        self.device, self.fail_at = 'cpu', fail_at
        self.cfg = SimpleNamespace(control_dt=.02, decimation=8, physics_dt=.0025, target_slew_rad=.04,
                                   episode_seconds=20., action_scale_rad=.35, declaration=lambda: {'num_envs': n})
        self.model = json.loads((MODEL/'model_rs05_mass_corrected.json').read_text())
        self.geometry_path = MODEL/'geometry/geometry.json'
        self.reference_metadata, self.reset_height = {}, .08
        self.origins = torch.tensor([[2.*i, 0., 0.] for i in range(n)])
        self.joint_names, self.native_body_names = list(JOINT_NAMES), list(BODY_NAMES)
        self.lower, self.upper, self.neutral = torch.full((18,), -1.), torch.full((18,), 1.), torch.zeros(18)
        self.held, self.commands = torch.zeros(n, 18), torch.zeros(n, 3)
        self.episode_steps = torch.zeros(n, dtype=torch.long)
        self.total_controls, self.telemetry, self.capture = 0, {}, None
        self.sim, self.contact = Sim(), Contact(n)
        self.reset()

    def _read(self):
        root = torch.zeros(self.num_envs, 7)
        root[:, :3], root[:, 2], root[:, 6] = self.origins, self.reset_height, 1.
        zeros = torch.zeros(self.num_envs, 3)
        return {'q': self.held.clone(), 'dq': torch.zeros(self.num_envs, 18), 'root': root,
                'linear': zeros, 'angular': zeros, 'gravity': torch.tensor([0., 0., -1.]).expand(self.num_envs, -1)}

    def _observations(self, state):
        obs = torch.zeros(self.num_envs, 231)
        obs[:, 173] = -1.
        obs[:, 210:213] = self.commands
        return {'obs': obs, 'critic': torch.cat((obs, state['linear']), -1)}

    def reset(self, indices=None):
        indices = torch.arange(self.num_envs) if indices is None else torch.as_tensor(indices, dtype=torch.long)
        self.held[indices], self.episode_steps[indices] = 0, 0
        self.current = self._read()
        return self._observations(self.current)

    def step(self, action):
        if self.fail_at is not None and self.total_controls == self.fail_at:
            raise FloatingPointError('Stand-in native failure')
        n = self.num_envs
        target = self.held+(.35*action.clamp(-1, 1)-self.held).clamp(-.04, .04)
        for substep in range(8):
            self.sim.step(render=False)
            state = self._read()
            forces = self.contact.get_contact_force_matrix(self.cfg.physics_dt).reshape(n, 19, 3)
            if self.capture is not None:
                zeros = torch.zeros(n, 18)
                self.capture(self, state, zeros, zeros, zeros, zeros, zeros, target, forces, substep, zeros)
        self.held = target
        self.episode_steps += 1
        self.total_controls += 1
        terminated = torch.zeros(n, dtype=torch.bool)
        terminated[0] = self.total_controls % 24 == 12
        zeros = torch.zeros(n, 18)
        self.current = state
        self.telemetry = {'linear_velocity_nav': torch.zeros(n, 3), 'angular_velocity_body': torch.zeros(n, 3),
            'root_pose_xyzw': state['root'], 'joint_velocity_rad_s': zeros, 'joint_position_rad': target.clone(),
            'joint_target_rad': target.clone(), 'torque_square_sum_400hz': zeros, 'saturation_count_400hz': zeros,
            'other_body_force_max_400hz': torch.zeros(n), 'command': self.commands.clone()}
        result = self._observations(state)
        result.update(terminated=terminated, truncated=self.episode_steps >= 1000)
        return result


class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1


class PhaseSpy:
    """Record the profile phase at each barrier and peak reset."""

    def __init__(self):
        self.driver, self.events = None, []

    def __call__(self):
        self.events.append(('barrier', self.driver.phase if self.driver.index < len(self.driver.plan) else 'done'))

    def reset_peaks(self):
        self.events.append(('reset', self.driver.phase))

    def sample(self):
        return {}


def training_stack(num_envs=2, fail_at=None):
    from rsl_rl.runners import OnPolicyRunner
    torch.manual_seed(3)
    env = StandIn(num_envs, fail_at)
    task = TrainingTask(env, TaskConfig(seed=5))
    wrapped = VanillaVecEnv(task)
    with redirect_stdout(io.StringIO()):
        runner = OnPolicyRunner(wrapped, ppo_config(5), None, device='cpu')
    env.capture = TrainingLoads(env)
    return env, task, wrapped, runner


def profile(env, task, wrapped, runner, *, warmup=2, measured=2, synchronize=None, deadline=None,
            memory=None, record_update=None):
    return throughput.Profile(env=env, task=task, wrapped=wrapped, runner=runner, task_module=task_module,
        warmup=warmup, measured=measured, synchronize=synchronize or Counter(),
        memory=memory or throughput.MemoryProbe('cpu'),
        record_update=record_update or (lambda update, values: {'update': update}), deadline=deadline)


def report(rows, driver, env, census=None, num_envs=None):
    identity = {key: 'fixture' for key in throughput.IDENTITY_KEYS}
    identity.update(learner=throughput.LEARNER, standing_admission={'num_envs': 32})
    settings = {'num_envs': num_envs or env.num_envs, 'warmup_updates': 2, 'measured_updates_per_pass': 2,
                'census_controls': 2}
    return throughput.build_report(rows, identity=identity, settings=settings, startup={},
        samples={'after_app_launch': {'system_used_bytes': 100}},
        census=census or {'capacity_patches': 1, 'max_capacity_fraction': 0., 'reported_patches_per_robot': {},
                          'policy': 'fixture', 'scope': 'fixture'},
        steps=driver.steps, num_envs=env.num_envs, stock_saves=driver.saves,
        platform_versions=throughput.versions(), training_loads=env.capture.report())


def assert_restored(case, env, task, wrapped, runner, sim, contact, loads):
    for owner, names in ((env, ('step', '_read', '_observations', 'reset')),
                         (task, ('step', 'reset', '_metrics', '_next_command_observation')),
                         (task.proximity, ('check',)), (wrapped, ('step',)),
                         (runner.alg, ('act', 'process_env_step', 'compute_returns', 'update')),
                         (runner.logger, ('process_env_step', 'log')), (runner, ('save',))):
        for name in names:
            case.assertNotIn(name, vars(owner), name)
    case.assertIs(env.sim, sim)
    case.assertIs(env.contact, contact)
    case.assertIs(env.capture, loads)
    case.assertIs(task_module.measured_reward, ORIGINAL_REWARD)


ORIGINAL_REWARD = task_module.measured_reward


class SectionTimerTests(unittest.TestCase):
    def timer(self, times):
        sync = Counter()
        clock = iter(times).__next__
        return throughput.SectionTimer(sync, clock), sync

    def test_nested_sections_partition_the_root_interval(self):
        timer, sync = self.timer([0., 1., 2., 4., 7., 8., 9., 10.])
        timer.begin('runner_other')
        timer.begin('env_control')
        timer.begin('physics_step')
        timer.end()
        timer.end()
        timer.begin('env_control')
        timer.end()
        self.assertEqual(timer.end(), 10.)
        seconds, calls, barriers = timer.take()
        self.assertEqual(seconds, {'physics_step': 2., 'env_control': 5., 'runner_other': 3.})
        self.assertEqual(calls, {'physics_step': 1, 'env_control': 2, 'runner_other': 1})
        self.assertEqual(sum(seconds.values()), 10.)
        self.assertEqual(sync.count, 8)
        # The root's opening barrier precedes its clock; each other barrier falls in one label.
        self.assertEqual(barriers, {'physics_step': 1, 'env_control': 3, 'runner_other': 3})

    def test_same_label_nesting_is_not_counted_twice(self):
        timer, _ = self.timer([0., 1., 3., 6., 8., 10.])
        timer.begin('runner_other')
        timer.begin('reset')
        timer.begin('reset')
        timer.end()
        timer.end()
        timer.end()
        seconds, calls, _ = timer.take()
        self.assertEqual(seconds, {'reset': 7., 'runner_other': 3.})
        self.assertEqual(calls['reset'], 2)

    def test_wrapped_failure_closes_its_section_and_unknown_labels_fail(self):
        timer, _ = self.timer([0., 1., 2., 3.])
        timer.begin('runner_other')
        with self.assertRaises(KeyError):
            timer.wrap('reward', lambda: {}['missing'])()
        self.assertEqual(len(timer.stack), 1)
        with self.assertRaises(RuntimeError):
            timer.take()
        timer.end()
        self.assertEqual(timer.take()[0], {'reward': 1., 'runner_other': 2.})
        with self.assertRaises(ValueError):
            timer.begin('undeclared')

    def test_idle_barrier_cost_is_the_median_interval(self):
        times = iter([0., 1., 3., 7.]).__next__
        sync = Counter()
        self.assertEqual(throughput.barrier_cost(sync, times, samples=2), 2.5)
        self.assertEqual(sync.count, 3)

    def test_distribution_rejects_empty_and_nonfinite_samples(self):
        self.assertEqual(throughput.distribution([1, 2, 3])['coefficient_of_variation'], .5)
        for values in ([], [1., float('nan')]):
            with self.assertRaises(ValueError):
                throughput.distribution(values)


class ProfileTests(unittest.TestCase):
    def test_stock_ppo_profile_partitions_components_and_restores_hooks(self):
        env, task, wrapped, runner = training_stack()
        sim, contact, loads = env.sim, env.contact, env.capture
        spy, hooked = PhaseSpy(), []

        def record(update, values):
            hooked.append((driver.phase, 'step' in vars(env), env.sim is sim,
                           task_module.measured_reward is ORIGINAL_REWARD))
            return {'update': update}
        driver = profile(env, task, wrapped, runner, synchronize=spy, memory=spy, record_update=record)
        spy.driver = driver
        rows = driver.run()
        assert_restored(self, env, task, wrapped, runner, sim, contact, loads)
        plan = ['warmup', 'warmup', 'throughput', 'components', 'throughput', 'components']
        self.assertEqual([row['phase'] for row in rows], plan)
        self.assertEqual(sim.count, 6*24*8)
        # Uninstrumented updates carry no hooks and one barrier at each boundary.
        self.assertEqual([event for event in spy.events if event[1] == 'throughput'],
                         [('reset', 'throughput'), ('barrier', 'throughput'), ('barrier', 'throughput')]*2)
        self.assertEqual([phase for kind, phase in spy.events if kind == 'reset'], plan)
        self.assertEqual(spy.events[0], ('reset', 'warmup'))
        self.assertEqual(hooked, [(phase, phase == 'components', phase != 'components', phase != 'components')
                                  for phase in plan])
        for row in rows:
            self.assertEqual(row['transitions'], 48)
            self.assertEqual(row['component_seconds'] is None, row['phase'] != 'components')
        for row in rows[3::2]:
            calls = row['component_calls']
            for label in ('physics_step', 'contact_readback', 'training_loads'):
                self.assertEqual(calls[label], 24*8, label)
            for label in ('policy_inference', 'env_control', 'vec_env_other', 'task_other', 'reward',
                          'task_metrics', 'rollout_storage', 'episode_logging'):
                self.assertEqual(calls[label], 24, label)
            for label in ('returns', 'ppo_update', 'update_logging', 'runner_other'):
                self.assertEqual(calls[label], 1, label)
            # One stand-in robot falls in the middle of each update; the task and the
            # environment each record that reset.
            self.assertEqual(calls['reset'], 2)
            self.assertEqual(calls['state_readback'], 24*8+1)
            self.assertEqual(calls['observation'], 2*24+2)
            self.assertEqual(calls['proximity_guard'], 2*24+3)
            self.assertEqual(row['component_barriers']['env_control'], 24*(4*8+1+1))
        result = report(rows, driver, env)
        components = result['components']
        self.assertAlmostEqual(components['component_sum_seconds'], components['wall_seconds']['sum'], places=12)
        self.assertAlmostEqual(sum(value['share'] for value in components['categories'].values()), 1., places=12)
        self.assertEqual(components['components']['env_control']['barriers'], 2*24*34)
        self.assertEqual(result['throughput']['transitions'], 96)
        self.assertEqual(result['throughput']['robot_physics_steps_per_second'],
                         8*result['throughput']['training_transitions_per_second'])
        self.assertEqual(result['learner']['label'], 'stock_ppo')
        self.assertEqual(result['amp_learner']['status'], 'pending')
        self.assertTrue(result['warmup']['excluded_from_statistics'])
        self.assertEqual(len(result['warmup']['wall_seconds']), 2)
        json.dumps(result, allow_nan=False)

    def test_native_failure_inside_a_component_update_restores_the_stack(self):
        env, task, wrapped, runner = training_stack(fail_at=2*24+30)
        sim, contact, loads = env.sim, env.contact, env.capture
        driver = profile(env, task, wrapped, runner)
        with self.assertRaisesRegex(FloatingPointError, 'Stand-in'):
            driver.run()
        assert_restored(self, env, task, wrapped, runner, sim, contact, loads)
        self.assertEqual([row['phase'] for row in driver.rows], ['warmup', 'warmup', 'throughput'])

    def test_deadline_stops_after_a_complete_update(self):
        env, task, wrapped, runner = training_stack()
        driver = profile(env, task, wrapped, runner, deadline=0.)
        with self.assertRaises(TimeoutError):
            driver.run()
        self.assertEqual(len(driver.rows), 1)

    def test_last_update_completes_after_the_deadline(self):
        env, task, wrapped, runner = training_stack()
        # The deadline check runs only before another update starts.
        times = iter([0., 0., 0., 10.]).__next__
        driver = throughput.Profile(env=env, task=task, wrapped=wrapped, runner=runner, task_module=task_module,
            warmup=2, measured=1, synchronize=Counter(), memory=throughput.MemoryProbe('cpu'),
            record_update=lambda update, values: {}, deadline=5., monotonic=times)
        self.assertEqual(len(driver.run()), 4)

    def stock_writer(self, runner, directory):
        def init(logger):
            logger.writer = SimpleNamespace(add_scalar=lambda *args: None)
            logger.logger_type, logger.log_dir = 'tensorboard', str(directory)
        return patch.object(type(runner.logger), 'init_logging_writer', init)

    def test_stock_saves_stay_in_warmup_or_after_the_final_update(self):
        env, task, wrapped, runner = training_stack()
        driver = profile(env, task, wrapped, runner)
        with tempfile.TemporaryDirectory() as directory, self.stock_writer(runner, directory), \
                redirect_stdout(io.StringIO()):
            rows = driver.run()
            self.assertEqual(sorted(path.name for path in Path(directory).glob('*.pt')), ['model_0.pt', 'model_5.pt'])
        self.assertEqual(driver.saves, [1, 6])
        warmup = report(rows, driver, env)['warmup']
        self.assertEqual(warmup['stock_runner_save_update_windows'], [2])
        self.assertEqual(warmup['stock_runner_saves_after_final_update'], 1)
        env, task, wrapped, runner = training_stack()
        runner.cfg['save_interval'] = 2
        driver = profile(env, task, wrapped, runner)
        with tempfile.TemporaryDirectory() as directory, self.stock_writer(runner, directory), \
                redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'stock saves'):
            driver.run()

    def test_partition_and_hook_coverage_are_enforced(self):
        env, task, wrapped, runner = training_stack()
        driver = profile(env, task, wrapped, runner)
        rows = driver.run()
        broken = copy.deepcopy(rows)
        broken[3]['component_seconds']['physics_step'] += 1.
        with self.assertRaisesRegex(ValueError, 'partition'):
            throughput.pass_summary(broken, 'components', transitions=48, steps=24)
        broken = copy.deepcopy(rows)
        broken[3]['component_calls']['physics_step'] -= 1
        with self.assertRaisesRegex(ValueError, 'missed'):
            throughput.pass_summary(broken, 'components', transitions=48, steps=24)

    def test_contact_census_counts_reported_and_loaded_patches_outside_timing(self):
        env, task, wrapped, runner = training_stack()
        census = throughput.contact_census(env, wrapped, runner, 3)
        self.assertEqual(census['capacity_patches'], 2048)
        self.assertEqual(census['max_capacity_fraction'], 24/2048)
        self.assertEqual(census['reported_patches_per_robot']['mean'], 12.)
        self.assertEqual(census['nonzero_force_patches_per_robot']['mean'], 6.)
        self.assertIsNone(census['physx_scene_statistics'])

        class Full(Contact):
            def get_contact_data(self, dt):
                data = super().get_contact_data(dt)
                counts, starts = data[4].value, data[5].value
                counts[:], starts[:] = 0, 0
                counts[0] = len(data[0].value)
                return data
        env.contact = Full(env.num_envs)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            throughput.contact_census(env, wrapped, runner, 1)

    def test_hooks_target_the_maintained_call_sites(self):
        sources = {name: inspect.getsource(function) for name, function in (
            ('env.step', LocomotionEnv.step), ('env.reset', LocomotionEnv.reset),
            ('task.step', TrainingTask.step), ('task.reset', TrainingTask.reset))}
        for call in ('self.sim.step(render=False)', 'self._read()', 'self.contact.get_contact_force_matrix(',
                     'self.capture(', 'self._observations(state)'):
            self.assertIn(call, sources['env.step'])
        for call in ('measured_reward(', 'self.proximity.check(', 'self._metrics(', 'self._next_command_observation(',
                     'self.env.step(action)'):
            self.assertIn(call, sources['task.step'])
        self.assertIn('self._read()', sources['env.reset'])
        self.assertIn('self.env.reset(selected)', sources['task.reset'])


class MemoryTests(unittest.TestCase):
    def test_probe_combines_allocator_pool_and_heap_counters(self):
        cuda = SimpleNamespace(is_available=lambda: True, reset_peak_memory_stats=lambda device: None,
            memory_allocated=lambda device: 10, memory_reserved=lambda device: 20,
            max_memory_allocated=lambda device: 15, max_memory_reserved=lambda device: 30)
        warp = lambda: {'warp_pool_used_bytes': 5, 'warp_pool_peak_bytes': 7}
        physx = lambda: {'physx_gpu_mem_heap': 100, 'physx_gpu_mem_heap_solver': 40, 'physx_nb_new_pairs': 3}
        resets = []
        cuda.reset_peak_memory_stats = resets.append
        probe = throughput.MemoryProbe('cuda:0', SimpleNamespace(cuda=cuda), warp, physx)
        probe.reset_peaks()
        self.assertEqual(resets, ['cuda:0'])
        sample = probe.sample()
        self.assertEqual(sample['torch_peak_reserved_bytes'], 30)
        self.assertEqual(sample['physx_gpu_mem_heap_solver'], 40)
        self.assertNotIn('physx_nb_new_pairs', sample)
        smaller = {**sample, 'torch_peak_reserved_bytes': 25, 'warp_pool_peak_bytes': 9}
        peaks = throughput.memory_peaks([sample, smaller])
        self.assertEqual(peaks['accounted_gpu_peak_bytes'], 30+9+100)
        self.assertIsNone(throughput.memory_peaks([{'process_peak_rss_bytes': 1}])['accounted_gpu_peak_bytes'])
        self.assertIsInstance(throughput.MemoryProbe('cpu').sample()['process_peak_rss_bytes'], int)

    def test_native_readers_report_unavailable_sources_on_cpu(self):
        env = SimpleNamespace(sim=SimpleNamespace(stage=None), scene_prim_path='/physicsScene')
        for reader, status in (throughput.warp_memory('cuda:0'), throughput.physx_statistics(env)):
            if reader is None:
                self.assertTrue(status.startswith(('unavailable', 'Warp memory pool disabled')))
        self.assertIsNone(throughput.cuda_contexts(env, 'cuda:0')['shared'])

    def test_physx_reader_counts_unanswered_reads(self):
        answers = iter((False, True))

        def fill(stage, scene, value):
            value.__dict__.update({name: index for index, name in enumerate(throughput.PHYSX_FIELDS)})
            return next(answers)
        reader = throughput.PhysxStatistics(SimpleNamespace(get_physx_scene_statistics=fill),
                                            SimpleNamespace, 1, 2)
        self.assertEqual(reader(), {})
        self.assertEqual(reader()['physx_gpu_mem_heap'], throughput.PHYSX_FIELDS.index('gpu_mem_heap'))
        self.assertEqual(reader.status(), {'reads': 2, 'unanswered_reads': 1})


class GuardTests(unittest.TestCase):
    def args(self, **overrides):
        values = dict(headless=True, device='cuda:0', seed=1, warmup_updates=2, updates=5, max_wall_seconds=6200.)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_profile_keeps_replica_and_update_guards(self):
        self.assertEqual(throughput.REPLICA_COUNTS, (1, 32, 128, 512, 1024))
        self.assertEqual(EnvConfig(num_envs=1024).num_envs, 1024)
        with self.assertRaises(ValueError):
            EnvConfig(num_envs=1025)
        parser = throughput.parser_for(['--preflight-only'])
        required = ['--mode', 'throughput', '--preflight-only', '--source-freeze-sha256', 'a'*64]
        for name in ('asset', 'model', 'geometry', 'geometry-extrema', 'stance', 'output', 'standing-admission'):
            required += ['--'+name, 'x']
        for count in throughput.REPLICA_COUNTS:
            self.assertEqual(parser.parse_args(required+['--num-envs', str(count)]).num_envs, count)
        with self.assertRaises(SystemExit), patch('sys.stderr', io.StringIO()):
            parser.parse_args(required+['--num-envs', '64'])
        throughput.validate(self.args())
        for overrides in ({'warmup_updates': 1}, {'warmup_updates': 11}, {'updates': 0}, {'updates': 21},
                          {'headless': False}, {'device': 'cuda:1'}, {'max_wall_seconds': 7000.}, {'seed': -1}):
            with self.subTest(**overrides), self.assertRaises(ValueError):
                throughput.validate(self.args(**overrides))

    def test_large_counts_reach_diagnostics_and_profiles_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, count in enumerate((512, 1024)):
                diagnostic = prepare(root/f'diag{index}', REMOTE, mode='diagnostic', num_envs=count)
                self.assertEqual(diagnostic['command_args'][diagnostic['command_args'].index('--num-envs')+1], str(count))
                profile = prepare(root/f'profile{index}', REMOTE, mode='throughput', num_envs=count, updates=5, warmup_updates=2)
                self.assertEqual(profile['module'], 'locomotion.throughput')
                self.assertEqual(profile['command_args'][profile['command_args'].index('--num-envs')+1], str(count))
            for index, bad in enumerate((dict(mode='train', num_envs=512), dict(mode='probe', num_envs=1024),
                                         dict(mode='diagnostic', num_envs=256), dict(mode='throughput', num_envs=2048, updates=5, warmup_updates=2))):
                with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'replica count'):
                    prepare(root/f'bad{index}', REMOTE, **bad)

    def test_prepare_binds_a_profile_with_admission_and_bounded_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            binding = prepare(Path(directory)/'pack', REMOTE, mode='throughput', num_envs=32,
                              updates=5, warmup_updates=3, seed=7)
            self.assertEqual(binding['module'], 'locomotion.throughput')
            args = binding['command_args']
            for option, value in (('--num-envs', '32'), ('--updates', '5'), ('--warmup-updates', '3'),
                                  ('--seed', '7'), ('--standing-admission', '/admission/admission.json')):
                self.assertEqual(args[args.index(option)+1], value)
            source = Path(directory)/'pack/source/locomotion'
            self.assertEqual(sha(source/'throughput.py'), sha(ROOT/'locomotion/throughput.py'))
            for name, options in (('bad_updates', {'updates': 512, 'warmup_updates': 2}),
                                  ('bad_warmup', {'updates': 5, 'warmup_updates': 1}),
                                  ('bad_count', {'updates': 5, 'warmup_updates': 2, 'num_envs': 64}),
                                  ('no_warmup', {'updates': 5})):
                with self.subTest(name), self.assertRaises(ValueError):
                    prepare(Path(directory)/name, REMOTE, mode='throughput', **{'num_envs': 32, **options})
            with self.assertRaises(ValueError):
                prepare(Path(directory)/'train', REMOTE, updates=5, warmup_updates=2)

    def test_launcher_accepts_the_profile_module_only_in_its_mode(self):
        root = '/home/orionh/HEXAPOD_runs/restart_20260914/launcher_fixture'
        binding = {'schema': 'hexapod_locomotion_launch_v1', 'root_review_complete': True,
                   'mode': 'throughput', 'module': 'locomotion.throughput', 'max_seconds': 6600,
                   **{key: root+'/'+key for key in ('source', 'output', 'asset', 'prior', 'geometry_source')},
                   'source_freeze_sha256': 'a'*64, 'input_files': {}, 'stage2_complete': False,
                   'physical_admission': False,
                   'command_args': ['--mode', 'throughput', '--source-freeze-sha256', 'a'*64,
                                    '--asset', '/asset', '--model', '/asset/source/model.json',
                                    '--device', 'cuda:0', '--headless']}

        def verify(value):
            with ExitStack() as stack:
                stack.enter_context(patch.object(reservation, 'canonical_path', side_effect=Path))
                for name in ('verify_tree', 'pinned_file'):
                    stack.enter_context(patch.object(reservation, name, return_value=None))
                return launch.verify(value, Path(value['source']))
        verify(binding)
        for mode, module in (('throughput', 'locomotion.train'), ('train', 'locomotion.throughput')):
            changed = copy.deepcopy(binding)
            changed['mode'], changed['module'] = mode, module
            changed['command_args'][1] = mode
            with self.subTest(mode=mode, module=module), self.assertRaises(ValueError):
                verify(changed)


class IdentityTests(unittest.TestCase):
    def run_module(self, source, module, *options):
        command = [sys.executable, '-B', '-m', module, *options, '--preflight-only', '--headless',
                   '--asset', str(MODEL/'usd'), '--model', str(MODEL/'usd/source/model.json'),
                   '--geometry', str(MODEL/'geometry/geometry.json'),
                   '--geometry-extrema', str(MODEL/'geometry/geometry_extrema.npz'),
                   '--stance', str(MODEL/'stance.json'), '--output', str(source.parent/'unused'),
                   '--source-freeze-sha256', sha(source/'FREEZE_SHA256.json')]
        result = subprocess.run(command, cwd=source, capture_output=True, text=True, timeout=120,
                                env={**os.environ, 'PYTHONPATH': str(source), 'PYTHONDONTWRITEBYTECODE': '1'})
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def admission(self, directory, identity):
        envelope = {'schema': 'hexapod_locomotion_standing_admission_v1', 'num_envs': 32,
                    **{key: identity[key] for key in ('model_sha256', 'usd_sha256', 'physics_source_files',
                        'physics_config', 'stance_sha256', 'geometry_sha256', 'geometry_extrema_sha256')}}
        for name, count in (('one', 1), ('batch', 32)):
            state = {'status': 'completed', 'standing_gate_pass': True, 'errors': [],
                     'identity': {**identity, 'config': EnvConfig(num_envs=count).declaration()}}
            report_value = {'num_envs': count, 'controls': 1000, 'substeps': 8000, 'all_pass': True,
                            'replicas': [{'env': e, 'pass': True, 'failed_physical_bounds': [],
                                          'quiet': {'pass': True, 'failed_bounds': []}} for e in range(count)]}
            row = envelope[name] = {}
            for kind, value in (('state', state), ('report', report_value)):
                path = directory/(name+'_'+kind+'.json')
                path.write_text(json.dumps(value)+'\n')
                row[kind+'_path'], row[kind+'_sha256'] = str(path), sha(path)
        path = directory/'admission.json'
        path.write_text(json.dumps(envelope)+'\n')
        return path

    def test_profile_identity_matches_training_admission_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            prepare(directory/'pack', REMOTE, mode='throughput', num_envs=32, updates=5, warmup_updates=2)
            source = directory/'pack/source'
            training = self.run_module(source, 'locomotion.train', '--mode', 'diagnostic', '--num-envs', '32')
            admission = self.admission(directory, training)
            result = self.run_module(source, 'locomotion.throughput', '--mode', 'throughput', '--num-envs', '32',
                                     '--standing-admission', str(admission))
            profile_identity = result['identity']
            for key in ('model_sha256', 'usd_sha256', 'physics_source_files', 'physics_config', 'stance_sha256',
                        'geometry_sha256', 'geometry_extrema_sha256', 'source_files', 'adapter_sha256'):
                self.assertEqual(profile_identity[key], training[key], key)
            self.assertEqual(profile_identity['standing_admission']['num_envs'], 32)
            self.assertEqual(profile_identity['config']['episode_seconds'], 20.)
            self.assertEqual(profile_identity['learner']['label'], 'stock_ppo')
            self.assertEqual(result['settings']['measured_updates_per_pass'], 5)


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env, task, wrapped, runner = training_stack()
        driver = profile(env, task, wrapped, runner)
        rows = driver.run()
        cls.base = report(rows, driver, env, throughput.contact_census(env, wrapped, runner, 2))

    def write(self, root, num_envs, job='completed', audit=True, binding=None, platform=None, **identity):
        """Lay out one launcher allocation: run/launch_binding.json, run/jobs and run/standing."""
        run = root/f'allocation_{num_envs}_{len(list(root.iterdir()))}'/'run'
        directory = run/'standing'
        (run/'jobs').mkdir(parents=True)
        directory.mkdir()
        value = copy.deepcopy(self.base)
        value['settings']['num_envs'] = num_envs
        value['identity'].update(identity)
        value['platform'].update(platform or {})
        (directory/'profile.json').write_text(json.dumps(value)+'\n')
        (directory/'state.json').write_text(json.dumps({'status': 'completed',
            'profile_sha256': sha(directory/'profile.json')})+'\n')
        launch = {'mode': 'throughput', 'module': 'locomotion.throughput', 'source_freeze_sha256': 'fixture',
                  'command_args': ['--mode', 'throughput', '--num-envs', str(num_envs), '--updates', '2',
                                   '--warmup-updates', '2']}
        launch.update(binding or {})
        (run/'launch_binding.json').write_text(json.dumps(launch)+'\n')
        (run/'jobs/standing.json').write_text(json.dumps({'status': job})+'\n')
        (run/'jobs/standing_contact_data_audit.json').write_text(json.dumps({'passed': audit})+'\n')
        return directory

    def test_summary_orders_counts_and_binds_each_allocation(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            runs = [self.write(root, 128), self.write(root, 32)]
            result = throughput.summarize(runs, root/'summary.json')
            self.assertEqual([row['num_envs'] for row in result['runs']], [32, 128])
            first = result['runs'][0]
            self.assertEqual(first['profile_sha256'], sha(runs[1]/'profile.json'))
            self.assertEqual(first['allocation_files']['jobs/standing.json'], sha(runs[1].parent/'jobs/standing.json'))
            self.assertFalse(first['component_split_verified'])
            self.assertEqual(first['contact_census']['scope'], self.base['contact_census']['scope'])
            scaling = result['scaling'][0]
            self.assertEqual(scaling['training_transitions_per_second_ratio'], 1.)
            self.assertIn('system_used_above_launch_bytes_per_added_robot', scaling)
            self.assertEqual(result['amp_learner']['status'], 'pending')
            text = throughput.table(result)
            for name in throughput.CATEGORIES:
                self.assertIn(name, text)
            self.assertIn('PhysX synchronization unverified', text)
            with self.assertRaises(FileExistsError):
                throughput.summarize(runs, root/'summary.json')

    def test_summary_rejects_failed_unbound_mixed_or_changed_profiles(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            one = self.write(root, 1)
            cases = (('physics_config', [one, self.write(root, 32, physics_config='other')]),
                     ('isaacsim', [one, self.write(root, 32, platform={'isaacsim': 'other'})]),
                     ('distinct', [one, one]),
                     ('admitted', [self.write(root, 64)]),
                     ('did not complete', [self.write(root, 32, job='failed')]),
                     ('did not complete', [self.write(root, 32, audit=False)]),
                     ('differ', [self.write(root, 32, binding={'source_freeze_sha256': 'other'})]),
                     ('differ', [self.write(root, 32, binding={'command_args': ['--num-envs', '128']})]))
            for index, (message, runs) in enumerate(cases):
                with self.subTest(message=message, index=index), self.assertRaisesRegex(ValueError, message):
                    throughput.summarize(runs, root/f'rejected_{index}.json')
            (one.parent/'jobs/standing.json').unlink()
            with self.assertRaisesRegex(ValueError, 'lacks'):
                throughput.summarize([one], root/'unbound.json')
            other = self.write(root, 32)
            with (other/'profile.json').open('a') as stream:
                stream.write(' ')
            with self.assertRaisesRegex(ValueError, 'changed'):
                throughput.summarize([other], root/'changed.json')

if __name__ == '__main__':
    unittest.main()
