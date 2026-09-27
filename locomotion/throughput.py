"""Profile stock-PPO training cost, throughput and memory at one admitted replica count.

The profiler runs the unchanged simulator, task and RSL-RL runner. It attaches
timing hooks at update boundaries and removes them before an uninstrumented
update. `summarize` compares completed allocations on a CPU host.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import functools
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import resource
import statistics
import sys
import time
import traceback

SCHEMA = 'hexapod_locomotion_throughput_v1'
SUMMARY_SCHEMA = 'hexapod_locomotion_throughput_summary_v1'
# The profiler measures the existing replica guards; it adds no scale.
REPLICA_COUNTS = (1, 32, 128)
WARMUP_UPDATES = (2, 10)
MEASURED_UPDATES = (1, 20)
CENSUS_CONTROLS = 50
PHYSICS_STEPS_PER_CONTROL = 8
LEARNER = {'label': 'stock_ppo', 'algorithm': 'RSL-RL 5.0.1 PPO', 'adapter': 'ppo.py',
           'networks': 'MLP [256, 256, 128] actor and critic', 'motion_prior': False}
AMP_LEARNER = {'status': 'pending', 'measured': False,
               'scope': 'This source has no AMP discriminator or style reward; their learner cost remains unmeasured.'}
# Each label owns exclusive time, so the labels partition every measured update.
COMPONENTS = {
    'physics_step': ('physics_contact', 'PhysX step: broadphase, SDF contact generation, solver and integration; '
                     'GPU work counts here only when PhysX shares the synchronized CUDA context'),
    'contact_readback': ('physics_contact', 'Floor contact-force matrix query'),
    'training_loads': ('physics_contact', 'Training accumulation of 400 Hz contact force and motor torque'),
    'state_readback': ('observation', 'Articulation kinematics, state getters and frame transforms'),
    'observation': ('observation', 'Actor/critic assembly and next-command fields'),
    'env_control': ('simulation_other', 'Target limiter, motor model, actuation I/O, force checks and termination'),
    'reset': ('simulation_other', 'Selected replica resets and their command draws'),
    'proximity_guard': ('task', 'Inter-replica proximity checks'),
    'reward': ('task', 'Measured task reward'),
    'task_metrics': ('task', 'Task reporting accumulators'),
    'task_other': ('task', 'Command holds and task bookkeeping'),
    'policy_inference': ('learner', 'Actor sample and critic value during collection'),
    'rollout_storage': ('learner', 'Observation normalizers and transition storage'),
    'returns': ('learner', 'Generalized advantage estimation'),
    'ppo_update': ('learner', 'PPO epochs: forward, backward and optimizer steps'),
    'vec_env_other': ('bookkeeping', 'Done mask and observation wrapper'),
    'episode_logging': ('bookkeeping', 'RSL-RL episode reward and length buffers'),
    'update_logging': ('bookkeeping', 'RSL-RL logger plus the training metrics row, scalars and state file'),
    'runner_other': ('bookkeeping', 'Runner loop, NaN checks and device transfers'),
}
CATEGORIES = tuple(dict.fromkeys(category for category, _ in COMPONENTS.values()))
MEMORY_SCOPE = ('PyTorch allocator peaks reset before each update. The Warp pool high-water mark and process '
                'peak RSS span the process. PhysX heap groups hold live allocations; gpu_mem_heap is heap '
                'capacity. system_used_bytes is MemTotal minus MemAvailable for the whole host and includes other '
                'processes. On GB10 unified memory, process RSS omits CUDA allocations. No per-process counter '
                'isolates the CUDA context or tensor-API staging buffers.')
PLATFORM_KEYS = ('python', 'machine', 'torch', 'rsl-rl-lib', 'tensordict', 'warp-lang', 'isaacsim', 'isaaclab',
                 'cuda_device', 'cuda_capability', 'cuda_runtime')
SETTING_KEYS = ('warmup_updates', 'measured_updates_per_pass', 'census_controls', 'steps_per_update')
IDENTITY_KEYS = ('source_files', 'source_freeze_sha256', 'model_sha256', 'usd_sha256', 'stance_sha256',
                 'geometry_sha256', 'geometry_extrema_sha256', 'physics_source_files', 'physics_config', 'seed',
                 'learner', 'ppo_config', 'upstream_source_files')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


class SectionTimer:
    """Exclusive wall time per label; a nested section subtracts from its parent.

    Each boundary synchronizes the device, so queued GPU work in the
    synchronized contexts belongs to the section that launched it. A child's
    opening barrier falls in its parent's time and each closing barrier in its
    own section; `barriers` counts both for the overhead estimate.
    """

    def __init__(self, synchronize, clock=time.perf_counter):
        self.synchronize, self.clock = synchronize, clock
        self.stack = []
        self.seconds, self.calls, self.barriers = defaultdict(float), defaultdict(int), defaultdict(int)

    def begin(self, label):
        if label not in COMPONENTS:
            raise ValueError('Undeclared profile component: '+label)
        self.synchronize()
        if self.stack:
            self.barriers[self.stack[-1][0]] += 1
        self.stack.append([label, self.clock(), 0.])

    def end(self):
        self.synchronize()
        label, start, children = self.stack.pop()
        elapsed = self.clock()-start
        self.seconds[label] += elapsed-children
        self.calls[label] += 1
        self.barriers[label] += 1
        if self.stack:
            self.stack[-1][2] += elapsed
        return elapsed

    def wrap(self, label, function):
        @functools.wraps(function)
        def timed(*args, **kwargs):
            self.begin(label)
            try:
                return function(*args, **kwargs)
            finally:
                self.end()
        return timed

    def take(self):
        if self.stack:
            raise RuntimeError('A profile section is still open')
        result = dict(self.seconds), dict(self.calls), dict(self.barriers)
        self.seconds, self.calls, self.barriers = defaultdict(float), defaultdict(int), defaultdict(int)
        return result


def barrier_cost(synchronize, clock=time.perf_counter, samples=50):
    """Median cost of one idle device barrier, measured outside any timed update."""
    synchronize()
    values = []
    for _ in range(samples):
        start = clock()
        synchronize()
        values.append(clock()-start)
    return statistics.median(values)


class TimedProxy:
    """Forward a native object and time its named methods."""

    def __init__(self, target, labels, timer):
        self._target = target
        self._timed = {name: timer.wrap(label, getattr(target, name)) for name, label in labels.items()}

    def __getattr__(self, name):
        return self._timed[name] if name in self._timed else getattr(self._target, name)


def instrument(timer, env, task, wrapped, runner, task_module):
    """Time the training path in place; return a function that restores it."""
    restores = []

    def method(owner, name, label):
        if name in vars(owner) or not callable(getattr(owner, name, None)):
            raise RuntimeError('Cannot attach a timing hook: '+name)
        setattr(owner, name, timer.wrap(label, getattr(owner, name)))
        restores.append(lambda: delattr(owner, name))

    def replace(owner, name, build):
        original = getattr(owner, name)
        setattr(owner, name, build(original))
        restores.append(lambda: setattr(owner, name, original))

    try:
        replace(env, 'sim', lambda sim: TimedProxy(sim, {'step': 'physics_step'}, timer))
        replace(env, 'contact', lambda view: TimedProxy(view, {'get_contact_force_matrix': 'contact_readback'}, timer))
        if env.capture is not None:
            replace(env, 'capture', lambda capture: timer.wrap('training_loads', capture))
        replace(task_module, 'measured_reward', lambda function: timer.wrap('reward', function))
        for owner, name, label in (
                (env, 'step', 'env_control'), (env, '_read', 'state_readback'),
                (env, '_observations', 'observation'), (env, 'reset', 'reset'),
                (task, 'step', 'task_other'), (task, 'reset', 'reset'), (task, '_metrics', 'task_metrics'),
                (task, '_next_command_observation', 'observation'),
                (task.proximity, 'check', 'proximity_guard'), (wrapped, 'step', 'vec_env_other'),
                (runner.alg, 'act', 'policy_inference'), (runner.alg, 'process_env_step', 'rollout_storage'),
                (runner.alg, 'compute_returns', 'returns'), (runner.alg, 'update', 'ppo_update'),
                (runner.logger, 'process_env_step', 'episode_logging')):
            method(owner, name, label)
    except BaseException:
        for restore in reversed(restores):
            restore()
        raise

    def restore_all():
        while restores:
            restores.pop()()
    return restore_all


def distribution(values):
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError('A profile distribution needs finite samples')
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.
    return {'count': len(values), 'sum': math.fsum(values), 'mean': mean,
            'median': statistics.median(values), 'min': min(values), 'max': max(values),
            'stdev': deviation, 'coefficient_of_variation': deviation/mean if mean > 0 else None}


def _proc_bytes(path, names):
    """Read kB fields from a Linux /proc file; other hosts return no values."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return {}
    values = {}
    for line in lines:
        key, _, rest = line.partition(':')
        if key in names:
            values[names[key]] = int(rest.split()[0])*1024
    return values


class MemoryProbe:
    """Sample allocator, memory-pool, PhysX-heap and host memory between timed intervals.

    The PyTorch, Warp and PhysX counters cover their own allocations. Process
    RSS omits CUDA allocations on GB10 unified memory. cudaMemGetInfo and
    /proc/meminfo both describe the whole device or host, so `system_used_bytes`
    includes other processes.
    """

    def __init__(self, device, torch=None, warp=None, physx=None):
        self.device, self.torch, self.warp, self.physx = device, torch, warp, physx
        self.cuda = torch is not None and str(device).startswith('cuda') and torch.cuda.is_available()

    def reset_peaks(self):
        if self.cuda:
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def sample(self):
        row = _proc_bytes('/proc/self/status', {'VmRSS': 'process_rss_bytes', 'VmHWM': 'process_peak_rss_bytes'})
        if 'process_peak_rss_bytes' not in row:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            row['process_peak_rss_bytes'] = peak*(1 if sys.platform == 'darwin' else 1024)
        host = _proc_bytes('/proc/meminfo', {'MemTotal': 'total', 'MemAvailable': 'available'})
        if len(host) == 2:
            row['system_used_bytes'] = host['total']-host['available']
        if self.cuda:
            cuda = self.torch.cuda
            row.update(torch_allocated_bytes=cuda.memory_allocated(self.device),
                       torch_reserved_bytes=cuda.memory_reserved(self.device),
                       torch_peak_allocated_bytes=cuda.max_memory_allocated(self.device),
                       torch_peak_reserved_bytes=cuda.max_memory_reserved(self.device))
        if self.warp is not None:
            row.update(self.warp())
        if self.physx is not None:
            row.update({key: value for key, value in self.physx().items() if key.startswith('physx_gpu_mem_heap')})
        return row


def memory_peaks(samples):
    keys = sorted({key for sample in samples for key, value in sample.items() if type(value) is int})
    peaks = {key: max(sample[key] for sample in samples if type(sample.get(key)) is int) for key in keys}
    parts = ('torch_peak_reserved_bytes', 'warp_pool_peak_bytes', 'physx_gpu_mem_heap')
    # A sum of separate maxima bounds their simultaneous use from above.
    peaks['accounted_gpu_peak_bytes'] = sum(peaks[key] for key in parts) if all(key in peaks for key in parts) else None
    return peaks


class Profile:
    """Drive the stock runner and switch hooks at update boundaries.

    Warmup updates come first. Measured updates then alternate between an
    uninstrumented update, timed only at its boundaries, and a component update.
    """

    def __init__(self, *, env, task, wrapped, runner, task_module, warmup, measured, synchronize,
                 memory, record_update, deadline=None, clock=time.perf_counter, monotonic=time.monotonic):
        self.env, self.task, self.wrapped, self.runner, self.task_module = env, task, wrapped, runner, task_module
        self.plan = ['warmup']*warmup + ['throughput', 'components']*measured
        self.synchronize, self.memory, self.record_update = synchronize, memory, record_update
        self.deadline, self.clock, self.monotonic = deadline, clock, monotonic
        self.timer = SectionTimer(synchronize, clock)
        self.index, self.start, self.restore = 0, None, None
        self.rows, self.saves, self.barrier_costs = [], [], []
        self.steps = runner.cfg['num_steps_per_env']
        self.transitions = self.steps*env.num_envs
        if 'log' in vars(runner.logger) or 'save' in vars(runner):
            raise RuntimeError('The runner already has an update hook')
        self.original_log, self.original_save = runner.logger.log, runner.save

    @property
    def phase(self):
        return self.plan[self.index]

    def save(self, *args, **kwargs):
        # The stock runner saves after update 1; warmup keeps that write out of measurements.
        self.saves.append(self.index)
        return self.original_save(*args, **kwargs)

    def run(self):
        self.runner.logger.log, self.runner.save = self.log, self.save
        try:
            self.begin()
            self.runner.learn(len(self.plan), init_at_random_ep_len=False)
        finally:
            if self.restore is not None:
                self.restore()
                self.restore = None
            del self.runner.logger.log, self.runner.save
        if self.index != len(self.plan) or any(0 < index < len(self.plan) and self.plan[index] != 'warmup'
                                                for index in self.saves):
            raise RuntimeError('Profile updates or stock saves differ from the declared plan')
        return self.rows

    def begin(self):
        self.memory.reset_peaks()
        if self.phase == 'components':
            self.barrier_costs.append(barrier_cost(self.synchronize, self.clock))
            self.restore = instrument(self.timer, self.env, self.task, self.wrapped, self.runner, self.task_module)
            self.timer.begin('runner_other')
        else:
            self.synchronize()
            self.start = self.clock()

    def log(self, **values):
        if values['it'] != self.index:
            raise RuntimeError('Stock runner update index differs from the profile plan')
        components = self.phase == 'components'
        if components:
            self.timer.begin('update_logging')
        try:
            self.original_log(**values)
            status = self.record_update(self.index+1, values)
        finally:
            if components:
                self.timer.end()
        if components:
            wall = self.timer.end()
            seconds, calls, barriers = self.timer.take()
            self.restore()
            self.restore = None
            cost = self.barrier_costs[-1]
        else:
            self.synchronize()
            wall = self.clock()-self.start
            seconds = calls = barriers = cost = None
        row = {'update': self.index+1, 'phase': self.phase, 'wall_seconds': wall,
               'rsl_collection_seconds': values['collect_time'], 'rsl_learning_seconds': values['learn_time'],
               'transitions': self.transitions, 'component_seconds': seconds, 'component_calls': calls,
               'component_barriers': barriers, 'barrier_cost_seconds': cost,
               'memory': self.memory.sample(), 'task_interval': status}
        self.rows.append(row)
        self.index += 1
        if self.index < len(self.plan):
            if self.deadline is not None and self.monotonic() >= self.deadline:
                raise TimeoutError('Throughput profile ended after a complete PPO update')
            self.begin()


def pass_summary(rows, phase, *, transitions, steps):
    rows = [row for row in rows if row['phase'] == phase]
    wall = distribution(row['wall_seconds'] for row in rows)
    count = sum(row['transitions'] for row in rows)
    if count != transitions*len(rows):
        raise ValueError('Profile transition count differs from the runner plan')
    result = {'updates': len(rows), 'transitions': count, 'wall_seconds': wall,
              'rsl_collection_seconds': distribution(row['rsl_collection_seconds'] for row in rows),
              'rsl_learning_seconds': distribution(row['rsl_learning_seconds'] for row in rows),
              'training_transitions_per_second': count/wall['sum'],
              'collection_transitions_per_second': count/math.fsum(row['rsl_collection_seconds'] for row in rows),
              'scene_physics_steps_per_second': PHYSICS_STEPS_PER_CONTROL*steps*len(rows)/wall['sum'],
              'robot_physics_steps_per_second': PHYSICS_STEPS_PER_CONTROL*count/wall['sum'],
              'simulated_robot_seconds_per_wall_second': .02*count/wall['sum'],
              'memory_peaks': memory_peaks([row['memory'] for row in rows])}
    if phase != 'components':
        return result
    seconds, calls = defaultdict(float), defaultdict(int)
    for row in rows:
        for label, value in row['component_seconds'].items():
            seconds[label] += value
        for label, value in row['component_calls'].items():
            calls[label] += value
        substeps = PHYSICS_STEPS_PER_CONTROL*steps
        exact = {'physics_step': substeps, 'contact_readback': substeps, 'policy_inference': steps,
                 'env_control': steps, 'vec_env_other': steps, 'task_other': steps, 'reward': steps,
                 'task_metrics': steps, 'rollout_storage': steps, 'episode_logging': steps,
                 'returns': 1, 'ppo_update': 1, 'update_logging': 1, 'runner_other': 1}
        # Resets add reads, observations and proximity checks; captures follow each substep when attached.
        minimum = {'state_readback': substeps, 'observation': 2*steps, 'proximity_guard': 2*steps}
        found = row['component_calls']
        if (any(found.get(label) != value for label, value in exact.items())
                or any(found.get(label, 0) < value for label, value in minimum.items())
                or found.get('training_loads', substeps) != substeps):
            raise ValueError('Profile hooks missed part of an update: '+json.dumps(found))
        if not math.isclose(math.fsum(row['component_seconds'].values()), row['wall_seconds'], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('Profile components do not partition their update')
    total = wall['sum']
    per_transition = lambda value: 1e6*value/count
    barriers = defaultdict(int)
    for row in rows:
        for label, value in row['component_barriers'].items():
            barriers[label] += value
    cost = statistics.median(row['barrier_cost_seconds'] for row in rows)
    result['components'] = {}
    for label, (category, description) in COMPONENTS.items():
        value, overhead = seconds.get(label, 0.), barriers.get(label, 0)*cost
        result['components'][label] = {'category': category, 'description': description,
            'seconds': value, 'share': value/total, 'microseconds_per_transition': per_transition(value),
            'calls': calls.get(label, 0), 'barriers': barriers.get(label, 0),
            'estimated_barrier_seconds': overhead,
            'microseconds_per_transition_less_barriers': per_transition(value-overhead)}
    result['categories'] = {}
    for category in CATEGORIES:
        members = [row for row in result['components'].values() if row['category'] == category]
        value = math.fsum(row['seconds'] for row in members)
        overhead = math.fsum(row['estimated_barrier_seconds'] for row in members)
        result['categories'][category] = {'seconds': value, 'share': value/total,
            'microseconds_per_transition': per_transition(value), 'estimated_barrier_seconds': overhead,
            'microseconds_per_transition_less_barriers': per_transition(value-overhead)}
    result['component_sum_seconds'] = math.fsum(seconds.values())
    result['barrier_cost_seconds'] = distribution(row['barrier_cost_seconds'] for row in rows)
    result['barrier_scope'] = ('Each barrier waits for PyTorch and Warp work. The estimate multiplies each label\'s '
                               'barrier count by the median idle barrier cost measured before each component update; '
                               'Python hook overhead stays in the parent labels.')
    return result


def contact_census(env, wrapped, runner, controls, physx=None):
    """Count reported floor-contact patches under the profiled policy, outside timing."""
    import numpy as np
    import torch
    policy = runner.get_inference_policy()
    observation = wrapped.get_observations()
    reported, active, statistics_rows = [], [], []
    capacity = None
    for _ in range(controls):
        with torch.inference_mode():
            observation, _, _, _ = wrapped.step(policy(observation))
        force, _, _, _, counts, starts = [np.array(value.numpy(), copy=True)
                                          for value in env.contact.get_contact_data(env.cfg.physics_dt)]
        capacity = len(force)
        used = [index for start, count in zip(starts[:, 0], counts[:, 0]) for index in range(int(start), int(start+count))]
        if len(used) >= capacity:
            raise ValueError('Contact capacity exhausted during the census')
        reported.append(len(used))
        active.append(int(np.count_nonzero(force[used, 0])) if used else 0)
        if physx is not None:
            statistics_rows.append(physx())
    per_robot = lambda values: distribution(value/env.num_envs for value in values)
    return {'controls': controls, 'policy': 'deterministic actor mean after the measured updates',
            'capacity_patches': capacity, 'max_capacity_fraction': max(reported)/capacity,
            'reported_patches_per_robot': per_robot(reported),
            'nonzero_force_patches_per_robot': per_robot(active),
            'physx_scene_statistics': statistics_rows or None,
            'scope': 'Patches reported by the floor contact view at each control end; the census runs after all timed updates.'}


def physics_config(cfg, metadata):
    """Admission-bound physics settings; train.py records the same fields."""
    return {'physics_dt': cfg.physics_dt, 'decimation': cfg.decimation,
            'spacing_m': cfg.spacing_m, 'target_slew_rad': cfg.target_slew_rad, 'action_scale_rad': cfg.action_scale_rad,
            'solver_position_iterations': 32, 'solver_velocity_iterations': 0,
            'floor': '80m_two_triangle_mesh_y_equals_x_seam', 'material_friction': [1., 1.],
            'restitution': 0., 'external_forces_every_iteration': True,
            'neutral_joint_position_rad': metadata['nominal_joint_position_rad'],
            'reset_root_height_m': metadata['reset_root_height_m']}


def versions():
    result = {'python': platform.python_version(), 'machine': platform.machine()}
    for name in ('torch', 'rsl-rl-lib', 'tensordict', 'warp-lang', 'isaacsim', 'isaaclab'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def parser_for(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--mode', choices=['throughput'], required=True)
    for name in ('asset', 'model', 'geometry', 'geometry-extrema', 'stance', 'output', 'standing-admission'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-freeze-sha256', required=True)
    parser.add_argument('--num-envs', type=int, choices=REPLICA_COUNTS, required=True)
    parser.add_argument('--seed', type=int, default=20260917)
    parser.add_argument('--warmup-updates', type=int, default=2)
    parser.add_argument('--updates', type=int, default=5, help='Measured updates in each of the two alternating passes.')
    parser.add_argument('--max-wall-seconds', type=float, default=6200.)
    parser.add_argument('--preflight-only', action='store_true')
    if any(flag in argv for flag in ('--preflight-only', '--help', '-h')):
        parser.add_argument('--headless', action='store_true')
        parser.add_argument('--device', default='cuda:0')
    else:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    return parser


def validate(args):
    if (not args.headless or args.device != 'cuda:0' or args.seed < 0
            or not WARMUP_UPDATES[0] <= args.warmup_updates <= WARMUP_UPDATES[1]
            or not MEASURED_UPDATES[0] <= args.updates <= MEASURED_UPDATES[1]
            or not 0 < args.max_wall_seconds <= 6600):
        raise ValueError('A bounded headless profile needs 2-10 warmup and 1-20 measured updates per pass')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['summarize']:
        return summarize_main(argv[1:])
    source = Path(__file__).resolve().parent
    prefix = __package__
    from . import admission
    configuration = importlib.import_module(prefix+'.env_config')
    args = parser_for(argv).parse_args(argv)
    validate(args)
    configuration.verify_assets(args.asset, args.model)
    if sha(source.parent/'FREEZE_SHA256.json') != args.source_freeze_sha256:
        raise ValueError('Frozen source identity differs')
    metadata = json.loads(args.stance.read_text())
    # The profile uses the training allocation: 20-second episodes, no motion features or rendering.
    cfg = configuration.EnvConfig(num_envs=args.num_envs, seed=args.seed, record_motion_features=False,
                                  render=False, episode_seconds=20., device=args.device)
    identity = {'schema': SCHEMA, 'source_files': {p.name: sha(p) for p in sorted(source.glob('*.py'))},
        'model_sha256': configuration.MODEL_SHA256, 'usd_sha256': configuration.USD_SHA256,
        'stance_sha256': sha(args.stance), 'geometry_sha256': sha(args.geometry),
        'geometry_extrema_sha256': sha(args.geometry_extrema), 'config': cfg.declaration(),
        'motion_prior': False, 'behavior_cloning': False, 'seed': args.seed,
        'rsl_rl_required_version': '5.0.1', 'adapter_sha256': sha(source/'ppo.py'),
        'entry_sha256': sha(__file__), 'source_freeze_sha256': args.source_freeze_sha256, 'learner': LEARNER}
    identity['physics_source_files'] = {k: identity['source_files'][k] for k in ('env.py', 'env_config.py')}
    identity['physics_config'] = physics_config(cfg, metadata)
    identity['standing_admission'] = admission.require_admission(args.standing_admission, identity, cfg)
    settings = {'num_envs': args.num_envs, 'seed': args.seed, 'warmup_updates': args.warmup_updates,
                'measured_updates_per_pass': args.updates, 'pass_order': 'warmup, then alternating throughput and components',
                'census_controls': CENSUS_CONTROLS, 'max_wall_seconds': args.max_wall_seconds, 'device': args.device}
    if args.preflight_only:
        print(json.dumps({'identity': identity, 'settings': settings}, indent=2))
        return 0
    return run(args, identity, settings, cfg, metadata, prefix)


def run(args, identity, settings, cfg, metadata, prefix):
    args.output.mkdir(parents=True, exist_ok=False)
    state = {'mode': 'throughput', 'status': 'initializing', 'identity': identity, 'settings': settings,
             'errors': [], 'stage2_complete': False, 'physical_admission': False,
             'runtime_binding': {'runtime_tree_sha256': args.source_freeze_sha256}}
    save(args.output/'state.json', state)
    app = loads = None
    started = time.monotonic()
    startup = {}
    try:
        import faulthandler
        from isaaclab.app import AppLauncher
        args.enable_cameras = False
        args.headless_explicit = False
        with (args.output/'startup_tracebacks.log').open('w') as stream:
            faulthandler.enable(file=stream)
            faulthandler.dump_traceback_later(45., repeat=True, file=stream)
            try:
                app = AppLauncher(args).app
            finally:
                faulthandler.cancel_dump_traceback_later(); faulthandler.disable()
        startup['app_launch_seconds'] = time.monotonic()-started
        print('REFERENCE_SCREEN_APP_READY', flush=True)
        import numpy as np
        import torch
        vanilla = importlib.import_module(prefix+'.ppo')
        native = importlib.import_module(prefix+'.env')
        task_module = importlib.import_module(prefix+'.task')
        train = importlib.import_module(prefix+'.train')
        from rsl_rl.runners import OnPolicyRunner
        version = importlib.metadata.version('rsl-rl-lib')
        if version != '5.0.1':
            raise ValueError('RSL-RL version differs: '+version)
        synchronize = device_barrier(torch, args.device)
        warp_reader, sources = warp_memory(args.device)
        memory = MemoryProbe(args.device, torch, warp_reader)
        samples = {'after_app_launch': memory.sample()}
        torch.manual_seed(args.seed); np.random.seed(args.seed)

        def stage(name, build):
            synchronize(); begin = time.perf_counter()
            value = build()
            synchronize(); startup[name+'_seconds'] = time.perf_counter()-begin
            samples['after_'+name] = memory.sample()
            return value
        env = stage('environment', lambda: native.LocomotionEnv(cfg, args.asset, args.model, args.geometry,
                                                                  args.output/'native', reference_metadata=metadata))
        physx, physx_status = physx_statistics(env)
        memory.physx = physx
        contexts = cuda_contexts(env, args.device)
        task = stage('task', lambda: task_module.TrainingTask(env, task_module.TaskConfig(seed=args.seed), args.output/'task'))
        config = vanilla.ppo_config(args.seed)
        save(args.output/'ppo_config.json', config)
        wrapped, runner = stage('learner', lambda: learner(vanilla, task, config, args, OnPolicyRunner))
        import rsl_rl
        upstream = Path(rsl_rl.__file__).parent
        identity['upstream_source_files'] = {name: sha(upstream/name) for name in (
            'runners/on_policy_runner.py', 'algorithms/ppo.py', 'models/mlp_model.py', 'storage/rollout_storage.py',
            'utils/logger.py')}
        identity['ppo_config'] = config
        loads = vanilla.TrainingLoads(env)
        env.capture = loads
        state['status'] = 'running'; save(args.output/'state.json', state)

        def record_update(update, values):
            # Mirror train.py's per-update metrics row, scalars and state file.
            row = {'update': update, 'transitions': update*runner.cfg['num_steps_per_env']*env.num_envs,
                   'collection_seconds': values['collect_time'], 'learning_seconds': values['learn_time'],
                   'loss': values['loss_dict'], 'learning_rate': values['learning_rate'],
                   'mean_action_std': float(values['action_std'].mean()),
                   'task': task.status(reset_interval=True)}
            with (args.output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            if runner.logger.writer is not None:
                for tag, value in train.scalars(row['task'], 'Task').items():
                    runner.logger.writer.add_scalar(tag, value, values['it'])
            state.update(updates=update, wall_seconds=time.monotonic()-started)
            save(args.output/'state.json', state)
            interval = row['task']['interval_metrics']
            return {key: interval.get(key) for key in ('environment_controls', 'terminations', 'truncations',
                                                       'nonfoot_event_rows', 'moving_command_rows')}
        profile = Profile(env=env, task=task, wrapped=wrapped, runner=runner, task_module=task_module,
                          warmup=args.warmup_updates, measured=args.updates, synchronize=synchronize,
                          memory=memory, record_update=record_update, deadline=started+args.max_wall_seconds)
        try:
            rows = profile.run()
        finally:
            with (args.output/'updates.jsonl').open('w') as stream:
                for row in profile.rows:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
        training_loads = loads.report()
        census = contact_census(env, wrapped, runner, CENSUS_CONTROLS, physx)
        report = build_report(rows, identity=identity, settings=settings, startup=startup, samples=samples,
                              census=census, steps=profile.steps, num_envs=env.num_envs, stock_saves=profile.saves,
                              platform_versions={**versions(), **device_record(torch, args.device),
                                                 'warp_memory': sources, 'cuda_contexts': contexts,
                                                 'physx_statistics': {'status': physx_status,
                                                     **(physx.status() if physx is not None else {})}},
                              training_loads=training_loads)
        save(args.output/'profile.json', report)
        env.verify_native_recipe('after_controlled_steps')
        state.update(status='completed', profile_sha256=sha(args.output/'profile.json'))
    except BaseException as error:
        state['status'] = 'failed'; state['errors'].append(repr(error))
        state['traceback'] = traceback.format_exc()
        print(state['traceback'], flush=True)
    finally:
        if loads is not None:
            try:
                save(args.output/'force_metrics.json', loads.report())
            except Exception as error:
                state['errors'].append('force_metrics: '+repr(error))
        state['startup'] = startup
        state['wall_seconds'] = time.monotonic()-started
        save(args.output/'state.json', state)
        if app is not None:
            app.close()
    return 0 if state['status'] == 'completed' else 1


def learner(vanilla, task, config, args, runner_class):
    wrapped = vanilla.VanillaVecEnv(task)
    runner = runner_class(wrapped, copy.deepcopy(config), str(args.output/'learner'), device=args.device)
    return wrapped, runner


def device_barrier(torch, device):
    """Wait for PyTorch and Warp work on the device, as Isaac Lab's step timer does."""
    try:
        import warp as wp
        warp_barrier = lambda: wp.synchronize_device(device)
    except ImportError:
        warp_barrier = lambda: None

    def synchronize():
        torch.cuda.synchronize(device)
        warp_barrier()
    return synchronize


def cuda_contexts(env, device):
    """Record whether PhysX runs in the primary CUDA context that the barriers wait for."""
    try:
        import warp as wp
        values = [int(getattr(value, 'value', value)) for value in
                  (env.sim.physics_sim_view.cuda_context, wp.get_device(device).context)]
        return {'physx_context': hex(values[0]), 'primary_context': hex(values[1]), 'shared': values[0] == values[1]}
    except Exception as error:
        return {'shared': None, 'status': 'unavailable: '+repr(error)}


def warp_memory(device):
    """Return a Warp memory-pool reader and its status; the pool high-water mark spans the process."""
    try:
        import warp as wp
        if not wp.is_mempool_enabled(device):
            return None, 'Warp memory pool disabled'
        read = lambda: {'warp_pool_used_bytes': int(wp.get_mempool_used_mem_current(device)),
                        'warp_pool_peak_bytes': int(wp.get_mempool_used_mem_high(device))}
        read()
        return read, 'available; Warp '+wp.__version__
    except Exception as error:
        return None, 'unavailable: '+repr(error)


PHYSX_FIELDS = ('nb_discrete_contact_pairs_total', 'nb_new_pairs', 'nb_lost_pairs', 'nb_new_touches',
                'nb_lost_touches', 'gpu_mem_rigid_contact_count', 'gpu_mem_rigid_patch_count', 'gpu_mem_heap',
                'gpu_mem_heap_broadphase', 'gpu_mem_heap_narrowphase', 'gpu_mem_heap_solver',
                'gpu_mem_heap_articulation', 'gpu_mem_heap_simulation', 'gpu_mem_heap_simulation_articulation',
                'gpu_mem_heap_other')


class PhysxStatistics:
    """Read PhysX scene statistics after a completed step and count unanswered reads.

    The statistics describe the last physics substep. `gpu_mem_heap` is heap
    capacity; the named heap groups hold live allocations.
    """

    def __init__(self, interface, statistics_class, stage, scene):
        self.interface, self.statistics_class, self.stage, self.scene = interface, statistics_class, stage, scene
        self.reads = self.unanswered = 0

    def __call__(self):
        value = self.statistics_class()
        self.reads += 1
        if not self.interface.get_physx_scene_statistics(self.stage, self.scene, value):
            self.unanswered += 1
            return {}
        return {'physx_'+name: int(getattr(value, name)) for name in PHYSX_FIELDS}

    def status(self):
        return {'reads': self.reads, 'unanswered_reads': self.unanswered}


def physx_statistics(env):
    """Return a PhysX statistics reader, or None and the reason it is unavailable."""
    try:
        from omni.physx import get_physx_statistics_interface
        from omni.physx.bindings._physx import PhysicsSceneStats
        from pxr import PhysicsSchemaTools, UsdUtils
        return PhysxStatistics(get_physx_statistics_interface(), PhysicsSceneStats,
                               UsdUtils.StageCache.Get().GetId(env.sim.stage).ToLongInt(),
                               PhysicsSchemaTools.sdfPathToInt(env.scene_prim_path)), 'available'
    except Exception as error:
        return None, 'unavailable: '+repr(error)


def device_record(torch, device):
    if not torch.cuda.is_available():
        return {'cuda_device': None}
    properties = torch.cuda.get_device_properties(device)
    return {'cuda_device': properties.name, 'cuda_capability': list(torch.cuda.get_device_capability(device)),
            'cuda_runtime': torch.version.cuda, 'device_total_memory_bytes': int(properties.total_memory),
            'integrated_memory': bool(getattr(properties, 'is_integrated', False))}


def build_report(rows, *, identity, settings, startup, samples, census, steps, num_envs,
                 stock_saves, platform_versions, training_loads):
    transitions = steps*num_envs
    warmup = [row for row in rows if row['phase'] == 'warmup']
    return {'schema': SCHEMA, 'status': 'completed', 'learner': LEARNER, 'amp_learner': AMP_LEARNER,
        'identity': identity, 'settings': {**settings, 'steps_per_update': steps, 'transitions_per_update': transitions},
        'platform': platform_versions, 'startup_seconds': startup, 'startup_memory': samples,
        'warmup': {'excluded_from_statistics': True, 'wall_seconds': [row['wall_seconds'] for row in warmup],
                   'stock_runner_save_update_windows': [index+1 for index in stock_saves if index < len(rows)],
                   'stock_runner_saves_after_final_update': sum(index >= len(rows) for index in stock_saves)},
        'throughput': {'synchronization': 'Device synchronization at update boundaries only; no component hooks.',
                       **pass_summary(rows, 'throughput', transitions=transitions, steps=steps)},
        'components': {'synchronization': ('Device synchronization at every component boundary; exclusive times '
                                           'partition each update. platform.cuda_contexts records whether the '
                                           'barriers also wait for PhysX kernels.'),
                       **pass_summary(rows, 'components', transitions=transitions, steps=steps)},
        'contact_census': census, 'training_loads': training_loads,
        'memory_scope': MEMORY_SCOPE,
        'excluded': ['Project checkpoints, which train.py writes every 50 updates and at the end.',
                     'W&B logging, policy evaluation and video capture.',
                     'The contact census controls, which run after training_loads is reported.'],
        'stage2_complete': False, 'physical_admission': False}


def allocation_files(directory, profile):
    """Require the launcher's completed job and matching binding beside a profile directory."""
    run = directory.parent
    binding_path, job_path, audit_path = run/'launch_binding.json', run/'jobs/standing.json', run/'jobs/standing_contact_data_audit.json'
    if not all(path.is_file() for path in (binding_path, job_path, audit_path)):
        raise ValueError('Profile lacks its launcher binding, job record or contact audit: '+str(directory))
    binding, job, audit = (json.loads(path.read_text()) for path in (binding_path, job_path, audit_path))
    if job.get('status') != 'completed' or audit.get('passed') is not True:
        raise ValueError('The launcher did not complete this allocation or its contact audit: '+str(directory))
    arguments = binding.get('command_args', [])
    value = lambda option: arguments[arguments.index(option)+1] if arguments.count(option) == 1 else None
    settings = profile['settings']
    if (binding.get('mode') != 'throughput' or binding.get('module') != 'locomotion.throughput'
            or binding.get('source_freeze_sha256') != profile['identity']['source_freeze_sha256']
            or value('--num-envs') != str(settings['num_envs'])
            or value('--updates') != str(settings['measured_updates_per_pass'])
            or value('--warmup-updates') != str(settings['warmup_updates'])):
        raise ValueError('Launch binding and profile settings differ: '+str(directory))
    files = {'launch_binding.json': sha(binding_path), 'jobs/standing.json': sha(job_path),
             'jobs/standing_contact_data_audit.json': sha(audit_path)}
    if (run/'cleanup.json').is_file():
        files['cleanup.json'] = sha(run/'cleanup.json')
    return files


def summarize(runs, output):
    """Compare completed profiles of one source and platform at distinct admitted replica counts."""
    output = Path(output)
    entries = []
    for directory in map(Path, runs):
        state_path, profile_path = directory/'state.json', directory/'profile.json'
        state = json.loads(state_path.read_text())
        profile = json.loads(profile_path.read_text())
        if state.get('status') != 'completed' or state.get('profile_sha256') != sha(profile_path):
            raise ValueError('Profile is incomplete or its bytes changed: '+str(directory))
        if profile.get('schema') != SCHEMA or profile.get('status') != 'completed' or profile.get('learner') != LEARNER:
            raise ValueError('Unsupported or incomplete profile: '+str(directory))
        entries.append((directory, state_path, profile_path, profile, allocation_files(directory, profile)))
    if not entries:
        raise ValueError('Name at least one completed profile')
    first = entries[0][3]
    for directory, _, _, profile, _ in entries:
        differs = ([key for key in IDENTITY_KEYS if profile['identity'].get(key) != first['identity'].get(key)]
                   + [key for key in PLATFORM_KEYS if profile['platform'].get(key) != first['platform'].get(key)]
                   + [key for key in SETTING_KEYS if profile['settings'].get(key) != first['settings'].get(key)])
        if differs:
            raise ValueError('Profiles differ in source, inputs, learner, platform or settings: '+', '.join(differs))
    counts = [entry[3]['settings']['num_envs'] for entry in entries]
    if len(set(counts)) != len(counts) or not set(counts) <= set(REPLICA_COUNTS):
        raise ValueError('Each profile needs a distinct admitted replica count')
    runs_out = []
    for directory, state_path, profile_path, profile, files in sorted(entries, key=lambda entry: entry[3]['settings']['num_envs']):
        throughput, components = profile['throughput'], profile['components']
        launch = profile['startup_memory'].get('after_app_launch', {}).get('system_used_bytes')
        peak = throughput['memory_peaks'].get('system_used_bytes')
        runs_out.append({'num_envs': profile['settings']['num_envs'], 'directory': str(directory),
            'profile_sha256': sha(profile_path), 'state_sha256': sha(state_path), 'allocation_files': files,
            'standing_admission_num_envs': profile['identity']['standing_admission']['num_envs'],
            'component_split_verified': profile['platform'].get('cuda_contexts', {}).get('shared') is True,
            'training_transitions_per_second': throughput['training_transitions_per_second'],
            'collection_transitions_per_second': throughput['collection_transitions_per_second'],
            'robot_physics_steps_per_second': throughput['robot_physics_steps_per_second'],
            'update_wall_seconds': throughput['wall_seconds'],
            'synchronization_overhead_ratio': components['wall_seconds']['mean']/throughput['wall_seconds']['mean'],
            'barrier_cost_seconds': components['barrier_cost_seconds']['median'],
            'category_microseconds_per_transition': {name: value['microseconds_per_transition']
                                                     for name, value in components['categories'].items()},
            'category_microseconds_per_transition_less_barriers': {
                name: value['microseconds_per_transition_less_barriers'] for name, value in components['categories'].items()},
            'component_microseconds_per_transition': {name: value['microseconds_per_transition']
                                                      for name, value in components['components'].items()},
            'memory_peaks': {'throughput': throughput['memory_peaks'], 'components': components['memory_peaks']},
            'system_used_above_launch_bytes': peak-launch if None not in (peak, launch) else None,
            'contact_census': {key: profile['contact_census'][key] for key in
                               ('capacity_patches', 'max_capacity_fraction', 'reported_patches_per_robot', 'policy', 'scope')}})
    scaling = []
    for low, high in zip(runs_out, runs_out[1:]):
        added = high['num_envs']-low['num_envs']
        memory = {}
        for key in ('torch_peak_reserved_bytes', 'accounted_gpu_peak_bytes', 'physx_gpu_mem_heap'):
            values = [row['memory_peaks']['throughput'].get(key) for row in (low, high)]
            memory[key+'_per_added_robot'] = (values[1]-values[0])/added if None not in values else None
        values = [row['system_used_above_launch_bytes'] for row in (low, high)]
        memory['system_used_above_launch_bytes_per_added_robot'] = (values[1]-values[0])/added if None not in values else None
        scaling.append({'from_num_envs': low['num_envs'], 'to_num_envs': high['num_envs'],
            'training_transitions_per_second_ratio': high['training_transitions_per_second']/low['training_transitions_per_second'],
            'update_wall_seconds_ratio': high['update_wall_seconds']['mean']/low['update_wall_seconds']['mean'],
            **memory})
    result = {'schema': SUMMARY_SCHEMA, 'learner': LEARNER, 'amp_learner': AMP_LEARNER,
              'identity': {key: first['identity'].get(key) for key in IDENTITY_KEYS},
              'platform': {key: first['platform'].get(key) for key in PLATFORM_KEYS},
              'settings': {key: first['settings'].get(key) for key in SETTING_KEYS},
              'runs': runs_out, 'scaling': scaling, 'memory_scope': MEMORY_SCOPE,
              'scope': ('Measured admitted replica counts only. The summary extrapolates to no other count and changes '
                        'no guard. Host memory slopes use each allocation\'s own post-launch baseline and still include '
                        'other processes. A row without component_split_verified may place PhysX GPU time in '
                        'state_readback or contact_readback.'),
              'stage2_complete': False, 'physical_admission': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def table(summary):
    lines = ['Microseconds per transition by category, from the component pass',
             'replicas  transitions/s  '+'  '.join(CATEGORIES)]
    for row in summary['runs']:
        category = row['category_microseconds_per_transition']
        lines.append(f"{row['num_envs']:>8}  {row['training_transitions_per_second']:>13.1f}  "
                     + '  '.join(f"{category[name]:>{len(name)}.1f}" for name in CATEGORIES)
                     + ('' if row['component_split_verified'] else '  (PhysX synchronization unverified)'))
    return '\n'.join(lines)


def summarize_main(argv):
    parser = argparse.ArgumentParser(prog='python -m locomotion.throughput summarize',
                                     description=summarize.__doc__, allow_abbrev=False)
    parser.add_argument('--run', type=Path, action='append', required=True,
                        help='Directory holding state.json and profile.json; repeat per replica count.')
    parser.add_argument('--output', type=Path, required=True, help='Fresh summary JSON path.')
    args = parser.parse_args(argv)
    print(table(summarize(args.run, args.output)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
