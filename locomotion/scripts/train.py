"""Train or evaluate standard PPO on the admitted canonical simulator."""
from __future__ import annotations

import argparse
import copy
import faulthandler
from functools import partial
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np

from locomotion.evaluation.admission import require_admission
from locomotion.evaluation.camera import NativePolicyCamera, prepare_policy_scene
# locomotion/tests/native_parity.py calls require_learner_identity through this module.
from locomotion.evaluation.checkpoint import (DECLARATION_SHA, checkpoint_reward_version, read_compatibility_receipt,
    require_checkpoint_identity, require_compatibility_configuration, require_compatibility_identity,
    require_embedded_declaration, require_learner_configuration, require_learner_identity)
from locomotion.omniwalk.agents.rsl_rl_ppo_cfg import RECIPES, ppo_config
from locomotion.omniwalk.mdp.commands import GAIT_CLOCK_CONTROLS, TaskConfig
from locomotion.omniwalk.env_cfg import EnvConfig
from locomotion.robots.hexapod import MODEL_SHA256, USD_SHA256, verify_assets
from locomotion.spark.sources import physics_files, source_files, verify_runtime
from locomotion.spark.validation import (ALLOCATION_PROFILES, LEARNER_OPTION_DEFAULTS, OPTION_DEFAULTS,
    collected_transitions, decay_updates, finish_training_update, resolve_learner_options, update_limit,
    validate_deadline, validate_decay)

SOURCE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def scalars(value, prefix):
    """Return finite numeric leaves of a task status; drop lists, text and missing values."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(scalars(item, f'{prefix}/{key}'))
        return result
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return {prefix: value}
    return {}


def print_progress(value):
    print(json.dumps(value), flush=True)


class ConfigRecord(dict):
    """Give RSL-RL's W&B writer the to_dict() it expects from an environment config."""
    def to_dict(self):
        return dict(self)


def parse_args(argv=None):
    """Parse the command line and resolve the recipe and the gait-clock default for the mode.

    A preflight or help run declares the two launcher options itself, so it needs no Isaac installation.
    """
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--mode', choices=['diagnostic', 'train', 'evaluate', 'probe'], required=True)
    for name in ('asset', 'model', 'geometry', 'geometry-extrema', 'stance', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--standing-admission', type=Path)
    parser.add_argument('--source-freeze-sha256', required=True)
    parser.add_argument('--num-envs', type=int, choices=[1, 32, 128, 512, 1024], required=True,
                        help='Replica count. Training and probes accept 128 and 512; diagnostics and profiles accept each listed count.')
    parser.add_argument('--seed', type=int, default=20260917)
    parser.add_argument('--updates', type=int, default=512)
    parser.add_argument('--extended-updates', action='store_true',
                        help='Accept up to 5000 training updates; without this opt-in training stops at 2000.')
    parser.add_argument('--max-wall-seconds', type=float, default=6200.)
    parser.add_argument('--allocation-profile', choices=ALLOCATION_PROFILES, default='standard')
    parser.add_argument('--eval-scope', choices=['focus', 'probes', 'full'], default='focus')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--checkpoint-compatibility', type=Path)
    parser.add_argument('--checkpoint-compatibility-sha256')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--logger', choices=['tensorboard', 'wandb'], default='tensorboard')
    parser.add_argument('--wandb-project')
    parser.add_argument('--wandb-mode', choices=['offline', 'online'], default='offline')
    parser.add_argument('--learner', choices=['ppo', 'amp'], default='ppo')
    parser.add_argument('--networks', choices=['mlp', 'paper'], default='mlp')
    parser.add_argument('--recipe', choices=[*RECIPES, 'none'], default='c48_long',
                        help='Learner options of PPO training and evaluation (%(default)s by default) from omniwalk/agents/'
                             'rsl_rl_ppo_cfg.py; option flags replace its values. Other runs and the recipe none keep the defaults.')
    parser.add_argument('--action-mean', choices=['unbounded', 'tanh'],
                        help='Use the historical Gaussian mean or bound the mean with tanh.')
    parser.add_argument('--observation-normalization', choices=['empirical', 'none'],
                        help='Use running observation statistics or retain the raw PPO observations.')
    parser.add_argument('--observation-scaling', choices=['none', 'fixed'],
                        help='Pass raw observations or apply the fixed input scales declared in wrapper.py.')
    parser.add_argument('--command-segments', choices=['continuous', 'bootstrap'],
                        help='Carry returns across command changes or bootstrap the return at each change.')
    parser.add_argument('--learning-rate-max', type=float,
                        help='Ceiling for the adaptive learning rate; the stock schedule allows 1e-2.')
    parser.add_argument('--action-std', type=float,
                        help='Initial standard deviation of the Gaussian action distribution.')
    parser.add_argument('--episode-seconds', type=float, default=20.,
                        help='Training episode length; 20 s by default. Shorter episodes bound how far a robot walks.')
    parser.add_argument('--action-smoothing', choices=['none', 'mean2'],
                        help='mean2 sends the mean of each action and the previous one to the environment.')
    parser.add_argument('--velocity-noise', type=float,
                        help='Standard deviation in rad/s of Gaussian noise on the actor joint-velocity inputs in training.')
    parser.add_argument('--gait-clock', type=int,
                        help=f'Append gait phase with this period in controls; without a recipe value, training '
                             f'defaults to {GAIT_CLOCK_CONTROLS}, other modes to 0.')
    parser.add_argument('--action-std-final', type=float,
                        help='Lower the action deviation linearly to this value over the run; PPO then does not learn it.')
    parser.add_argument('--action-std-decay-updates', type=int,
                        help='Reach the final action deviation after this many updates and hold it; the run length by default.')
    parser.add_argument('--action-noise-correlation', type=float,
                        help='Share of each exploration noise sample carried to the next control (tanh mean only).')
    parser.add_argument('--steps-per-env', type=int,
                        help='Controls each robot collects per PPO update, a multiple of 6 from 12 to 96; '
                             'mini-batches scale with it.')
    parser.add_argument('--video-case',
                        help='Learning probe to record on video; the first selected probe by default.')
    if any(flag in (argv if argv is not None else sys.argv[1:]) for flag in ('--preflight-only', '--help', '-h')):
        parser.add_argument('--headless', action='store_true')
        parser.add_argument('--device', default='cuda:0')
    else:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    vars(args).update(resolve_learner_options(args.recipe, args.mode, args.learner, args.networks,
                                              {key: getattr(args, key) for key in OPTION_DEFAULTS}))
    return args


def learner_options(args):
    """Return the learner options that ppo_config records, in its keyword order."""
    return dict(observation_scaling=args.observation_scaling, command_segments=args.command_segments,
                learning_rate_max=args.learning_rate_max, action_std=args.action_std,
                action_noise_correlation=args.action_noise_correlation,
                action_std_final=args.action_std_final, gait_clock=args.gait_clock,
                action_smoothing=args.action_smoothing, velocity_noise=args.velocity_noise,
                action_std_decay_updates=args.action_std_decay_updates, steps_per_env=args.steps_per_env)


def validate(args):
    """Reject option combinations, changed assets and a changed frozen source.

    Return the compatibility receipt, or None when the run names no receipt.
    """
    if (args.networks == 'paper' and args.learner != 'amp') or (args.learner == 'amp' and args.mode not in ('train', 'evaluate')):
        raise ValueError('The paper networks need the AMP learner for training or evaluation')
    if args.mode == 'train' and args.networks == 'mlp' and args.gait_clock != GAIT_CLOCK_CONTROLS:
        raise ValueError(f'The active reward requires the {GAIT_CLOCK_CONTROLS}-control gait clock')
    compatibility = None
    if args.checkpoint_compatibility is not None or args.checkpoint_compatibility_sha256 is not None:
        if args.mode != 'evaluate' or args.eval_scope != 'probes' or args.num_envs != 1 or args.checkpoint_compatibility is None or args.checkpoint_compatibility_sha256 is None:
            raise ValueError('Compatibility requires evaluation and both receipt arguments')
        compatibility = read_compatibility_receipt(args.checkpoint_compatibility, args.checkpoint_compatibility_sha256)
    if args.logger == 'wandb':
        if (args.mode != 'train' or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', args.wandb_project or '')
                or importlib.util.find_spec('wandb') is None
                or (args.wandb_mode == 'online' and not os.environ.get('WANDB_API_KEY'))):
            raise ValueError('W&B logging needs train mode, a project name, the wandb package '
                             'and WANDB_API_KEY when online')
    elif args.wandb_project is not None or args.wandb_mode != 'offline':
        raise ValueError('W&B options require --logger wandb')
    if args.action_mean != 'unbounded' and args.mode not in ('train', 'evaluate'):
        raise ValueError('Bounded action means apply to training and evaluation')
    if args.observation_normalization != 'empirical' and args.mode not in ('train', 'evaluate'):
        raise ValueError('Observation normalization selection requires PPO training or evaluation')
    if learner_options(args) != LEARNER_OPTION_DEFAULTS and args.mode not in ('train', 'evaluate'):
        raise ValueError('Observation scaling, command segments, the rate ceiling, the action deviation and '
                         'the noise correlation require PPO training or evaluation')
    if not 12 <= args.steps_per_env <= 96 or args.steps_per_env % 6:
        raise ValueError('The rollout length is a multiple of 6 from 12 to 96 controls per robot')
    if args.video_case is not None and (args.mode != 'evaluate' or args.eval_scope == 'full'):
        raise ValueError('A video case applies to a learning-probe evaluation')
    if args.episode_seconds != 20. and (args.mode != 'train' or not 5. <= args.episode_seconds <= 20.):
        raise ValueError('Episode length selection applies to training, between 5 and 20 seconds')
    if args.action_std_decay_updates is not None and args.action_std_final is None:
        raise ValueError('The deviation decay length needs a final action deviation')
    validate_decay(args.mode, args.updates, args.action_std_decay_updates)
    limit = update_limit(args.mode, args.extended_updates)
    verify_assets(args.asset, args.model)
    validate_deadline(args.mode, args.allocation_profile, args.max_wall_seconds)
    if (not args.headless or args.device != 'cuda:0' or not 1 <= args.updates <= limit
            or args.seed < 0
            or (args.mode in ('train', 'probe') and (args.num_envs not in (128, 512) or args.checkpoint is not None))
            or (args.mode == 'evaluate' and (args.num_envs != 1 or args.checkpoint is None))):
        raise ValueError('A bounded scratch training, a 128-replica probe or single-replica evaluation is required')
    if sha(SOURCE.parent/'FREEZE_SHA256.json') != args.source_freeze_sha256:
        raise ValueError('Frozen source identity differs')
    return compatibility


def environment_config(args):
    """Build the environment configuration for the mode; full evaluation uses the evaluation configuration."""
    from locomotion.evaluation.runner import EvaluationEnvConfig
    config_class = EvaluationEnvConfig if args.mode == 'evaluate' and args.eval_scope == 'full' else EnvConfig
    if args.mode == 'evaluate':
        episode_seconds = 90. if args.eval_scope == 'full' else 60.
    else:
        episode_seconds = 60. if args.mode == 'diagnostic' else args.episode_seconds
    return config_class(num_envs=args.num_envs, seed=args.seed,
        record_motion_features=args.mode == 'evaluate' or args.learner == 'amp', render=args.mode == 'evaluate',
        episode_seconds=episode_seconds, device=args.device)


def build_identity(args, cfg, metadata):
    """Record the sources, inputs, physics and learner that the run and its checkpoints bind."""
    identity = {'schema': 'hexapod_locomotion_ppo_v2', 'source_files': source_files(SOURCE),
        'runtime_sources': verify_runtime(SOURCE.parent),
        'model_sha256': MODEL_SHA256, 'usd_sha256': USD_SHA256,
        'stance_sha256': sha(args.stance),
        'geometry_sha256': sha(args.geometry), 'geometry_extrema_sha256': sha(args.geometry_extrema),
        'config': cfg.declaration(), 'motion_prior': args.learner == 'amp', 'behavior_cloning': False,
        'seed': args.seed, 'rsl_rl_required_version': '5.0.1',
        'adapter_sha256': sha(SOURCE/'rl/wrapper.py'), 'entry_sha256': sha(__file__),
        'reward_version': '4', 'reward_task': 'v4_yaw_gate', 'learner': args.learner, 'networks': args.networks}
    if args.extended_updates:
        # The exploration decay length enters ppo_config; the opt-in records the requested run length beside it.
        identity['training_updates'] = args.updates
    if args.learner == 'amp':
        from locomotion.priors.amp import AMPConfig, load_demonstrations
        identity['learner_source_files'] = {name: sha(SOURCE/name) for name in (
            'omniwalk/mdp/observations.py', 'priors/discriminator.py', 'priors/amp.py', 'priors/networks.py')}
        identity['amp_dataset'] = load_demonstrations(SOURCE.parent/AMPConfig().dataset)[1]
    identity['physics_source_files'] = physics_files(identity['source_files'])
    identity['physics_config'] = {'physics_dt': cfg.physics_dt, 'decimation': cfg.decimation,
        'spacing_m': cfg.spacing_m, 'target_slew_rad': cfg.target_slew_rad, 'action_scale_rad': cfg.action_scale_rad,
        'solver_position_iterations': 32, 'solver_velocity_iterations': 0,
        'floor': 'isaaclab_ground_plane', 'filter_collisions': True, 'material_friction': [1., 1.],
        'restitution': 0., 'external_forces_every_iteration': True,
        'neutral_joint_position_rad': metadata['nominal_joint_position_rad'],
        'reset_root_height_m': metadata['reset_root_height_m']}
    if args.mode != 'diagnostic':
        identity['standing_admission'] = require_admission(args.standing_admission, identity, cfg)
    return identity


def bind_checkpoint(args, identity, compatibility):
    """Check the checkpoint bytes and declaration, then record the reward it learned from.

    Return the checkpoint declaration.
    """
    if sha(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('Checkpoint bytes differ')
    record = json.loads(args.checkpoint.with_suffix('.json').read_text())
    if record['checkpoint_sha256'] != args.checkpoint_sha256:
        raise ValueError('Checkpoint declaration differs')
    if compatibility is None:
        require_checkpoint_identity(record, identity)
    else:
        if sha(args.checkpoint.with_suffix('.json')) != DECLARATION_SHA:
            raise ValueError('Compatibility checkpoint declaration differs')
        require_compatibility_identity(compatibility, record, identity)
    # Evaluation computes no training reward; record the one the checkpoint learned from.
    identity['reward_version'] = checkpoint_reward_version(record)
    for key in ('reward_task', 'reward_options'):
        identity.pop(key, None)
        if key in record['identity']:
            identity[key] = record['identity'][key]
    identity['behavior_cloning'] = record['identity'].get('behavior_cloning', False)
    if 'bc' in record['identity']:
        identity['bc'] = record['identity']['bc']
    return record


def initial_state(args, identity, compatibility, checkpoint_record):
    """Return the first state.json record of the run."""
    state = {'mode': args.mode, 'status': 'initializing', 'identity': identity, 'errors': [], 'stage2_complete': False,
        'runtime_binding': {'runtime_tree_sha256': args.source_freeze_sha256},
        'allocation': {'profile': args.allocation_profile, 'max_wall_seconds': args.max_wall_seconds},
        'experiment_logger': {'backend': args.logger, 'wandb_project': args.wandb_project,
                              'wandb_mode': args.wandb_mode if args.logger == 'wandb' else None}}
    if compatibility is not None:
        state['checkpoint_compatibility'] = {'receipt_sha256': args.checkpoint_compatibility_sha256,
            'original_identity': checkpoint_record['identity'], 'checkpoint_sha256': args.checkpoint_sha256}
    return state


def launch_app(args):
    """Start Isaac Sim; a stalled start writes the thread stacks to the output every 45 s."""
    from isaaclab.app import AppLauncher
    args.enable_cameras = args.mode == 'evaluate'
    args.headless_explicit = False
    with (args.output/'startup_tracebacks.log').open('w') as stream:
        faulthandler.enable(file=stream)
        faulthandler.dump_traceback_later(45., repeat=True, file=stream)
        try:
            return AppLauncher(args).app
        finally:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.disable()


def create_env(args, cfg, metadata):
    """Seed torch and NumPy, then build the native environment; evaluation adds the policy camera scene."""
    import torch
    from locomotion.omniwalk.adapter import OmniEnv
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = OmniEnv(cfg, args.asset, args.model, args.geometry,
        args.output/'native', reference_metadata=metadata)
    if args.mode == 'evaluate':
        prepare_policy_scene(env)
    return env


def run_diagnostic(env, args, state, started):
    """Hold a zero command for 1000 controls, capture the standing robot and score it."""
    import torch
    from locomotion.evaluation.capture import DiagnosticCapture, score_diagnostic
    env.commands.zero_()
    capture = DiagnosticCapture(env, args.output/'standing', args.geometry_extrema)
    try:
        for control in range(1000):
            env.step(torch.zeros((env.num_envs, 18), device=env.device))
            if (control+1) % 100 == 0:
                print(f"STANDING controls={control+1} replicas={env.num_envs}", flush=True)
            if time.monotonic()-started >= args.max_wall_seconds:
                raise TimeoutError('Standing allocation exceeded its deadline')
    finally:
        capture.close()
    report = score_diagnostic(args.output/'standing')
    save(args.output/'standing'/'standing_report.json', report)
    state['standing_gate_pass'] = report['all_pass']


def run_probe(env, args, state, started):
    """Run the noise probe inside the remaining allocation and record its summary without the cells."""
    from locomotion.evaluation import noise_probe
    summary = noise_probe.run(env, args.output/'probe', seed=args.seed,
        max_wall_seconds=max(.001, args.max_wall_seconds-(time.monotonic()-started)),
        progress=print_progress)
    state['probe'] = {key: value for key, value in summary.items() if key != 'cells'}


def learner_config(args, identity, state):
    """Check RSL-RL, write ppo_config.json and record the upstream sources the learner runs."""
    import rsl_rl
    version = importlib.metadata.version('rsl-rl-lib')
    if version != '5.0.1':
        raise ValueError('RSL-RL version differs: '+version)
    factory, options = ppo_config, {}
    if args.learner == 'amp':
        from locomotion.priors.amp import amp_ppo_config
        factory, options = amp_ppo_config, {'networks': args.networks}
    config = factory(args.seed, **options, action_mean=args.action_mean,
                     observation_normalization=args.observation_normalization, **learner_options(args))
    save(args.output/'ppo_config.json', config)
    upstream = Path(rsl_rl.__file__).parent
    identity['upstream_source_files'] = {name: sha(upstream/name) for name in (
        'runners/on_policy_runner.py', 'algorithms/ppo.py', 'models/mlp_model.py', 'storage/rollout_storage.py')}
    identity['ppo_config'] = config
    state['status'] = 'running'
    save(args.output/'state.json', state)
    return config


def training_wrapper(env, args, task):
    """Wrap the task for RSL-RL; AMP adds collision capture and the discriminator configuration."""
    from locomotion.rl.wrapper import OmniVecEnv
    wrapper, options = OmniVecEnv, {}
    if args.learner == 'amp':
        from locomotion.priors.amp import AMPConfig, AMPVecEnv, CollisionCapture
        wrapper = AMPVecEnv
        options = {'collision': CollisionCapture(env), 'amp_config': AMPConfig().resolve(args.seed)}
    return wrapper(task, **options, observation_scaling=args.observation_scaling,
        command_segments=args.command_segments, gait_clock=args.gait_clock,
        action_smoothing=args.action_smoothing, velocity_noise=args.velocity_noise)


def write_checkpoint(runner, args, config, identity, state, num_envs, update):
    """Save the runner with its identity, write the JSON declaration and name the checkpoint in state.json."""
    path = args.output/f'checkpoint_update{update:06d}.pt'
    runner.save(str(path), infos={'identity': identity, 'updates': update})
    save(path.with_suffix('.json'), {'identity': identity, 'updates': update,
        'checkpoint_sha256': sha(path), 'transitions': collected_transitions(update, config, num_envs)})
    state['checkpoint'] = str(path)
    state['checkpoint_sha256'] = sha(path)
    save(args.output/'state.json', state)


def metrics_row(values, runner, env, task, diagnostics, config):
    """Return the metrics.jsonl row for the values RSL-RL logs after one update."""
    from locomotion.rl.ppo import action_metrics
    update = values['it']+1
    return {'update': update, 'transitions': collected_transitions(update, config, env.num_envs),
        'collection_seconds': values['collect_time'], 'learning_seconds': values['learn_time'],
        'loss': values['loss_dict'], 'learning_rate': values['learning_rate'],
        'mean_action_std': float(values['action_std'].mean()),
        'actions': action_metrics(runner.alg.storage, env.joint_names),
        'task': task.status(reset_interval=True), 'policy_update': diagnostics.latest}


def name_wandb_run(args, identity):
    """Name the W&B run after the seed and the frozen source; RSL-RL names it after its log directory."""
    import wandb
    wandb.run.name = f'seed{args.seed}-{args.source_freeze_sha256[:12]}'
    wandb.config.update({'identity': identity}, allow_val_change=True)


def training_runner(env, args, config, identity, state, started):
    """Build the RSL-RL runner with update diagnostics, load capture and a record after each update.

    Return the runner and the load accumulator.
    """
    from rsl_rl.runners import OnPolicyRunner
    from locomotion.evaluation.loads import TrainingLoads
    from locomotion.rl.ppo import DeviationSchedule, UpdateDiagnostics
    from locomotion.omniwalk.task import TrainingTask
    task = TrainingTask(env, TaskConfig(seed=args.seed), args.output/'task')
    wrapped = training_wrapper(env, args, task)
    if args.networks == 'mlp' and args.gait_clock != task.reward_config.schedule_period_controls:
        raise ValueError('The contact-schedule reward needs a gait clock of the same period')
    runner_config = copy.deepcopy(config)
    # The wrapper consumes its own options; the runner receives the stock keys.
    runner_config.pop('environment_wrapper', None)
    runner_config.pop('exploration', None)
    if args.logger == 'wandb':
        # The logger choice stays out of ppo_config, which checkpoint loading compares.
        # Copy saved checkpoints: container paths do not exist where offline runs sync.
        os.environ.update(WANDB_DIR=str(args.output), WANDB_MODE=args.wandb_mode, WANDB_SYMLINK='false')
        runner_config.update(logger='wandb', wandb_project=args.wandb_project)
        wrapped.cfg = ConfigRecord(wrapped.cfg)
    runner = OnPolicyRunner(wrapped, runner_config, str(args.output/'learner'), device=args.device)
    diagnostics = UpdateDiagnostics(runner.alg)
    runner.alg.update = diagnostics.update
    if args.action_std_final is not None:
        # The schedule wraps the diagnostic update, so each record reads the deviation it trained with.
        runner.alg.update = DeviationSchedule(runner.alg, args.action_std, args.action_std_final,
            decay_updates(args.updates, args.action_std_decay_updates)).update
    loads = TrainingLoads(env)
    if args.learner == 'amp':
        wrapped.collision.inner = loads
        env.capture = wrapped.collision
    else:
        env.capture = loads
    checkpoint = partial(write_checkpoint, runner, args, config, identity, state, env.num_envs)
    log_update = runner.logger.log

    def log(**values):
        log_update(**values)
        row = metrics_row(values, runner, env, task, diagnostics, config)
        with (args.output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        if runner.logger.writer is not None:
            for tag, value in scalars(row['task'], 'Task').items():
                runner.logger.writer.add_scalar(tag, value, values['it'])
        if args.logger == 'wandb' and row['update'] == 1:
            name_wandb_run(args, identity)
        state.update(updates=row['update'], transitions=row['transitions'], wall_seconds=time.monotonic()-started)
        save(args.output/'state.json', state)
        finish_training_update(row['update'], args.updates, time.monotonic()-started,
            args.max_wall_seconds, checkpoint,
            lambda: save(args.output/'force_metrics.json', loads.report()))
    runner.logger.log = log
    return runner, loads


def actor_input(env, args, observation):
    """Scale the actor observation and append the gait clock, as the training wrapper does."""
    import torch
    from locomotion.rl.wrapper import clock_features, scale_observation
    scaled = scale_observation(observation, args.observation_scaling)
    if args.gait_clock:
        clock = clock_features(env.episode_steps, args.gait_clock, observation[:, 210:213]).to(scaled)
        scaled = torch.cat((scaled, clock), -1)
    return scaled


def evaluation_policy(env, args, config, identity, checkpoint_record, compatibility):
    """Check the checkpoint configuration, load its actor and return the policy that the evaluator calls."""
    import torch
    from tensordict import TensorDict
    from locomotion.rl.ppo import load_actor
    from locomotion.rl.wrapper import smoothed_action
    check = require_compatibility_configuration if compatibility is not None else require_learner_configuration
    check(checkpoint_record, identity, config)
    observation = env.observations(env.current)['obs']
    scaled = actor_input(env, args, observation)
    if args.networks == 'paper':
        from locomotion.priors.networks import actor_observation
        observations = actor_observation(observation)
    else:
        observations = TensorDict({'policy': scaled}, batch_size=[env.num_envs])
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    require_embedded_declaration(payload.get('infos'), checkpoint_record)
    actor, _ = load_actor(payload, config, observations, args.device)
    previous = None

    def policy(observation):
        nonlocal previous
        with torch.inference_mode():
            if args.networks == 'paper':
                return actor(actor_observation(observation))
            action = actor(TensorDict({'policy': actor_input(env, args, observation)}, batch_size=[env.num_envs]))
            if args.action_smoothing == 'mean2':
                # The same two-control mean as the training wrapper, restarted with each episode.
                last = torch.zeros_like(action) if previous is None else previous
                previous = action.clone()
                action = smoothed_action(action, last, env.episode_steps)
            return action
    return policy


def run_evaluation(env, args, policy, state, started):
    """Score the policy on the selected cases, then require a complete set of load summaries."""
    from locomotion.evaluation.runner import evaluate_suite, run_learning_probe_suite
    env.render = NativePolicyCamera(env)
    remaining = max(.001, args.max_wall_seconds-(time.monotonic()-started))
    options = dict(progress=print_progress, max_wall_seconds=remaining)
    if args.eval_scope == 'full':
        result = evaluate_suite(env, policy, args.output/'evaluation',
            args.geometry_extrema, args.checkpoint, **options)
    else:
        selected = (['learning:translate_0.05_0deg', 'learning:quiet_20s',
            'learning:forward_0.05_to_stop'] if args.eval_scope == 'focus' else None)
        result = run_learning_probe_suite(env, policy, args.output/'evaluation',
            args.geometry_extrema, args.checkpoint, selected_case_ids=selected,
            record_video=True, video_case_id=args.video_case, seed=args.seed, **options)
    state['evaluation'] = result
    summaries = list((args.output/'evaluation').rglob('force_metrics.json'))
    if not summaries or any(json.loads(p.read_text())['status'] != 'available' for p in summaries):
        raise ValueError('Evaluation has no complete set of load summaries')


def main(argv=None):
    args = parse_args(argv)
    compatibility = validate(args)
    metadata = json.loads(args.stance.read_text())
    cfg = environment_config(args)
    identity = build_identity(args, cfg, metadata)
    checkpoint_record = bind_checkpoint(args, identity, compatibility) if args.mode == 'evaluate' else None
    if args.preflight_only:
        print(json.dumps(identity, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    state = initial_state(args, identity, compatibility, checkpoint_record)
    save(args.output/'state.json', state)
    app = loads = None
    started = time.monotonic()
    try:
        app = launch_app(args)
        print('REFERENCE_SCREEN_APP_READY', flush=True)
        env = create_env(args, cfg, metadata)
        if args.mode == 'diagnostic':
            run_diagnostic(env, args, state, started)
        elif args.mode == 'probe':
            run_probe(env, args, state, started)
        else:
            config = learner_config(args, identity, state)
            if args.mode == 'train':
                runner, loads = training_runner(env, args, config, identity, state, started)
                runner.learn(args.updates, init_at_random_ep_len=False)
                save(args.output/'force_metrics.json', loads.report())
            else:
                policy = evaluation_policy(env, args, config, identity, checkpoint_record, compatibility)
                run_evaluation(env, args, policy, state, started)
        env.verify_native_recipe('after_controlled_steps')
        state['status'] = 'completed'
    except BaseException as error:
        state['status'] = 'failed'
        state['errors'].append(repr(error))
        state['traceback'] = traceback.format_exc()
        print(state['traceback'], flush=True)
    finally:
        if loads is not None:
            save(args.output/'force_metrics.json', loads.report())
        state['wall_seconds'] = time.monotonic()-started
        save(args.output/'state.json', state)
        if app is not None:
            app.close()
    return 0 if state['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
