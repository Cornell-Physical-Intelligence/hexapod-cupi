"""Train or evaluate standard PPO on the admitted canonical simulator."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


from locomotion.spark.validation import ALLOCATION_PROFILES, CLEANUP_MARGIN_SECONDS, UPDATE_LIMIT, EXTENDED_UPDATE_LIMIT, update_limit, decay_updates, validate_decay, validate_deadline, finish_training_update, collected_transitions, LEARNER_OPTION_DEFAULTS


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


from locomotion.spark.sources import source_files, physics_files, verify_runtime
import locomotion.evaluation.checkpoint as compat
from locomotion.evaluation.checkpoint import checkpoint_reward_version, LEARNER_DEFAULTS, LEARNER_KEYS, json_shape, require_learner_identity, require_learner_configuration, require_embedded_declaration, require_checkpoint_identity


class ConfigRecord(dict):
    """Give RSL-RL's W&B writer the to_dict() it expects from an environment config."""
    def to_dict(self):
        return dict(self)


def main(argv=None):
    source = Path(__file__).resolve().parents[1]
    prefix = 'locomotion'
    import locomotion.evaluation.admission as admission
    from locomotion.evaluation.camera import prepare_policy_scene
    configuration = importlib.import_module(prefix+'.omni.config')
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
    parser.add_argument('--action-mean', choices=['unbounded', 'tanh'], default='unbounded',
                        help='Use the historical Gaussian mean or bound the mean with tanh.')
    parser.add_argument('--observation-normalization', choices=['empirical', 'none'], default='empirical',
                        help='Use running observation statistics or retain the raw PPO observations.')
    parser.add_argument('--observation-scaling', choices=['none', 'fixed'], default='none',
                        help='Pass raw observations or apply the fixed input scales declared in ppo.py.')
    parser.add_argument('--command-segments', choices=['continuous', 'bootstrap'], default='continuous',
                        help='Carry returns across command changes or bootstrap the return at each change.')
    parser.add_argument('--learning-rate-max', type=float,
                        help='Ceiling for the adaptive learning rate; the stock schedule allows 1e-2.')
    parser.add_argument('--action-std', type=float, default=.15,
                        help='Initial standard deviation of the Gaussian action distribution.')
    parser.add_argument('--episode-seconds', type=float, default=20.,
                        help='Training episode length; 20 s by default. Shorter episodes bound how far a robot walks.')
    parser.add_argument('--action-smoothing', choices=['none', 'mean2'], default='none',
                        help='mean2 sends the mean of each action and the previous one to the environment.')
    parser.add_argument('--velocity-noise', type=float, default=0.,
                        help='Standard deviation in rad/s of Gaussian noise on the actor joint-velocity inputs in training.')
    parser.add_argument('--gait-clock', type=int,
                        help='Append gait phase with this period in controls; training defaults to 60, other modes to 0.')
    parser.add_argument('--action-std-final', type=float,
                        help='Lower the action deviation linearly to this value over the run; PPO then does not learn it.')
    parser.add_argument('--action-std-decay-updates', type=int,
                        help='Reach the final action deviation after this many updates and hold it; the run length by default.')
    parser.add_argument('--action-noise-correlation', type=float, default=0.,
                        help='Share of each exploration noise sample carried to the next control (tanh mean only).')
    parser.add_argument('--steps-per-env', type=int, default=24,
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
    if (args.networks == 'paper' and args.learner != 'amp') or (args.learner == 'amp' and args.mode not in ('train', 'evaluate')):
        raise ValueError('The paper networks need the AMP learner for training or evaluation')
    if args.gait_clock is None:
        args.gait_clock = 60 if args.mode == 'train' and args.networks == 'mlp' else 0
    if args.mode == 'train' and args.networks == 'mlp' and args.gait_clock != 60:
        raise ValueError('The active reward requires the 60-control gait clock')
    compatibility = None
    if args.checkpoint_compatibility is not None or args.checkpoint_compatibility_sha256 is not None:
        if args.mode != 'evaluate' or args.eval_scope != 'probes' or args.num_envs != 1 or args.checkpoint_compatibility is None or args.checkpoint_compatibility_sha256 is None:
            raise ValueError('Compatibility requires evaluation and both receipt arguments')
        compatibility = compat.read_receipt(args.checkpoint_compatibility, args.checkpoint_compatibility_sha256)
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
    if args.observation_normalization != 'empirical' and (args.mode not in ('train', 'evaluate')):
        raise ValueError('Observation normalization selection requires PPO training or evaluation')
    learner_options = dict(observation_scaling=args.observation_scaling, command_segments=args.command_segments,
                           learning_rate_max=args.learning_rate_max, action_std=args.action_std,
                           action_noise_correlation=args.action_noise_correlation,
                           action_std_final=args.action_std_final, gait_clock=args.gait_clock,
                           action_smoothing=args.action_smoothing, velocity_noise=args.velocity_noise,
                           action_std_decay_updates=args.action_std_decay_updates, steps_per_env=args.steps_per_env)
    if (learner_options != LEARNER_OPTION_DEFAULTS
            and (args.mode not in ('train', 'evaluate'))):
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
    configuration.verify_assets(args.asset, args.model)
    validate_deadline(args.mode, args.allocation_profile, args.max_wall_seconds)
    if (not args.headless or args.device != 'cuda:0' or not 1 <= args.updates <= limit
            or args.seed < 0
            or (args.mode in ('train', 'probe') and (args.num_envs not in (128, 512) or args.checkpoint is not None))
            or (args.mode == 'evaluate' and (args.num_envs != 1 or args.checkpoint is None))):
        raise ValueError('A bounded scratch training, a 128-replica probe or single-replica evaluation is required')
    if sha(source.parent/'FREEZE_SHA256.json') != args.source_freeze_sha256:
        raise ValueError('Frozen source identity differs')
    metadata = json.loads(args.stance.read_text())
    from locomotion.evaluation.runner import EvaluationEnvConfig
    config_class = EvaluationEnvConfig if args.mode == 'evaluate' and args.eval_scope == 'full' else configuration.EnvConfig
    cfg = config_class(num_envs=args.num_envs, seed=args.seed,
        record_motion_features=args.mode == 'evaluate' or args.learner == 'amp', render=args.mode == 'evaluate', episode_seconds=(90. if args.eval_scope == 'full' else 60.) if args.mode == 'evaluate' else (60. if args.mode == 'diagnostic' else args.episode_seconds), device=args.device)
    identity = {'schema': 'hexapod_locomotion_ppo_v2', 'source_files': source_files(source),
        'runtime_sources': verify_runtime(source.parent),
        'model_sha256': configuration.MODEL_SHA256, 'usd_sha256': configuration.USD_SHA256,
        'stance_sha256': sha(args.stance),
        'geometry_sha256': sha(args.geometry), 'geometry_extrema_sha256': sha(args.geometry_extrema),
        'config': cfg.declaration(), 'motion_prior': args.learner == 'amp', 'behavior_cloning': False,
        'seed': args.seed, 'rsl_rl_required_version': '5.0.1',
        'adapter_sha256': sha(source/'omni/ppo.py'), 'entry_sha256': sha(__file__),
        'reward_version': '4', 'reward_task': 'v4_yaw_gate', 'learner': args.learner, 'networks': args.networks}
    if args.extended_updates:
        # The exploration decay length enters ppo_config; the opt-in records the requested run length beside it.
        identity['training_updates'] = args.updates
    amp_module = None
    if args.learner == 'amp':
        amp_module = importlib.import_module(prefix+'.priors.amp')
        identity['learner_source_files'] = {name: sha(source/name) for name in (
            'omni/features.py', 'priors/discriminator.py', 'priors/amp.py', 'priors/networks.py')}
        identity['amp_dataset'] = amp_module.load_demonstrations(source.parent/amp_module.AMPConfig().dataset)[1]
    identity['physics_source_files'] = physics_files(identity['source_files'])
    identity['physics_config'] = {'physics_dt': cfg.physics_dt, 'decimation': cfg.decimation,
        'spacing_m': cfg.spacing_m, 'target_slew_rad': cfg.target_slew_rad, 'action_scale_rad': cfg.action_scale_rad,
        'solver_position_iterations': 32, 'solver_velocity_iterations': 0,
        'floor': 'isaaclab_ground_plane', 'filter_collisions': True, 'material_friction': [1., 1.],
        'restitution': 0., 'external_forces_every_iteration': True,
        'neutral_joint_position_rad': metadata['nominal_joint_position_rad'],
        'reset_root_height_m': metadata['reset_root_height_m']}
    if args.mode != 'diagnostic':
        identity['standing_admission'] = admission.require_admission(args.standing_admission, identity, cfg)
    checkpoint_record = None
    if args.mode == 'evaluate':
        if sha(args.checkpoint) != args.checkpoint_sha256:
            raise ValueError('Checkpoint bytes differ')
        checkpoint_record = json.loads(args.checkpoint.with_suffix('.json').read_text())
        if checkpoint_record['checkpoint_sha256'] != args.checkpoint_sha256:
            raise ValueError('Checkpoint declaration differs')
        if compatibility is None:
            require_checkpoint_identity(checkpoint_record, identity)
        else:
            if sha(args.checkpoint.with_suffix('.json')) != compat.DECLARATION_SHA:
                raise ValueError('Compatibility checkpoint declaration differs')
            compat.require_identity(compatibility, checkpoint_record, identity)
        # Evaluation computes no training reward; record the one the checkpoint learned from.
        identity['reward_version'] = checkpoint_reward_version(checkpoint_record)
        for key in ('reward_task', 'reward_options'):
            identity.pop(key, None)
            if key in checkpoint_record['identity']:
                identity[key] = checkpoint_record['identity'][key]
        identity['behavior_cloning'] = checkpoint_record['identity'].get('behavior_cloning', False)
        if 'bc' in checkpoint_record['identity']:
            identity['bc'] = checkpoint_record['identity']['bc']
    if args.preflight_only:
        print(json.dumps(identity, indent=2)); return 0
    args.output.mkdir(parents=True, exist_ok=False)
    state = {'mode': args.mode, 'status': 'initializing', 'identity': identity, 'errors': [], 'stage2_complete': False,
        'runtime_binding': {'runtime_tree_sha256': args.source_freeze_sha256},
        'allocation': {'profile': args.allocation_profile, 'max_wall_seconds': args.max_wall_seconds},
        'experiment_logger': {'backend': args.logger, 'wandb_project': args.wandb_project,
                              'wandb_mode': args.wandb_mode if args.logger == 'wandb' else None}}
    if compatibility is not None:
        state['checkpoint_compatibility'] = {'receipt_sha256': args.checkpoint_compatibility_sha256,
            'original_identity': checkpoint_record['identity'], 'checkpoint_sha256': args.checkpoint_sha256}
    save(args.output/'state.json', state)
    app = env = runner = loads = None
    started = time.monotonic()
    try:
        import faulthandler
        args.enable_cameras = args.mode == 'evaluate'
        args.headless_explicit = False
        with (args.output/'startup_tracebacks.log').open('w') as stream:
            faulthandler.enable(file=stream)
            faulthandler.dump_traceback_later(45., repeat=True, file=stream)
            try:
                app = AppLauncher(args).app
            finally:
                faulthandler.cancel_dump_traceback_later(); faulthandler.disable()
        print('REFERENCE_SCREEN_APP_READY', flush=True)
        import torch
        import numpy as np
        vanilla = importlib.import_module(prefix+'.omni.ppo')
        native = importlib.import_module(prefix+'.omni.env')
        from locomotion.evaluation.capture import DiagnosticCapture, score_diagnostic
        from locomotion.evaluation.loads import TrainingLoads
        if args.mode in ('train', 'probe'):
            task_module = importlib.import_module(prefix+'.omni.task')
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        env = native.LocomotionEnv(cfg, args.asset, args.model, args.geometry,
            args.output/'native', reference_metadata=metadata)
        if args.mode == 'evaluate':
            prepare_policy_scene(env)
        if args.mode == 'diagnostic':
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
            env.verify_native_recipe('after_controlled_steps')
            state['status'] = 'completed'
            return 0
        if args.mode == 'probe':
            probe = importlib.import_module(prefix+'.evaluation.noise_probe')
            summary = probe.run(env, args.output/'probe', seed=args.seed,
                max_wall_seconds=max(.001, args.max_wall_seconds-(time.monotonic()-started)),
                progress=lambda value: print(json.dumps(value), flush=True))
            state['probe'] = {key: value for key, value in summary.items() if key != 'cells'}
            env.verify_native_recipe('after_controlled_steps')
            state['status'] = 'completed'
            return 0
        from tensordict import TensorDict
        version = importlib.metadata.version('rsl-rl-lib')
        if version != '5.0.1':
            raise ValueError('RSL-RL version differs: '+version)
        config_factory = amp_module.amp_ppo_config if amp_module is not None else vanilla.ppo_config
        config_options = {'networks': args.networks} if amp_module is not None else {}
        config = config_factory(args.seed, **config_options, action_mean=args.action_mean,
                                    observation_normalization=args.observation_normalization, **learner_options)
        save(args.output/'ppo_config.json', config)
        import rsl_rl
        upstream = Path(rsl_rl.__file__).parent
        identity['upstream_source_files'] = {name: sha(upstream/name) for name in (
            'runners/on_policy_runner.py', 'algorithms/ppo.py', 'models/mlp_model.py', 'storage/rollout_storage.py')}
        identity['ppo_config'] = config
        state['status'] = 'running'; save(args.output/'state.json', state)
        if args.mode == 'train':
            from rsl_rl.runners import OnPolicyRunner
            task = task_module.TrainingTask(env, task_module.TaskConfig(seed=args.seed), args.output/'task')
            wrapper = vanilla.VanillaVecEnv
            wrapper_options = {}
            if amp_module is not None:
                wrapper = amp_module.AMPVecEnv
                wrapper_options = {'collision': amp_module.CollisionCapture(env),
                                   'amp_config': amp_module.AMPConfig().resolve(args.seed)}
            wrapped = wrapper(task, **wrapper_options, observation_scaling=args.observation_scaling,
                command_segments=args.command_segments, gait_clock=args.gait_clock,
                action_smoothing=args.action_smoothing, velocity_noise=args.velocity_noise)
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
            diagnostics = vanilla.UpdateDiagnostics(runner.alg)
            runner.alg.update = diagnostics.update
            if args.action_std_final is not None:
                # The schedule wraps the diagnostic update, so each record reads the deviation it trained with.
                runner.alg.update = vanilla.DeviationSchedule(runner.alg, args.action_std, args.action_std_final,
                    decay_updates(args.updates, args.action_std_decay_updates)).update
            def checkpoint(update):
                path = args.output/f'checkpoint_update{update:06d}.pt'
                runner.save(str(path), infos={'identity': identity, 'updates': update})
                save(path.with_suffix('.json'), {'identity': identity, 'updates': update,
                    'checkpoint_sha256': sha(path), 'transitions': collected_transitions(update, config, env.num_envs)})
                state['checkpoint'] = str(path); state['checkpoint_sha256'] = sha(path)
                save(args.output/'state.json', state)
            loads = TrainingLoads(env)
            if amp_module is not None:
                wrapped.collision.inner = loads
                env.capture = wrapped.collision
            else:
                env.capture = loads
            original_log = runner.logger.log
            def log(**values):
                original_log(**values)
                update = values['it']+1
                row = {'update': update, 'transitions': collected_transitions(update, config, env.num_envs),
                    'collection_seconds': values['collect_time'], 'learning_seconds': values['learn_time'],
                    'loss': values['loss_dict'], 'learning_rate': values['learning_rate'],
                    'mean_action_std': float(values['action_std'].mean()),
                    'actions': vanilla.action_metrics(runner.alg.storage, env.joint_names),
                    'task': task.status(reset_interval=True)}
                if diagnostics is not None:
                    row['policy_update'] = diagnostics.latest
                with (args.output/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                if runner.logger.writer is not None:
                    for tag, value in scalars(row['task'], 'Task').items():
                        runner.logger.writer.add_scalar(tag, value, values['it'])
                if args.logger == 'wandb' and update == 1:
                    # RSL-RL names the run after its log directory; bind it to the frozen source instead.
                    import wandb
                    wandb.run.name = f'seed{args.seed}-{args.source_freeze_sha256[:12]}'
                    wandb.config.update({'identity': identity}, allow_val_change=True)
                state.update(updates=update, transitions=row['transitions'], wall_seconds=time.monotonic()-started)
                save(args.output/'state.json', state)
                finish_training_update(update, args.updates, time.monotonic()-started,
                    args.max_wall_seconds, checkpoint,
                    lambda: save(args.output/'force_metrics.json', loads.report()))
            runner.logger.log = log
            runner.learn(args.updates, init_at_random_ep_len=False)
            save(args.output/'force_metrics.json', loads.report())
        else:
            (compat.require_configuration if compatibility is not None else require_learner_configuration)(checkpoint_record, identity, config)
            observation = env._observations(env.current)['obs']
            observations = vanilla.scale_observation(observation, args.observation_scaling)
            if args.gait_clock:
                clock = vanilla.clock_features(env.episode_steps, args.gait_clock, observation[:, 210:213]).to(observations)
                observations = torch.cat((observations, clock), -1)
            if args.networks == 'paper':
                networks = importlib.import_module(prefix+'.priors.networks')
                observations = networks.actor_observation(observation)
            else:
                observations = TensorDict({'policy': observations}, batch_size=[env.num_envs])
            payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
            require_embedded_declaration(payload.get('infos'), checkpoint_record)
            actor, _ = vanilla.load_actor(payload, config,
                observations, args.device)
            previous_action = [None]
            def policy(observation):
                with torch.inference_mode():
                    if args.networks == 'paper':
                        return actor(networks.actor_observation(observation))
                    scaled = vanilla.scale_observation(observation, args.observation_scaling)
                    if args.gait_clock:
                        clock = vanilla.clock_features(env.episode_steps, args.gait_clock, observation[:, 210:213]).to(scaled)
                        scaled = torch.cat((scaled, clock), -1)
                    action = actor(TensorDict({'policy': scaled}, batch_size=[env.num_envs]))
                    if args.action_smoothing == 'mean2':
                        # The same two-control mean as the training wrapper, restarted with each episode.
                        last = torch.zeros_like(action) if previous_action[0] is None else previous_action[0]
                        previous_action[0] = action.clone()
                        action = vanilla.smoothed_action(action, last, env.episode_steps)
                    return action
            evaluation = importlib.import_module(prefix+'.evaluation.runner')
            camera = importlib.import_module(prefix+'.evaluation.camera')
            env.render = camera.NativePolicyCamera(env)
            remaining = max(.001, args.max_wall_seconds-(time.monotonic()-started))
            options = dict(progress=lambda value: print(json.dumps(value), flush=True), max_wall_seconds=remaining)
            if args.eval_scope == 'full':
                result = evaluation.evaluate_suite(env, policy, args.output/'evaluation',
                    args.geometry_extrema, args.checkpoint, **options)
            else:
                selected = (['learning:translate_0.05_0deg', 'learning:quiet_20s',
                    'learning:forward_0.05_to_stop'] if args.eval_scope == 'focus' else None)
                result = evaluation.run_learning_probe_suite(env, policy, args.output/'evaluation',
                    args.geometry_extrema, args.checkpoint, selected_case_ids=selected,
                    record_video=True, video_case_id=args.video_case, seed=args.seed, **options)
            state['evaluation'] = result
            summaries = list((args.output/'evaluation').rglob('force_metrics.json'))
            if not summaries or any(json.loads(p.read_text())['status'] != 'available' for p in summaries):
                raise ValueError('Evaluation has no complete set of load summaries')
        env.verify_native_recipe('after_controlled_steps')
        state['status'] = 'completed'
    except BaseException as error:
        state['status'] = 'failed'; state['errors'].append(repr(error))
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
