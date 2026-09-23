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


class ConfigRecord(dict):
    """Give RSL-RL's W&B writer the to_dict() it expects from an environment config."""
    def to_dict(self):
        return dict(self)


def main(argv=None):
    source = Path(__file__).resolve().parent
    prefix = __package__
    from . import admission
    from .camera import prepare_policy_scene
    configuration = importlib.import_module(prefix+'.env_config')
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--mode', choices=['diagnostic', 'train', 'evaluate'], required=True)
    for name in ('asset', 'model', 'geometry', 'geometry-extrema', 'stance', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--standing-admission', type=Path)
    parser.add_argument('--source-freeze-sha256', required=True)
    parser.add_argument('--num-envs', type=int, choices=[1, 32, 128], required=True)
    parser.add_argument('--seed', type=int, default=20260917)
    parser.add_argument('--updates', type=int, default=512)
    parser.add_argument('--max-wall-seconds', type=float, default=6200.)
    parser.add_argument('--eval-scope', choices=['focus', 'probes', 'full'], default='focus')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--logger', choices=['tensorboard', 'wandb'], default='tensorboard')
    parser.add_argument('--wandb-project')
    parser.add_argument('--wandb-mode', choices=['offline', 'online'], default='offline')
    if any(flag in (argv if argv is not None else sys.argv[1:]) for flag in ('--preflight-only', '--help', '-h')):
        parser.add_argument('--headless', action='store_true')
        parser.add_argument('--device', default='cuda:0')
    else:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.logger == 'wandb':
        if (args.mode != 'train' or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', args.wandb_project or '')
                or importlib.util.find_spec('wandb') is None
                or (args.wandb_mode == 'online' and not os.environ.get('WANDB_API_KEY'))):
            raise ValueError('W&B logging needs train mode, a project name, the wandb package '
                             'and WANDB_API_KEY when online')
    elif args.wandb_project is not None or args.wandb_mode != 'offline':
        raise ValueError('W&B options require --logger wandb')
    configuration.verify_assets(args.asset, args.model)
    if (not args.headless or args.device != 'cuda:0' or not 1 <= args.updates <= 2000
            or not 0 < args.max_wall_seconds <= 6600 or args.seed < 0
            or (args.mode == 'train' and (args.num_envs != 128 or args.checkpoint is not None))
            or (args.mode == 'evaluate' and (args.num_envs != 1 or args.checkpoint is None))):
        raise ValueError('A bounded scratch training or single-replica evaluation is required')
    if sha(source.parent/'FREEZE_SHA256.json') != args.source_freeze_sha256:
        raise ValueError('Frozen source identity differs')
    metadata = json.loads(args.stance.read_text())
    from .evaluation_config import EvaluationEnvConfig
    config_class = EvaluationEnvConfig if args.mode == 'evaluate' and args.eval_scope == 'full' else configuration.EnvConfig
    cfg = config_class(num_envs=args.num_envs, seed=args.seed,
        record_motion_features=args.mode == 'evaluate', render=args.mode == 'evaluate', episode_seconds=(90. if args.eval_scope == 'full' else 60.) if args.mode == 'evaluate' else (60. if args.mode == 'diagnostic' else 20.), device=args.device)
    identity = {'schema': 'hexapod_locomotion_ppo_v1', 'source_files': {p.name: sha(p) for p in sorted(source.glob('*.py'))},
        'model_sha256': configuration.MODEL_SHA256, 'usd_sha256': configuration.USD_SHA256,
        'stance_sha256': sha(args.stance),
        'geometry_sha256': sha(args.geometry), 'geometry_extrema_sha256': sha(args.geometry_extrema),
        'config': cfg.declaration(), 'motion_prior': False, 'behavior_cloning': False,
        'seed': args.seed, 'rsl_rl_required_version': '5.0.1',
        'adapter_sha256': sha(source/'ppo.py'), 'entry_sha256': sha(__file__)}
    identity['physics_source_files'] = {k: identity['source_files'][k] for k in ('env.py', 'env_config.py')}
    identity['physics_config'] = {'physics_dt': cfg.physics_dt, 'decimation': cfg.decimation,
        'spacing_m': cfg.spacing_m, 'target_slew_rad': cfg.target_slew_rad, 'action_scale_rad': cfg.action_scale_rad,
        'solver_position_iterations': 32, 'solver_velocity_iterations': 0,
        'floor': '80m_two_triangle_mesh_y_equals_x_seam', 'material_friction': [1., 1.],
        'restitution': 0., 'external_forces_every_iteration': True,
        'neutral_joint_position_rad': metadata['nominal_joint_position_rad'],
        'reset_root_height_m': metadata['reset_root_height_m']}
    if args.mode != 'diagnostic':
        identity['standing_admission'] = admission.require_admission(args.standing_admission, identity, cfg)
    from contracts.release import COMPATIBILITY_KEYS
    compatibility_keys = COMPATIBILITY_KEYS
    checkpoint_record = None
    if args.mode == 'evaluate':
        if sha(args.checkpoint) != args.checkpoint_sha256:
            raise ValueError('Checkpoint bytes differ')
        checkpoint_record = json.loads(args.checkpoint.with_suffix('.json').read_text())
        if checkpoint_record['checkpoint_sha256'] != args.checkpoint_sha256:
            raise ValueError('Checkpoint declaration differs')
        if any(checkpoint_record['identity'][key] != identity[key] for key in compatibility_keys):
            raise ValueError('Checkpoint model, physics, seed or implementation differs')
    if args.preflight_only:
        print(json.dumps(identity, indent=2)); return 0
    args.output.mkdir(parents=True, exist_ok=False)
    state = {'mode': args.mode, 'status': 'initializing', 'identity': identity, 'errors': [], 'stage2_complete': False,
        'runtime_binding': {'runtime_tree_sha256': args.source_freeze_sha256},
        'experiment_logger': {'backend': args.logger, 'wandb_project': args.wandb_project,
                              'wandb_mode': args.wandb_mode if args.logger == 'wandb' else None}}
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
        vanilla = importlib.import_module(prefix+'.ppo')
        native = importlib.import_module(prefix+'.env')
        task_module = importlib.import_module(prefix+'.task')
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        env = native.LocomotionEnv(cfg, args.asset, args.model, args.geometry,
            args.output/'native', reference_metadata=metadata)
        if args.mode == 'evaluate':
            prepare_policy_scene(env)
        if args.mode == 'diagnostic':
            env.commands.zero_()
            capture = native.DiagnosticCapture(env, args.output/'standing', args.geometry_extrema)
            try:
                for control in range(1000):
                    env.step(torch.zeros((env.num_envs, 18), device=env.device))
                    if (control+1) % 100 == 0:
                        print(f"STANDING controls={control+1} replicas={env.num_envs}", flush=True)
                    if time.monotonic()-started >= args.max_wall_seconds:
                        raise TimeoutError('Standing allocation exceeded its deadline')
            finally:
                capture.close()
            report = native.score_diagnostic(args.output/'standing')
            save(args.output/'standing'/'standing_report.json', report)
            state['standing_gate_pass'] = report['all_pass']
            env.verify_native_recipe('after_controlled_steps')
            state['status'] = 'completed'
            return 0
        from rsl_rl.runners import OnPolicyRunner
        from tensordict import TensorDict
        version = importlib.metadata.version('rsl-rl-lib')
        if version != '5.0.1':
            raise ValueError('RSL-RL version differs: '+version)
        task = task_module.TrainingTask(env, task_module.TaskConfig(seed=args.seed), args.output/'task')
        wrapped = vanilla.VanillaVecEnv(task)
        config = vanilla.ppo_config(args.seed)
        save(args.output/'ppo_config.json', config)
        runner_config = copy.deepcopy(config)
        if args.logger == 'wandb':
            # The logger choice stays out of ppo_config, which checkpoint loading compares.
            os.environ.update(WANDB_DIR=str(args.output), WANDB_MODE=args.wandb_mode)
            runner_config.update(logger='wandb', wandb_project=args.wandb_project)
            wrapped.cfg = ConfigRecord(wrapped.cfg)
        runner = OnPolicyRunner(wrapped, runner_config, str(args.output/'learner'), device=args.device)
        import rsl_rl
        upstream = Path(rsl_rl.__file__).parent
        identity['upstream_source_files'] = {name: sha(upstream/name) for name in (
            'runners/on_policy_runner.py', 'algorithms/ppo.py', 'models/mlp_model.py', 'storage/rollout_storage.py')}
        identity['ppo_config'] = config
        state['status'] = 'running'; save(args.output/'state.json', state)
        if args.mode == 'train':
            def checkpoint(update):
                path = args.output/f'checkpoint_update{update:06d}.pt'
                runner.save(str(path), infos={'identity': identity, 'updates': update})
                save(path.with_suffix('.json'), {'identity': identity, 'updates': update,
                    'checkpoint_sha256': sha(path), 'transitions': update*24*env.num_envs})
                state['checkpoint'] = str(path); state['checkpoint_sha256'] = sha(path)
            loads = vanilla.TrainingLoads(env)
            env.capture = loads
            original_log = runner.logger.log
            def log(**values):
                original_log(**values)
                update = values['it']+1
                row = {'update': update, 'transitions': update*24*env.num_envs,
                    'collection_seconds': values['collect_time'], 'learning_seconds': values['learn_time'],
                    'loss': values['loss_dict'], 'learning_rate': values['learning_rate'],
                    'mean_action_std': float(values['action_std'].mean()),
                    'task': task.status(reset_interval=True)}
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
                if update % 50 == 0 or update == args.updates:
                    checkpoint(update)
                    save(args.output/'force_metrics.json', loads.report())
                if time.monotonic()-started >= args.max_wall_seconds:
                    raise TimeoutError('Training allocation ended after a complete PPO update')
            runner.logger.log = log
            runner.learn(args.updates, init_at_random_ep_len=False)
            save(args.output/'force_metrics.json', loads.report())
        else:
            if (checkpoint_record['identity']['upstream_source_files'] != identity['upstream_source_files']
                    or checkpoint_record['identity']['ppo_config'] != config):
                raise ValueError('Checkpoint learner dependency or configuration differs')
            infos = runner.load(str(args.checkpoint), strict=True)
            if infos != {'identity': checkpoint_record['identity'], 'updates': checkpoint_record['updates']}:
                raise ValueError('Embedded checkpoint declaration differs')
            actor = runner.get_inference_policy()
            def policy(observation):
                with torch.inference_mode():
                    return actor(TensorDict({'policy': observation}, batch_size=[env.num_envs]))
            evaluation = importlib.import_module(prefix+'.evaluate')
            camera = importlib.import_module(prefix+'.camera')
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
                    record_video=True, seed=args.seed, **options)
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
