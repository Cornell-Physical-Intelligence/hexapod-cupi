"""Train or evaluate standard PPO on the admitted canonical simulator."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main(argv=None):
    source = Path(__file__).resolve().parent
    sys.path.insert(0, str(source.parent))
    prefix = source.name
    original = importlib.import_module(prefix+'.paper_train')
    configuration = importlib.import_module(prefix+'.env_config')
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True)
    for name in ('asset', 'model', 'geometry', 'geometry-extrema', 'prior', 'prior-metadata',
                 'standing-admission', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-freeze-sha256', required=True)
    parser.add_argument('--num-envs', type=int, choices=[1, 128], required=True)
    parser.add_argument('--seed', type=int, default=20260917)
    parser.add_argument('--updates', type=int, default=512)
    parser.add_argument('--max-wall-seconds', type=float, default=6200.)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--preflight-only', action='store_true')
    if '--preflight-only' in (argv if argv is not None else sys.argv[1:]):
        parser.add_argument('--headless', action='store_true')
        parser.add_argument('--device', default='cuda:0')
    else:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    configuration.verify_assets(args.asset, args.model)
    if (not args.headless or args.device != 'cuda:0' or not 1 <= args.updates <= 2000
            or not 0 < args.max_wall_seconds <= 6600 or args.seed < 0
            or (args.mode == 'train' and (args.num_envs != 128 or args.checkpoint is not None))
            or (args.mode == 'evaluate' and (args.num_envs != 1 or args.checkpoint is None))):
        raise ValueError('A bounded scratch training or single-replica evaluation is required')
    if sha(source/'FREEZE_SHA256.json') != args.source_freeze_sha256:
        raise ValueError('Frozen source identity differs')
    metadata = json.loads(args.prior_metadata.read_text())
    cfg = configuration.EnvConfig(num_envs=args.num_envs, seed=args.seed,
        render=args.mode == 'evaluate', episode_seconds=60. if args.mode == 'evaluate' else 20., device=args.device)
    identity = {'schema': 'canonical_vanilla_ppo_v1', 'source_files': original.source_identity(),
        'model_sha256': configuration.MODEL_SHA256, 'usd_sha256': configuration.USD_SHA256,
        'prior_sha256': sha(args.prior), 'prior_metadata_sha256': sha(args.prior_metadata),
        'geometry_sha256': sha(args.geometry), 'geometry_extrema_sha256': sha(args.geometry_extrema),
        'config': cfg.declaration(), 'motion_prior': False, 'behavior_cloning': False,
        'seed': args.seed, 'rsl_rl_required_version': '5.0.1',
        'adapter_sha256': sha(source/'vanilla.py'), 'entry_sha256': sha(__file__)}
    identity['physics_source_files'] = {k: identity['source_files'][k] for k in ('env.py', 'env_config.py')}
    identity['physics_config'] = {'physics_dt': cfg.physics_dt, 'decimation': cfg.decimation,
        'spacing_m': cfg.spacing_m, 'target_slew_rad': cfg.target_slew_rad, 'action_scale_rad': cfg.action_scale_rad,
        'solver_position_iterations': 32, 'solver_velocity_iterations': 0,
        'floor': '80m_two_triangle_mesh_y_equals_x_seam', 'material_friction': [1., 1.],
        'restitution': 0., 'external_forces_every_iteration': True,
        'neutral_joint_position_rad': metadata['nominal_joint_position_rad'],
        'reset_root_height_m': metadata['reset_root_height_m']}
    identity['standing_admission'] = original.require_admission(args.standing_admission, identity, cfg)
    compatibility_keys = ('schema', 'model_sha256', 'usd_sha256', 'physics_source_files', 'physics_config',
        'prior_metadata_sha256', 'geometry_sha256', 'geometry_extrema_sha256', 'adapter_sha256', 'entry_sha256', 'seed')
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
    state = {'status': 'initializing', 'identity': identity, 'errors': [], 'stage2_complete': False,
        'runtime_binding': {'runtime_tree_sha256': args.source_freeze_sha256}}
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
        from rsl_rl.runners import OnPolicyRunner
        from tensordict import TensorDict
        version = importlib.metadata.version('rsl-rl-lib')
        if version != '5.0.1':
            raise ValueError('RSL-RL version differs: '+version)
        vanilla = importlib.import_module(prefix+'.vanilla')
        native = importlib.import_module(prefix+'.env')
        task_module = importlib.import_module(prefix+'.task')
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        env = native.PaperWalkEnv(cfg, args.asset, args.model, args.geometry,
            args.output/'native', reference_metadata=metadata)
        original.prepare_policy_scene(env)
        task = task_module.TrainingTask(env, task_module.TaskConfig(seed=args.seed), args.output/'task')
        wrapped = vanilla.VanillaVecEnv(task)
        config = vanilla.ppo_config(args.seed)
        save(args.output/'ppo_config.json', config)
        runner = OnPolicyRunner(wrapped, copy.deepcopy(config), str(args.output/'learner'), device=args.device)
        import rsl_rl
        upstream = Path(rsl_rl.__file__).parent
        identity['upstream_source_files'] = {name: sha(upstream/name) for name in (
            'runners/on_policy_runner.py', 'algorithms/ppo.py', 'models/mlp_model.py', 'storage/rollout_storage.py')}
        identity['ppo_config'] = config
        state['status'] = 'running'; save(args.output/'state.json', state)
        if args.mode == 'train':
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
                state.update(updates=update, transitions=row['transitions'], wall_seconds=time.monotonic()-started)
                save(args.output/'state.json', state)
                if update % 50 == 0 or update == args.updates:
                    path = args.output/f'checkpoint_update{update:06d}.pt'
                    runner.save(str(path), infos={'identity': identity, 'updates': update})
                    save(path.with_suffix('.json'), {'identity': identity, 'updates': update,
                        'checkpoint_sha256': sha(path), 'transitions': row['transitions']})
                    state['checkpoint'] = str(path); state['checkpoint_sha256'] = sha(path)
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
            reporter = importlib.import_module(prefix+'.force_metrics')
            camera = importlib.import_module(prefix+'.camera')
            env.render = camera.NativePolicyCamera(env)
            selected = {'learning:translate_0.05_0deg', 'learning:quiet_20s', 'learning:forward_0.05_to_stop'}
            state['evaluations'] = []
            for index, case in enumerate(c for c in evaluation.learning_probe_cases() if c['case_id'] in selected):
                result = reporter.run_batch(env, policy, [case], args.output/f'evaluation_{index:02d}',
                    args.geometry_extrema, checkpoint_sha256=args.checkpoint_sha256, source_sha256=sha(__file__),
                    seed=args.seed, video_case_id=case['case_id'],
                    progress=lambda value: print(json.dumps(value), flush=True),
                    max_wall_seconds=max(.001, args.max_wall_seconds-(time.monotonic()-started)))
                state['evaluations'].append(result); save(args.output/'state.json', state)
                if result['force_metrics']['status'] != 'available':
                    raise ValueError('Evaluation has no load summary')
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
