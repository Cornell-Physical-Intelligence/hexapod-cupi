"""Replay optimized motor targets through the admitted native controller.

The source pack installs this entry point as train.py and preserves the original
entry point as paper_train.py. The original launcher and reservation guard keep
their bytes and retain ownership of locks, the container, and cleanup.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def paper_modules():
    here = Path(__file__).resolve().parent
    if (here/'paper_train.py').is_file():
        sys.path.insert(0, str(here.parent))
        return here.name, importlib.import_module(here.name+'.paper_train')
    return 'experiments.paper_walk', importlib.import_module('experiments.paper_walk.train')


def validate_trajectory(path, expected_sha, model_sha):
    import numpy as np
    path = Path(path)
    if sha(path) != expected_sha:
        raise ValueError('Trajectory bytes differ')
    declaration = json.loads(path.with_name('INPUT.json').read_text())
    result = json.loads(path.with_name('RESULT.json').read_text())
    if (declaration.get('schema') != 'canonical_full_body_trajectory_optimization_v1'
            or declaration['model']['model_sha256'] != model_sha
            or result.get('trajectory_sha256') != expected_sha
            or result.get('status') != 'solved' or result.get('audit', {}).get('passed') is not True):
        raise ValueError('Trajectory lacks a solved, model-bound CPU feasibility result')
    with np.load(path, allow_pickle=False) as data:
        target = data['target'].copy()
    n = round(declaration['config']['period_s']/.02)
    if target.shape != (n, 18) or not np.isfinite(target).all():
        raise ValueError('Incomplete finite target cycle')
    neutral = np.tile([0., -.3, .4], 6)
    if abs(target-neutral).max() > .3500001 or abs(np.roll(target, -1, axis=0)-target).max() > .0400001:
        raise ValueError('Trajectory exceeds the unchanged action range or target slew')
    return declaration, target


def main(argv=None):
    prefix, original = paper_modules()
    env_config = importlib.import_module(prefix+'.env_config')
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--mode', choices=['replay'], required=True)
    for key in ('asset', 'model', 'geometry', 'geometry-extrema', 'prior', 'prior-metadata',
                'output', 'standing-admission', 'trajectory'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--trajectory-sha256', required=True)
    parser.add_argument('--source-freeze-sha256', required=True)
    parser.add_argument('--num-envs', type=int, choices=[1], default=1)
    parser.add_argument('--max-wall-seconds', type=float, default=1500.)
    parser.add_argument('--preflight-only', action='store_true')
    if '--preflight-only' in (argv if argv is not None else sys.argv[1:]):
        parser.add_argument('--headless', action='store_true')
        parser.add_argument('--device', default='cuda:0')
    else:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    env_config.verify_assets(args.asset, args.model)
    if not args.headless or not 0 < args.max_wall_seconds <= 3600:
        raise ValueError('Bounded headless allocation required')
    source = Path(__file__).resolve().parent
    if sha(source/'FREEZE_SHA256.json') != args.source_freeze_sha256:
        raise ValueError('Source-freeze identity differs')
    metadata = json.loads(args.prior_metadata.read_text())
    cfg = env_config.EnvConfig(num_envs=1, render=True, episode_seconds=60., device=args.device)
    identity = {'source_files': original.source_identity(), 'model_sha256': env_config.MODEL_SHA256,
        'usd_sha256': env_config.USD_SHA256, 'prior_sha256': sha(args.prior),
        'prior_metadata_sha256': sha(args.prior_metadata), 'geometry_sha256': sha(args.geometry),
        'geometry_extrema_sha256': sha(args.geometry_extrema), 'config': cfg.declaration()}
    identity['physics_source_files'] = {k: identity['source_files'][k] for k in ('env.py', 'env_config.py')}
    identity['physics_config'] = {'physics_dt': cfg.physics_dt, 'decimation': cfg.decimation,
        'spacing_m': cfg.spacing_m, 'target_slew_rad': cfg.target_slew_rad, 'action_scale_rad': cfg.action_scale_rad,
        'solver_position_iterations': 32, 'solver_velocity_iterations': 0,
        'floor': '80m_two_triangle_mesh_y_equals_x_seam', 'material_friction': [1., 1.],
        'restitution': 0., 'external_forces_every_iteration': True,
        'neutral_joint_position_rad': metadata['nominal_joint_position_rad'],
        'reset_root_height_m': metadata['reset_root_height_m']}
    identity['standing_admission'] = original.require_admission(args.standing_admission, identity, cfg)
    declaration, targets = validate_trajectory(args.trajectory, args.trajectory_sha256, env_config.MODEL_SHA256)
    identity.update(controller_kind='optimized_periodic_motor_targets', learned_policy=False,
        trajectory_sha256=args.trajectory_sha256, optimization_input_sha256=sha(args.trajectory.with_name('INPUT.json')),
        optimization_result_sha256=sha(args.trajectory.with_name('RESULT.json')))
    if args.preflight_only:
        print(json.dumps(identity, indent=2)); return 0
    args.output.mkdir(parents=True, exist_ok=False)
    state = {'schema': 'canonical_optimized_trajectory_replay_v1', 'status': 'initializing',
        'identity': identity, 'errors': [], 'stage2_complete': False,
        'runtime_binding': {'runtime_tree_sha256': args.source_freeze_sha256}}
    save(args.output/'state.json', state)
    app = None
    try:
        import faulthandler
        args.enable_cameras = True
        args.headless_explicit = False
        with (args.output/'startup_tracebacks.log').open('w') as stream:
            faulthandler.enable(file=stream)
            faulthandler.dump_traceback_later(45., repeat=True, file=stream)
            try:
                app = AppLauncher(args).app
            finally:
                faulthandler.cancel_dump_traceback_later(); faulthandler.disable()
        print('REFERENCE_SCREEN_APP_READY', flush=True)
        import numpy as np
        import torch
        env_module = importlib.import_module(prefix+'.env')
        capture_module = importlib.import_module(prefix+'.evaluate')
        camera_module = importlib.import_module(prefix+'.camera')
        torch.manual_seed(cfg.seed)
        env = env_module.PaperWalkEnv(cfg, args.asset, args.model, args.geometry,
            args.output/'native', reference_metadata=metadata)
        original.prepare_policy_scene(env)
        env.render = camera_module.NativePolicyCamera(env)
        target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=env.device)
        control = 0
        def controller(observation):
            nonlocal control
            target = target_tensor[control % len(target_tensor)]
            control += 1
            return ((target-env.neutral)/env.cfg.action_scale_rad)[None, :]
        case = {'case_id': 'trajectory:forward_0.05', 'profile': 'omni_static',
                'command': [declaration['config']['forward_mps'], 0., 0.], 'controls': 1000}
        state['status'] = 'running'; save(args.output/'state.json', state)
        def progress(value):
            state['progress'] = value; save(args.output/'state.json', state)
            print(json.dumps(value), flush=True)
        report = capture_module.run_batch(env, controller, [case], args.output/'evaluation',
            args.geometry_extrema, checkpoint_sha256=None, source_sha256=sha(__file__),
            seed=cfg.seed, video_case_id=case['case_id'], progress=progress,
            max_wall_seconds=args.max_wall_seconds)
        with np.load(args.output/'evaluation/control_trace.npz', allow_pickle=False) as trace:
            # Retain the whole native record, including the cold-start prefix.
            np.savez_compressed(args.output/'recorded_transitions.npz',
                states=trace['amp_state_before'][:, 0], next_states=trace['amp_state_after'][:, 0],
                commands=trace['command'][:, 0])
        state.update(status='completed' if report['acquisition_complete'] else 'failed',
            replay=report, eligible_for_motion_prior=bool(report['results'][0]['pass']),
            recorded_transitions_sha256=sha(args.output/'recorded_transitions.npz'))
        save(args.output/'state.json', state)
        return 0 if report['acquisition_complete'] else 1
    except BaseException as error:
        state['status'] = 'failed'; state['errors'].append(repr(error))
        state['traceback'] = traceback.format_exc(); save(args.output/'state.json', state)
        raise
    finally:
        if app is not None:
            app.close()


if __name__ == '__main__':
    raise SystemExit(main())
