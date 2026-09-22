"""Evaluate the finite prescribed-controller sweep through native capture."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from .env_config import EnvConfig, JOINT_NAMES, MODEL_SHA256, USD_SHA256, sha, verify_assets
from .tripod import TripodController, supported
from .tripod_config import SWEEP


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def cases_for(suite):
    from .evaluate import case_manifest
    if suite == 'screen':
        return [{'case_id': 'tripod:screen:'+name, 'profile': 'omni_static',
                 'command': command, 'controls': 1000, 'clearance': 'low'}
                for name, command in [('forward', [.05, 0., 0.]),
                                       ('left', [0., 0., .2]), ('right', [0., 0., -.2])]]
    if suite == 'qualification':
        return [c for c in case_manifest() if c['profile'] != 'transition' and supported(c['command'])]
    if suite == 'clearance':
        return [{'case_id': 'tripod:'+mode+':'+name, 'profile': 'omni_static',
                 'command': command, 'controls': 1800 if mode == 'switch' else 1000,
                 'clearance': mode,
                 **({'score_control_windows': [[0, 1000], [800, 1800]]} if mode == 'switch' else {})}
                for mode in ('low', 'raised', 'switch')
                for name, command in [('forward', [.05, 0., 0.]),
                                       ('left', [0., 0., .2]), ('right', [0., 0., -.2])]]
    raise ValueError('Unknown prescribed-controller suite')


def clearance_at(case, control):
    mode = case.get('clearance', 'low')
    if mode == 'switch':
        return 'raised' if 300 <= control < 1050 else 'low'
    return mode


class NativeController:
    """Read prior captured foot patches; emit ordinary motor-target actions."""
    def __init__(self, env, case, config, stream):
        self.env, self.case, self.stream = env, case, stream
        self.controller = TripodController(env.neutral.detach().cpu().numpy(),
            env.lower.detach().cpu().numpy(), env.upper.detach().cpu().numpy(), config)
        self.control = 0

    def __call__(self, observation):
        import torch
        samples = self.env.capture.last_control
        # No contact exists before the first actual physics sample.
        force = np.zeros(6)
        if samples:
            if len(samples) != 8:
                raise ValueError('Touchdown requires the previous complete native control')
            force = np.mean([np.linalg.norm(s['distal_force_world_n'][0], axis=-1)
                             for s in samples], axis=0)
        command = self.env.commands[0].detach().cpu().numpy().astype(float)
        # Native command tensors carry float32 rounding at the envelope endpoints.
        command = np.round(command, 7)
        target = self.controller.step(command, force, clearance_at(self.case, self.control))
        self.stream.write(json.dumps({'control': self.control, 'measured_force_n': force.tolist(),
                                      **self.controller.snapshot()}, allow_nan=False)+'\n')
        self.stream.flush()
        self.control += 1
        if self.controller.fault:
            raise RuntimeError('Controller fault: '+self.controller.fault)
        value = torch.as_tensor(target, dtype=self.env.neutral.dtype, device=self.env.device)
        return ((value-self.env.neutral)/self.env.cfg.action_scale_rad).reshape(1, 18)


def clearance_metrics(directory, case):
    """Measure settled height and per-foot lift without replacing native gates."""
    directory = Path(directory)
    path = directory/'control_trace.npz'
    if not path.exists():
        return {'pass': False, 'reason': 'missing_capture'}
    states = [json.loads(line) for line in (directory/'controller.jsonl').read_text().splitlines()]
    with np.load(path, allow_pickle=False) as trace:
        n = len(trace['time_s'])
        states = states[:n]
        height = trace['root_pose_xyzw'][:, 0, 2]
        feet = trace['toe_xyz_world_m'][:, 0, :, 2]
        contacts = trace['distal_contact'][:, 0]
    values = {}
    for mode in ('low', 'raised'):
        mask = np.array([s['mode'] == mode and s['state'] == 'walk' for s in states])
        # Omit the first two seconds after entry to each contiguous mode segment.
        stable = np.zeros(n, bool); count = 0
        for i, keep in enumerate(mask):
            count = count+1 if keep else 0
            stable[i] = count > 100
        if stable.sum() < 100:
            continue
        lifts = []
        for leg in range(6):
            support = stable & contacts[:, leg]
            swing = stable & ~contacts[:, leg]
            lifts.append(float(feet[swing, leg].max()-np.median(feet[support, leg]))
                         if support.any() and swing.any() else None)
        values[mode] = {'median_root_height_m': float(np.median(height[stable])),
                        'toe_lift_m': lifts, 'controls': int(stable.sum())}
    expected = ('low', 'raised') if case['clearance'] == 'switch' else (case['clearance'],)
    passed = all(mode in values and all(x is not None and x >= (.012 if mode == 'low' else .016)
                 for x in values[mode]['toe_lift_m']) for mode in expected)
    delta = None
    if case['clearance'] == 'switch':
        if all(mode in values for mode in ('low', 'raised')):
            delta = values['raised']['median_root_height_m']-values['low']['median_root_height_m']
        passed = passed and delta is not None and delta >= .008
        # Confirm both requested transitions reached their endpoint.
        passed = passed and any(s['mode'] == 'raised' for s in states) and states[-1]['mode'] == 'low'
    receipt = json.loads((directory/'native400hz/capture.json').read_text())
    passed = passed and receipt['nonfoot_contact_steps_400hz'] == [0] and receipt['joint_bound_violation_steps'] == [0]
    return {'pass': bool(passed), 'modes': values, 'raised_minus_low_m': delta,
            'scope': 'Additional geometry targets; original numerical gates also apply.'}


def run_suite(env, output, geometry, config, suite, seed, max_seconds, identity):
    from .evaluate import case_manifest, run_batch
    output = Path(output)
    cases = cases_for(suite)
    full = case_manifest()
    declaration = {'schema': 'zhang_tripod_native_suite_v1', 'suite': suite,
        'configuration': config.declaration(), 'identity': identity, 'cases': cases,
        'repeats': 1 if suite == 'screen' else 2, 'seed': seed,
        'unsupported_case_ids': [c['case_id'] for c in full
            if c['profile'] == 'transition' or not supported(c['command'])],
        'stage2_complete': False, 'terrain_complete': False, 'visual_acceptance': False}
    save(output/'allocation.json', declaration)
    results = []; deadline = time.monotonic()+max_seconds; acquisition_failed = False
    for repeat in range(declaration['repeats']):
        for case in cases:
            remaining = deadline-time.monotonic()
            if remaining <= 0 or env.native_errors:
                break
            name = f'trial_{len(results):03d}'
            log_path = output/(name+'_controller.jsonl')
            with log_path.open('x') as stream:
                policy = NativeController(env, case, config, stream)
                report = run_batch(env, policy, [case], output/name, geometry,
                    checkpoint_sha256=None, source_sha256=sha(__file__), seed=case.get('seed', seed+repeat),
                    video_case_id=case['case_id'], max_wall_seconds=remaining)
            log_path.rename(output/name/'controller.jsonl')
            result = {'case_id': case['case_id'], 'repeat': repeat, 'path': name,
                      'pass': report['results'][0]['pass'], 'acquisition_complete': report['acquisition_complete'],
                      'controller_fault': policy.controller.fault}
            if suite == 'clearance':
                result['clearance'] = clearance_metrics(output/name, case)
                result['pass'] = result['pass'] and result['clearance']['pass']
            result['pass'] = bool(result['pass'] and not result['controller_fault'])
            result['files'] = {p.relative_to(output/name).as_posix(): sha(p)
                              for p in sorted((output/name).rglob('*')) if p.is_file()}
            results.append(result)
            save(output/'progress.json', results)
            if (report['native_capture_failure'] or report.get('allocation_limit_reached')
                    or (report['failure_kind'] == 'acquisition_error' and not policy.controller.fault)):
                acquisition_failed = True
                break
        if acquisition_failed or time.monotonic() >= deadline or env.native_errors:
            break
    summary = {**declaration, 'results': results,
               'acquisition_complete': len(results) == len(cases)*declaration['repeats']
                   and all(r['acquisition_complete'] for r in results),
               'pass': len(results) == len(cases)*declaration['repeats'] and all(r['pass'] for r in results)}
    save(output/'summary.json', summary)
    return summary


def main(argv=None):
    from . import admission
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--mode', choices=['tripod'], required=True)
    for name in ('asset', 'model', 'geometry', 'geometry-extrema', 'stance', 'output', 'standing-admission'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-freeze-sha256', required=True)
    parser.add_argument('--num-envs', type=int, choices=[1], default=1)
    parser.add_argument('--candidate', type=int, choices=range(len(SWEEP)), required=True)
    parser.add_argument('--suite', choices=['screen', 'qualification', 'clearance'], required=True)
    parser.add_argument('--seed', type=int, default=27057)
    parser.add_argument('--max-wall-seconds', type=float, default=6200.)
    parser.add_argument('--preflight-only', action='store_true')
    if any(flag in (sys.argv[1:] if argv is None else argv) for flag in ('--preflight-only', '--help', '-h')):
        parser.add_argument('--headless', action='store_true')
        parser.add_argument('--device', default='cuda:0')
    else:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if not args.headless or args.device != 'cuda:0' or not 0 < args.max_wall_seconds <= 6200 or args.seed < 0:
        raise ValueError('A bounded headless native evaluation is required')
    verify_assets(args.asset, args.model)
    source = Path(__file__).resolve().parent
    if sha(source.parent/'FREEZE_SHA256.json') != args.source_freeze_sha256:
        raise ValueError('Frozen source identity differs')
    metadata = json.loads(args.stance.read_text())
    if metadata['joint_names'] != list(JOINT_NAMES):
        raise ValueError('Stance joint order differs')
    cfg = EnvConfig(num_envs=1, render=True, record_motion_features=True, episode_seconds=60., seed=args.seed)
    identity = {'model_sha256': MODEL_SHA256, 'usd_sha256': USD_SHA256,
        'source_files': {p.name: sha(p) for p in sorted(source.glob('*.py'))},
        'source_freeze_sha256': args.source_freeze_sha256,
        'stance_sha256': sha(args.stance), 'geometry_sha256': sha(args.geometry),
        'geometry_extrema_sha256': sha(args.geometry_extrema), 'config': cfg.declaration(),
        'controller': SWEEP[args.candidate].declaration(), 'candidate': args.candidate,
        'controller_kind': 'prescribed_tripod', 'learned_policy': False}
    identity['physics_source_files'] = {k: identity['source_files'][k] for k in ('env.py', 'env_config.py')}
    identity['physics_config'] = {'physics_dt': cfg.physics_dt, 'decimation': cfg.decimation,
        'spacing_m': cfg.spacing_m, 'target_slew_rad': cfg.target_slew_rad, 'action_scale_rad': cfg.action_scale_rad,
        'solver_position_iterations': 32, 'solver_velocity_iterations': 0,
        'floor': '80m_two_triangle_mesh_y_equals_x_seam', 'material_friction': [1., 1.],
        'restitution': 0., 'external_forces_every_iteration': True,
        'neutral_joint_position_rad': metadata['nominal_joint_position_rad'],
        'reset_root_height_m': metadata['reset_root_height_m']}
    identity['standing_admission'] = admission.require_admission(args.standing_admission, identity, cfg)
    if args.preflight_only:
        print(json.dumps(identity, indent=2)); return 0
    args.output.mkdir(parents=True, exist_ok=False)
    state = {'schema': 'zhang_tripod_native_run_v1', 'status': 'initializing', 'identity': identity,
             'errors': [], 'stage2_complete': False,
             'runtime_binding': {'runtime_tree_sha256': args.source_freeze_sha256}}
    save(args.output/'state.json', state)
    app = None
    try:
        import faulthandler
        args.enable_cameras = True; args.headless_explicit = False
        with (args.output/'startup_tracebacks.log').open('w') as stream:
            faulthandler.enable(file=stream)
            faulthandler.dump_traceback_later(45., repeat=True, file=stream)
            try:
                app = AppLauncher(args).app
            finally:
                faulthandler.cancel_dump_traceback_later(); faulthandler.disable()
        print('REFERENCE_SCREEN_APP_READY', flush=True)
        from .env import LocomotionEnv
        from .camera import NativePolicyCamera, prepare_policy_scene
        env = LocomotionEnv(cfg, args.asset, args.model, args.geometry, args.output/'native', reference_metadata=metadata)
        prepare_policy_scene(env); env.render = NativePolicyCamera(env)
        state['status'] = 'running'; save(args.output/'state.json', state)
        summary = run_suite(env, args.output, args.geometry_extrema, SWEEP[args.candidate],
                            args.suite, args.seed, args.max_wall_seconds, identity)
        state['status'] = 'completed' if summary['acquisition_complete'] else 'failed'
        state['numerical_pass'] = summary['pass']
        save(args.output/'state.json', state)
        return 0 if summary['acquisition_complete'] else 1
    except BaseException as error:
        state.update(status='failed', errors=[repr(error)], traceback=traceback.format_exc())
        save(args.output/'state.json', state)
        raise
    finally:
        if app is not None:
            app.close()


if __name__ == '__main__':
    raise SystemExit(main())
