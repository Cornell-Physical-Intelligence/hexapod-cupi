"""Run a source-bound RS05 admission inside the preserved allocation guard."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def loads_report(directory, model_sha, controls, source_sha):
    import numpy as np
    from experiments.trajectory_optimization.force_metrics import FIELDS, MASS_KG, summarize
    directory = Path(directory)
    receipt = json.loads((directory/'session.json').read_text())
    chunks, inputs = [], {'session.json': sha(directory/'session.json')}
    for name in receipt['substep_files']:
        with np.load(directory/name, allow_pickle=False) as data:
            chunks.append({key: data[key].copy() for key in FIELDS})
        inputs[name] = sha(directory/name)
    raw = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in FIELDS}
    cases = [{'case_id': 'standing:'+str(i), 'controls': controls} for i in range(len(receipt['root_paths']))]
    report = {'schema': 'canonical_rs05_load_metrics_v1', 'status': 'available',
        'model_sha256': model_sha, 'source_freeze_sha256': source_sha, 'source_files': inputs,
        'physics_hz': 400, 'settle_seconds': 2., 'reference_mass_kg': MASS_KG,
        'definitions': 'Normal contact forces exclude friction. Torque uses absolute values. Locomotion requires a nonzero recorded command after 2 s.',
        'capture_failure': receipt['failure'], 'cases': summarize(raw, cases)}
    save(directory/'force_metrics.json', report)
    return report


def verify_native(env, model):
    import numpy as np
    from hexapod_env.tasks.mkii_rs05.env import tensor
    view = env._robot.root_view
    def read(name):
        value = getattr(view, name)()
        return np.asarray(value.numpy() if hasattr(value, 'numpy') else tensor(value).cpu().numpy())
    masses = read('get_masses')
    expected = {row['name']: row['mass'] for row in model['links']}
    if not np.allclose(masses, [[expected[name] for name in env._body_names]], atol=1e-6, rtol=0):
        raise ValueError('Native masses differ from the approved model')
    for name in ('get_dof_stiffnesses', 'get_dof_dampings', 'get_dof_armatures'):
        if np.any(read(name) != 0):
            raise ValueError('Unexpected implicit drive or armature: '+name)
    sdf = env.sim.physics_sim_view.create_sdf_shape_view('/World/envs/env_*/Robot/*/collisions/part_*', 1)
    if sdf.count != 153*env.num_envs or not sdf.check():
        raise ValueError('Detailed collision geometry differs')
    from pxr import PhysxSchema
    for root in env.scene.env_prim_paths:
        prim = env.sim.stage.GetPrimAtPath(root+'/Robot/body')
        api = PhysxSchema.PhysxArticulationAPI(prim)
        if [api.GetSolverPositionIterationCountAttr().Get(), api.GetSolverVelocityIterationCountAttr().Get()] != [32, 0]:
            raise ValueError('Native solver settings differ')
        if prim.GetAttribute('physxArticulation:enabledSelfCollisions').Get() is not True:
            raise ValueError('Native self-collision setting differs')
    return {'mass_per_replica_kg': masses.sum(axis=1).tolist(), 'sdf_shapes': sdf.count,
            'implicit_drive_and_armature_zero': True, 'solver_iterations': [32, 0], 'self_collision': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--mode', choices=['diagnostic'], required=True)
    for name in ('asset', 'model', 'geometry', 'geometry-extrema', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-freeze-sha256', required=True)
    parser.add_argument('--num-envs', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=20260917)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    if not (root/'FREEZE_SHA256.json').is_file() or sha(root/'FREEZE_SHA256.json') != args.source_freeze_sha256:
        raise ValueError('Frozen source identity differs')
    sys.path.insert(0, str(root))
    from hexapod_env.assets.mkii_rs05 import MKII_RS05_MODEL_SHA256, MKII_RS05_USD_SHA256
    if (sha(args.model) != MKII_RS05_MODEL_SHA256 or sha(args.asset/'robot.usda') != MKII_RS05_USD_SHA256
            or args.num_envs < 1 or not args.headless or args.device != 'cuda:0'):
        raise ValueError('Approved model and bounded headless CUDA configuration required')
    identity = {'source_freeze_sha256': args.source_freeze_sha256,
        'model_sha256': MKII_RS05_MODEL_SHA256, 'usd_sha256': MKII_RS05_USD_SHA256,
        'geometry_sha256': sha(args.geometry), 'geometry_extrema_sha256': sha(args.geometry_extrema),
        'num_envs': args.num_envs, 'seed': args.seed, 'mode': args.mode}
    if args.preflight_only:
        print(json.dumps(identity, indent=2)); return 0
    args.output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'initializing', 'identity': identity, 'errors': [],
        'stage2_complete': False, 'runtime_binding': {'runtime_tree_sha256': args.source_freeze_sha256}}
    save(args.output/'state.json', state)
    app = env = None
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device=args.device).app
        print('RS05_APP_READY', flush=True)
        import torch
        from hexapod_env.tasks.mkii_rs05.config import HexapodMkiiRs05FlatEnvCfg
        from hexapod_env.tasks.mkii_rs05.env import HexapodMkiiRs05Env
        cfg = HexapodMkiiRs05FlatEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        cfg.standing_only = True
        cfg.episode_length_s = 60.
        cfg.robot.spawn.usd_path = str(args.asset/'robot.usda')
        env = HexapodMkiiRs05Env(cfg)
        env.reset()
        state['native_readback'] = verify_native(env, json.loads(args.model.read_text()))
        env.begin_diagnostic_capture(args.output/'native400hz', geometry_path=args.geometry,
            geometry_extrema_path=args.geometry_extrema, scope='Fresh RS05 standing admission; unchanged scorer.')
        state['status'] = 'running'; save(args.output/'state.json', state)
        action = torch.zeros(env.num_envs, 18, device=env.device)
        with torch.inference_mode():
            for control in range(1000):
                env.step(action)
                if (control+1) % 100 == 0:
                    state['controls'] = control+1; save(args.output/'state.json', state)
                    print(json.dumps({'controls': control+1}), flush=True)
        env.end_diagnostic_capture(scope='Full native standing capture; unchanged scorer.')
        from experiments.paper_walk.env import score_diagnostic
        report = score_diagnostic(args.output/'native400hz')
        save(args.output/'standing_report.json', report)
        loads_report(args.output/'native400hz', MKII_RS05_MODEL_SHA256, 1000, args.source_freeze_sha256)
        state.update(status='completed', standing_gate_pass=report['all_pass'])
        save(args.output/'state.json', state)
        return 0 if report['all_pass'] else 1
    except BaseException as error:
        state.update(status='failed', traceback=traceback.format_exc())
        state['errors'].append(repr(error)); save(args.output/'state.json', state)
        raise
    finally:
        if env is not None:
            if env._capture is not None:
                env.end_diagnostic_capture(scope='Preserved failed or interrupted native capture.')
            env.close()
        if app is not None:
            app.close()


if __name__ == '__main__':
    raise SystemExit(main())
