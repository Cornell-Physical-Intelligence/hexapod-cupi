#!/usr/bin/env python3
"""Matched-origin standing acquisition; quiet/rate outcomes stay separate from admission."""
from __future__ import annotations
import argparse
import faulthandler
import json
from pathlib import Path
import time
import traceback
from c_study_runtime import bootstrap_c_study_runtime
RUNTIME = bootstrap_c_study_runtime()
from isaaclab.app import AppLauncher
from screen_contract import OPTIONS, STARTUP, save, digest
from origin_contract import CASES, PROTOCOL, preflight, load_initial

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--package', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--geometry-reference', type=Path, required=True)
parser.add_argument('--variant', default='f050_t060')
parser.add_argument('--stance-index', type=int, default=0)
parser.add_argument('--mode', choices=('origin',), required=True)
parser.add_argument('--num-envs', type=int, required=True)
parser.add_argument('--steps', type=int, required=True)
parser.add_argument('--admission', type=Path, required=True)
parser.add_argument('--case', choices=tuple(CASES), required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
SOURCE = Path(__file__).resolve().parents[1]
identity, geometry_reference = preflight(args, SOURCE)
args.output.mkdir(parents=True)
faulthandler.enable()
faulthandler.dump_traceback_later(45, repeat=True)
print('REFERENCE_SCREEN_APP_START', flush=True)
app = AppLauncher(args).app
faulthandler.cancel_dump_traceback_later()
print('REFERENCE_SCREEN_APP_READY', flush=True)

import numpy as np
import torch
from reference_physics_env import build_reference_environment
from physics_telemetry import capture_measured_state, pre_reset_capture, require_single_pre_reset_sample, array
from screen_metrics import standing_screen
from omni_quiet_review import quiet_metrics, QUIET_GATES
from matched_origin import patched_factory, initial_readback, ground_readback
from canonical_stance_startup import CanonicalStanceStartup
from physics_substeps import PhysicsSubstepRecorder
from origin_metrics import compare_measurements


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def main():
    env = None
    physics_recorder = None
    state = dict(status='initializing', mode=args.mode, started_unix=time.time(), identity=identity,
                 runtime_binding=RUNTIME, policy_training_started=False, stage2_complete=False,
                 terrain_qualified=False, scope='matched-origin diagnostic acquisition; never wave/PPO admission', origin_protocol=PROTOCOL, case=args.case)
    samples, references = [], []
    failure = None
    try:
        payload=load_initial(SOURCE)
        with patched_factory(payload,args.case):
            env, manifest, plan, record, stance, layout, asset_audit, controller_contract = build_reference_environment(args, OPTIONS)
        state.update(asset_audit=asset_audit, controller=controller_contract, layout=layout)
        save(args.output / 'state.json', state)
        points_local = np.asarray(geometry_reference['stance']['toes'], dtype=float)
        if tuple(geometry_reference['joint_names']) != layout['joint_names_leg_major']:
            raise ValueError('Reference toe and joint leg-order contract differs')
        env.reset(seed=0)
        env.episode_length_buf.zero_()
        state['initial_readback']=initial_readback(env,payload,args.case)
        save(args.output/'initial_readback.json',state['initial_readback'])
        state['ground_readback']=ground_readback(env)
        save(args.output/'ground_readback.json',state['ground_readback'])
        if not state['initial_readback']['passed'] or not state['ground_readback']['passed']:
            raise RuntimeError('Matched reset or infinite-plane readback failed before control0')
        zero = torch.zeros((args.num_envs, 18), device=env.device)
        reset_target = array(env.reference_residual_controller.reference_position)
        nominal = np.asarray([stance['joint_positions_rad'][name] for name in layout['joint_names_runtime']],dtype=np.float32).astype(np.float64)
        nominal = np.broadcast_to(nominal,reset_target.shape).copy()
        if not np.array_equal(nominal,array(env._robot.data.default_joint_pos).astype(np.float64)):
            raise RuntimeError('Named plan stance differs from actual nominal articulation target')
        soft_limits = array(env._robot.data.soft_joint_pos_limits)
        startup = CanonicalStanceStartup(reset_target,nominal,soft_limits[...,0],soft_limits[...,1],
            duration_s=STARTUP['duration_s'],dt=env.step_dt)
        startup_contract = {**startup.contract(),'joint_names_runtime':list(layout['joint_names_runtime']),
            'canonical_hold_before_scoring_s':STARTUP['canonical_hold_before_scoring_s'],
            'physical_reset_preserved':False,'reset_scope':'explicit common audited009env6 initial state inside cold reset only'}
        save(args.output / 'startup_reference.json',startup_contract)
        state['startup_reference'] = startup_contract
        clock = {'step': 0}

        def capture():
            row = capture_measured_state(env, layout, points_local, time_s=(clock['step']+1)*env.step_dt)
            target = env.reference_residual_target
            if target is None:
                raise RuntimeError('No actual target result at pre-reset capture')
            row.update({key: array(value) for key, value in target.items()})
            return row

        state['status'] = 'running'
        save(args.output / 'state.json', state)
        physics_recorder = PhysicsSubstepRecorder(env)
        with physics_recorder, pre_reset_capture(env, capture) as captured:
            for step in range(args.steps):
                clock['step'] = step
                requested = np.zeros(3)
                startup_ref = startup.sample(step+1)
                env.set_reference_targets(startup_ref['q_ref'],startup_ref['analytic_velocity_rad_s'],
                    startup_ref['analytic_acceleration_rad_s2'],startup_ref['valid'])
                env.set_evaluation_targets(torch.as_tensor(requested, device=env.device, dtype=zero.dtype).expand(args.num_envs, 3))
                before = len(captured)
                physics_recorder.begin_control(step)
                with torch.inference_mode():
                    obs, reward, term, trunc, _ = env.step(zero)
                row = require_single_pre_reset_sample(captured, before, term, trunc)
                row['requested_command'] = np.broadcast_to(requested, (args.num_envs, 3)).copy()
                row['reference_source_code'] = np.full(args.num_envs,
                    0 if step+1 < startup.steps else 1,dtype=np.int8)
                samples.append(row)
                physics_recorder.end_control(row)
                if not all(torch.isfinite(v).all() for v in obs.values()) or not torch.isfinite(reward).all():
                    failure = 'Nonfinite observation or reward'
                    break
                if term.any() or trunc.any():
                    failure = 'Physical terminal event; last sample is pre-reset; no continued run'
                    break
                if (step+1) % 100 == 0:
                    state['control_steps'] = step+1
                    save(args.output / 'state.json', state)
                    print(f'REFERENCE_SCREEN {args.mode} {step+1}/{args.steps}', flush=True)

        state['physics_substep_review'] = physics_recorder.export(args.output)
        data = {key: np.stack([row[key] for row in samples]) for key in samples[0]} if samples else {}
        if samples:
            np.savez_compressed(args.output / 'trace.npz', **data,
                                joint_names=np.asarray(layout['joint_names_runtime']), legs=np.asarray(('lf','lm','lr','rf','rm','rr')))
        save(args.output / 'reference_states.json', references)
        gate = None; quiet = None
        if len(samples) > 200:
            gate = standing_screen(data)
            quiet = quiet_metrics(data,0,200,list(layout['joint_names_runtime']),env.step_dt)
            quiet['bounds']=QUIET_GATES
        if len(samples)==1000:
            raw=physics_recorder.data();raw['joint_names']=np.asarray(layout['joint_names_runtime'])
            diagnostic=compare_measurements(raw,CASES[args.case])
            save(args.output/'rate_discrepancy.json',diagnostic)
        state['unchanged_physical_gate']=gate
        state['unchanged_quiet_gate']=quiet
        state['all_existing_bounds_met']=bool(gate and gate['passed'] and quiet and quiet['pass'] and quiet['window_duration_s']>=10.)
        state['quiet_failure_does_not_become_an_admission']=True
        state['velocity_fidelity_qualified']=False
        passed = bool(failure is None and len(samples)==args.steps and gate and gate['passed'])
        state.update(status='completed' if passed else 'rejected', failure=failure,
                     acquisition_complete=passed, control_steps=len(samples), finished_unix=time.time())
        save(args.output / 'state.json', state)
        return 0 if passed else 1
    except Exception as exc:
        if physics_recorder is not None and physics_recorder.rows:
            try:
                state['physics_substep_review'] = physics_recorder.export(args.output)
            except Exception as export_error:
                state['physics_substep_export_error'] = repr(export_error)
        if samples:
            np.savez_compressed(args.output / 'partial_trace.npz', **{key: np.stack([row[key] for row in samples]) for key in samples[0]})
        save(args.output / 'reference_states.json', references)
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(),
                     control_steps=len(samples), finished_unix=time.time())
        save(args.output / 'state.json', state)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    try:
        code = main()
    finally:
        app.close()
    raise SystemExit(code)
