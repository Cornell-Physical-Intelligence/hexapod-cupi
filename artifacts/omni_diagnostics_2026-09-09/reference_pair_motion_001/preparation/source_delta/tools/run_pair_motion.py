#!/usr/bin/env python3
"""Bounded contact-aware opposing-pair motion; newfour-support contract, no PPO."""
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
from screen_contract import OPTIONS, WAVE, STARTUP, save, digest
from pair_motion_contract import CASES, PROTOCOL, preflight, STARTUP_CONTROLS, MOTION_START, MOTION_END

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--package', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--geometry-reference', type=Path, required=True)
parser.add_argument('--variant', default='f050_t060')
parser.add_argument('--stance-index', type=int, default=0)
parser.add_argument('--mode', choices=('paired_motion',), required=True)
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
from pair_motion_metrics import score_pair_motion
from pair_motion_measurements import measured_support, check_sensor_row, torque_window
from sensor_freshness import ContactFreshness, verify_installed_sources
import warp as wp
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent/'paired_runtime'))
from pair_motion import PairContactReference
from canonical_stance_startup import CanonicalStanceStartup
from physics_substeps import PhysicsSubstepRecorder


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


def save_runtime_json(path, value):
    """Normalize NumPy values, retaining the existing strict finite-JSON writer."""
    save(path, serializable(value))


def save_failure_state(path, state, exc, traceback_text, controls):
    """Always retain a strict failed receipt even if its computed gate is malformed."""
    failed = dict(state, status='failed', error=f'{type(exc).__name__}: {exc}',
                  traceback=traceback_text, control_steps=int(controls), finished_unix=time.time())
    try:
        payload = serializable(failed)
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as serialization_error:
        payload = dict(status='failed', error=failed['error'], traceback=traceback_text,
                       control_steps=int(controls), finished_unix=failed['finished_unix'],
                       finalization_serialization_error=repr(serialization_error),
                       invalid_state_repr=repr(failed), gate=None, gate_serialization_valid=False,
                       policy_training_started=False, stage2_complete=False, terrain_qualified=False)
    save(path, payload)
    return payload


def main():
    env = None
    physics_recorder = None
    state = dict(status='initializing', mode=args.mode, started_unix=time.time(), identity=identity,
                 runtime_binding=RUNTIME, policy_training_started=False, stage2_complete=False,
                 terrain_qualified=False, scope='fullC zero-residual bounded contact-physics screen')
    samples, references, support_checks = [], [], []
    failure = None
    try:
        env, manifest, plan, record, stance, layout, asset_audit, controller_contract = build_reference_environment(args, OPTIONS)
        state.update(asset_audit=asset_audit, controller=controller_contract, layout=layout)
        save_runtime_json(args.output / 'state.json', state)
        points_local = np.asarray(geometry_reference['stance']['toes'], dtype=float)
        if tuple(geometry_reference['joint_names']) != layout['joint_names_leg_major']:
            raise ValueError('Reference toe and joint leg-order contract differs')
        env.reset(seed=0)
        env.episode_length_buf.zero_()
        state['installed_sensor_sources']=verify_installed_sources()
        freshness=ContactFreshness(env, wp.to_torch)
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
            'canonical_hold_before_scoring_s':STARTUP['canonical_hold_before_scoring_s']}
        save_runtime_json(args.output / 'startup_reference.json',startup_contract)
        state['startup_reference'] = startup_contract
        clock = {'step': 0}
        generator = None

        def capture():
            fresh=freshness.read_after_normal_updates(8)
            row = capture_measured_state(env, layout, points_local, time_s=(clock['step']+1)*env.step_dt)
            row.update(sensor_all_valid=array(fresh['all_sensors_valid']),sensor_contact_valid=array(fresh['contact_valid']),
                sensor_age_s=array(fresh['sensor_age_s']),sensor_outdated=array(fresh['sensor_outdated']),
                sensor_timestamp_s=array(fresh['sensor_timestamp_s']),sensor_last_update_s=array(fresh['sensor_last_update_s']),
                sensor_expected_timestamp_s=array(fresh['expected_timestamp_s']))
            target = env.reference_residual_target
            if target is None:
                raise RuntimeError('No actual target result at pre-reset capture')
            row.update({key: array(value) for key, value in target.items()})
            return row

        state['status'] = 'running'
        save_runtime_json(args.output / 'state.json', state)
        physics_recorder = PhysicsSubstepRecorder(env)
        with physics_recorder, pre_reset_capture(env, capture) as captured:
            for step in range(args.steps):
                clock['step'] = step
                requested = np.zeros(3)
                if step >= STARTUP_CONTROLS:
                    if generator is None:
                        generator = PairContactReference(layout['joint_names_runtime'])
                        reset_ref = generator.reset(samples[-1])
                        if not np.array_equal(reset_ref['q_ref'], samples[-1]['joint_target_rad']):
                            raise RuntimeError('Generator reset changed loaded standing target/preload')
                        references.append({'physical_step': step, 'reset': serializable(reset_ref)})
                    if MOTION_START <= step < MOTION_END:
                        requested[:] = CASES[args.case]
                    ref = generator.step(samples[-1], requested, dt=env.step_dt)
                    references.append({'physical_step': step, 'result': serializable(ref)})
                    if not np.asarray(ref['valid']).all():
                        failure = 'Reference rejected before target emission: ' + str(ref.get('failure_reason'))
                        break
                    env.set_reference_targets(ref['q_ref'], ref['v_ref'], ref['a_ref'], ref['valid'])
                else:
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
                    2 if generator is not None else (0 if step+1 < startup.steps else 1),dtype=np.int8)
                row['reference_admitted_target_twist'] = np.asarray(ref['admitted_target_command'])[None].copy() if generator is not None else np.zeros((1,3))
                row['reference_filtered_twist'] = np.asarray(ref['admitted_command'])[None].copy() if generator is not None else np.zeros((1,3))
                row['actual_executed_body_navigation_twist'] = np.stack((-row['velocity_body_mps'][:,1], row['velocity_body_mps'][:,0], row['gyro_body_rad_s'][:,2]),axis=-1)
                samples.append(row)
                physics_recorder.end_control(row)
                check_sensor_row(row)
                if not all(torch.isfinite(v).all() for v in obs.values()) or not torch.isfinite(reward).all():
                    failure = 'Nonfinite observation or reward'
                    break
                if term.any() or trunc.any():
                    failure = 'Physical terminal event; last sample is pre-reset; no continued run'
                    break
                if step >= STARTUP_CONTROLS:
                    # Inspect all8recorded torques and finalsupport before another target.
                    try:
                        torques=torque_window(physics_recorder.rows[-8:])
                        support=measured_support(generator,row)
                        support_checks.append({'physical_step':step,'support':support,'torque':torques})
                    except ValueError as error:
                        support_checks.append({'physical_step':step,'failure':str(error)})
                        failure=str(error)
                        break
                if (step+1) % 100 == 0:
                    state['control_steps'] = step+1
                    save_runtime_json(args.output / 'state.json', state)
                    print(f'REFERENCE_SCREEN {args.mode} {step+1}/{args.steps}', flush=True)

        state['physics_substep_review'] = physics_recorder.export(args.output)
        data = {key: np.stack([row[key] for row in samples]) for key in samples[0]} if samples else {}
        if samples:
            np.savez_compressed(args.output / 'trace.npz', **data,
                                joint_names=np.asarray(layout['joint_names_runtime']), legs=np.asarray(('lf','lm','lr','rf','rm','rr')))
        save_runtime_json(args.output / 'reference_states.json', references)
        save_runtime_json(args.output / 'paired_poststep_checks.json',support_checks)
        state['failure']=failure
        gate = None
        if len(samples) > 200:
            gate = score_pair_motion(data,references,physics_recorder.data(),support_checks,
                joint_names=layout['joint_names_runtime'],failure=failure,dt=env.step_dt)
        passed = bool(failure is None and len(samples)==args.steps and gate and gate['passed'])
        state.update(status='completed' if passed else 'rejected', gate=gate, failure=failure,
                     control_steps=len(samples), finished_unix=time.time())
        save_runtime_json(args.output / 'state.json', state)
        return 0 if passed else 1
    except Exception as exc:
        failure_traceback = traceback.format_exc()
        if physics_recorder is not None and physics_recorder.rows:
            try:
                state['physics_substep_review'] = physics_recorder.export(args.output)
            except Exception as export_error:
                state['physics_substep_export_error'] = repr(export_error)
        if samples:
            try:
                np.savez_compressed(args.output / 'partial_trace.npz', **{key: np.stack([row[key] for row in samples]) for key in samples[0]})
            except Exception as export_error:
                state['partial_trace_export_error'] = repr(export_error)
        try:
            save_runtime_json(args.output / 'reference_states.json', references)
        except Exception as export_error:
            state['reference_states_export_error'] = repr(export_error)
        try:
            save_runtime_json(args.output / 'paired_poststep_checks.json',support_checks)
        except Exception as export_error:
            state['paired_checks_export_error']=repr(export_error)
        state = save_failure_state(args.output / 'state.json', state, exc, failure_traceback, len(samples))
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
