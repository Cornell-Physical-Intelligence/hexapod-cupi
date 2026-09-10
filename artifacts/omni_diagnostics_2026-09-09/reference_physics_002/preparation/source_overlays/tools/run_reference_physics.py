#!/usr/bin/env python3
"""Bounded full-C zero-residual standing and measured-contact wave proof only."""
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
from screen_contract import OPTIONS, WAVE, STARTUP, preflight, save, digest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--package', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--geometry-reference', type=Path, required=True)
parser.add_argument('--variant', default='f050_t060')
parser.add_argument('--stance-index', type=int, default=0)
parser.add_argument('--mode', choices=('standing', 'wave'), required=True)
parser.add_argument('--num-envs', type=int, required=True)
parser.add_argument('--steps', type=int, required=True)
parser.add_argument('--admission', type=Path)
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
from screen_metrics import standing_screen, physical_metrics, measured_flight_touchdowns, measured_progress
from canonical_stance_startup import CanonicalStanceStartup


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
    state = dict(status='initializing', mode=args.mode, started_unix=time.time(), identity=identity,
                 runtime_binding=RUNTIME, policy_training_started=False, stage2_complete=False,
                 terrain_qualified=False, scope='fullC zero-residual bounded contact-physics screen')
    samples, references = [], []
    failure = None
    try:
        env, manifest, plan, record, stance, layout, asset_audit, controller_contract = build_reference_environment(args, OPTIONS)
        state.update(asset_audit=asset_audit, controller=controller_contract, layout=layout)
        save(args.output / 'state.json', state)
        points_local = np.asarray(geometry_reference['stance']['toes'], dtype=float)
        if tuple(geometry_reference['joint_names']) != layout['joint_names_leg_major']:
            raise ValueError('Reference toe and joint leg-order contract differs')
        env.reset(seed=0)
        env.episode_length_buf.zero_()
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
        save(args.output / 'startup_reference.json',startup_contract)
        state['startup_reference'] = startup_contract
        clock = {'step': 0}
        generator = None

        def capture():
            row = capture_measured_state(env, layout, points_local, time_s=(clock['step']+1)*env.step_dt)
            target = env.reference_residual_target
            if target is None:
                raise RuntimeError('No actual target result at pre-reset capture')
            row.update({key: array(value) for key, value in target.items()})
            return row

        state['status'] = 'running'
        save(args.output / 'state.json', state)
        with pre_reset_capture(env, capture) as captured:
            for step in range(args.steps):
                clock['step'] = step
                requested = np.zeros(3)
                if args.mode == 'wave' and step >= WAVE['settle_steps']:
                    if generator is None:
                        from wave_reference import WaveContactReference
                        generator = WaveContactReference(layout['joint_names_runtime'])
                        reset_ref = generator.reset(samples[-1])
                        if not np.array_equal(reset_ref['q_ref'], samples[-1]['joint_target_rad']):
                            raise RuntimeError('Generator reset changed loaded standing target/preload')
                        references.append({'physical_step': step, 'reset': serializable(reset_ref)})
                    if step < WAVE['settle_steps'] + WAVE['forward_steps']:
                        requested[0] = WAVE['requested_forward_mps']
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
                with torch.inference_mode():
                    obs, reward, term, trunc, _ = env.step(zero)
                row = require_single_pre_reset_sample(captured, before, term, trunc)
                row['requested_command'] = np.broadcast_to(requested, (args.num_envs, 3)).copy()
                row['reference_source_code'] = np.full(args.num_envs,
                    2 if generator is not None else (0 if step+1 < startup.steps else 1),dtype=np.int8)
                samples.append(row)
                if not all(torch.isfinite(v).all() for v in obs.values()) or not torch.isfinite(reward).all():
                    failure = 'Nonfinite observation or reward'
                    break
                if term.any() or trunc.any():
                    failure = 'Physical terminal event; last sample is pre-reset; no continued run'
                    break
                if args.mode == 'wave' and step >= WAVE['settle_steps']:
                    unwanted = any(np.any(row[name]) for name in ('shaft_contact', 'coxa_contact', 'femur_contact', 'base_contact'))
                    if np.abs(row['computed_torque_nm']).max() > 1.6:
                        failure = 'Wave requested torque exceeds1.6Nm; no physical reference admission'
                        break
                    if unwanted or row['distal_contact'].sum(-1).min() < 5:
                        failure = 'Wave lost required five-foot support or made nonfoot contact'
                        break
                if (step+1) % 100 == 0:
                    state['control_steps'] = step+1
                    save(args.output / 'state.json', state)
                    print(f'REFERENCE_SCREEN {args.mode} {step+1}/{args.steps}', flush=True)

        data = {key: np.stack([row[key] for row in samples]) for key in samples[0]} if samples else {}
        if samples:
            np.savez_compressed(args.output / 'trace.npz', **data,
                                joint_names=np.asarray(layout['joint_names_runtime']), legs=np.asarray(('lf','lm','lr','rf','rm','rr')))
        save(args.output / 'reference_states.json', references)
        gate = None
        if len(samples) > 200:
            if args.mode == 'standing':
                gate = standing_screen(data)
            else:
                gate = physical_metrics(data)
                events = measured_flight_touchdowns(data, start_step=WAVE['settle_steps'])
                complete = [event for event in events if event['confirmed_measured_touchdown']
                            and event['measured_reference_point_lift_m'] >= .002]
                movement = slice(WAVE['settle_steps'], min(len(samples), WAVE['settle_steps']+WAVE['forward_steps']))
                actual_forward = float(-data['velocity_body_mps'][movement, :, 1].mean())
                final_ref = references[-1].get('result', {}) if references else {}
                final_state = final_ref.get('state', {})
                progress = None
                if len(samples) >= WAVE['settle_steps']+WAVE['forward_steps']:
                    progress = measured_progress(data, start_step=WAVE['settle_steps']-1,
                        end_step=WAVE['settle_steps']+WAVE['forward_steps']-1, dt=env.step_dt)
                quiet = None
                quiet_time = final_state.get('reference_quiet_time_s')
                quiet_start = None if quiet_time is None else int(np.ceil(quiet_time/env.step_dt))+100
                if quiet_start is not None and len(samples) >= quiet_start+500:
                    from omni_quiet_review import quiet_metrics, QUIET_GATES
                    if np.any(data['requested_command'][quiet_start:]):
                        raise RuntimeError('Quiet window includes a nonzero requested command')
                    quiet = quiet_metrics(data, 0, quiet_start, list(layout['joint_names_runtime']), env.step_dt)
                    quiet['bounds'] = QUIET_GATES
                gate.update(kind='single_leg_wave_bounded_physics_screen_not_stage2', measured_flight_events=events,
                    completed_measured_leg_indices=sorted(set(event['leg_index'] for event in complete)),
                    moving_window_forward_mps=actual_forward,
                    generator_confirmed_touchdowns=final_state.get('confirmed_touchdowns', 0),
                    generator_final_mode=final_state.get('mode'),
                    independent_link_progress=progress, final_quiet_stop_window=quiet,
                    passed=bool(failure is None and len(samples)==args.steps
                        and gate['original_basic_standing_physics_pass']
                        and gate['post_settle_max_requested_torque_nm'] <= 1.6
                        and gate['post_settle_min_distal_support_count'] >= 5
                        and len(set(event['leg_index'] for event in complete)) == 6
                        and actual_forward >= .5*WAVE['requested_forward_mps']
                        and progress is not None
                        and progress['measured_forward_displacement_m'] >= .5*WAVE['requested_forward_mps']*progress['duration_s']
                        and progress['displacement_integral_difference_m'] <= .005
                        and quiet is not None and quiet['pass'] and quiet['window_duration_s'] >= 10.
                        and gate['max_reference_to_executable_lag_rad'] <= 1e-12
                        and final_state.get('mode') == 'reference_quiet_hold'))
        passed = bool(failure is None and len(samples)==args.steps and gate and gate['passed'])
        state.update(status='completed' if passed else 'rejected', gate=gate, failure=failure,
                     control_steps=len(samples), finished_unix=time.time())
        save(args.output / 'state.json', state)
        if args.mode == 'standing':
            save(args.output / 'admission.json', state)
        return 0 if passed else 1
    except Exception as exc:
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
