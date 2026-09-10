#!/usr/bin/env python3
"""External progress recording for the byte-immutable candidate003 PPO source.

No training, reference injection, source edits or qualification claims. Root must
schedule this bounded GPU job through the existing Spark workload controls.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from recording_contract import (IDENTITY, SOURCE_SHA256, SOURCE_MANIFEST_SHA256, DT, FPS, SEQUENCE, digest,
    save_json, verify_source, verify_checkpoint_bytes, verify_pilot_prerequisites, verify_pilot_run, verify_admitted_package,
    schedule, actual_pose_xy_heading, assert_event_snapshot, check_observations,
    external_source_hashes)


def parser_base():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--runner-smoke', type=Path, required=True)
    parser.add_argument('--pilot-state', type=Path, required=True)
    parser.add_argument('--probe-campaign', type=Path, required=True)
    parser.add_argument('--study-tree-receipt', type=Path, required=True)
    parser.add_argument('--pilot-inputs-receipt', type=Path, required=True)
    parser.add_argument('--checkpoint-label', choices=('scratch_initial', 'scratch_50_update_final'), required=True)
    parser.add_argument('--seed', type=int, default=37057)
    parser.add_argument('--preflight-only', action='store_true')
    return parser


def preflight(args):
    root = verify_source(args.source_root)
    args.source_root = root
    for key in ('package', 'checkpoint', 'admission', 'calibration', 'runner_smoke', 'pilot_state',
                'probe_campaign', 'study_tree_receipt', 'pilot_inputs_receipt', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    admitted_package = verify_admitted_package(args)
    if args.output.exists():
        raise FileExistsError('Use a fresh recording output directory')
    if args.output == root or root in args.output.parents:
        raise ValueError('Recording output cannot be inside frozen training source')
    if args.checkpoint_label == 'scratch_50_update_final' and args.checkpoint.name != 'final.pt':
        raise ValueError('Final pilot label requires the immutable final.pt artifact')
    if args.checkpoint_label == 'scratch_initial' and args.checkpoint.name != 'initial.pt':
        raise ValueError('Initial label requires initial.pt; no stand-smoke relabeling')
    sidecar = verify_checkpoint_bytes(args.checkpoint, args.checkpoint_sha256)
    verify_pilot_prerequisites(args.calibration, args.runner_smoke)
    pilot_state = verify_pilot_run(args.pilot_state, args.checkpoint_sha256, args.checkpoint_label)
    sys.path.insert(0, str(root / 'tools'))
    from candidate_runner import preflight_candidate
    if Path(sys.modules['candidate_runner'].__file__).resolve() != root / 'tools/candidate_runner.py':
        raise RuntimeError('Candidate runner imported from a different source')
    # A recording is a separate use of the existing strict evaluation admission,
    # not a new mode accepted by, or modification to, the frozen training CLI.
    args.mode = 'evaluate'
    args.variant = IDENTITY['variant']
    args.stance_index = IDENTITY['stance_index']
    args.iterations = None
    identity, config = preflight_candidate(args)
    if identity != IDENTITY:
        raise ValueError('Actual frozen preflight returned a different identity')
    provenance = {
        'kind': 'candidate003_early_progress_video_not_qualification',
        'checkpoint_label': args.checkpoint_label,
        'training_identity': identity,
        'training_source_manifest_sha256': SOURCE_MANIFEST_SHA256,
        'admitted_package': admitted_package,
        'checkpoint_path': str(args.checkpoint),
        'checkpoint_sha256': args.checkpoint_sha256,
        'checkpoint_sidecar_sha256': digest(args.checkpoint.with_suffix('.pt.json')),
        'admission_sha256': digest(args.admission),
        'calibration_sha256': digest(args.calibration),
        'runner_smoke_sha256': digest(args.runner_smoke),
        'pilot_state_sha256': digest(args.pilot_state),
        'pilot_updates': pilot_state['iterations'],
        'recording_source_hashes': external_source_hashes(Path(__file__).resolve().parent),
        'strict_tensor_readback_completed': False,
        'stage2_complete': False, 'qualification_performed': False,
        'command_sequence': [dict(segment=n, seconds=d, forward_left_yaw=list(c)) for n, d, c in SEQUENCE],
        'dt_s': DT, 'fps': FPS, 'seed': args.seed,
        'quaternion_convention': 'raw Isaac Lab XYZW; explicit WXYZ conversion retained separately',
        'rendering_scope': 'one full19-body18-joint6-foot C study robot; unchanged dynamics/controller/rewards',
    }
    return identity, config, provenance


def annotate(raw, row, admitted, measured, frames, provenance, *, event=None):
    from PIL import Image, ImageDraw
    frame = Image.fromarray(np.asarray(raw[:, :, :3], dtype=np.uint8))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((8, 8, frame.width - 8, 122), fill=(14, 20, 29))
    requested = row['requested_command']
    lines = [
        f"CANDIDATE003 PPO | {provenance['checkpoint_label']} | {row['segment']} | t={row['time_start_s'] + DT:.2f}s",
        f"REQUEST forward={requested[0]:+.3f} m/s  left={requested[1]:+.3f} m/s  yaw={requested[2]:+.3f} rad/s",
        f"ACTOR COMMAND forward={admitted[0]:+.3f}  left={admitted[1]:+.3f}  yaw={admitted[2]:+.3f} | MEASURED {measured[0]:+.3f}, {measured[1]:+.3f}, {measured[2]:+.3f}",
        'Actual deterministic PPO actions / full contact physics / command-only, no pose correction / camera follows robot',
        f"Checkpoint {provenance['checkpoint_sha256'][:16]} | EARLY PROGRESS ONLY - STAGE 2 INCOMPLETE",
    ]
    for i, text in enumerate(lines):
        draw.text((18, 16 + i * 20), text, fill='white')
    if event:
        draw.rectangle((8, 136, frame.width - 8, 184), fill=(150, 15, 20))
        draw.text((18, 150), event + ' | last valid rendered frame; no post-reset continuation', fill='white')
    return np.asarray(frame)


def reference_schedule(initial_pose, rows, device):
    """Draw expected command integration only; never impose it on the robot."""
    import torch
    from omni_flat_math import integrate_body_twist, slew_commands
    pose = torch.tensor(initial_pose, device=device, dtype=torch.float32).reshape(1, 3)
    command = torch.zeros_like(pose)
    poses = []
    commands = []
    for row in rows:
        # Env scores current command, then slews it after rewards for next action.
        pose = integrate_body_twist(pose, command, DT)
        poses.append(pose[0].clone())
        commands.append(command[0].clone())
        target = torch.tensor(row['requested_command'], device=device).reshape(1, 3)
        command = slew_commands(command, target, DT)
    return torch.stack(poses).cpu().numpy(), torch.stack(commands).cpu().numpy()


def draw_live_command(drawing, actual, requested, label):
    """Reuse only non-colliding USD mesh helpers for on-ground commands."""
    x, y, heading = map(float, actual)
    forward, left, yaw = requested
    drawing.text('LiveCommand', f'{label}: FWD {forward:+.2f}  LEFT {left:+.2f} M/S  YAW {yaw:+.2f} RAD/S',
                 x, y + .48, color=(.8, .95, 1.), pitch=.0025)
    travel = []
    if math.hypot(forward, left) > 1e-8:
        travel = [(x, y, heading + math.atan2(left, forward))]
    drawing.arrows('LiveTravelArrow', travel, (.08, .48, 1.), length=.20, z=.010)
    if abs(yaw) > 1e-8:
        angles = np.linspace(heading, heading + math.copysign(math.pi / 2, yaw), 14)
        turn = np.column_stack((x + .34*np.cos(angles), y + .34*np.sin(angles)))
        tip_heading = float(angles[-1] + math.copysign(math.pi / 2, yaw))
        yaw_arrow = [(float(turn[-1, 0]), float(turn[-1, 1]), tip_heading)]
    else:
        turn = np.empty((0, 2))
        yaw_arrow = []
    drawing.line('LiveYawArc', turn, (1., .65, .05), width=.006, z=.011)
    drawing.arrows('LiveYawArrow', yaw_arrow, (1., .65, .05), length=.065, z=.011)


def initial_rgb_frame(env, *, max_attempts=24):
    """Warm the lazily attached RGB annotator with renders only, no physics."""
    attempts = []
    deadline = time.monotonic() + 30.
    for index in range(max_attempts):
        raw = env.render()
        valid = raw is not None and raw.ndim == 3 and np.std(raw) >= 1
        attempts.append(dict(attempt=index + 1, shape=None if raw is None else list(raw.shape),
                             std=None if raw is None or raw.size == 0 else float(np.std(raw)), valid=bool(valid)))
        if valid:
            return raw, dict(render_attempts=attempts, physics_steps=0, ready=True)
        if time.monotonic() >= deadline:
            break
        # DirectRLEnv's first RGB call creates the render product/annotator.
        # A later render is required before data exist; this API advances no physics.
        env.sim.render()
    raise RuntimeError('RGB annotator did not become ready after bounded render-only warmup: ' + json.dumps(attempts))


def record(env, runner, args, provenance):
    import torch
    import imageio.v2 as imageio
    from tensordict import TensorDict
    from omni_path_demo import GroundDrawing
    from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view
    rows = schedule()
    snapshots = []
    raw_actions = []
    telemetry = []
    frames = 0
    terminal_event = None
    runner.alg.eval_mode()
    policy = runner.get_inference_policy(device=env.device)
    env.omni_diagnostic_enabled = True
    with torch.inference_mode():
        env.reset(seed=args.seed)
        env.episode_length_buf.zero_()
        env.set_evaluation_targets(torch.zeros((1, 3), device=env.device))
        data = env._robot.data
        initial = actual_pose_xy_heading(data.root_pos_w.torch[0].cpu().numpy(),
                                        data.root_quat_w.torch[0].cpu().numpy(), convention='xyzw')
        poses, commands = reference_schedule(initial, rows, env.device)
        drawing = GroundDrawing()
        drawing.reference(poses, commands, 'CANDIDATE003 PPO / REQUESTED TWIST SEQUENCE / NOT QUALIFICATION')
        trail = []
        p = data.root_pos_w.torch[0].cpu().tolist()
        set_kit_renderer_camera_view(eye=(p[0] + .95, p[1] + 1.25, 1.20), target=(p[0], p[1] - .05, .12))
        last_raw, warmup = initial_rgb_frame(env)
        provenance['initial_rgb_warmup'] = warmup
        save_json(args.output / 'rgb_warmup.json', warmup)
        with imageio.get_writer(str(args.output / 'rollout.mp4'), fps=FPS, codec='libx264', quality=8) as writer:
            for step, row in enumerate(rows):
                requested = torch.tensor(row['requested_command'], device=env.device, dtype=torch.float32).reshape(1, 3)
                env.set_evaluation_targets(requested)
                admitted = env._commands[0].detach().cpu().numpy().copy()
                obs = env._get_observations()
                check_observations(obs)
                actions = policy(TensorDict(obs, batch_size=[1]))
                if tuple(actions.shape) != (1, 18) or not bool(torch.isfinite(actions).all()):
                    raise RuntimeError('Nonfinite or wrong-size deterministic candidate action')
                raw_actions.append(actions.detach().cpu().numpy().copy())
                _, _, terminated, truncated, _ = env.step(actions)
                sample = {key: value.copy() for key, value in env.omni_diagnostic_sample.items()}
                event = assert_event_snapshot(sample, terminated.cpu().numpy(), truncated.cpu().numpy())
                snapshots.append(sample)
                heading_observable = True
                try:
                    actual = actual_pose_xy_heading(sample['position_world_m'][0], sample['quaternion_world_xyzw'][0], convention='xyzw')
                except ValueError:
                    # A terminal fall can make projected heading undefined.
                    # Keep its position and event evidence; do not invent a
                    # recorded heading or render the auto-reset articulation.
                    if not event or not np.isfinite(sample['position_world_m']).all():
                        raise
                    actual = np.r_[sample['position_world_m'][0, :2], 0.]
                    heading_observable = False
                # Use pre-reset telemetry, never a freshly reset articulation state.
                measured = [float(sample['velocity_navigation_mps'][0, 0]),
                            float(sample['velocity_navigation_mps'][0, 1]),
                            float(sample['gyro_navigation_rad_s'][0, 2])]
                trail.append(actual[:2].copy())
                telemetry.append({**row, 'time_end_s': (step + 1) * DT,
                    'actor_observed_command': admitted.tolist(),
                    'actual_pose_xy_heading': actual.tolist() if heading_observable else [float(actual[0]), float(actual[1]), None],
                    'body_heading_observable': heading_observable,
                    'command_integrated_reference_pose': poses[step].tolist(),
                    'measured_forward_left_yaw': measured,
                    'position_reference_error_m': float(np.linalg.norm(actual[:2] - poses[step, :2])),
                    'terminated': bool(terminated[0]), 'truncated': bool(truncated[0])})
                if event:
                    terminal_event = dict(step=step, segment=row['segment'],
                                          terminated=bool(terminated[0]), truncated=bool(truncated[0]))
                    writer.append_data(annotate(last_raw, row, admitted, measured, frames, provenance,
                                               event='TERMINAL EVENT - RECORDING STOPPED'))
                    frames += 1
                    break
                if step % 2 == 1:
                    if step % 10 == 1:
                        drawing.line('ActualTrail', np.array(trail), (1., .25, .08), width=.007, z=.008)
                        draw_live_command(drawing, actual, row['requested_command'], row['segment'])
                    pos = sample['position_world_m'][0]
                    set_kit_renderer_camera_view(eye=(float(pos[0]) + .95, float(pos[1]) + 1.25, 1.20),
                                                 target=(float(pos[0]), float(pos[1]) - .05, .12))
                    last_raw = env.render()
                    if last_raw is None or last_raw.ndim != 3 or np.std(last_raw) < 1:
                        raise RuntimeError('Missing or blank Isaac recording frame')
                    frame = annotate(last_raw, row, admitted, measured, frames, provenance)
                    writer.append_data(frame)
                    if frames == 0:
                        imageio.imwrite(str(args.output / 'first_frame.png'), frame)
                    if step + 1 == len(rows):
                        imageio.imwrite(str(args.output / 'last_frame.png'), frame)
                    frames += 1
                if (step + 1) % 100 == 0:
                    save_json(args.output / 'progress.json', dict(complete=False, control_steps=step + 1,
                              frames=frames, checkpoint_sha256=args.checkpoint_sha256, stage2_complete=False))
                    print(f'RECORD {step+1}/{len(rows)} {row["segment"]}', flush=True)
    env.omni_diagnostic_enabled = False
    np.savez_compressed(args.output / 'trace.npz',
        **{key: np.stack([snapshot[key] for snapshot in snapshots]) for key in snapshots[0]},
        deterministic_actor_action=np.stack(raw_actions),
        joint_names=np.array(env._robot.joint_names),
        segment_index=np.array([row['segment_index'] for row in telemetry]),
        time_s=np.arange(1, len(snapshots) + 1) * DT)
    verify_source(args.source_root)
    verify_checkpoint_bytes(args.checkpoint, args.checkpoint_sha256)
    if verify_admitted_package(args) != provenance['admitted_package']:
        raise RuntimeError('Admitted package/receipts changed during recording')
    if external_source_hashes(Path(__file__).resolve().parent) != provenance['recording_source_hashes']:
        raise RuntimeError('Recording code changed during the run')
    report = {**provenance, 'complete': terminal_event is None and len(snapshots) == len(rows),
        'planned_control_steps': len(rows), 'recorded_control_steps': len(snapshots),
        'frames': frames, 'playback_duration_s': frames / FPS, 'physics_duration_s': len(snapshots) * DT,
        'terminal_event': terminal_event, 'telemetry': telemetry, 'trace_file': 'trace.npz',
        'source_and_checkpoint_reverified_after_recording': True,
        'video_sha256': digest(args.output / 'rollout.mp4'),
        'notes': ['Single continuous episode; no segment resets or reference-driven body constraints.',
                  'Deterministic actor output retains matched observation noise and controller semantics.',
                  'A terminal event ends recording; final event frame is the last valid render with an explicit banner.',
                  'Blue/gold show command-integrated travel/heading; orange is actual pre-reset robot trail.',
                  'Short quiet windows are visual checks, not the full quiet/stop qualification.']}
    save_json(args.output / 'video.json', report)
    return report


def main():
    parser = parser_base()
    early, _ = parser.parse_known_args()
    identity, candidate_config, provenance = preflight(early)
    if early.preflight_only:
        print(json.dumps(provenance, indent=2))
        return
    from c_study_runtime import bootstrap_c_study_runtime
    runtime = bootstrap_c_study_runtime()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument('-h', '--help', action='help')
    args = parser.parse_args()
    # The checked values are explicitly retained; no source/identity field is rewritten.
    for key in ('source_root', 'package', 'output', 'checkpoint', 'admission', 'calibration', 'runner_smoke', 'pilot_state',
                'probe_campaign', 'study_tree_receipt', 'pilot_inputs_receipt',
                'mode', 'variant', 'stance_index', 'iterations'):
        setattr(args, key, getattr(early, key))
    args.enable_cameras = True
    args.output.mkdir(parents=True, exist_ok=False)
    provenance['runtime_binding'] = runtime
    provenance['started_unix'] = time.time()
    save_json(args.output / 'provenance.json', provenance)
    app = env = None
    try:
        app = AppLauncher(args).app
        from recording_env import build_env_and_runner
        from candidate_runner import load_candidate
        env, runner, plan, audit = build_env_and_runner(args, candidate_config)
        loaded = load_candidate(runner, args.checkpoint, candidate_config, identity)
        provenance.update(strict_tensor_readback_completed=True, controller=env.candidate_contract,
                          inertia_audit=audit, observed_joint_order=list(env._robot.joint_names),
                          checkpoint_loader_result=loaded)
        save_json(args.output / 'provenance.json', provenance)
        result = record(env, runner, args, provenance)
        print('RECORDING COMPLETE' if result['complete'] else 'RECORDING STOPPED ON TERMINAL EVENT', flush=True)
    except Exception as exc:
        save_json(args.output / 'failure.json', dict(error=repr(exc), checkpoint_sha256=args.checkpoint_sha256,
                  stage2_complete=False, qualification_performed=False, time_unix=time.time()))
        raise
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == '__main__':
    main()
