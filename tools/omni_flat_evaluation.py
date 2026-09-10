"""Held-out command matrix, uninterrupted transitions and actual Isaac video."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from omni_flat_math import (evaluation_scenarios, transition_sequence, scenario_gate,
                            trajectory_command, integrate_body_twist)


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def measurements(env):
    d = env._robot.data
    v = env._vector_in_command_frame(d.root_lin_vel_b.torch)
    distal, shaft, slip = env._get_foot_contact_state()
    coxa = torch.linalg.vector_norm(env._coxa_contact_sensor.data.net_forces_w_history.torch, dim=-1).amax(1) > 1
    femur = torch.cat([torch.linalg.vector_norm(s.data.net_forces_w_history.torch, dim=-1).amax(1) > 1
                       for s in env._femur_contact_sensors], 1)
    return {
        'velocity': v[:, :2], 'yaw': d.root_ang_vel_b.torch[:, 2],
        'tilt2': torch.acos((-d.projected_gravity_b.torch[:, 2]).clamp(-1, 1)).square(),
        'vertical2': d.root_lin_vel_w.torch[:, 2].square(),
        'saturation': (d.computed_torque.torch.abs() > 1.6).float().mean(-1),
        'requested_torque_max': d.computed_torque.torch.abs().amax(-1),
        'nonfoot': torch.cat((coxa, femur, shaft), 1).any(-1).float(),
        'power': (d.applied_torque.torch * d.joint_vel.torch).clamp_min(0).sum(-1),
        'slip': (slip * distal).sum(-1) / distal.sum(-1).clamp_min(1),
    }


@torch.inference_mode()
def evaluate_omni(env, runner, plan, output, checkpoint_sha):
    if "diagnostics" in plan["omni"]:
        from omni_diagnostics import evaluate_diagnostics
        return evaluate_diagnostics(env, runner, plan, output, checkpoint_sha)
    scenarios = evaluation_scenarios()
    if env.num_envs % len(scenarios):
        raise RuntimeError('Evaluation environments must contain equal replicas of all scenarios')
    replicas = env.num_envs // len(scenarios)
    targets = torch.tensor([s['command'] for s in scenarios], device=env.device).repeat_interleave(replicas, 0)
    policy = runner.get_inference_policy(device=env.device)
    env.omni_measurements_enabled = True
    report = {'complete': False, 'architecture': 'omni_history_direct_v1',
              'checkpoint_sha256': checkpoint_sha, 'replicas_per_scenario': replicas,
              'static_duration_s': 20., 'motion_metrics_omit_first_s': 2.,
              'measurement_capture': 'before_automatic_reset',
              'settling_exclusion': 'first_2_seconds_of_every_episode',
              'static': [], 'transitions': []}
    for seed in plan['omni']['evaluation_seeds']:
        env.reset(seed=seed); env.episode_length_buf.zero_()
        env.set_evaluation_targets(targets)
        terminations = torch.zeros(env.num_envs, device=env.device)
        truncations = torch.zeros_like(terminations)
        sums = {}; maxima = torch.zeros_like(terminations); counts = torch.zeros_like(terminations)
        for step in range(1000):
            obs = TensorDict(env._get_observations(), batch_size=[env.num_envs])
            _, _, term, trunc, _ = env.step(policy(obs))
            # A failed environment may reset; failures remain recorded, and it
            # receives the original scenario again rather than a random command.
            env.set_evaluation_targets(targets)
            terminations += term; truncations += trunc
            m = env.omni_measurement_sample
            if not all(torch.isfinite(x).all() for x in m.values()):
                raise RuntimeError('Nonfinite omni evaluation measurement')
            maxima = torch.maximum(maxima, m['requested_torque_max'])
            eligible = env.omni_measurement_age_s >= 2.
            m['planar_error'] = (m['velocity'] - targets[:, :2]).norm(dim=-1)
            m['yaw_error'] = (m['yaw'] - targets[:, 2]).abs()
            for k, v in m.items():
                if k not in sums: sums[k] = torch.zeros_like(v)
                mask = eligible.unsqueeze(-1) if v.ndim > eligible.ndim else eligible
                sums[k] += torch.where(mask, v, 0.)
            counts += eligible
        for i, scenario in enumerate(scenarios):
            ids = slice(i * replicas, (i + 1) * replicas)
            denominator = counts[ids].clamp_min(1)
            mean = lambda name: float((sums[name][ids] / denominator).mean())
            row = {**scenario, 'seed': seed, 'replicas': replicas,
                   'post_settle_samples_per_replica': counts[ids].int().cpu().tolist(),
                   'mean_velocity_mps': (sums['velocity'][ids] / denominator[:, None]).mean(0).cpu().tolist(),
                   'mean_yaw_rad_s': mean('yaw'), 'planar_error_mps': mean('planar_error'),
                   'yaw_error_rad_s': mean('yaw_error'), 'tilt_rms_deg': math.degrees(math.sqrt(mean('tilt2'))),
                   'vertical_velocity_rms_mps': math.sqrt(mean('vertical2')),
                   'torque_saturation_fraction': mean('saturation'),
                   'max_requested_torque_nm': float(maxima[ids].max()),
                   'nonfoot_fraction': mean('nonfoot'), 'positive_mechanical_power_w': mean('power'),
                   'contact_slip_mps': mean('slip'), 'terminations': int(terminations[ids].sum()),
                   'truncations': int(truncations[ids].sum())}
            row['pass'] = bool((counts[ids] > 0).all()) and scenario_gate(row)
            report['static'].append(row)
        save(Path(output)/'evaluation.json', report)
        print(f'OMNI_STATIC_DONE seed={seed} passed={sum(r["pass"] for r in report["static"])}', flush=True)

    # All segments share one physical episode; count errors during the ramp too.
    env.reset(seed=plan['omni']['evaluation_seeds'][0] + 999)
    env.episode_length_buf.zero_()
    # Physical forward is local -Y. Reference pose uses heading of that vector
    # projected into the world plane, not the URDF body's +X yaw convention.
    from isaaclab.utils.math import quat_apply
    forward = quat_apply(env._robot.data.root_quat_w.torch,
                         torch.tensor([0., -1., 0.], device=env.device).expand(env.num_envs, 3))
    reference_pose = torch.cat((env._robot.data.root_pos_w.torch[:, :2].clone(),
                                torch.atan2(forward[:, 1], forward[:, 0])[:, None]), -1)
    for name, duration, command in transition_sequence():
        target = torch.tensor(command, device=env.device).expand(env.num_envs, 3)
        env.set_evaluation_targets(target)
        total_planar = total_yaw = late_planar = late_yaw = 0.
        late_count = nterm = ntrunc = 0
        pose_error_sum = 0.
        n = round(duration/env.step_dt)
        for step in range(n):
            target = torch.tensor(trajectory_command(name, step*env.step_dt, duration, command),
                                  device=env.device).expand(env.num_envs, 3)
            env.set_evaluation_targets(target)
            observed_command = env._commands.clone()
            reference_pose = integrate_body_twist(reference_pose, observed_command, env.step_dt)
            obs = TensorDict(env._get_observations(), batch_size=[env.num_envs])
            _, _, term, trunc, _ = env.step(policy(obs))
            env.set_evaluation_targets(target)
            nterm += int(term.sum()); ntrunc += int(trunc.sum())
            m = env.omni_measurement_sample
            pose_error = (env.omni_measurement_position[:, :2] - reference_pose[:, :2]).norm(dim=-1)
            pose_error_sum += float(pose_error.mean())
            total_planar += float((m['velocity'] - observed_command[:, :2]).norm(dim=-1).mean())
            total_yaw += float((m['yaw'] - observed_command[:, 2]).abs().mean())
            if step >= n - round(1./env.step_dt):
                late_planar += float((m['velocity'] - target[:, :2]).norm(dim=-1).mean())
                late_yaw += float((m['yaw'] - target[:, 2]).abs().mean()); late_count += 1
        row = {'name': name, 'duration_s': duration, 'command': command, 'num_envs': env.num_envs,
               'terminations': nterm, 'truncations': ntrunc,
               'ramped_command_planar_error_mps': total_planar/n,
               'ramped_command_yaw_error_rad_s': total_yaw/n,
               'mean_feedforward_path_error_m': pose_error_sum/n,
               'endpoint_feedforward_path_error_p95_m': float(torch.quantile(pose_error, .95)),
               'last_second_target_planar_error_mps': late_planar/late_count,
               'last_second_target_yaw_error_rad_s': late_yaw/late_count}
        row['pass'] = nterm == 0 and ntrunc == 0 and row['last_second_target_planar_error_mps'] <= .03 and row['last_second_target_yaw_error_rad_s'] <= .08
        report['transitions'].append(row)
        save(Path(output)/'evaluation.json', report)
    report['complete'] = True
    report['all_scenarios_pass'] = all(r['pass'] for r in report['static'] + report['transitions'])
    save(Path(output)/'evaluation.json', report)
    env.omni_measurements_enabled = False
    return report


@torch.inference_mode()
def record_omni(env, runner, plan, output, checkpoint_sha):
    from omni_path_demo import record_path_demo
    return record_path_demo(env, runner, plan, output, checkpoint_sha)


@torch.inference_mode()
def record_twist_sequence(env, runner, plan, output, checkpoint_sha):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view
    policy = runner.get_inference_policy(device=env.device)
    env.reset(seed=plan['omni']['evaluation_seeds'][0]); env.episode_length_buf.zero_()
    if env.render() is None: raise RuntimeError('No Isaac RGB output')
    trajectory = []; step = 0; frames = 0
    with imageio.get_writer(str(Path(output)/'rollout.mp4'), fps=25, codec='libx264', quality=8) as writer:
        for name, duration, command in transition_sequence():
            target = torch.tensor(command, device=env.device).reshape(1, 3)
            env.set_evaluation_targets(target)
            for segment_step in range(round(duration/env.step_dt)):
                active_command = trajectory_command(name, segment_step*env.step_dt, duration, command)
                target = torch.tensor(active_command, device=env.device).reshape(1, 3)
                env.set_evaluation_targets(target)
                obs = TensorDict(env._get_observations(), batch_size=[1])
                _, _, term, trunc, _ = env.step(policy(obs)); env.set_evaluation_targets(target)
                d = env._robot.data; pos = d.root_pos_w.torch[0].cpu().tolist()
                velocity = env._vector_in_command_frame(d.root_lin_vel_b.torch)[0].cpu().tolist()
                step += 1
                trajectory.append({'time_s': step * env.step_dt, 'segment': name, 'target': active_command,
                                   'ramped_command': env._commands[0].cpu().tolist(), 'position_m': pos,
                                   'velocity_mps': velocity, 'yaw_rad_s': float(d.root_ang_vel_b.torch[0, 2]),
                                   'terminated': bool(term[0]), 'truncated': bool(trunc[0])})
                if step % 2 == 0:
                    set_kit_renderer_camera_view(eye=(pos[0]+.9, pos[1]+1., .72), target=(pos[0], pos[1]-.10, .13))
                    raw = env.render()
                    if raw is None or raw.ndim != 3 or float(np.std(raw)) < 1: raise RuntimeError('Blank Isaac frame')
                    frame = Image.fromarray(raw[:, :, :3]); draw = ImageDraw.Draw(frame)
                    draw.rectangle((8, 8, 600, 65), fill=(10, 18, 28))
                    label = ' / '.join(f'{x:+.2f}' for x in active_command)
                    draw.text((18, 18), f'Omni policy | {name} | target forward/left/yaw: {label}', fill='white')
                    draw.text((18, 40), f'Actual forward {velocity[0]:+.3f} m/s | left {velocity[1]:+.3f} m/s', fill='white')
                    writer.append_data(np.asarray(frame)); frames += 1
            print(f'OMNI_VIDEO segment={name} t={step * env.step_dt:.1f}', flush=True)
    report = {'complete': True, 'variant': 'f050_t060', 'architecture': 'omni_history_direct_v1',
              'checkpoint_sha256': checkpoint_sha, 'fps': 25, 'frames': frames,
              'duration_s': step * env.step_dt, 'trajectory': trajectory}
    save(Path(output)/'video.json', report)
    return report
