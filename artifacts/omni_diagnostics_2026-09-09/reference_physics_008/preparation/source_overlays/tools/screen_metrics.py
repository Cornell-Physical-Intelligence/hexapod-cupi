"""Independent bounded physics-screen metrics; not Stage2/terrain qualification."""
import numpy as np


def measured_progress(data, *, start_step, end_step, dt=.02):
    """Project actual root-link displacement and its velocity integral independently."""
    if not 0 <= start_step < end_step < len(data['position_world_m']):
        raise ValueError('Complete bounded progress interval required')
    axis = -data['rotation_world_from_body'][start_step, 0, :, 1].copy()
    axis[2] = 0.
    norm = np.linalg.norm(axis)
    if norm < .5:
        raise ValueError('Projected actual initial forward heading is undefined')
    axis /= norm
    positions = data['position_world_m'][start_step:end_step+1, 0]
    velocity = data['velocity_world_mps'][start_step:end_step+1, 0]
    displacement = positions[-1]-positions[0]
    integrated = .5*(velocity[:-1]+velocity[1:]).sum(axis=0)*dt
    return dict(duration_s=(end_step-start_step)*dt,
        measured_world_displacement_m=displacement.tolist(), integrated_link_velocity_displacement_m=integrated.tolist(),
        actual_initial_forward_axis_world=axis.tolist(),
        measured_forward_displacement_m=float(displacement @ axis),
        integrated_forward_displacement_m=float(integrated @ axis),
        displacement_integral_difference_m=float(np.linalg.norm(displacement-integrated)),
        interval_terminations=int(data['terminated'][start_step:end_step+1].sum()),
        interval_truncations=int(data['truncated'][start_step:end_step+1].sum()))


def physical_metrics(data, *, settle_steps=200, dt=.02):
    """Retain every failure; per-joint saturation cannot hide in an all-motor mean."""
    torque = np.asarray(data['computed_torque_nm'])
    if torque.ndim != 3 or torque.shape[-1] != 18 or len(torque) <= settle_steps:
        raise ValueError('Complete time/environment/18-motor trace after settling required')
    for key, value in data.items():
        if key != 'contact_point_world_m' and not np.isfinite(value).all():
            raise ValueError('Nonfinite screen trace: ' + key)
    selected = slice(settle_steps, None)
    saturation = (np.abs(torque[selected]) > 1.6).mean(axis=0)
    nonfoot = (data['coxa_contact'].any(-1) | data['femur_contact'].any(-1)
               | data['shaft_contact'].any(-1) | data['base_contact'])
    body = data['velocity_body_mps'][selected]
    supports = data['distal_contact'][selected].sum(-1)
    return {
        'control_steps': len(torque), 'num_envs': torque.shape[1], 'settle_steps': settle_steps,
        'control_dt_s': dt, 'post_settle_measured_duration_s': (len(torque)-settle_steps)*dt,
        'max_applied_torque_nm': float(np.abs(data['applied_torque_nm']).max()),
        'post_settle_max_requested_torque_nm': float(np.abs(torque[selected]).max()),
        'post_settle_joint_saturation_fraction': saturation.tolist(),
        'post_settle_max_joint_saturation_fraction': float(saturation.max()),
        'post_settle_all_motor_saturation_fraction': float(saturation.mean()),
        'post_settle_nonfoot_env_steps': int(nonfoot[selected].sum()),
        'post_settle_min_distal_support_count': int(supports.min()),
        'terminations': int(data['terminated'].sum()), 'truncations': int(data['truncated'].sum()),
        'min_root_height_m': float(data['position_world_m'][..., 2].min()),
        'mean_measured_forward_mps': float(-body[..., 1].mean()),
        'mean_measured_left_mps': float(body[..., 0].mean()),
        'max_reference_to_executable_lag_rad': float(np.abs(data['reference_to_executable_lag_rad']).max()),
        'max_target_cast_error_rad': float(np.abs(data['position_target_cast_error_rad']).max()),
        'original_basic_standing_physics_pass': bool(
            np.abs(data['applied_torque_nm']).max() <= 1.61 and saturation.mean() <= .005
            and not nonfoot[selected].any() and not data['terminated'].any()
            and not data['truncated'].any() and data['position_world_m'][..., 2].min() >= .055),
    }


def standing_screen(data, *, settle_steps=200, dt=.02):
    metrics = physical_metrics(data, settle_steps=settle_steps, dt=dt)
    metrics['kind'] = 'zero_residual_standing_physics_screen_no_policy_admission'
    metrics['passed'] = bool(metrics['original_basic_standing_physics_pass']
        and metrics['post_settle_max_joint_saturation_fraction'] <= .005
        and metrics['post_settle_min_distal_support_count'] == 6
        and metrics['max_reference_to_executable_lag_rad'] <= 1e-12
        and metrics['max_target_cast_error_rad'] <= 2e-7)
    return metrics


def measured_flight_touchdowns(data, *, start_step, minimum_flight_samples=2, touchdown_samples=3):
    """Actual support loss and regained distal contact, independent of planner labels.

    A brief drop or an assumed timed touchdown cannot count as a completed step.
    Reference-point lift is a measured link-point displacement, not full-mesh
    terrain clearance or a physical footpad pressure measurement.
    """
    contacts = data['distal_contact'][:, 0]
    positions = data['reference_point_world_m'][:, 0]
    if contacts.shape[1:] != (6,) or positions.shape[1:] != (6, 3):
        raise ValueError('Single-replica named six-foot trace required')
    events = []
    for leg in range(6):
        index = max(1, start_step)
        while index < len(contacts):
            if contacts[index-1, leg] and not contacts[index, leg]:
                begin = index
                while index < len(contacts) and not contacts[index, leg]:
                    index += 1
                end = index
                regained = end+touchdown_samples <= len(contacts) and contacts[end:end+touchdown_samples, leg].all()
                if end-begin >= minimum_flight_samples:
                    events.append({'leg_index': leg, 'flight_begin_step': begin, 'flight_end_step': end,
                        'flight_samples': end-begin, 'confirmed_measured_touchdown': bool(regained),
                        'measured_reference_point_lift_m': float(positions[begin:end, leg, 2].max()-positions[begin-1, leg, 2]),
                        'measured_reference_point_planar_step_m': None if end == len(contacts) else float(np.linalg.norm(positions[end, leg, :2]-positions[begin-1, leg, :2]))})
            index += 1
    return events


def standing_quiet_review(data, joint_names, *, settle_steps=200, dt=.02):
    """Apply the unchanged measured quiet gates to every fresh standing replica."""
    from omni_quiet_review import quiet_metrics, QUIET_GATES
    if data['joint_position_rad'].shape != (1000, 32, 18):
        raise ValueError('Exact complete32x1000 standing trace required for quiet admission')
    if np.any(data['requested_command']):
        raise ValueError('Standing quiet admission requires zero requested command throughout')
    rows = [quiet_metrics(data, index, settle_steps, list(joint_names), dt) for index in range(32)]
    return {'kind':'unchanged_quiet_gates_all32_fresh_standing', 'bounds':QUIET_GATES,
            'settle_steps':settle_steps, 'num_envs':32, 'per_environment':rows,
            'passed':bool(all(row['pass'] and row['window_duration_s'] >= 10. for row in rows))}
