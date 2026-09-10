"""Read measured Isaac Lab 6 XYZW state before reset; never write robot pose.

No Isaac imports: this module reads the installed data properties already used
by the pinned C runtime. Missing contact points remain unknown, never feet.
"""
from contextlib import contextmanager
from types import MethodType
import numpy as np

LEGS = ('lf', 'lm', 'lr', 'rf', 'rm', 'rr')


def array(value):
    """Copy simulator storage so later integration or automatic reset cannot alter it."""
    value = value.torch if hasattr(value, 'torch') else value
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def rotation_xyzw(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape[-1:] != (4,) or not np.isfinite(q).all():
        raise ValueError('Finite SDK XYZW quaternions required')
    norm = np.linalg.norm(q, axis=-1)
    if np.any(np.abs(norm - 1.) > 1e-4):
        raise ValueError('Measured quaternion is not unit length')
    x, y, z, w = np.moveaxis(q / norm[..., None], -1, 0)
    return np.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    ), axis=-1).reshape(q.shape[:-1] + (3, 3))


def named_layout(manifest, joint_names, body_names, sensor_names):
    """Require the exact named C chains and explicitly permute reported indices."""
    joints = tuple(joint_names)
    bodies = tuple(body_names)
    expected_joints = tuple(manifest['link_joint_mapping'][leg]['joints'][part]
                            for leg in LEGS for part in ('coxa', 'femur', 'tibia'))
    expected_feet = tuple(manifest['link_joint_mapping'][leg]['links']['tibia'] for leg in LEGS)
    if len(joints) != 18 or len(set(joints)) != 18 or set(joints) != set(expected_joints):
        raise ValueError('Actual articulation must report all18 exact named C joints')
    expected_bodies = {'body_mock'} | {manifest['link_joint_mapping'][leg]['links'][part]
                                       for leg in LEGS for part in ('coxa', 'femur', 'tibia')}
    if len(bodies) != 19 or len(set(bodies)) != 19 or set(bodies) != expected_bodies:
        raise ValueError('Actual articulation must report all19 exact named C bodies')
    sensor_names = tuple(tuple(names) for names in sensor_names)
    if sensor_names != tuple((name,) for name in expected_feet):
        raise ValueError('The six contact sensors must retain explicit declared leg order')
    return {
        'joint_names_runtime': joints, 'joint_names_leg_major': expected_joints,
        'joint_runtime_to_leg_major': tuple(joints.index(name) for name in expected_joints),
        'foot_body_ids': tuple(bodies.index(name) for name in expected_feet),
        'foot_link_names': expected_feet,
    }


def capture_measured_state(env, layout, reference_points_local_m, *, time_s):
    """Actual measured frame for one or more replicas, in declared leg order.

    reference_points_local_m are the frozen kinematic reference's toe points;
    their world Z is a reference-point height, not complete pad/shaft clearance.
    Contact points, contact classifications and body transforms come from SDK.
    """
    if not np.isfinite(time_s) or time_s < 0:
        raise ValueError('Finite nonnegative physical sample time required')
    if tuple(env._robot.joint_names) != layout['joint_names_runtime']:
        raise ValueError('Articulation joint ordering changed after binding')
    d = env._robot.data
    position = array(d.root_pos_w)
    quaternion = array(d.root_quat_w)
    rotation = rotation_xyzw(quaternion)
    ids = list(layout['foot_body_ids'])
    leg_order = list(layout['joint_runtime_to_leg_major'])
    points = np.asarray(reference_points_local_m, dtype=float)
    if points.shape != (6, 3) or not np.isfinite(points).all():
        raise ValueError('Six finite local reference points required')
    link_position = array(d.body_link_pos_w)[:, ids]
    link_rotation = rotation_xyzw(array(d.body_link_quat_w)[:, ids])
    offsets = np.einsum('blij,lj->bli', link_rotation, points)
    reference_world = link_position + offsets
    reference_velocity = array(d.body_link_lin_vel_w)[:, ids] + np.cross(
        array(d.body_link_ang_vel_w)[:, ids], offsets)
    contact_raw = np.stack([array(s.data.contact_pos_w)[:, 0, 0] for s in env._feet_contact_sensors], axis=1)
    contact_valid = np.isfinite(contact_raw).all(-1)
    # NaN means no measured contact point. Store the validity mask explicitly;
    # JSON wrappers may substitute null but must never substitute a foot anchor.
    normal_force = np.stack([array(s.data.force_matrix_w)[:, 0, 0] for s in env._feet_contact_sensors], axis=1)
    distal, shaft, slip, _, reaction, friction_xy = env._get_foot_contact_state(include_ground_wrench=True)
    distal, shaft = array(distal), array(shaft)
    if np.any((distal | shaft) & ~contact_valid):
        raise ValueError('Runtime claims a measured contact with an unknown contact point')
    coxa = np.linalg.norm(array(env._coxa_contact_sensor.data.net_forces_w_history), axis=-1).max(axis=1) > 1.
    femur = np.concatenate([np.linalg.norm(array(s.data.net_forces_w_history), axis=-1).max(axis=1) > 1.
                             for s in env._femur_contact_sensors], axis=1)
    base_force = np.linalg.norm(array(env._base_contact_sensor.data.net_forces_w_history), axis=-1).max(axis=1)[:, 0]
    sample = {
        'time_s': np.full(position.shape[0], float(time_s)),
        'position_world_m': position, 'quaternion_world_xyzw': quaternion,
        'quaternion_world_wxyz': quaternion[:, [3, 0, 1, 2]].copy(),
        'rotation_world_from_body': rotation,
        'velocity_world_mps': array(d.root_link_lin_vel_w), 'gyro_world_rad_s': array(d.root_link_ang_vel_w),
        'com_velocity_world_mps': array(d.root_lin_vel_w),
        'velocity_body_mps': array(d.root_lin_vel_b), 'gyro_body_rad_s': array(d.root_ang_vel_b),
        'projected_gravity_body': array(d.projected_gravity_b),
        'joint_position_rad': array(d.joint_pos), 'joint_velocity_rad_s': array(d.joint_vel),
        'joint_target_rad': array(env._processed_actions),
        'soft_joint_pos_limits_rad': array(d.soft_joint_pos_limits),
        'joint_position_leg_major_rad': array(d.joint_pos)[:, leg_order].reshape(-1, 6, 3),
        'joint_velocity_leg_major_rad_s': array(d.joint_vel)[:, leg_order].reshape(-1, 6, 3),
        'joint_target_leg_major_rad': array(env._processed_actions)[:, leg_order].reshape(-1, 6, 3),
        'computed_torque_nm': array(d.computed_torque), 'applied_torque_nm': array(d.applied_torque),
        'reference_point_world_m': reference_world, 'reference_point_velocity_world_mps': reference_velocity,
        'contact_point_world_m': contact_raw, 'contact_point_valid': contact_valid,
        'normal_force_world_n': normal_force, 'reaction_force_world_n': array(reaction),
        'friction_force_xy_n': array(friction_xy),
        'distal_contact': distal, 'shaft_contact': shaft,
        'distal_contact_slip_mps': array(slip), 'coxa_contact': coxa, 'femur_contact': femur,
        'base_contact': base_force > 1., 'reason_base_contact': base_force > 5.,
        'reason_too_low': position[:, 2] < .055,
        'reason_upside_down': array(d.projected_gravity_b)[:, 2] > -.45,
    }
    for name, value in sample.items():
        if name != 'contact_point_world_m' and not np.isfinite(value).all():
            raise ValueError(f'Nonfinite measured {name}')
    controller = getattr(env, 'reference_residual_controller', None)
    if controller is not None:
        sample['executable_target_velocity_rad_s'] = array(controller.reference_velocity + controller.residual_velocity)
        if not np.isfinite(sample['executable_target_velocity_rad_s']).all():
            raise ValueError('Nonfinite executed target velocity')
    return sample


@contextmanager
def pre_reset_capture(env, capture):
    """Call the original done predicate exactly once; snapshot before auto reset."""
    original = env._get_dones
    samples = []

    def wrapped(instance):
        terminated, truncated = original()
        sample = capture()
        sample['terminated'] = array(terminated)
        sample['truncated'] = array(truncated)
        samples.append(sample)
        return terminated, truncated

    env._get_dones = MethodType(wrapped, env)
    try:
        yield samples
    finally:
        env._get_dones = original


def require_single_pre_reset_sample(samples, previous_count, terminated, truncated):
    if len(samples) != previous_count + 1:
        raise RuntimeError('Expected exactly one measured pre-reset snapshot per control step')
    sample = samples[-1]
    if not np.array_equal(sample['terminated'], array(terminated)) or not np.array_equal(sample['truncated'], array(truncated)):
        raise RuntimeError('Pre-reset termination snapshot differs from returned control-step flags')
    return sample
