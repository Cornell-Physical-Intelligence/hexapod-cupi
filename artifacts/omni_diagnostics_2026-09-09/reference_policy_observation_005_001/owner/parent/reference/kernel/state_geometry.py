"""233-value geometric block, NOT the740-value actor or a new policy schema.

Body-relative geometry from fixed batched tensors. Mode/counters/contact/stop
and history blocks remain required in a future complete encoder; none is
implicitly inferred or omitted from an admitted actor. Inputs are simulated
measurements and ideal reference state, not qualified estimator signals.
"""
import torch
from tensor_kernel import finite_rows

SHAPES = {
    'position_world_m': (3,), 'rotation_world_from_body': (3, 3), 'time_s': (),
    'target_position_rad': (18,), 'target_velocity_rad_s': (18,),
    'residual_position_rad': (18,), 'residual_velocity_rad_s': (18,),
    'reference_position_rad': (18,), 'reference_velocity_rad_s': (18,),
    'reference_anchors_world_m': (6, 3), 'measured_anchors_world_m': (6, 3),
    'reference_minus_measured_preload_world_m': (6, 3), 'neutral_reference_toes_body_m': (6, 3),
    'initial_joint_preload_leg_major_rad': (6, 3), 'desired_position_world_m': (3,),
    'desired_rotation_world_from_body': (3, 3), 'initial_desired_rotation_world_from_body': (3, 3),
    'trajectory_coefficients': (2, 6, 3), 'trajectory_endpoints_world_m': (2, 3),
    'trajectory_start_s': (2,), 'trajectory_duration_s': (2,), 'trajectory_lift_m': (2,),
}


def extract_geometry_state(g, batch):
    """Original swing is trajectory index0; provisional landing is index1.

    Every field must be finite, including inactive finite-zero sentinels.
    Numerical violations mask only that replica, with NaN output to avoid
    accidentally feeding invalid partial observations to a learner.
    """
    missing = set(SHAPES) | {'trajectory_active'}
    missing -= set(batch)
    if missing:
        raise ValueError('Missing complete geometric state: '+','.join(sorted(missing)))
    n = batch['position_world_m'].shape[0]
    valid = torch.ones(n, device=g.device, dtype=torch.bool)
    for key, tail in SHAPES.items():
        g.check(batch[key], (n, *tail), key)
        valid &= finite_rows(batch[key])
    active = g.check(batch['trajectory_active'], (n, 2), 'trajectory presence', torch.bool)
    p, R = batch['position_world_m'], batch['rotation_world_from_body']
    valid &= g.rotation_valid(R)
    valid &= g.rotation_valid(batch['desired_rotation_world_from_body'])
    valid &= g.rotation_valid(batch['initial_desired_rotation_world_from_body'])
    points = lambda x: (x-p[:, None, :]) @ R
    point = lambda x: ((x-p)[:, None, :] @ R).squeeze(1)
    q, v = batch['target_position_rad'], batch['target_velocity_rad_s']
    rq, rv = batch['residual_position_rad'], batch['residual_velocity_rad_s']
    valid &= (q-batch['reference_position_rad']-rq).abs().amax(-1) <= 1e-9
    valid &= (v-batch['reference_velocity_rad_s']-rv).abs().amax(-1) <= 1e-9
    valid &= (rq.abs().amax(-1) <= .02+1e-7) & (rv.abs().amax(-1) <= .25+1e-7)
    valid &= v.abs().amax(-1) <= g.total_velocity+1e-5
    anchors, measured = batch['reference_anchors_world_m'], batch['measured_anchors_world_m']
    preload = batch['reference_minus_measured_preload_world_m']
    valid &= (anchors-measured-preload).abs().amax((-1, -2)) <= 1e-8
    values, fields = [], []
    def add(name, value):
        flat = value.reshape(n, -1)
        start = sum(v.shape[1] for v in values)
        fields.append(dict(name=name, start=start, stop=start+flat.shape[1]))
        values.append(flat)
    add('executable_target_offset', q-g.observation_nominal)
    add('executable_target_velocity', v/g.total_velocity)
    add('residual_position', rq/.02)
    add('residual_velocity', rv/.25)
    add('reference_anchors_body', points(anchors)*5)
    add('measured_anchors_body', points(measured)*5)
    add('preload_body', preload@R*5)
    add('neutral_toes_initial_body', batch['neutral_reference_toes_body_m']*5)
    add('desired_position_error_body', point(batch['desired_position_world_m'])*5)
    add('desired_rotation_relative', R.transpose(-1, -2) @ batch['desired_rotation_world_from_body'])
    add('initial_desired_rotation_relative', R.transpose(-1, -2) @ batch['initial_desired_rotation_world_from_body'])
    for i, key in enumerate(('swing', 'landing')):
        coeff = batch['trajectory_coefficients'][:, i]
        endpoint = batch['trajectory_endpoints_world_m'][:, i]
        start = batch['trajectory_start_s'][:, i]
        duration = batch['trajectory_duration_s'][:, i]
        lift = batch['trajectory_lift_m'][:, i]
        elapsed = batch['time_s']-start
        valid_active = (duration > 0) & (elapsed >= -1e-7) & ((coeff.sum(1)-endpoint).abs().amax(-1) <= 1e-8)
        valid_active &= lift == (g.configuration['lift_m'] if key == 'swing' else 0.)
        sentinel = ((coeff == 0).all((-1, -2)) & (endpoint == 0).all(-1) & (start == 0) & (duration == 0) & (lift == 0))
        valid &= torch.where(active[:, i], valid_active, sentinel)
        relative = coeff @ R
        relative = torch.cat((point(coeff[:, 0])[:, None, :], relative[:, 1:]), 1)
        safe_duration = torch.where(active[:, i] & (duration > 0), duration, 1.)
        block = torch.cat((active[:, i, None].to(g.dtype), (elapsed/safe_duration)[:, None], (duration/2)[:, None],
                           point(endpoint)*5, relative.flatten(1)*5, (lift*100)[:, None]), -1)
        add(key+'_trajectory', torch.where(active[:, i, None], block, 0.))
    valid &= ~active[:, 1] | active[:, 0]  # Landing never discards original swing.
    add('initial_joint_preload_leg_major_rad', batch['initial_joint_preload_leg_major_rad'])
    output = torch.cat(values, -1)
    valid &= finite_rows(output)
    return dict(values=torch.where(valid[:, None], output, torch.nan), checked_values=output,
                valid=valid, fields=fields, width=output.shape[-1],
                source_binding=g.identity, full_actor=False, policy_training_allowed=False,
                measurement_provenance='simulator_truth_and_ideal_reference_unqualified_for_deployment')
