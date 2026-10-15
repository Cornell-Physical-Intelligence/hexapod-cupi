"""Build policy observations and the shared 61-value flat-ground motion features from measured state."""
import copy

import numpy as np
import torch

from locomotion.omni.frames import navigation
from locomotion.omni.robot import JOINT_NAMES, LEGS

CONTRACT = {
    'schema': 'hexapod_amp_features_v1', 'width': 61, 'control_dt_s': .02,
    'joint_names': list(JOINT_NAMES), 'leg_order': list(LEGS),
    'body_axes': ['left', 'backward', 'up'],
    'fields': [
        {'name': 'joint_position', 'slice': [0, 18], 'unit': 'rad', 'frame': 'native_joint_absolute'},
        {'name': 'joint_velocity', 'slice': [18, 36], 'unit': 'rad/s', 'frame': 'native_joint'},
        {'name': 'linear_velocity', 'slice': [36, 39], 'unit': 'm/s', 'frame': 'body_root_origin'},
        {'name': 'angular_velocity', 'slice': [39, 42], 'unit': 'rad/s', 'frame': 'body'},
        {'name': 'height', 'slice': [42, 43], 'unit': 'm', 'frame': 'world_z_above_flat_zero_plane'},
        {'name': 'toe_position', 'slice': [43, 61], 'unit': 'm', 'frame': 'body_root_relative_xyz'},
    ],
    'normalization': 'none; learner normalization must preserve this raw contract',
    'interpretation': 'James retains six XYZ toe positions; six scalar heights would give 49 values.',
    'paper': 'https://arxiv.org/html/2511.03167v1#S3.SS1',
}


def feature_contract():
    return copy.deepcopy(CONTRACT)


def extract_features(state):
    """Pack native body-frame measurements without commands or gait phase."""
    batch = state['q'].shape[:-1]
    shapes = {'q': (18,), 'dq': (18,), 'linear': (3,), 'angular': (3,), 'toe_body': (6, 3)}
    for name, tail in shapes.items():
        if state[name].shape != batch+tail:
            raise ValueError('AMP field shape differs: '+name)
    if state['root'].shape not in (batch+(3,), batch+(7,)):
        raise ValueError('AMP root requires XYZ position or XYZ/XYZW pose')
    parts = (state['q'], state['dq'], state['linear'], state['angular'],
             state['root'][..., 2:3], state['toe_body'].reshape(batch+(18,)))
    if isinstance(state['q'], torch.Tensor):
        return torch.cat(parts, dim=-1)
    return np.concatenate(parts, axis=-1)


def proprio(state, neutral):
    """One 42-value frame: body rates and gravity in navigation axes, joint offsets from neutral, joint rates."""
    return torch.cat((navigation(state["angular"]), navigation(state["gravity"]),
                      state["q"] - neutral, state["dq"]), dim=-1)


def observations(history, commands, previous_action, state, record_motion_features):
    """Pack the 231-value actor input, the 234-value critic input and, when recorded, the motion features."""
    obs = torch.cat((history.flatten(1), commands, previous_action), dim=-1)
    critic = torch.cat((obs, navigation(state["linear"])), dim=-1)
    result = {"obs": obs, "critic": critic}
    if record_motion_features:
        result["amp"] = extract_features(state)
    return result
