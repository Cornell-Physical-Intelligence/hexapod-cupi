"""Derive tripod coefficients from the admitted model's joint transforms."""
import math
import numpy as np

from .env_config import JOINT_NAMES, LEGS


def stance_geometry(model, toe_local_points, pose):
    """Return body-frame toes and analytic position Jacobians at one stance."""
    joints = {joint['name']: joint for joint in model['joints']}
    toes = np.asarray(toe_local_points, float).reshape(6, 3)
    pose = np.asarray(pose, float).reshape(6, 3)
    if set(joints) != set(JOINT_NAMES) or not np.isfinite([toes, pose]).all():
        raise ValueError('Expected the admitted joint model and six finite toe points')
    points, jacobians = [], []
    for leg, angles, toe in zip(LEGS, pose, toes):
        position, rotation = np.zeros(3), np.eye(3)
        origins, axes = [], []
        parent = 'body'
        for name, angle in zip((n for n in JOINT_NAMES if n.startswith(leg+'_')), angles):
            joint = joints[name]
            if joint['parent'] != parent:
                raise ValueError('Joint chain differs from the admitted leg order')
            x, y, z, w = joint['quaternion_xyzw']
            origin_rotation = np.array([
                [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
            position = position+rotation@joint['xyz']
            rotation = rotation@origin_rotation
            origins.append(position.copy()); axes.append(rotation@joint['axis'])
            x, y, z = joint['axis']
            cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
            rotation = rotation@(np.eye(3)+math.sin(angle)*cross+(1-math.cos(angle))*(cross@cross))
            parent = joint['child']
        point = position+rotation@toe
        points.append(point)
        jacobians.append(np.column_stack([np.cross(axis, point-origin)
                                          for axis, origin in zip(axes, origins)]))
    return np.array(points), np.array(jacobians)
