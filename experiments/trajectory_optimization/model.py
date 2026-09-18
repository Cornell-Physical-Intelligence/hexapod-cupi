"""Floating-base Newton-Euler dynamics for the approved direct-drive robot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import casadi as ca
import numpy as np

from locomotion.env_config import JOINT_NAMES, KD, MODEL_SHA256, URDF_SHA256

ROOT = Path(__file__).resolve().parents[2]
LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")
NEUTRAL = np.tile([0., -.30, .40], 6)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def skew(v):
    return ca.vertcat(ca.horzcat(0, -v[2], v[1]),
                      ca.horzcat(v[2], 0, -v[0]),
                      ca.horzcat(-v[1], v[0], 0))


def axis_rotation(axis, angle):
    k = skew(ca.DM(axis))
    return ca.DM.eye(3) + ca.sin(angle)*k + (1-ca.cos(angle))*(k@k)


def quaternion_matrix(q):
    q = np.asarray(q, dtype=float)
    if not np.isclose(q@q, 1., atol=1e-8):
        raise ValueError("Joint quaternion is not unit length")
    x, y, z, w = q
    return ca.DM([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class RobotModel:
    """Use world forces and moments, ZYX root angles, and the named joint order.

    Coordinates are root XYZ, root roll/pitch/yaw, then 18 joint angles.
    The root residual contains force followed by world moment. Joint residuals
    contain motor torques. The contact model uses one point in each source toe.
    Native replay retains the source mesh and its separate contact-patch checks.
    """
    def __init__(self, root=ROOT):
        root = Path(root)
        active = json.loads((root/'robot/active_model.json').read_text())
        self.model_path = root/active['model']['path']
        self.urdf_path = root/active['urdf']['path']
        if digest(self.model_path) != MODEL_SHA256 or digest(self.urdf_path) != URDF_SHA256:
            raise ValueError("Optimizer requires the approved model and URDF bytes")
        self.data = json.loads(self.model_path.read_text())
        self.links = {b['name']: b for b in self.data['links']}
        by_name = {j['name']: j for j in self.data['joints']}
        self.joints = [by_name[n] for n in JOINT_NAMES]
        self.lower = np.array([j['lower'] for j in self.joints])
        self.upper = np.array([j['upper'] for j in self.joints])
        self.mass = sum(b['mass'] for b in self.links.values())
        self.kd = np.asarray(KD)
        self.geometry_path = root/'artifacts/restart_2026-09-14/standing_translation_preparation_001/source/geometry/geometry.json'
        geometry = json.loads(self.geometry_path.read_text())
        shapes = {s['body']: s for s in geometry['shapes']}
        centers = []
        for leg in LEGS:
            shape = shapes[leg+'_tibia']
            center = np.mean(shape['cap_bounds_m'], axis=0)
            centers.append((np.array(shape['shape_to_link'])@np.r_[center, 1])[:3])
        self.toe_local = np.array(centers)
        self._build()
        # Select a source mesh point at each toe's neutral ground contact.
        nominal = np.r_[np.zeros(6), NEUTRAL]
        nominal[2] = .09780231400684256
        self.extrema_path = self.geometry_path.with_name('geometry_extrema.npz')
        if digest(self.extrema_path) != geometry['extrema_sha256']:
            raise ValueError("Geometry extrema bytes differ")
        positions, rotations = (np.asarray(x) for x in self.body_poses(nominal))
        with np.load(self.extrema_path, allow_pickle=False) as clouds:
            for i, leg in enumerate(LEGS):
                column = list(self.links).index(leg+'_tibia')
                rotation = rotations[:, column].reshape(3, 3, order='F')
                cloud = clouds['all__'+leg+'_tibia']
                height = cloud@rotation[2]
                self.toe_local[i] = cloud[height <= height.min()+1e-7].mean(axis=0)
        self._build()
        self.contact_height = np.zeros(6)

    def _build(self):
        q, v, a = (ca.SX.sym(n, 24) for n in ('q', 'v', 'a'))
        contact = ca.SX.sym('contact', 3, 6)
        rx = axis_rotation([1, 0, 0], q[3])
        ry = axis_rotation([0, 1, 0], q[4])
        rz = axis_rotation([0, 0, 1], q[5])
        r0 = rz@ry@rx
        angular_map = ca.horzcat(rz@ry[:, 0], rz[:, 1], ca.DM([0, 0, 1]))
        w0 = angular_map@v[3:6]
        alpha0 = ca.jtimes(w0, q, v) + angular_map@a[3:6]
        pose = {'body': (q[:3], r0, w0, alpha0, a[:3])}
        axes = []
        for i, joint in enumerate(self.joints):
            p, r, w, alpha, linear_a = pose[joint['parent']]
            displacement = r@ca.DM(joint['xyz'])
            basis = r@quaternion_matrix(joint['quaternion_xyzw'])
            axis = basis@ca.DM(joint['axis'])
            child_r = basis@axis_rotation(joint['axis'], q[6+i])
            child_w = w + axis*v[6+i]
            child_alpha = alpha + axis*a[6+i] + ca.cross(w, axis*v[6+i])
            child_a = linear_a + ca.cross(alpha, displacement) + ca.cross(w, ca.cross(w, displacement))
            pose[joint['child']] = (p+displacement, child_r, child_w, child_alpha, child_a)
            axes.append(axis)
        forces, moments, coms, potential = {}, {}, [], 0
        for name, body in self.links.items():
            p, r, w, alpha, linear_a = pose[name]
            c = r@ca.DM(body['com'])
            world_i = r@ca.DM(body['inertia'])@r.T
            f = body['mass']*(linear_a + ca.cross(alpha, c)
                + ca.cross(w, ca.cross(w, c)) - ca.DM([0, 0, -9.81]))
            forces[name] = f
            moments[name] = world_i@alpha + ca.cross(w, world_i@w) + ca.cross(c, f)
            coms.append(body['mass']*(p+c))
            potential += body['mass']*9.81*(p+c)[2]
        toes = []
        for i, leg in enumerate(LEGS):
            name = leg+'_tibia'
            p, r, *_ = pose[name]
            offset = r@ca.DM(self.toe_local[i])
            toes.append(p+offset)
            forces[name] -= contact[:, i]
            moments[name] -= ca.cross(offset, contact[:, i])
        tau = [None]*18
        for i in reversed(range(18)):
            joint = self.joints[i]
            name, parent = joint['child'], joint['parent']
            tau[i] = ca.dot(axes[i], moments[name])
            forces[parent] += forces[name]
            moments[parent] += moments[name] + ca.cross(pose[name][0]-pose[parent][0], forces[name])
        residual = ca.vertcat(forces['body'], moments['body'], *tau)
        self.inverse_dynamics = ca.Function('inverse_dynamics', [q, v, a, contact], [residual])
        self.kinematics = ca.Function('kinematics', [q], [ca.horzcat(*toes), sum(coms)/self.mass])
        self.body_poses = ca.Function('body_poses', [q], [
            ca.horzcat(*(pose[name][0] for name in self.links)),
            ca.horzcat(*(ca.reshape(pose[name][1], 9, 1) for name in self.links))])
        self.potential = ca.Function('potential', [q], [potential])
        self.generalized_dynamics = ca.Function('generalized_dynamics', [q, v, a, contact],
            [ca.vertcat(residual[:3], angular_map.T@residual[3:6], residual[6:])])

    def identity(self):
        return {'model_sha256': digest(self.model_path), 'urdf_sha256': digest(self.urdf_path),
                'geometry_sha256': digest(self.geometry_path),
                'geometry_extrema_sha256': digest(self.extrema_path), 'mass_kg': self.mass,
                'joint_names': list(JOINT_NAMES), 'contact_point_link_m': self.toe_local.tolist(),
                'contact_point_ground_height_m': self.contact_height.tolist()}
