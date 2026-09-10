"""Synthetic kinematic/contact fixtures only; no physics pass is implied."""
import unittest
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
from serial_geometry import SerialGeometry, tensor


class Fixture:
    def __init__(self, names=None, preload=.0):
        self.g = SerialGeometry()
        self.leg_names = tuple(n for leg in self.g.names for n in leg)
        self.names = tuple(names or self.leg_names)
        self.to_runtime = [self.leg_names.index(n) for n in self.names]
        self.to_leg = [self.names.index(n) for n in self.leg_names]
        q = self.g.q0.numpy().copy()
        self.qtarget = q.reshape(18)[self.to_runtime].copy()
        self.q = q.copy(); self.q[:, 1] -= preload
        self.p = np.array([0., 0., .13053251856352807]); self.R = np.eye(3)
        self.time = 0.; self.v = np.zeros(18); self.cmd = np.zeros(3)
        self.previous_feet = None; self.force_no_touchdown = False; self.force_never_flight = False
        feet = self.g.fk(tensor(self.q))[0].numpy() + self.p
        self.floor = feet[:, 2].copy()

    def snapshot(self):
        local = self.g.fk(tensor(self.q))[0].numpy()
        feet = (self.R @ local.T).T+self.p
        velocity = np.zeros((6, 3)) if self.previous_feet is None else (feet-self.previous_feet)/.02
        contact = feet[:, 2] < self.floor+1e-5
        if self.force_never_flight:
            contact[:] = True
        if self.force_no_touchdown and self.time > .2:
            contact[0] = False
        points = feet.copy(); points[~contact] = np.nan
        limits = np.stack((self.g.lower.numpy(), self.g.upper.numpy()), axis=-1).reshape(18, 2)[self.to_runtime]
        sample = dict(time_s=np.array([self.time]), position_world_m=self.p[None].copy(),
            quaternion_world_xyzw=Rotation.from_matrix(self.R).as_quat()[None], rotation_world_from_body=self.R[None].copy(),
            velocity_body_mps=np.array([[self.cmd[1], -self.cmd[0], 0.]]), gyro_body_rad_s=np.array([[0., 0., self.cmd[2]]]),
            joint_position_rad=self.q.reshape(18)[self.to_runtime][None].copy(), joint_target_rad=self.qtarget[None].copy(),
            executable_target_velocity_rad_s=self.v[None].copy(), soft_joint_pos_limits_rad=limits[None],
            reference_point_world_m=feet[None], reference_point_velocity_world_mps=velocity[None],
            contact_point_world_m=points[None], contact_point_valid=contact[None].copy(), distal_contact=contact[None].copy(),
            shaft_contact=np.zeros((1, 6), bool), coxa_contact=np.zeros((1, 6), bool), femur_contact=np.zeros((1, 6), bool),
            base_contact=np.array([False]), terminated=np.array([False]), truncated=np.array([False]))
        normal = np.zeros((1,6,3)); normal[0,contact,2] = 8.26081134*9.81/max(1,contact.sum())
        sample.update(computed_torque_nm=np.full((1,18),.7), applied_torque_nm=np.full((1,18),.7),
                      normal_force_world_n=normal, reaction_force_world_n=normal.copy(),
                      distal_contact_slip_mps=np.zeros((1,6)), joint_velocity_rad_s=self.v[None].copy(),
                      quaternion_world_wxyz=sample["quaternion_world_xyzw"][:,[3,0,1,2]].copy())
        return sample

    def advance(self, output):
        self.previous_feet = self.snapshot()['reference_point_world_m'][0].copy()
        self.qtarget = output['q_ref'][0].copy(); self.q = self.qtarget[self.to_leg].reshape(6, 3)
        self.v = output['v_ref'][0].copy(); self.time = output['target_time_s']

