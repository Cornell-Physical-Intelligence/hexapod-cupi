from copy import deepcopy
from types import SimpleNamespace as NS
from pathlib import Path
import unittest
import json
import xml.etree.ElementTree as ET
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from device_pack import DeviceTelemetry
from physics_telemetry_oracle import capture_measured_state
from fixtures import Rig

HERE=Path(__file__).parent

def sdk_fixture(n=3):
    rig=Rig(n);names=tuple(reversed(rig.names));urdf=ET.parse(HERE/'reference/oracle/geometry/f050_t060.urdf').getroot()
    bodies=tuple(reversed([l.attrib['name'] for l in urdf.findall('link')]))
    children={j.attrib['name']:j.find('child').attrib['link'] for j in urdf.findall('joint')}
    feet=tuple(children[rig.names[i]] for i in range(2,18,3))
    layout=dict(joint_names_runtime=names,joint_runtime_to_leg_major=tuple(names.index(name) for name in rig.names),foot_link_names=feet,foot_body_ids=tuple(bodies.index(name) for name in feet))
    gen=torch.Generator().manual_seed(31)
    rnd=lambda *shape:torch.randn(*shape,generator=gen,dtype=torch.float64)*.01
    quats=torch.tensor(Rotation.from_rotvec(rnd(n,3).numpy()).as_quat(),dtype=torch.float64)
    bodyq=torch.tensor(Rotation.from_rotvec(rnd(n*19,3).numpy()).as_quat().reshape(n,19,4),dtype=torch.float64)
    data=NS(root_pos_w=rnd(n,3).float(),root_quat_w=quats,root_link_lin_vel_w=rnd(n,3),root_link_ang_vel_w=rnd(n,3),root_lin_vel_w=rnd(n,3),root_lin_vel_b=rnd(n,3),root_ang_vel_b=rnd(n,3),projected_gravity_b=rnd(n,3),joint_pos=rnd(n,18),joint_vel=rnd(n,18),soft_joint_pos_limits=torch.tensor([[[-1.,2.]]*18]*n),computed_torque=rnd(n,18),applied_torque=rnd(n,18),body_link_pos_w=rnd(n,19,3),body_link_quat_w=bodyq,body_link_lin_vel_w=rnd(n,19,3),body_link_ang_vel_w=rnd(n,19,3))
    env=NS(_robot=NS(joint_names=names,body_names=bodies,data=data),_processed_actions=rnd(n,18),reference_residual_controller=NS(reference_velocity=rnd(n,18),residual_velocity=rnd(n,18)))
    env._feet_contact_sensors=[NS(body_names=[name],data=NS(contact_pos_w=rnd(n,1,1,3),force_matrix_w=rnd(n,1,1,3))) for name in feet]
    env._coxa_contact_sensor=NS(data=NS(net_forces_w_history=rnd(n,3,6,3)))
    env._femur_contact_sensors=[NS(data=NS(net_forces_w_history=rnd(n,3,1,3))) for _ in range(6)]
    env._base_contact_sensor=NS(data=NS(net_forces_w_history=rnd(n,3,1,3)))
    env.contact=torch.ones(n,6,dtype=torch.bool);env.shaft=torch.zeros_like(env.contact);env.slip=rnd(n,6);env.reaction=rnd(n,6,3);env.friction=rnd(n,6)
    env._get_foot_contact_state=lambda include_ground_wrench:(env.contact,env.shaft,env.slip,None,env.reaction,env.friction)
    toes=rnd(6,3)
    return env,layout,toes

def capture(adapter,env):
    n=adapter.n
    return adapter.capture(env,time_s=torch.ones(n,dtype=torch.float64),terminated=torch.zeros(n,dtype=torch.bool),truncated=torch.zeros(n,dtype=torch.bool),contact_valid=torch.ones(n,6,dtype=torch.bool),contact_age_s=torch.zeros(n,6,dtype=torch.float64))

class PackingTests(unittest.TestCase):
    def test_named_device_snapshot_matches_independent_numpy_sdk_reader(self):
        env,layout,toes=sdk_fixture();a=DeviceTelemetry(env,layout,toes);new=capture(a,env)['measurement'];old=capture_measured_state(env,layout,toes.numpy(),time_s=1.)
        self.assertTrue(new['measurement_valid'].all());self.assertTrue(new['position_is_float32'].all())
        common=set(new)&set(old)
        self.assertGreater(len(common),30)
        for key in common:np.testing.assert_allclose(new[key].numpy(),old[key],atol=3e-17,rtol=1e-12,err_msg=key)
        saved=new['joint_position_rad'].clone();env._robot.data.joint_pos.add_(1.)
        torch.testing.assert_close(new['joint_position_rad'],saved,atol=0,rtol=0)
    def test_unknown_points_have_explicit_mask_no_invented_anchor_and_contact_claim_rejects(self):
        env,layout,toes=sdk_fixture();a=DeviceTelemetry(env,layout,toes)
        env.contact[1,2]=False;env._feet_contact_sensors[2].data.contact_pos_w[1]=torch.nan
        m=capture(a,env)['measurement'];self.assertTrue(m['measurement_valid'].all());self.assertFalse(m['contact_point_valid'][1,2]);self.assertTrue((m['contact_point_world_m'][1,2]==0).all())
        self.assertTrue(torch.isnan(m['contact_point_world_m_raw'][1,2]).all())
        env.contact[1,2]=True;self.assertEqual(capture(a,env)['measurement']['measurement_valid'].tolist(),[True,False,True])
    def test_bad_names_maps_shapes_and_order_fail_closed(self):
        env,layout,toes=sdk_fixture();bad=deepcopy(layout);bad['foot_body_ids']=(0,)*6
        with self.assertRaises(ValueError):DeviceTelemetry(env,bad,toes)
        a=DeviceTelemetry(env,layout,toes);env._robot.body_names=tuple(reversed(env._robot.body_names))
        with self.assertRaises(ValueError):capture(a,env)
        env,layout,toes=sdk_fixture();a=DeviceTelemetry(env,layout,toes);env._robot.data.body_link_pos_w=torch.zeros(3,18,3)
        with self.assertRaises(ValueError):capture(a,env)
    def test_nonfinite_row_fails_without_hiding_other_replicas(self):
        env,layout,toes=sdk_fixture();a=DeviceTelemetry(env,layout,toes);env._robot.data.joint_vel[0,0]=torch.nan
        result=capture(a,env);self.assertEqual(result['measurement']['measurement_valid'].tolist(),[False,True,True]);self.assertFalse(result['deployment_qualified'])
        self.assertTrue(torch.isnan(result['measurement']['joint_velocity_rad_s'][0,0]))
    def test_nonfinite_contact_force_history_is_not_silently_classified_false(self):
        env,layout,toes=sdk_fixture();a=DeviceTelemetry(env,layout,toes)
        env._femur_contact_sensors[0].data.net_forces_w_history[1,0,0,2]=torch.nan
        self.assertEqual(capture(a,env)['measurement']['measurement_valid'].tolist(),[True,False,True])
if __name__=='__main__':unittest.main()
