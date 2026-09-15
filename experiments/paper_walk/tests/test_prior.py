"""Independent URDF/FK checks of the kinematic prior, not dynamics admission."""
from pathlib import Path
import hashlib
import json
import math
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO/'experiments/paper_walk'))
from generate_prior import Kinematics, LEGS, NAMES, NOMINAL

URDF = REPO/'robot/hexapod_mkii_updated_v1/urdf/hexapod_updated_rs05_mass_corrected.urdf'
MODEL = REPO/'robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json'
GEOMETRY = REPO/'artifacts/restart_2026-09-14/standing_translation_preparation_001/source/geometry'
PRIOR = REPO/'artifacts/restart_2026-09-14/paper_tripod_prior_001'


def numbers(value):
    return np.array([float(x) for x in value.split()])


def axis_rotation(axis, angle):
    """Rodrigues implemented independently of the generator's scipy Rotation."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x,y,z = axis
    cross = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return np.eye(3) + math.sin(angle)*cross + (1-math.cos(angle))*(cross@cross)


def urdf_origin(joint):
    origin = joint.find('origin')
    xyz = numbers(origin.get('xyz','0 0 0'))
    roll,pitch,yaw = numbers(origin.get('rpy','0 0 0'))
    transform = np.eye(4)
    # URDF fixed-axis roll/pitch/yaw, parent-to-joint frame.
    transform[:3,:3] = axis_rotation([0,0,1],yaw) @ axis_rotation([0,1,0],pitch) @ axis_rotation([1,0,0],roll)
    transform[:3,3] = xyz
    return transform


def quaternion_matrix(q):
    x,y,z,w = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def all_link_transforms(joints, q):
    values = dict(zip(NAMES,q))
    transforms = {'body':np.eye(4)}
    remaining = list(joints.values())
    while remaining:
        ready = [j for j in remaining if j.find('parent').get('link') in transforms]
        if not ready:
            raise ValueError('Disconnected URDF')
        for joint in ready:
            turn = np.eye(4)
            turn[:3,:3] = axis_rotation(numbers(joint.find('axis').get('xyz')), values[joint.get('name')])
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            transforms[child] = transforms[parent] @ urdf_origin(joint) @ turn
            remaining.remove(joint)
    return transforms


class PriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.urdf = ET.parse(URDF).getroot()
        cls.joints = {j.get('name'):j for j in cls.urdf.findall('joint')}
        cls.model = json.loads(MODEL.read_text())
        cls.geometry = json.loads((GEOMETRY/'geometry.json').read_text())
        cls.meta = json.loads((PRIOR/'prior_metadata.json').read_text())
        cls.kin = Kinematics(cls.model,cls.geometry)
        with np.load(PRIOR/'tripod_prior.npz',allow_pickle=False) as arrays:
            cls.data = {key:arrays[key].copy() for key in arrays.files}

    def test_all_model_joint_fields_match_exact_mass_corrected_urdf(self):
        self.assertEqual(hashlib.sha256(URDF.read_bytes()).hexdigest(),self.meta['urdf_sha256'])
        self.assertEqual(hashlib.sha256(MODEL.read_bytes()).hexdigest(),self.meta['model_sha256'])
        self.assertEqual(set(self.joints),set(NAMES))
        self.assertEqual({j['name'] for j in self.model['joints']},set(NAMES))
        for model_joint in self.model['joints']:
            joint = self.joints[model_joint['name']]
            self.assertEqual(joint.get('type'),'revolute')
            self.assertEqual(joint.find('parent').get('link'),model_joint['parent'])
            self.assertEqual(joint.find('child').get('link'),model_joint['child'])
            np.testing.assert_allclose(numbers(joint.find('axis').get('xyz')),model_joint['axis'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(urdf_origin(joint)[:3,3],model_joint['xyz'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(urdf_origin(joint)[:3,:3],quaternion_matrix(model_joint['quaternion_xyzw']),atol=1e-10,rtol=0)
            for key in ('lower','upper'):
                self.assertEqual(float(joint.find('limit').get(key)),model_joint[key])
            self.assertEqual(float(joint.find('limit').get('effort')),5.5)
            self.assertAlmostEqual(float(joint.find('limit').get('velocity')),self.model['actuator_specification']['urdf_max_velocity_rad_s'])
        links = {x['name']:x for x in self.model['links']}
        self.assertEqual(set(links),{x.get('name') for x in self.urdf.findall('link')})
        total = 0.
        for link in self.urdf.findall('link'):
            mass = float(link.find('inertial/mass').get('value'))
            self.assertAlmostEqual(mass,links[link.get('name')]['mass'],places=12)
            total += mass
        self.assertAlmostEqual(total,self.meta['mass_kg'],places=10)

    def test_nonzero_joint_fk_matches_urdf_parent_origin_axis_order(self):
        poses = [np.zeros(18), NOMINAL, NOMINAL+np.tile([.15,.07,-.11],6),
                 NOMINAL+np.tile([-.12,-.04,.08],6)]
        for q in poses:
            transforms = all_link_transforms(self.joints,q)
            for i,leg in enumerate(LEGS):
                np.testing.assert_allclose(self.kin.transform(i,q[i*3:i*3+3]),transforms[leg+'_tibia'],atol=1e-10,rtol=0)

    def test_amp_shapes_named_order_limits_and_no_command_boundary_pairs(self):
        self.assertEqual(self.meta['joint_names'],NAMES)
        self.assertEqual(self.meta['toe_names'],[leg+'_tibia' for leg in LEGS])
        self.assertEqual(self.meta['amp_order'],'q18,dq18,native_body_linear3,native_body_angular3,root_height1,native_body_toe_xyz18')
        count = round(self.meta['period_s']/self.meta['dt_s'])
        n = len(self.meta['commands'])*count
        for key,width in [('states',61),('next_states',61),('commands',3),('q',18)]:
            self.assertEqual(self.data[key].shape,(n,width))
            self.assertEqual(self.data[key].dtype,np.float32)
            self.assertTrue(np.isfinite(self.data[key]).all())
        states = self.data['states']
        np.testing.assert_array_equal(states[:,:18],self.data['q'])
        self.assertTrue(np.all(states[:,:18]>=self.kin.lower))
        self.assertTrue(np.all(states[:,:18]<=self.kin.upper))
        np.testing.assert_allclose(states[:,36:39],self.data['commands'][:,[1,0,2]]*np.array([1,-1,0]),atol=1e-8,rtol=0)
        np.testing.assert_array_equal(states[:,39:41],np.zeros((n,2)))
        np.testing.assert_array_equal(states[:,41],self.data['commands'][:,2])
        np.testing.assert_allclose(states[:,42],self.meta['root_height_m'],atol=1e-8,rtol=0)
        for start in range(0,n,count):
            cycle = states[start:start+count]
            np.testing.assert_array_equal(self.data['next_states'][start:start+count],np.roll(cycle,-1,axis=0))
            q = cycle[:,:18].astype(float)
            self.assertLessEqual(float(np.max(np.abs(np.roll(q,-1,axis=0)-q))),.04)
            np.testing.assert_allclose(cycle[:,18:36],(np.roll(q,-1,axis=0)-np.roll(q,1,axis=0))/(2*self.meta['dt_s']),atol=4e-6,rtol=0)
        self.assertLess(float(np.max(np.abs(states[:,:18]-NOMINAL))),.35)

    def test_stored_toes_match_independent_urdf_fk_for_every_transition(self):
        local = np.asarray(self.meta['toe_local_points_m'])
        for row in self.data['states']:
            transforms = all_link_transforms(self.joints,row[:18])
            expected = np.array([(transforms[leg+'_tibia']@np.r_[local[i],1])[:3] for i,leg in enumerate(LEGS)])
            np.testing.assert_allclose(row[43:].reshape(6,3),expected,atol=5e-8,rtol=0)

    def test_stance_mesh_height_and_sampled_touchdown_velocity_continuity(self):
        np.testing.assert_array_equal(self.meta['nominal_joint_position_rad'],NOMINAL)
        transforms = all_link_transforms(self.joints,NOMINAL)
        with np.load(GEOMETRY/'geometry_extrema.npz') as arrays:
            minima = {name:float((arrays['all__'+name]@t[:3,:3].T+t[:3,3])[:,2].min()) for name,t in transforms.items()}
        expected_height = -min(minima.values())
        self.assertAlmostEqual(expected_height,self.meta['root_height_m'],places=10)
        self.assertGreaterEqual(min(minima.values())+self.meta['reset_root_height_m'],.004999999)
        count = round(self.meta['period_s']/self.meta['dt_s'])
        dt = self.meta['dt_s']
        # Second-order one-sided derivatives at lift-off/touchdown detect a
        # discontinuous swing/stance join without copying the generator polynomial.
        for start in range(0,len(self.data['states']),count):
            toe = self.data['states'][start:start+count,43:].astype(float).reshape(count,6,3)
            for boundary in (0,count//2):
                before = (3*toe[boundary]-4*toe[(boundary-1)%count]+toe[(boundary-2)%count])/(2*dt)
                after = (-3*toe[boundary]+4*toe[(boundary+1)%count]-toe[(boundary+2)%count])/(2*dt)
                self.assertLess(float(np.max(np.abs(before-after))),.002)


if __name__=='__main__':
    unittest.main()
