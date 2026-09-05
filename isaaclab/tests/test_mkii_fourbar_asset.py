"""Physical four-bar source, coordinate bridge and portable USD regressions."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
try:
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    import mkii_fourbar_kinematics as kin
    import prepare_mkii_fourbar_usd as builder
    import audit_mkii_fourbar_usd as audit
    from audit_mkii_stance import _origin, _rotation
except ImportError:
    np = None


@unittest.skipIf(np is None, 'NumPy and OpenUSD are required for physical asset audits')
class FourbarKinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.reference, cls.source = kin.load_model()
        cls.contract = kin.make_contract()
        cls.frames = cls.contract['joint_frames']

    def test_committed_contract_matches_regenerated_math(self):
        comparison = audit.compare_kinematic_contract(json.loads(kin.CONTRACT.read_text()), self.contract)
        self.assertTrue(comparison['pass'], comparison['errors'])
        self.assertEqual(len(self.contract['active_joint_names']), 18)
        self.assertEqual(len(self.contract['tree_joint_names']), 30)
        self.assertEqual(len(self.contract['body_paths']), 31)
        self.assertEqual(len(self.contract['closure_joint_names']), 6)

    def test_observed_platform_roundoff_is_accepted_without_coercion(self):
        other = json.loads(json.dumps(self.contract))
        other['joint_limits_rad']['lm_tibia_pitch'][1] += 4.440892098500626e-16
        other['nominal_foot_clearance_m']['lm'] += 2.7755575615628914e-17
        other['phase_bridge']['lm']['stance_full_pose_orientation_residual_rad'] -= 7.895150523206263e-12
        comparison = audit.compare_kinematic_contract(other, self.contract)
        self.assertTrue(comparison['pass'], comparison['errors'])
        self.assertEqual(len(comparison['bounded_numeric_differences']), 3)

    def test_meaningful_contract_changes_and_nonfinite_values_stay_rejected(self):
        for value in [1e-12, float('nan'), float('inf')]:
            other = copy.deepcopy(self.contract)
            other['default_joint_positions_rad']['lf_coxa_yaw'] = value
            self.assertFalse(audit.compare_kinematic_contract(other, self.contract)['pass'])
        for mutate in [
            lambda x: x['active_joint_names'].reverse(),
            lambda x: x['source_sha256'].update(linkage_urdf='different'),
            lambda x: x.update(extra='not permitted'),
            lambda x: x.update(reset_joint_jitter_rad=False),
            lambda x: x['phase_bridge']['lm'].update(stance_full_pose_orientation_residual_rad=1e-7),
        ]:
            other = copy.deepcopy(self.contract)
            mutate(other)
            self.assertFalse(audit.compare_kinematic_contract(other, self.contract)['pass'])

    def test_upstream_frames_preserve_original_axis_and_motion(self):
        for leg in kin.LEGS:
            for suffix in ['coxa_yaw', 'femur_pitch']:
                name = f'{leg}_{suffix}'
                joint = self.source[name]
                axis = np.array([float(x) for x in joint.find('axis').get('xyz').split()])
                for q in [-.7, 0., .4]:
                    rotated = np.eye(4)
                    rotated[:3, :3] = _rotation(axis, q)
                    self.assertTrue(np.allclose(kin.relative_pose(self.frames[name], q), _origin(joint) @ rotated, atol=1e-14, rtol=0))

    def test_new_zero_recovers_actual_distal_cad_relative_to_femur(self):
        poses = kin.forward_kinematics(self.frames, {name: 0. for name in self.contract['tree_joint_names']})
        for leg in kin.LEGS:
            bodies = self.reference['legs'][leg]['bodies']
            femur = f'{leg}_femur'
            cad_femur = np.array(bodies[femur]['export_from_link_matrix'])
            for suffix in ['tibia', 'tibia_push_lever', 'tibia_pushrod']:
                name = f'{leg}_{suffix}'
                actual = np.linalg.inv(poses[femur]) @ poses[name]
                expected = np.linalg.inv(cad_femur) @ np.array(bodies[name]['export_from_link_matrix'])
                self.assertTrue(np.allclose(actual, expected, atol=2e-14, rtol=0), name)

    def test_phase_bridge_maps_stance_and_both_limits_not_approximate_offset(self):
        for leg in kin.LEGS:
            knee = f'{leg}_tibia_pitch'
            bridge = self.contract['phase_bridge'][leg]
            values = list(zip(bridge['old_knee_limits_rad'], bridge['canonical_limits_rad']))
            values.append((-.55, bridge['canonical_stance_rad']))
            for old, new in values:
                self.assertAlmostEqual(kin.old_knee_phase(self.source[knee], self.frames[knee], new), old, places=13)
                self.assertAlmostEqual(kin.inverse_old_phase(self.source[knee], self.frames[knee], old), new, places=13)
            self.assertGreater(bridge['stance_full_pose_orientation_residual_rad'], 0.)
            # Submilliradian frame corrections are recorded, never hidden as
            # an exact match of the old complete body pose.
            self.assertLess(bridge['stance_full_pose_orientation_residual_rad'], .001)
            lo, hi = self.contract['joint_limits_rad'][knee]
            toggles = bridge['toggle_canonical_q_rad']
            self.assertLess(toggles[0], lo)
            self.assertGreater(toggles[1], hi)

    def test_passive_branch_satisfies_all_six_physical_closures(self):
        rng = np.random.default_rng(274913)
        for _ in range(64):
            active = {name: rng.uniform(*self.contract['joint_limits_rad'][name]) for name in self.contract['active_joint_names']}
            positions = kin.expand_active(active)
            for leg in kin.LEGS:
                q = active[f'{leg}_tibia_lever_pivot']
                self.assertEqual(positions[f'{leg}_tibia_pitch'], q)
                self.assertEqual(positions[f'{leg}_tibia_rod_pivot'], -q)
            for row in kin.closure_errors(self.frames, positions).values():
                self.assertLess(row['point_error_m'], 1e-14)
                self.assertLess(row['axis_error_rad'], 1e-13)
                self.assertGreater(row['axis_dot'], .9999999999)

    def test_wrong_passive_sign_and_pin_offset_are_detectable(self):
        positions = dict(self.contract['default_joint_positions_rad'])
        positions['lf_tibia_rod_pivot'] *= -1
        self.assertGreater(kin.closure_errors(self.frames, positions)['lf_tibia_loop_closure']['point_error_m'], .01)
        frames = copy.deepcopy(self.frames)
        frames['lf_tibia_loop_closure']['body0_from_hinge_matrix'][0][3] += .001
        self.assertGreater(kin.closure_errors(frames, self.contract['default_joint_positions_rad'])['lf_tibia_loop_closure']['point_error_m'], .0009)

    def test_nominal_reset_uses_actual_all_link_collision_geometry(self):
        self.assertEqual(self.contract['reset_joint_jitter_rad'], 0.)
        poses = kin.forward_kinematics(self.frames, self.contract['default_joint_positions_rad'])
        rows = kin.collision_heights(self.root, poses, self.contract['reset_root_height_m'])
        self.assertEqual(len(rows), 171)
        self.assertEqual(sum(row['foot'] for row in rows), 12)
        self.assertGreaterEqual(min(row['bottom_z_m'] for row in rows if row['foot']), .005)
        self.assertLess(min(row['bottom_z_m'] for row in rows if row['foot']), .005001)
        self.assertGreater(min(row['bottom_z_m'] for row in rows if not row['foot']), .03)
        for name, q in self.contract['default_joint_positions_rad'].items():
            lo, hi = self.contract['joint_limits_rad'][name]
            self.assertLess(lo, q)
            self.assertGreater(hi, q)


@unittest.skipIf(np is None, 'NumPy and OpenUSD are required for physical asset audits')
class FourbarUsdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='fourbar_tests_')
        cls.base = Path(cls.directory.name)
        cls.output = cls.base/'fresh'/'robot.usda'
        cls.inputs_before = {str(p): kin.sha256(p) for p in [kin.URDF, kin.PINS, kin.CONTRACT]}
        cls.report = builder.prepare(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def clone(self, label):
        target = self.base/label
        shutil.copytree(self.output.parent, target)
        return target/self.output.name

    def test_full_asset_cpu_roundtrip_and_portable_dependency_manifest(self):
        self.assertTrue(self.report['pass'], self.report['errors'])
        self.assertEqual((self.report['rigid_bodies'], self.report['tree_joints'], self.report['closure_joints'], self.report['active_joints']), (31, 30, 6, 18))
        self.assertEqual((self.report['visual_instances'], self.report['unique_source_meshes'], self.report['collision_primitives']), (1927, 77, 171))
        self.assertAlmostEqual(self.report['mass_kg'], 8.260811322, places=10)
        self.assertEqual(self.report['physical_validation'], 'not_performed')
        self.assertEqual(self.report['prescribed_closure_samples'], 1206)
        for row in self.report['dependencies']:
            self.assertFalse(Path(row['path']).is_absolute())
            self.assertEqual(kin.sha256(self.output.parent/row['path']), row['sha256'])
        for path, expected in self.inputs_before.items():
            self.assertEqual(kin.sha256(path), expected)

    def test_relocation_and_overwrite_refusal(self):
        relocated = self.clone('relocated')
        report = audit.validate(kin.URDF, kin.PINS, relocated)
        self.assertTrue(report['pass'], report['errors'])
        self.assertEqual(report['dependencies'], self.report['dependencies'])
        before = kin.sha256(relocated)
        with self.assertRaisesRegex(ValueError, 'immutable'):
            builder.prepare(relocated)
        self.assertEqual(kin.sha256(relocated), before)

    def test_builder_accepts_platform_roundoff_and_preserves_supplied_identity(self):
        # Inject exactly two platform differences relative to the supplied
        # bytes. Host-regenerated math can already differ at other fields;
        # its real platform comparison is covered separately above.
        derived = json.loads(kin.CONTRACT.read_text())
        derived['joint_limits_rad']['lm_tibia_pitch'][1] += 4.440892098500626e-16
        derived['phase_bridge']['lm']['stance_full_pose_orientation_residual_rad'] -= 7.895150523206263e-12
        output = self.base/'platform_build'/'robot.usda'
        with patch.object(kin, 'make_contract', return_value=derived):
            report = builder.prepare(output)
        self.assertTrue(report['pass'], report['errors'])
        self.assertEqual((output.parent/'kinematics.json').read_bytes(), kin.CONTRACT.read_bytes())
        self.assertEqual(report['kinematic_contract_sha256'], kin.sha256(kin.CONTRACT))
        self.assertEqual(len(report['kinematic_contract_comparison']['bounded_numeric_differences']), 2)

    def test_passive_motor_drive_is_rejected(self):
        output = self.clone('passive_drive')
        stage = Usd.Stage.Open(str(output))
        joint = stage.GetPrimAtPath('/Robot/Physics/lf_tibia_pitch')
        UsdPhysics.DriveAPI.Apply(joint, 'angular').CreateStiffnessAttr(30.)
        stage.GetRootLayer().Save()
        result = audit.validate(kin.URDF, kin.PINS, output)
        self.assertFalse(result['pass'])
        self.assertTrue(any('passive' in text for text in result['errors']), result['errors'])

    def test_unexcluded_loop_and_one_mm_frame_error_are_rejected(self):
        output = self.clone('broken_closure')
        stage = Usd.Stage.Open(str(output))
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/Robot/Physics/lf_tibia_loop_closure'))
        joint.CreateExcludeFromArticulationAttr(False)
        point = joint.GetLocalPos0Attr().Get()
        joint.CreateLocalPos0Attr(Gf.Vec3f(point[0]+.001, point[1], point[2]))
        stage.GetRootLayer().Save()
        result = audit.validate(kin.URDF, kin.PINS, output)
        self.assertFalse(result['pass'])
        self.assertTrue(any('exclusion' in text for text in result['errors']), result['errors'])
        self.assertTrue(any('hinge frame' in text for text in result['errors']), result['errors'])

    def test_wrong_inertia_and_collision_dimensions_are_rejected(self):
        output = self.clone('wrong_physics')
        stage = Usd.Stage.Open(str(output))
        body = stage.GetPrimAtPath('/Robot/Geometry/body/lf_coxa/lf_femur/lf_tibia')
        UsdPhysics.MassAPI(body).CreateDiagonalInertiaAttr(Gf.Vec3f(.1, .1, .1))
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Sphere):
                UsdGeom.Sphere(prim).CreateRadiusAttr(.05)
                break
        stage.GetRootLayer().Save()
        result = audit.validate(kin.URDF, kin.PINS, output)
        self.assertFalse(result['pass'])
        self.assertTrue(any('inertia=False' in text for text in result['errors']), result['errors'])
        self.assertTrue(any('sphere differs' in text for text in result['errors']), result['errors'])

    def test_altered_stl_mesh_points_are_rejected(self):
        output = self.clone('wrong_mesh')
        stage = Usd.Stage.Open(str(output.parent/'geometry.usdc'))
        prim = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        points[0] = points[0]+Gf.Vec3f(.001, 0, 0)
        mesh.CreatePointsAttr(points)
        stage.GetRootLayer().Save()
        result = audit.validate(kin.URDF, kin.PINS, output)
        self.assertFalse(result['pass'])
        self.assertTrue(any('STL vertices differ' in text for text in result['errors']), result['errors'])


if __name__ == '__main__':
    unittest.main()
