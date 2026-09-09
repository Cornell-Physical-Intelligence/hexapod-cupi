"""Candidate layout uses actual SDK grid math and real composed v5 USD on CPU."""
import ast
import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from test_mkii_fourbar_collision_isolation import ROOT, fixture, verify
from hexapod_core import fourbar_v1 as contract
from hexapod_env.tasks.mkii_fourbar_v1 import environment_layout as layout
from hexapod_env.tasks.mkii_fourbar_v1.collision_isolation import verify_physical_environment_ownership

KIN = contract.load_kinematics(ROOT/contract.KINEMATICS_PATH)
USD = ROOT/'robot/hexapod_mkii_assy/usd/hexapod_mkii_fourbar_v5/hexapod_mkii_fourbar_v5.usda'


def configured(count=2):
    scene = fixture(count)
    for env in scene.env_prim_paths:
        UsdGeom.Xform.Define(scene.stage, env)
        scene.stage.GetPrimAtPath(env+'/Robot').GetReferences().AddReference(str(USD))
    scene._default_env_pose = torch.zeros(count, 7); scene._default_env_pose[:, 6] = 1.
    cfg = SimpleNamespace(environment_layout=layout.COINCIDENT_LAYOUT,
        scene=SimpleNamespace(env_spacing=0.), terrain=SimpleNamespace(env_spacing=0., terrain_type='plane', prim_path='/World/ground'))
    names = list(KIN['body_paths'])
    links = [[env+'/Robot/'+KIN['body_paths'][name] for name in names] for env in scene.env_prim_paths]
    poses = torch.zeros(count, 7); poses[:, 2] = KIN['reset_root_height_m']; poses[:, 6] = 1.
    view = SimpleNamespace(prim_paths=[env+'/Robot' for env in scene.env_prim_paths], link_paths=links,
                          get_root_transforms=lambda: poses)
    sensors = {name: SimpleNamespace(body_physx_view=SimpleNamespace(prim_paths=[row[i] for row in links]))
               for i, name in enumerate(names)}
    raw = SimpleNamespace(cfg=cfg, scene=scene, num_envs=count,
        _terrain=SimpleNamespace(env_origins=torch.zeros(count, 3)),
        _robot=SimpleNamespace(root_view=view, body_names=names), _body_contact_sensors=sensors)
    return raw, poses


def complete_evidence(count=2):
    raw, poses = configured(count)
    raw.layout_report = layout.verify_scene_layout(raw)
    raw.layout_report['physical_ownership'] = verify_physical_environment_ownership(raw.scene.stage, verify(raw.scene), KIN)
    raw.layout_report['native_row_mapping'] = layout.verify_native_sensor_rows(raw, KIN['body_paths'])
    with patch.dict(sys.modules, {'warp': SimpleNamespace(to_torch=lambda value: value)}):
        layout.record_reset_readback(raw, torch.arange(count, dtype=torch.int32), poses.clone())
    return raw, poses


class CoincidentEnvironmentLayoutTests(unittest.TestCase):
    def test_actual_installed_sdk_grid_accepts_zero_for_32_and_512(self):
        evidence = json.loads((ROOT/'artifacts/mkii_fourbar_2026-09-06/coincident_layout/sdk_zero_spacing_evidence.json').read_text())
        source = evidence['sources']['cloner/cloner_utils.py']['functions']['grid_transforms']
        scope = {'torch': torch, 'math': math}; exec(compile(source, '<captured installed SDK grid>', 'exec'), scope)
        for count in (1, 2, 32, 512):
            positions, orientations = scope['grid_transforms'](count, spacing=0., device='cpu')
            self.assertTrue(torch.equal(positions, torch.zeros(count, 3)))
            self.assertTrue(torch.equal(orientations[:, :3], torch.zeros(count, 3)))
            self.assertTrue(torch.equal(orientations[:, 3], torch.ones(count)))
        self.assertGreater(float(scope['grid_transforms'](32, spacing=2.)[0].abs().max()), 0.)
        terrain = evidence['sources']['terrains/terrain_importer.py']['functions']
        self.assertIn('self.cfg.env_spacing is None', terrain['configure_env_origins'])
        self.assertIn('grid_transforms(num_envs, env_spacing', terrain['_compute_env_origins_grid'])

    def test_selection_is_opt_in_and_changes_only_placement_before_scene_creation(self):
        self.assertEqual(layout.select_environment_layout({}), layout.GRID_LAYOUT)
        with self.assertRaises(ValueError): layout.select_environment_layout({layout.ENVIRONMENT_VARIABLE:'typo'})
        raw, _ = configured(); raw.cfg.scene.env_spacing = raw.cfg.terrain.env_spacing = 2.
        raw.cfg.physics_token = {'dt': .00125, 'kp': 30, 'kd': .3}
        layout.configure_environment_layout(raw.cfg)
        self.assertEqual(raw.cfg.scene.env_spacing, 0.)
        self.assertEqual(raw.cfg.terrain.env_spacing, 0.)
        self.assertEqual(raw.cfg.physics_token, {'dt': .00125, 'kp':30, 'kd':.3})
        raw.cfg.terrain.terrain_type='generator'
        with self.assertRaises(ValueError): layout.configure_environment_layout(raw.cfg)
        tree=ast.parse((ROOT/'packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/env.py').read_text())
        init=next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name=='__init__')
        statements=[ast.unparse(n) for n in init.body]
        self.assertLess(next(i for i,s in enumerate(statements) if 'configure_environment_layout' in s),
                        next(i for i,s in enumerate(statements) if 'super().__init__' in s))

    def test_actual_v5_colliders_mimics_native_rows_and_full_reset_are_verified(self):
        raw, _ = complete_evidence(2)
        result=layout.validate_layout_evidence(raw.layout_report, 2)
        self.assertTrue(result['complete_initial_reset_verified'])
        for row in raw.layout_report['physical_ownership']['environments']:
            self.assertEqual(len(row['rigid_body_paths']),31)
            self.assertEqual(len(row['mimic_references']),12)
            self.assertGreaterEqual(len(row['collider_paths']),31)
        runtime={'resolved_environment_layout':layout.layout_runtime_descriptor()}
        self.assertEqual(layout.validate_layout_runtime(runtime,layout.COINCIDENT_LAYOUT), layout.layout_runtime_descriptor())
        self.assertTrue(layout.validate_layout_runtime({}, layout.GRID_LAYOUT)['historical_manifest_without_layout_field'])
        with self.assertRaises(ValueError): layout.validate_layout_runtime(runtime, layout.GRID_LAYOUT)
        with self.assertRaises(ValueError): layout.validate_layout_runtime({}, layout.COINCIDENT_LAYOUT)

    def test_nonzero_authored_or_terrain_placement_is_rejected(self):
        for where in ('terrain','scene_buffer','usd'):
            raw,_=configured()
            if where=='terrain': raw._terrain.env_origins[1,0]=.001
            elif where=='scene_buffer': raw.scene._default_env_pose[1,0]=.001
            else: UsdGeom.Xformable(raw.scene.stage.GetPrimAtPath(raw.scene.env_prim_paths[1])).AddTranslateOp().Set((.001,0.,0.))
            with self.subTest(where=where), self.assertRaises(ValueError): layout.verify_scene_layout(raw)

    def test_sensor_swapped_rows_and_foreign_native_links_are_rejected(self):
        raw,_=configured()
        raw._body_contact_sensors['lf_tibia'].body_physx_view.prim_paths.reverse()
        with self.assertRaisesRegex(ValueError,'sensor rows'): layout.verify_native_sensor_rows(raw,KIN['body_paths'])
        raw,_=configured();raw._robot.root_view.link_paths[1][5]=raw._robot.root_view.link_paths[0][5]
        with self.assertRaisesRegex(ValueError,'link row'): layout.verify_native_sensor_rows(raw,KIN['body_paths'])

    def test_foreign_mimic_and_orphan_collider_are_rejected_on_real_usd(self):
        for failure in ('mimic','orphan'):
            raw,_=configured();stage=raw.scene.stage
            if failure=='mimic':
                stage.GetPrimAtPath('/World/envs/env_0/Robot/Physics/lf_tibia_pitch').GetRelationship(
                    'physxMimicJoint:rotZ:referenceJoint').SetTargets(['/World/envs/env_1/Robot/Physics/lf_tibia_lever_pivot'])
            else: UsdPhysics.CollisionAPI.Apply(UsdGeom.Sphere.Define(stage,'/World/envs/env_0/orphan').GetPrim())
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                verify_physical_environment_ownership(stage,verify(raw.scene),KIN)

    def test_selected_reset_checks_actual_backend_and_host_rejects_partial_snapshot(self):
        raw,poses=complete_evidence()
        with patch.dict(sys.modules, {'warp':SimpleNamespace(to_torch=lambda value:value)}):
            layout.record_reset_readback(raw,torch.tensor([1],dtype=torch.int32),poses[1:].clone())
            self.assertEqual(raw.layout_report['latest_reset']['environment_row_indices'],[1])
            with self.assertRaises(ValueError): layout.validate_layout_evidence(raw.layout_report,2)
            expected=poses[1:].clone();poses[1,0]=.01
            with self.assertRaisesRegex(ValueError,'reset pose'): layout.record_reset_readback(raw,torch.tensor([1]),expected)

    def test_host_rejects_missing_mapping_ownership_and_nonfinite_reset(self):
        raw,_=complete_evidence();original=raw.layout_report
        for field,value in (('physical_ownership',{}),('native_row_mapping',{}),('reset_checks',0),
                            ('terrain_origins_m',[[0.,0.,0.]]),('max_reset_position_error_m',float('nan'))):
            bad=copy.deepcopy(original);bad[field]=value
            with self.subTest(field=field), self.assertRaises(ValueError): layout.validate_layout_evidence(bad,2)
        bad=copy.deepcopy(original);bad['latest_reset']['native_root_poses_xyzw'][1][2]=float('nan')
        with self.assertRaises(ValueError): layout.validate_layout_evidence(bad,2)


if __name__=='__main__': unittest.main()
