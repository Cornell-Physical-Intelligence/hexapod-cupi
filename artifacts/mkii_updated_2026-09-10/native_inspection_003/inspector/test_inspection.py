import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from pxr import Usd
import inspect_core as core
import inspection_contract as contract
import run_inspection as entry

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent
ASSET=REPO/'artifacts/mkii_updated_2026-09-10/usd_002/rs05_mass_corrected'


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=contract.read(ASSET/'source/model.json')
        cls.usd=core.usd_readback(Usd.Stage.Open(str(ASSET/'robot.usda')))

    def native(self):
        links=list(reversed(self.model['links']));joints=self.model['joints'][7:]+self.model['joints'][:7]
        return {'count':1,'fixed_base':False,'body_names':[m['name']for m in links],
                'joint_names':[m['name']for m in joints], 'masses':[[m['mass']for m in links]],
                'coms':[[m['com']+[0.,0.,0.,1.]for m in links]],
                'inertias':[[np.asarray(m['inertia']).reshape(-1,order='F').tolist()for m in links]],
                'limits':[[[m['lower'],m['upper']]for m in joints]],
                'stiffness':np.zeros((1,18)).tolist(),'damping':np.zeros((1,18)).tolist()}

    def test_exact_saved_usd_and_per_leg_limits(self):
        self.assertTrue(core.validate_usd(self.usd,self.model))
        limits={j['name']:j['limits_rad']for j in self.usd['joints']}
        self.assertAlmostEqual(limits['lf_tibia_pitch'][1],np.pi)
        self.assertNotEqual(limits['lf_coxa_yaw'],limits['lm_coxa_yaw'])
        self.assertEqual(len(self.usd['colliders']),153)

    def test_usd_frames_mass_drives_and_geometry_reject(self):
        for edit in ('mass','axis','drives','collider'):
            r=copy.deepcopy(self.usd)
            if edit=='mass':r['bodies'][0]['mass']+=.128
            if edit=='axis':r['joints'][0]['axis']='Y'
            if edit=='drives':r['drives']=['x:PhysicsDriveAPI:angular']
            if edit=='collider':r['colliders'][0]['approximation']='convexHull'
            with self.subTest(edit=edit),self.assertRaises(ValueError):core.validate_usd(r,self.model)

    def test_native_name_permutation_full_inertia(self):
        r=self.native();self.assertTrue(core.validate_native(r,self.model))
        # Erasing the real off-diagonal moments must not pass a principal-value-only comparison.
        r['inertias'][0][-1]=np.diag(np.diag(np.asarray(self.model['links'][0]['inertia']))).reshape(-1).tolist()
        with self.assertRaises(ValueError):core.validate_native(r,self.model)

    def test_native_missing_limits_gains_and_nonfinite(self):
        for edit in ('names','limits','gain','finite'):
            r=self.native()
            if edit=='names':r['joint_names'][0]=r['joint_names'][1]
            if edit=='limits':r['limits'][0][0][1]=-.1
            if edit=='gain':r['damping'][0][2]=1.
            if edit=='finite':r['masses'][0][3]=float('nan')
            with self.subTest(edit=edit),self.assertRaises(ValueError):core.validate_native(r,self.model)

    def test_sdf_paths_not_just_count(self):
        r={'count':153,'valid':True,'paths':[x['path']for x in self.usd['colliders']],
           'contract':'canonical_native_sdf_initialization_v2',
           'initialization_barrier':{'sim_reset_returned':True,'physics_view_valid':True,'articulation_view_valid':True},
           'legacy_task_counter':{'available':False,'pending':None}}
        self.assertTrue(core.validate_sdf(r,self.usd));r['paths'][0]='/Other'
        with self.assertRaises(ValueError):core.validate_sdf(r,self.usd)
        r['paths']=[x['path']for x in self.usd['colliders']];r['legacy_task_counter']={'available':True,'pending':1}
        with self.assertRaises(ValueError):core.validate_sdf(r,self.usd)

    def test_canonical_fk_against_usd_and_wrong_quaternion(self):
        from scipy.spatial.transform import Rotation
        names=[x['name']for x in self.usd['bodies']]
        poses=[]
        for body in self.usd['bodies']:
            T=np.asarray(body['world_transform']);poses.append([*T[:3,3],*Rotation.from_matrix(T[:3,:3]).as_quat()])
        r={'link_pose_xyzw':[poses],'joint_position':[np.zeros(18).tolist()]}
        joints=[x['name']for x in self.usd['joints']]
        self.assertTrue(core.validate_native_frames(r,names,joints,self.model))
        r['link_pose_xyzw'][0][1][0]+=.001
        with self.assertRaises(ValueError):core.validate_native_frames(r,names,joints,self.model)

    def test_samples_exclude_missing_and_nonfinite(self):
        rows=[{'explicit_step':n,'explicit_step_counter':100+n,'link_pose_xyzw':np.zeros((1,19,7)).tolist(),
               'link_com_velocity':np.zeros((1,19,6)).tolist(),'joint_position':np.zeros((1,18)).tolist(),
               'joint_velocity_sdk':np.zeros((1,18)).tolist()}for n in range(9)]
        self.assertTrue(core.validate_samples(rows))
        with self.assertRaises(ValueError):core.validate_samples(rows[1:])
        rows[3]['explicit_step_counter']=102
        with self.assertRaises(ValueError):core.validate_samples(rows)
        rows[3]['explicit_step_counter']=103
        rows[5]['joint_velocity_sdk'][0][0]=float('inf')
        with self.assertRaises(ValueError):core.validate_samples(rows)

    def test_runtime_has_no_control_or_state_setters(self):
        import ast
        tree=ast.parse((HERE/'run_inspection.py').read_text())
        calls=[n.func.attr for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)]
        self.assertFalse(any(x.startswith(('set_joint','write_joint','set_dof','set_masses','set_coms','set_inertias')) for x in calls))
        constructors=[n.func.id for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)]
        self.assertNotIn('Articulation',constructors)

    def test_failure_preserved_before_exit_close(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);state={'status':'completed','errors':[]}
            class App:
                def close(self):
                    self.asserted=contract.read(p/'state.json')['status']=='failed'
                    raise SystemExit(0)
            app=App()
            with patch.object(entry,'verify_inputs',side_effect=ValueError('changed')):
                with self.assertRaises(SystemExit):entry.finish(p,state,None,{},app)
            self.assertTrue(app.asserted);self.assertTrue((p/'failure.json').is_file())

    def test_nonfinite_raw_evidence_retained(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw.json';entry.save(p,{'rate':[float('nan')]})
            self.assertEqual(contract.read(p)['rate'][0],{'nonfinite':'nan'})

    def test_getter_failure_preserves_prior_native_properties(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            view=SimpleNamespace(count=1,shared_metatype=SimpleNamespace(fixed_base=False,link_names=['body'],dof_names=['joint']),
                get_masses=lambda:np.array([[7.4]]),get_coms=lambda:np.zeros((1,1,7)))
            with self.assertRaises(AttributeError):entry.collect_native(view,p)
            r=contract.read(p/'native_readback.json')
            self.assertEqual(r['masses'],[[7.4]]);self.assertIn('coms',r)
            self.assertEqual(r['failed_getter']['name'],'get_inertias')

    def test_stdlib_contract_import(self):
        code="import sys;sys.path.insert(0,"+repr(str(HERE))+");import inspection_contract;assert 'numpy' not in sys.modules and 'isaaclab' not in sys.modules"
        subprocess.run([sys.executable,'-S','-c',code],check=True)

    def test_failed_missing_receipt_cannot_complete(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);entry.save(p/'state.json',{'schema':contract.SCHEMA,'identity':{},'status':'failed'})
            with self.assertRaises(ValueError):contract.validate_result(p,{})
            entry.save(p/'state.json',{'schema':contract.SCHEMA,'identity':{},'status':'completed','inputs_unchanged':True,
                'physics_admitted':False,'physical_admission':False,'training_allowed':False,'explicit_steps_completed':8,'checks':{'finite_samples':True}})
            with self.assertRaises(ValueError):contract.validate_result(p,{})

    def test_post_close_native_error_rejects_completed_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            files=['usd_readback.json','native_readback.json','samples.json','sdf_readback.json','resolved_stage.usda','runtime_api.json']
            for name in files:entry.save(p/name,{})
            entry.save(p/'native_errors.json',[])
            state={'schema':contract.SCHEMA,'identity':{},'status':'completed','inputs_unchanged':True,
                   'errors':[],'native_error_events':[],'physics_admitted':False,'physical_admission':False,'training_allowed':False,
                   'explicit_steps_completed':8,'checks':{k:True for k in ['usd_identity','native_identity','native_frames','native_scene','native_sdf_paths','no_drive_gains','finite_samples','sdk_source_bound']},
                   'outputs':{name:contract.sha(p/name)for name in files}}
            entry.save(p/'state.json',state)
            self.assertEqual(contract.validate_result(p,{})['status'],'completed')
            entry.save(p/'native_errors.json',[{'type':1,'payload':'late shutdown error'}])
            with self.assertRaisesRegex(ValueError,'Late native'):contract.validate_result(p,{})
            entry.save(p/'native_errors.json',[]);state['native_error_events']=[{'type':1}];entry.save(p/'state.json',state)
            with self.assertRaisesRegex(ValueError,'Recorded native'):contract.validate_result(p,{})

    def test_scene_readback_rejects_gravity_dt_self_collision(self):
        r={'gravity_magnitude':0.,'time_steps_per_second':400,'manager_dt':.0025,'root_self_collisions':True}
        self.assertTrue(core.validate_scene(r))
        for key,value in [('gravity_magnitude',9.81),('time_steps_per_second',50),('manager_dt',.02),('root_self_collisions',False)]:
            x=dict(r);x[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):core.validate_scene(x)


if __name__=='__main__':unittest.main()
