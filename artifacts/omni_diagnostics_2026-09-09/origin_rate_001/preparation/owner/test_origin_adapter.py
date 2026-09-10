import ast,copy,json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'reference_physics_adapter_009/source_009/tools'))
sys.path.insert(0,str(HERE))
from matched_origin import expected_initial,matched_class,initial_readback,validate_ground_rows
from canonical_stance_startup import CanonicalStanceStartup
from origin_metrics import compare_measurements
P=json.loads((HERE/'origin_initial_state.json').read_text())

class Base:
 def __init__(self):
  self.device='cpu';self.num_envs=1;self._sim_step_counter=0;self.writes=[]
  self._terrain=SimpleNamespace(env_origins=torch.zeros(1,3));self.scene=SimpleNamespace(env_origins=self._terrain.env_origins)
  self._robot=SimpleNamespace(joint_names=P['joint_names_runtime'],body_names=['body'+str(i) for i in range(19)],data=SimpleNamespace())
  d=self._robot.data
  def write(key,value):setattr(d,key,value.clone());self.writes.append(key)
  def pose(*,root_pose,env_ids):
   write('root_link_pos_w',root_pose[:,:3]);write('root_link_quat_w',root_pose[:,3:])
   d.body_link_pos_w=root_pose[:,:3,None].transpose(1,2).expand(1,19,3).clone()
   d.body_link_quat_w=root_pose[:,None,3:].expand(1,19,4).clone()
  def vel(*,root_velocity,env_ids):write('root_com_lin_vel_w',root_velocity[:,:3]);write('root_link_ang_vel_w',root_velocity[:,3:])
  self._robot.write_root_pose_to_sim_index=pose;self._robot.write_root_velocity_to_sim_index=vel
  self._robot.write_joint_position_to_sim_index=lambda *,position,env_ids:write('joint_pos',position)
  self._robot.write_joint_velocity_to_sim_index=lambda *,velocity,env_ids:write('joint_vel',velocity)
  self._robot.set_joint_position_target_index=lambda *,target,env_ids:write('joint_pos_target',target)
  self._robot.set_joint_velocity_target_index=lambda *,target,env_ids:write('joint_vel_target',target)
  c=SimpleNamespace()
  def reset(q,r,env_ids):
   assert torch.equal(q,r);c.reference_position=q.double().clone()
   for key in ('reference_velocity','residual_position','residual_velocity'):setattr(c,key,torch.zeros_like(c.reference_position))
  c.reset=reset;self.reference_residual_controller=c
  for key in ('_previous_processed_joint_target','_processed_actions','omni_filtered_target'):setattr(self,key,torch.zeros(1,18))
  self._commands=torch.ones(1,3);self._has_previous_processed_joint_target=torch.zeros(1,dtype=torch.bool);self._reference_fresh=torch.ones(1,dtype=torch.bool)
 def _reset_idx(self,ids):self._commands.zero_()

class OriginTests(unittest.TestCase):
 def test_exact_common_state_only_xy_varies(self):
  a=expected_initial(P,'origin_a')
  for case,xy in [('near',(3,-5)),('far',(30,-50)),('near_opposite',(-3,5)),('origin_repeat',(0,0))]:
   b=expected_initial(P,case)
   for k in ('q','v','target','root_com_velocity'):np.testing.assert_array_equal(a[k],b[k])
   np.testing.assert_array_equal(a['root_pose'][:,2:],b['root_pose'][:,2:]);np.testing.assert_array_equal(b['root_pose'][0,:2],xy)
 def test_reset_writes_expected_sdk_keywords_and_readback(self):
  for case in ('origin_a','near','far','near_opposite','origin_repeat'):
   e=matched_class(Base,P,case)();e._reset_idx(None)
   self.assertTrue(initial_readback(e,P,case)['passed']);self.assertEqual(len(e.writes),8)
   for k in ('_processed_actions','_previous_processed_joint_target','omni_filtered_target'):np.testing.assert_array_equal(getattr(e,k).numpy(),expected_initial(P,case)['q'])
 def test_no_injection_after_first_substep(self):
  e=matched_class(Base,P,'near')();e._reset_idx(None);n=len(e.writes);e._sim_step_counter=8;e._reset_idx(None)
  self.assertEqual(len(e.writes),n);self.assertTrue(e.matched_origin_unexpected_reset)
 def test_counter0_state_mismatch_rejected(self):
  e=matched_class(Base,P,'near')();e._reset_idx(None);e._robot.data.joint_pos[0,0]+=.01
  self.assertFalse(initial_readback(e,P,'near')['passed'])
 def test_origin_alias_required(self):
  e=matched_class(Base,P,'far')();e._reset_idx(None);e.scene.env_origins=e._terrain.env_origins.clone()
  self.assertFalse(initial_readback(e,P,'far')['passed'])
 def test_invalid_initial_state_rejected(self):
  for key in ('joint_position_rad','joint_velocity_rad_s'):
   p=copy.deepcopy(P);p['initial'][key][0]=float('nan')
   with self.assertRaises(ValueError):expected_initial(p,'origin_a')
  p=copy.deepcopy(P);p['initial']['root_link_quaternion_world_xyzw']=[1,0,0,0]
  with self.assertRaises(ValueError):expected_initial(p,'origin_a')
 def test_first_knot_and_canonical_hold(self):
  q=expected_initial(P,'origin_a')['q'];nominal=np.asarray([0.]*6+[np.float32(np.deg2rad(40))]*6+[np.float32(np.deg2rad(120))]*6)[None,:]
  startup=CanonicalStanceStartup(q,nominal,np.full_like(q,-4),np.full_like(q,4))
  self.assertTrue(np.array_equal(startup.sample(0)['q_ref'],q));self.assertTrue(np.array_equal(startup.sample(200)['q_ref'],nominal))
  self.assertLess(startup.contract()['max_discrete_velocity_rad_s'],1.75);self.assertLess(startup.contract()['max_discrete_acceleration_rad_s2'],6)
 def test_ground_requires_infinite_z0_plane_no_extra_collider(self):
  row={'type':'Plane','collision_enabled':True,'world_normal':[0,0,1],'world_origin_m':[0,0,0]}
  self.assertTrue(validate_ground_rows([row],'plane'))
  self.assertFalse(validate_ground_rows([row,row],'plane'))
  for key,value in [('type','Mesh'),('collision_enabled',False),('world_normal',[0,1,0]),('world_origin_m',[0,0,.01])]:
   modified={**row,key:value};self.assertFalse(validate_ground_rows([modified],'plane'))
 def test_real009_angles_rate_exact_selected_bias(self):
  path=HERE.parent/'reference_physics_results_009/run/standing/physics_substeps.npz'
  with np.load(path) as z:
   d={k:(z[k][:,6:7].copy() if z[k].ndim==3 else z[k].copy()) for k in z.files}
  r=compare_measurements(d,(3,-5));row=next(x for x in r['joints'] if x['joint']=='revolute_2_1')
  self.assertAlmostEqual(row['reported_integral_minus_delta_rad']['trapezoid'],.4103967515123078,places=12)
  self.assertAlmostEqual(row['angle_range_rad'],.0007212162017822266,places=12)
 def test_entry_has_no_wave_or_pose_writes(self):
  tree=ast.parse((HERE/'run_origin_physics.py').read_text());text=ast.unparse(tree)
  self.assertNotIn('WaveContactReference',text);self.assertNotIn('write_root',text);self.assertNotIn('write_joint',text)
  self.assertIn('PhysicsSubstepRecorder',text);self.assertIn('require_single_pre_reset_sample',text)
  self.assertIn('initial_readback',text);self.assertIn('ground_readback',text)


class HostContractTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import importlib.util
  spec=importlib.util.spec_from_file_location('origin_host',HERE/'launch_origin_physics_spark.py');cls.host=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.host)
 def test_six_only_commands_and_readonly_mounts(self):
  h=self.host
  for phase in ('standing',*h.CASES):
   argv=h.command(Path('/source'),Path('/output'),'owned',phase)
   self.assertIn('/source:/workspace/hexapod:ro',argv);self.assertIn('/output/inputs/study:/study:ro',argv)
   self.assertIn('--steps',argv);self.assertEqual(argv[argv.index('--steps')+1],'1000')
   if phase=='standing':self.assertNotIn('--case',argv)
   else:self.assertEqual(argv[argv.index('--case')+1],phase);self.assertEqual(argv[argv.index('--mode')+1],'origin')
  for phase in ('wave','train','origin_typo'):
   with self.assertRaises(ValueError):h.command(Path('/source'),Path('/output'),'owned',phase)
 def test_unknown_docker_inspection_error_not_absence(self):
  from unittest.mock import patch
  result=SimpleNamespace(returncode=1,stderr='permission denied',stdout='')
  with patch.object(self.host.subprocess,'run',return_value=result):
   with self.assertRaises(RuntimeError):self.host.owned_container('owned')
 def test_cleanup_and_limits_parent_ast_preserved(self):
  p=ast.parse((HERE.parent/'reference_physics_adapter_009/source_009/tools/launch_reference_physics_spark.py').read_text());n=ast.parse((HERE/'launch_origin_physics_spark.py').read_text())
  for name in ('owned_container','run_owned','tree_hashes'):
   old=next(x for x in p.body if isinstance(x,ast.FunctionDef) and x.name==name);new=next(x for x in n.body if isinstance(x,ast.FunctionDef) and x.name==name)
   self.assertEqual(ast.dump(old),ast.dump(new))
 def test_case_contract_rejects_wrong_source_and_failed_quiet(self):
  import tempfile
  from unittest.mock import patch
  import origin_contract as c
  with tempfile.TemporaryDirectory() as td:
   admission=Path(td)/'admission.json';args=SimpleNamespace(mode='origin',case='near',num_envs=1,steps=1000,admission=admission)
   identity={'source':'new-source'}
   valid={'identity':identity,'mode':'standing','status':'completed','gate':{'passed':True,'num_envs':32,'control_steps':1000,'all_replica_quiet':{'passed':True}}}
   with patch.object(c,'standing_preflight',return_value=(identity,{})),patch.object(c,'load_initial',return_value=P):
    admission.write_text(json.dumps(valid));self.assertEqual(c.preflight(args,Path(td))[0]['case'],'near')
    for change in ('identity','quiet'):
     bad=copy.deepcopy(valid)
     if change=='identity':bad['identity']={'source':'old009'}
     else:bad['gate']['all_replica_quiet']['passed']=False
     admission.write_text(json.dumps(bad))
     with self.assertRaises(ValueError):c.preflight(args,Path(td))

class MetricConventionTests(unittest.TestCase):
 def test_constant_named_angle_and_world_xyzw_yaw_rate(self):
  n=8001;t=np.arange(n)*.0025;zeros=np.zeros((n,1,3));q=np.zeros((n,1,18));q[:,:,2]=t[:,None]*.01
  v=np.zeros_like(q);v[:,:,2]=.01;quat=np.zeros((n,1,4));quat[:,0,2]=np.sin(t*.1/2);quat[:,0,3]=np.cos(t*.1/2)
  angular=zeros.copy();angular[:,:,2]=.1
  d={'time_s':t,'relative_physics_index':np.arange(n),'control_index':np.r_[-1,np.repeat(np.arange(1000),8)],'substep_index':np.r_[0,np.tile(np.arange(1,9),1000)],'joint_position_rad':q,'joint_velocity_rad_s':v,'joint_names':np.asarray(P['joint_names_runtime']),'root_link_quaternion_world_xyzw':quat,'root_angular_velocity_world_rad_s':angular}
  for frame in ('link','com'):
   d['root_'+frame+'_position_world_m']=zeros.copy();d['root_'+frame+'_velocity_world_mps']=zeros.copy()
  r=compare_measurements(d,(0,0))
  self.assertLess(abs(r['joints'][2]['reported_integral_minus_delta_rad']['trapezoid']),1e-12)
  self.assertLess(max(r['world_angular']['interval_disagreement_rms_rad_s'][0]),1e-12)
  self.assertAlmostEqual(r['world_angular']['sum_actual_increment_rotation_vectors_rad'][0][2],1.6,places=12)

if __name__=='__main__':unittest.main()
