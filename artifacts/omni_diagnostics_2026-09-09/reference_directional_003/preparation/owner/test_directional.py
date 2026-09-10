"""Synthetic metric/host regressions; no physics trajectories are implied."""
import ast,copy,json,sys,tempfile,unittest,time,traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent/'reference_physics_adapter_009/source_009'
sys.path.insert(0,str(PARENT/'tools'));sys.path.insert(0,str(HERE))
from directional_contract import CASES,preflight
import directional_contract as contract
from directional_metrics import motion_evidence,score_direction
import launch_directional_physics_spark as host
from screen_contract import save

def runtime_functions(*names):
 tree=ast.parse((HERE/'run_directional_physics.py').read_text())
 namespace=dict(np=np,json=json,time=time,traceback=traceback,save=save)
 selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 exec(compile(ast.Module(body=selected,type_ignores=[]),'<actual runtime functions>','exec'),namespace)
 return namespace

def fixture(case,actual=None):
 req=np.array(CASES[case]);actual=np.array(actual if actual is not None else req);n=2400;dt=.02
 requested=np.zeros((n,1,3));requested[200:1400,0]=req
 twist=np.zeros((n,1,3));twist[200:1400,0]=actual
 angle=np.cumsum(twist[:,0,2])*dt;c=np.cos(angle);s=np.sin(angle)
 rot=np.zeros((n,1,3,3));rot[:,0,0,0]=c;rot[:,0,1,1]=c;rot[:,0,0,1]=-s;rot[:,0,1,0]=s;rot[:,0,2,2]=1
 body=np.stack([twist[...,1],-twist[...,0],np.zeros((n,1))],-1);vel=(rot@body[...,None])[...,0]
 pos=np.zeros((n,1,3));pos[...,2]=.13
 pos[1:]+=np.cumsum(.5*(vel[1:]+vel[:-1])*dt,axis=0)
 q=np.zeros((n,1,18));foot=np.zeros((n,1,6,3));contact=np.ones((n,1,6),bool)
 for leg in range(6):
  a=230+leg*120;contact[a:a+30,0,leg]=False;foot[a:a+30,0,leg,2]=.003
 d={'time_s':(np.arange(n)+1)[:,None]*dt,'position_world_m':pos,'rotation_world_from_body':rot,'velocity_world_mps':vel,'velocity_body_mps':body,
 'gyro_body_rad_s':np.stack([np.zeros((n,1)),np.zeros((n,1)),twist[...,2]],-1),
 'quaternion_world_wxyz':np.stack([np.cos(angle/2),np.zeros(n),np.zeros(n),np.sin(angle/2)],-1)[:,None],
 'joint_position_rad':q.copy(),'joint_velocity_rad_s':q.copy(),'joint_target_rad':q.copy(),'computed_torque_nm':np.full_like(q,.7),'applied_torque_nm':np.full_like(q,.7),
 'shaft_contact':np.zeros((n,1,6),bool),'coxa_contact':np.zeros((n,1,6),bool),'femur_contact':np.zeros((n,1,6),bool),'base_contact':np.zeros((n,1),bool),
 'terminated':np.zeros((n,1),bool),'truncated':np.zeros((n,1),bool),'distal_contact':contact,'reference_point_world_m':foot,
 'reference_to_executable_lag_rad':q.copy(),'position_target_cast_error_rad':q.copy(),'requested_command':requested,
 'reference_admitted_target_twist':requested.copy(),'reference_filtered_twist':requested.copy(),'actual_executed_body_navigation_twist':twist.copy()}
 ref=[{'result':{'valid':[True],'command_derating_factor':1.,'state':{'mode':'reference_quiet_hold','reference_quiet_time_s':32.,'confirmed_touchdowns':6}}}]
 return d,ref

class MotionTests(unittest.TestCase):
 def test_correct_signed_turn_arc(self):
  for case in CASES:
   d,r=fixture(case);g=score_direction(d,r,case=case,joint_names=[f'joint{i}' for i in range(18)])
   self.assertTrue(g['passed'],(case,g['independent_directional_motion']));self.assertFalse(g['all_omni_directions_qualified'])
 def test_wrong_translation_or_yaw_direction_fails(self):
  for case in CASES:
   d,_=fixture(case,-np.array(CASES[case]));m=motion_evidence(d,CASES[case])
   self.assertFalse(m['signed_translation_pass'] and m['signed_yaw_pass'],case)
 def test_pure_turn_needs_rotation_not_translation(self):
  d,_=fixture('left_turn');m=motion_evidence(d,CASES['left_turn'])
  self.assertTrue(m['signed_translation_pass']);self.assertTrue(m['signed_yaw_pass']);self.assertEqual(m['max_planar_excursion_from_start_m'],0.)
  d,_=fixture('left_turn',[0,0,0]);self.assertFalse(motion_evidence(d,CASES['left_turn'])['signed_yaw_pass'])
 def test_pure_turn_drift_rejected(self):
  d,_=fixture('left_turn',[.003,0,.015]);self.assertFalse(motion_evidence(d,CASES['left_turn'])['signed_translation_pass'])
 def test_same5mm_pose_rate_gate_rejects_despite_good_direction(self):
  d,r=fixture('left_turn');d['velocity_world_mps'][200:1400,0,0]+=.001
  g=score_direction(d,r,case='left_turn',joint_names=[f'joint{i}' for i in range(18)]);self.assertFalse(g['passed'])
  self.assertGreater(g['independent_directional_motion']['old_world_displacement_integral_evidence']['displacement_integral_difference_m'],.005)
 def test_command_time_and_admitted_and_actual_channels_bound(self):
  for key in ['requested_command','reference_admitted_target_twist','actual_executed_body_navigation_twist']:
   d,r=fixture('left_turn');d[key][200,0,1]+=.001
   with self.assertRaises(ValueError):score_direction(d,r,case='left_turn',joint_names=[f'joint{i}' for i in range(18)])
 def test_timebase_mismatch_rejected(self):
  d,r=fixture('left_turn');d['time_s'][201,0]+=.01
  with self.assertRaises(ValueError):score_direction(d,r,case='left_turn',joint_names=[f'joint{i}' for i in range(18)])
 def test_contact_flight_torque_terminal_and_quiet_not_waived(self):
  for kind in ['flight','torque','terminal','quiet']:
   d,r=fixture('left_turn')
   if kind=='flight':d['distal_contact'][:]=True
   if kind=='torque':d['computed_torque_nm'][900,0,0]=1.61
   if kind=='terminal':d['terminated'][900,0]=True
   if kind=='quiet':d['joint_velocity_rad_s'][1800:,0,0]=.04
   g=score_direction(d,r,case='left_turn',joint_names=[f'joint{i}' for i in range(18)]);self.assertFalse(g['passed'],kind)

class HostTests(unittest.TestCase):
 def test_only_previously_unmeasured_cases_allowed(self):
  self.assertEqual(list(CASES),['left_turn','forward_right_arc'])
  self.assertFalse(contract.PROTOCOL['prior_measured_case_not_retried']['admitted'])
  with self.assertRaises(ValueError):host.command(Path('/src'),Path('/out'),'owner','reverse')
  with self.assertRaises(ValueError):host.command(Path('/src'),Path('/out'),'owner','left_strafe')
 def test_exact_case_modes_counts_and_readonly_mounts(self):
  for case in ['standing',*CASES]:
   cmd=host.command(Path('/src'),Path('/out'),'owner',case)
   self.assertEqual(cmd[cmd.index('--steps')+1],'1000' if case=='standing' else '2400')
   self.assertEqual(cmd[cmd.index('--mode')+1],'standing' if case=='standing' else 'directional')
   self.assertIn('/src:/workspace/hexapod:ro',cmd)
  with self.assertRaises(ValueError):host.command(Path('/src'),Path('/out'),'owner','wave')
 def test_owned_cleanup_ast_and_no_origin_injection(self):
  old=ast.parse((PARENT/'tools/launch_reference_physics_spark.py').read_text());new=ast.parse((HERE/'launch_directional_physics_spark.py').read_text())
  for name in ['run_owned','owned_container','tree_hashes']:
   a=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name);b=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name);self.assertEqual(ast.dump(a),ast.dump(b))
  runtime=(HERE/'run_directional_physics.py').read_text();self.assertNotIn('write_root',runtime);self.assertNotIn('write_joint',runtime);self.assertNotIn('sim.step',runtime);self.assertNotIn('patched_factory',runtime)
 def test_exact_fresh_standing_and_wave_required(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'receipt.json';a=SimpleNamespace(mode='directional',case='left_turn',num_envs=1,steps=2400,admission=p);identity={'source':'new_directional'}
   good={'mode':'standing','identity':identity,'status':'completed','gate':{'passed':True,'num_envs':32,'control_steps':1000,'all_replica_quiet':{'passed':True}}}
   with patch.object(contract,'standing_preflight',return_value=(identity,{})),patch.object(contract,'check_wave'):
    p.write_text(json.dumps(good));self.assertEqual(preflight(a,Path(td))[0]['case'],'left_turn')
    for field in ['oldsource','quiet']:
     bad=copy.deepcopy(good)
     if field=='oldsource':bad['identity']={'source':'009'}
     else:bad['gate']['all_replica_quiet']['passed']=False
     p.write_text(json.dumps(bad))
     with self.assertRaises(ValueError):preflight(a,Path(td))

class SerializationTests(unittest.TestCase):
 def test_full_pass_and_rejected_gate_strict_json_for_every_case(self):
  for case in CASES:
   for bad in [False,True]:
    d,r=fixture(case)
    if bad:d['computed_torque_nm'][900,0,0]=1.61
    gate=score_direction(d,r,case=case,joint_names=[f'joint{i}' for i in range(18)])
    self.assertEqual(gate['passed'],not bad)
    state={'status':'rejected' if bad else 'completed','identity':{'directional_protocol':contract.PROTOCOL,'case':case},
           'gate':gate,'control_steps':2400,'failure':None,'physics_substep_review':{'passed':True}}
    encoded=json.dumps(state,allow_nan=False)
    self.assertEqual(json.loads(encoded),state)
    self.assertIs(type(gate['independent_directional_motion']['yaw_required']),bool)
 def test_actual_reverse001_gate_exception_is_strictly_recorded_without_admission(self):
  ns=runtime_functions('serializable','save_runtime_json','save_failure_state')
  gate=json.loads((HERE/'reverse001_replay.json').read_text())['original_gate_posthoc_replay']
  self.assertFalse(gate['passed'])
  gate['independent_directional_motion']['yaw_required']=np.bool_(False)
  state={'status':'running','gate':gate,'identity':{'source':'preserved001'},'control_steps':2400}
  with self.assertRaises(TypeError):json.dumps(state,allow_nan=False)
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'state.json'
   result=ns['save_failure_state'](p,state,TypeError('original NumPy finalization failure'),'original traceback',2400)
   self.assertEqual(json.loads(p.read_text()),result)
   self.assertEqual(result['status'],'failed');self.assertFalse(result['gate']['passed'])
   self.assertIn('original NumPy',result['error']);json.dumps(result,allow_nan=False)
 def test_invalid_nonfinite_or_object_gate_retains_failed_fallback_and_evidence(self):
  ns=runtime_functions('serializable','save_failure_state')
  for bad in [float('nan'),object()]:
   with tempfile.TemporaryDirectory() as td:
    p=Path(td)/'state.json';result=ns['save_failure_state'](p,{'gate':{'bad':bad}},ValueError('first failure'),'trace',2400)
    self.assertEqual(result['status'],'failed');self.assertIsNone(result['gate'])
    self.assertFalse(result['gate_serialization_valid']);self.assertIn('bad',result['invalid_state_repr'])
    self.assertEqual(json.loads(p.read_text()),result);json.dumps(result,allow_nan=False)
 def test_actual_exception_handler_keeps_original_error_when_reference_export_fails(self):
  ns=runtime_functions('serializable','save_runtime_json','save_failure_state','main')
  def setup_failure(*_):raise RuntimeError('first environment failure')
  original=ns['save_runtime_json']
  def export_failure(path,value):
   if path.name=='reference_states.json':raise TypeError('reference export failure')
   return original(path,value)
  with tempfile.TemporaryDirectory() as td:
   ns.update(args=SimpleNamespace(output=Path(td),mode='directional'),identity={'source':'test'},RUNTIME={},OPTIONS={},
             build_reference_environment=setup_failure,save_runtime_json=export_failure)
   with self.assertRaisesRegex(RuntimeError,'first environment failure'):ns['main']()
   saved=json.loads((Path(td)/'state.json').read_text())
   self.assertEqual(saved['status'],'failed');self.assertIn('first environment failure',saved['error'])
   self.assertIn('reference export failure',saved['reference_states_export_error'])
   self.assertEqual(saved['control_steps'],0);json.dumps(saved,allow_nan=False)
if __name__=='__main__':unittest.main()
