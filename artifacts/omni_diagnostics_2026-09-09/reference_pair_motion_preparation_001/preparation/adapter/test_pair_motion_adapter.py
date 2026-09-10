"""Synthetic end-to-end adapter/scoring seams; no Isaac or real contact admission."""
import ast,copy,json,sys,tempfile,unittest,time,traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent/'reference_physics_adapter_009/source_009';O=H.parent/'reference_pair_motion_001'
sys.path.insert(0,str(P/'tools'));sys.path.insert(0,str(O));sys.path.insert(0,str(H/'paired_runtime'));sys.path.insert(0,str(H))
from pair_motion import PairContactReference
from ideal_fixture import IdealMeasuredFixture
from pair_motion_contract import PROTOCOL,STEPS,preflight
import pair_motion_contract as contract
from pair_motion_metrics import score_pair_motion,replay_pair_reference,substep_evidence,expected_request
from pair_motion_measurements import measured_support,check_sensor_row,torque_window
import launch_pair_motion_spark as host
from screen_contract import save

def serial(v):
 if isinstance(v,np.ndarray):return serial(v.tolist())
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {k:serial(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [serial(x) for x in v]
 return v

def sensor_fields(row):
 row.update(sensor_all_valid=np.ones(1,bool),sensor_contact_valid=np.ones((1,6),bool),sensor_age_s=np.zeros((1,14)),sensor_outdated=np.zeros((1,14),bool),
  sensor_timestamp_s=np.full((1,14),row['time_s'][0]),sensor_last_update_s=np.full((1,14),row['time_s'][0]),sensor_expected_timestamp_s=np.full((1,14),row['time_s'][0]))
 return row

def fixture():
 f=IdealMeasuredFixture();g=None;rows=[];refs=[];checks=[]
 for step in range(STEPS):
  if step==200:
   g=PairContactReference(f.names);out=g.reset(f.snapshot());refs.append({'physical_step':step,'reset':serial(out)})
  if step>=200:
   out=g.step(rows[-1],expected_request(step));assert out['valid'][0],out['failure_reason'];refs.append({'physical_step':step,'result':serial(out)});f.advance(out)
  else:f.time=(step+1)*.02
  row=sensor_fields(f.snapshot());req=expected_request(step)
  row.update(velocity_world_mps=np.einsum('bij,bj->bi',row['rotation_world_from_body'],row['velocity_body_mps']),
   requested_command=req[None],reference_admitted_target_twist=req[None],reference_filtered_twist=(out['admitted_command'] if g else np.zeros(3))[None],
   actual_executed_body_navigation_twist=np.stack((-row['velocity_body_mps'][:,1],row['velocity_body_mps'][:,0],row['gyro_body_rad_s'][:,2]),-1),
   reference_to_executable_lag_rad=np.zeros((1,18)),position_target_cast_error_rad=np.zeros((1,18)))
  rows.append(row)
  if g is not None:checks.append({'physical_step':step,'support':measured_support(g,row),'torque':{'max_requested_nm':.7,'max_applied_nm':.7,'substeps':8}})
 d={k:np.stack([row[k] for row in rows]) for k in rows[0]}
 count=STEPS*8+1;physics={'relative_physics_index':np.arange(count),'time_s':np.arange(count)*.0025,'sim_step_counter':np.arange(count)+912,
 'control_index':np.r_[-1,np.repeat(np.arange(STEPS),8)],'substep_index':np.r_[0,np.tile(np.arange(1,9),STEPS)]}
 for field,key in [('root_link_position_world_m','position_world_m'),('root_link_quaternion_world_xyzw','quaternion_world_xyzw'),('root_link_velocity_world_mps','velocity_world_mps'),
  ('joint_position_rad','joint_position_rad'),('joint_velocity_rad_s','joint_velocity_rad_s'),('computed_torque_nm','computed_torque_nm'),('applied_torque_nm','applied_torque_nm')]:
  physics[field]=np.concatenate([d[key][:1],np.repeat(d[key],8,axis=0)])
 return d,refs,physics,checks,f.names

def functions(*names):
 ns=dict(np=np,json=json,time=time,traceback=traceback,save=save)
 tree=ast.parse((H/'run_pair_motion.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),'<runtime functions>','exec'),ns);return ns

class FullScore(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.data,cls.refs,cls.physics,cls.checks,cls.names=fixture()
 def test_full44_sequence_after4startup_scores_and_replays(self):
  g=score_pair_motion(self.data,self.refs,self.physics,self.checks,joint_names=self.names)
  self.assertTrue(g['passed'],g);self.assertEqual(g['completed_pairs'],10);self.assertEqual(g['generator_confirmed_touchdowns'],20)
  self.assertEqual(g['full_state_numerical_replay']['max_absolute_difference'],0);self.assertFalse(g['old846_849_packet_admitted']);json.dumps(g,allow_nan=False)
 def test_intermediate_substep_torque_spike_is_not_hidden_by_control_sample(self):
  p=copy.deepcopy(self.physics);p['computed_torque_nm'][320*8+3,0,2]=1.61
  c=copy.deepcopy(self.checks);c[120]={'physical_step':320,'failure':'Paired400Hz requested/applied torque limit exceeded'}
  g=score_pair_motion(self.data,self.refs,p,c,joint_names=self.names)
  self.assertFalse(g['passed']);self.assertEqual(g['complete400Hz']['poststartup_requested_excess_samples'],1);json.dumps(g,allow_nan=False)
 def test_original5mm_gate_not_replaced(self):
  d=copy.deepcopy(self.data);p=copy.deepcopy(self.physics);d['velocity_world_mps'][300:1500,0,0]+=.001;p['root_link_velocity_world_mps'][8::8]=d['velocity_world_mps']
  g=score_pair_motion(d,self.refs,p,self.checks,joint_names=self.names)
  self.assertFalse(g['passed']);self.assertGreater(g['independent_forward_motion']['displacement_integral_difference_m'],.005)
 def test_last_row_required_support_and_quiet_not_omitted(self):
  d=copy.deepcopy(self.data);p=copy.deepcopy(self.physics);d['joint_velocity_rad_s'][-600:,0,0]=.04;p['joint_velocity_rad_s'][8::8]=d['joint_velocity_rad_s']
  g=score_pair_motion(d,self.refs,p,self.checks,joint_names=self.names);self.assertFalse(g['passed']);self.assertFalse(g['final_quiet_stop_window']['pass'])
  c=copy.deepcopy(self.checks);c[-1]={'physical_step':2399,'failure':'A required measured retained support is missing'}
  d=copy.deepcopy(self.data);d['distal_contact'][-1,0,0]=False;d['contact_point_valid'][-1,0,0]=False;d['normal_force_world_n'][-1,0,0]=0.;d['contact_point_world_m'][-1,0,0]=np.nan
  g=score_pair_motion(d,self.refs,self.physics,c,joint_names=self.names);self.assertFalse(g['passed']);self.assertFalse(g['poststep_checks_complete'])
 def test_poststep_margin_is_recomputed_from_raw_not_trusted(self):
  c=copy.deepcopy(self.checks);c[120]['support']['support_margin_m']+=.01
  with self.assertRaisesRegex(ValueError,'poststep'):score_pair_motion(self.data,self.refs,self.physics,c,joint_names=self.names)
 def test_each_foot_state_replayed_and_tamper_rejected(self):
  r=copy.deepcopy(self.refs);row=next(x['result'] for x in r if 'result' in x and x['result']['state']['active_pair'])
  row['state']['foot_cycles'][1]['flight_seen']=not row['state']['foot_cycles'][1]['flight_seen']
  with self.assertRaises(ValueError):replay_pair_reference(self.data,r,self.names)
 def test_timestamp_command_and_substep_boundary_fail_closed(self):
  for kind in ('time','command','counter','boundary'):
   d=copy.deepcopy(self.data);p=copy.deepcopy(self.physics)
   if kind=='time':d['time_s'][301,0]+=.001
   elif kind=='command':d['requested_command'][300,0,0]=.015
   elif kind=='counter':p['sim_step_counter'][1000]+=1
   else:p['joint_position_rad'][800,0,0]+=.01
   with self.assertRaises(ValueError):score_pair_motion(d,self.refs,p,self.checks,joint_names=self.names)

class BoundaryTests(unittest.TestCase):
 def test_missing_contact_point_force_and_stale_sensor_rejected(self):
  base=sensor_fields(IdealMeasuredFixture().snapshot())
  for mode in ('point','force','age','clock','valid'):
   row=copy.deepcopy(base)
   if mode=='point':row['contact_point_valid'][0,0]=False
   elif mode=='force':row['normal_force_world_n'][0,0]=[0,0,1.]
   elif mode=='age':row['sensor_age_s'][0,0]=.0025
   elif mode=='clock':row['sensor_expected_timestamp_s'][0,0]+=.0025
   else:row['sensor_all_valid'][0]=False
   with self.assertRaises(ValueError):check_sensor_row(row)
 def test_poststep_checks_are_nonmutating_and_no_missing_foot_tolerated(self):
  f=IdealMeasuredFixture();g=PairContactReference(f.names);r=g.reset(f.snapshot());before=serial(r['state']);row=sensor_fields(f.snapshot())
  result=measured_support(g,row);self.assertEqual(result['required_leg_indices'],list(range(6)));self.assertEqual(before,serial(g._output(g.q,g.v,np.zeros_like(g.q),g._read(row))['state']))
  row['distal_contact'][0,0]=False;row['contact_point_valid'][0,0]=False
  with self.assertRaises(ValueError):measured_support(g,row)
 def test_exact8_torque_window_checks_each_before_next_target(self):
  rows=[{'computed_torque_nm':np.full((1,18),.7),'applied_torque_nm':np.full((1,18),.7)} for _ in range(8)]
  self.assertEqual(torque_window(rows)['substeps'],8)
  with self.assertRaises(ValueError):torque_window(rows[:7])
  rows[2]['computed_torque_nm'][0,2]=1.61
  with self.assertRaises(ValueError):torque_window(rows)
 def test_exact_new_fresh_standing_identity_required(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'admission.json';args=SimpleNamespace(mode='paired_motion',case='paired_forward',num_envs=1,steps=2400,admission=p)
   identity={'source':'pairednew'};good={'identity':identity,'mode':'standing','status':'completed','gate':{'passed':True,'num_envs':32,'control_steps':1000,'all_replica_quiet':{'passed':True}}}
   with patch.object(contract,'standing_preflight',return_value=(identity,{})),patch.object(contract,'check_pair'):
    p.write_text(json.dumps(good));self.assertEqual(preflight(args,Path(td))[0]['paired_motion_protocol'],PROTOCOL)
    good['identity']={'source':'009'};p.write_text(json.dumps(good))
    with self.assertRaises(ValueError):preflight(args,Path(td))
 def test_host_readonly_modes_and_original_owned_cleanup_ast(self):
  for phase in ('standing','paired_forward'):
   c=host.command(Path('/src'),Path('/out'),'owned',phase);self.assertIn('/src:/workspace/hexapod:ro',c)
   self.assertEqual(c[c.index('--steps')+1],'1000' if phase=='standing' else '2400')
   self.assertEqual(c[c.index('--mode')+1],'standing' if phase=='standing' else 'paired_motion')
  for phase in ('wave','train','left_strafe'):
   with self.assertRaises(ValueError):host.command(Path('/src'),Path('/out'),'owned',phase)
  old=ast.parse((P/'tools/launch_reference_physics_spark.py').read_text());new=ast.parse((H/'launch_pair_motion_spark.py').read_text())
  for name in ('run_owned','owned_container','tree_hashes'):
   pick=lambda tree:ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
   self.assertEqual(pick(old),pick(new))
 def test_finalization_preserves_first_failure_and_secondary_export(self):
  ns=functions('serializable','save_runtime_json','save_failure_state','main')
  def fail(*_):raise RuntimeError('firstenvironmentfailure')
  real=ns['save_runtime_json']
  def save_fail(path,value):
   if path.name=='paired_poststep_checks.json':raise TypeError('checks export failed')
   return real(path,value)
  with tempfile.TemporaryDirectory() as td:
   ns.update(args=SimpleNamespace(output=Path(td),mode='paired_motion'),identity={},RUNTIME={},OPTIONS={},build_reference_environment=fail,save_runtime_json=save_fail)
   with self.assertRaisesRegex(RuntimeError,'firstenvironmentfailure'):ns['main']()
   row=json.loads((Path(td)/'state.json').read_text());self.assertEqual(row['status'],'failed');self.assertIn('checks export failed',row['paired_checks_export_error']);self.assertEqual(row['control_steps'],0)
   json.dumps(row,allow_nan=False)
if __name__=='__main__':unittest.main()
