"""Synthetic evidence tests exercise actual scoring/host contracts, not physical feasibility."""
from pathlib import Path
import tempfile,unittest,json,copy
import numpy as np
from standing_score import score
from standing_contract import validate_result,SCHEMA,sha
from test_standing import NAMES,MODEL,save

class Evidence(unittest.TestCase):
 def fixture(self,d):
  t=8000;n=1;zeros=np.zeros((t,n,18),np.float32)
  r={k:zeros.copy()for k in ['joint_position_rad','joint_velocity_rad_s','pre_joint_position_rad','pre_joint_velocity_rad_s','computed_torque_nm','applied_torque_nm','native_input_pre_nm','joint_target_rad','interval_angle_rate_rad_s']}
  r['effort_ceiling_nm']=np.full_like(zeros,1.6);r['sequence']=np.arange(t);r['explicit_counter']=np.arange(32,8032);r['control_index']=np.arange(t)//8;r['substep_index']=np.arange(t)%8;r['time_s']=(np.arange(t)+1)*.0025
  r['root_pose_xyzw']=np.zeros((t,n,7));r['root_pose_xyzw'][...,2]=.076611091013;r['root_pose_xyzw'][...,6]=1
  for k in ['terminated','truncated','nonfoot_contact']:r[k]=np.zeros((t,n),bool)
  for k in ['contact_valid','interval_valid']:r[k]=np.ones((t,n),bool)
  r['distal_contact']=np.ones((t,n,6),bool);r['minimum_non_toe_floor_m']=np.full((t,n),.015)
  native={'joint_names':NAMES,'limits':[[[j['lower'],j['upper']]for j in MODEL['joints']]],'native_max_velocity':np.full((1,18),50.26548).tolist()};save(d/'native_readback.json',native)
  save(d/'session.json',{'steps':8000,'captured_steps':8000,'all_rows_recorded':True,'controls':1000,'reset_count':1,'failure':None,'substep_files':['substeps_000.npz'],'joint_names':NAMES,'root_paths':['/Robot']})
  save(d/'initial_reset.json',{'counter_after':31,'post_reset':{'joint_position_rad':np.zeros((1,18)).tolist(),'joint_velocity_rad_s':np.zeros((1,18)).tolist()}})
  return r
 def write_arrays(self,d,r):
  np.savez_compressed(d/'substeps_000.npz',**r);np.savez_compressed(d/'control_trace.npz',**{k:v[7::8]for k,v in r.items()})
 def test_full_clock_servo_endpoint_quiet_replay_and_startup_nonfoot_rejection(self):
  with tempfile.TemporaryDirectory()as x:
   d=Path(x);r=self.fixture(d);self.write_arrays(d,r);report=score(d);self.assertTrue(report['all_pass']);self.assertEqual(report['substeps'],8000)
   r['nonfoot_contact'][0,0]=True;self.write_arrays(d,r);report=score(d);self.assertFalse(report['all_pass']);self.assertIn('nonfoot_contact',report['replicas'][0]['failed_physical_bounds'])
   r['nonfoot_contact'][0,0]=False;r['minimum_non_toe_floor_m'][0,0]=-.00101;self.write_arrays(d,r);report=score(d);self.assertIn('non_toe_floor_clearance',report['replicas'][0]['failed_physical_bounds'])
 def test_interior_clock_native_input_and_late_missing_contact_fail(self):
  with tempfile.TemporaryDirectory()as x:
   d=Path(x);r=self.fixture(d)
   for field in ['explicit_counter','native_input_pre_nm']:
    old=r[field].copy();r[field][3]+=1;self.write_arrays(d,r)
    with self.assertRaises(ValueError):score(d)
    r[field]=old
   r['distal_contact'][1603,0,2]=False;self.write_arrays(d,r);self.assertFalse(score(d)['all_pass'])
 def test_complete_flag_cannot_hide_numeric_quiet_failure_or_late_native_error(self):
  with tempfile.TemporaryDirectory()as x:
   d=Path(x);r=self.fixture(d);self.write_arrays(d,r);report=score(d);save(d/'standing_report.json',report)
   for f in ['contact_view.json','sdf_readback.json']:save(d/f,{})
   (d/'contacts.jsonl').write_text('');save(d/'native_errors.json',[])
   identity={'num_envs':1};state={'schema':SCHEMA,'identity':identity,'status':'completed','inputs_unchanged':True,'errors':[],'native_error_events':[],'physical_admission':False,'physics_admitted':False,'training_allowed':False,'explicit_steps_completed':8000,'checks':{'test':True}}
   def seal():
    state['outputs']={p.name:sha(p)for p in d.iterdir()if p.name not in ['state.json','native_errors.json']};save(d/'state.json',state)
   seal();self.assertTrue(validate_result(d,identity)['standing_pass'])
   report['replicas'][0]['quiet']['max_joint_velocity_rms_rad_s']=.03001;save(d/'standing_report.json',report);seal()
   with self.assertRaises(ValueError):validate_result(d,identity)
   report['replicas'][0]['quiet']['max_joint_velocity_rms_rad_s']=0.;save(d/'standing_report.json',report);seal();save(d/'native_errors.json',[{'late':'error'}])
   with self.assertRaises(ValueError):validate_result(d,identity)
if __name__=='__main__':unittest.main()
