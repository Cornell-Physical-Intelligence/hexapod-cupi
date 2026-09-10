"""Synthetic driver integration at physical time4..26s; no simulated-physics evidence."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'reference_load_transfer_001'))
sys.path.insert(0,str(HERE/'source_pair_001/tools'))
from synthetic_fixture import Fixture
from load_transfer import PairLoadTransfer
from score_transfer import score_transfer,check_substep_batch
from reference_residual import ReferenceResidualTarget,ResidualConfig
from pair_rollout import emit_control,check_measured_pair_state

class CompletePairWiringTests(unittest.TestCase):
 def test_real_frozen_generator_core_driver_scorer_share_full_window(self):
  f=Fixture();f.time=4.;g=PairLoadTransfer(f.names);start=f.snapshot();g.reset(start)
  limits=start['soft_joint_pos_limits_rad'][0]
  core=ReferenceResidualTarget(f.names,dict(zip(f.names,limits[:,0])),dict(zip(f.names,limits[:,1])),1,ResidualConfig('formal_004',.02,.25,2.,8.))
  core.reset(start['joint_target_rad'],start['joint_target_rad'])
  captured=[];rows=[];refs=[];pending={};counter=1600
  keys=('computed_torque_nm','applied_torque_nm')
  subs=[dict(time_s=0.,sdk_sim_timestamp_s=4.,control_index=-1,substep_index=0,sim_step_counter=1600,**{k:start[k].copy() for k in keys})]
  def begin(k):pending['index']=k
  def targets(q,v,a,valid):
   pending.update(q=q,v=v,a=a,valid=valid)
  def step(zero):
   k=pending['index'];result=core.step(pending['q'],zero,reference_valid=pending['valid'])
   np.testing.assert_array_equal(result['target_position_rad'].numpy(),pending['ref']['q_ref'])
   f.advance(pending['ref']);row=f.snapshot()
   row.update(reference_to_executable_lag_rad=np.zeros((1,18)),position_target_cast_error_rad=np.zeros((1,18)))
   for i in range(1,9):
    elapsed=(8*k+i)*.0025
    subs.append(dict(time_s=elapsed,sdk_sim_timestamp_s=4.+elapsed,control_index=k,substep_index=i,sim_step_counter=1600+8*k+i,**{key:row[key].copy() for key in keys}))
   captured.append(row)
   return {'policy':torch.zeros(1,1)},torch.zeros(1),torch.tensor(row['terminated']),torch.tensor(row['truncated']),{}
  def end(row):
   for key in keys:np.testing.assert_array_equal(subs[-1][key],row[key])
   self.assertIs(rows[-1],row)
  recorder=SimpleNamespace(begin_control=begin,end_control=end)
  env=SimpleNamespace(device='cpu',set_reference_targets=targets,set_evaluation_targets=lambda a:self.assertFalse(a.any()),step=step)
  for k in range(1100):
   ref=g.step(rows[-1] if rows else start);self.assertTrue(ref['valid'][0],ref['failure_reason']);pending['ref']=ref;refs.append(ref)
   row,term,trunc=emit_control(env,ref,captured,recorder,k,rows,torch.zeros(1,18))
   check_substep_batch(subs[-8:],k,initial_counter=counter,control_row=row)
   self.assertFalse(term.any() or trunc.any());check_measured_pair_state(g,row)
  data={key:np.stack([r[key] for r in rows]) for key in rows[0]};data['joint_names']=np.asarray(f.names)
  subdata={key:np.stack([s[key] for s in subs]) for key in subs[0]}
  scored=score_transfer(data,refs,substeps=subdata)
  self.assertTrue(scored['proposed_criteria_met'],scored['failed_proposed_criteria'])
  self.assertFalse(scored['physics_qualification']);self.assertFalse(scored['stage2_complete'])
  self.assertAlmostEqual(rows[0]['time_s'][0],4.02);self.assertAlmostEqual(rows[-1]['time_s'][0],26.)
  self.assertAlmostEqual(refs[-1]['state']['reference_quiet_time_s'],14.)
  self.assertGreaterEqual(scored['quiet_review']['window_duration_s'],10.-1e-7)
  self.assertEqual(len(subs),8801)

if __name__=='__main__':unittest.main()
