"""CPU target/state contract tests; recorded contacts are not changed physics."""
import ast,copy,importlib.util,json,math,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parent;SOURCE=HERE/'source_rr_preload_001';PARENT=HERE.parent/'reference_directional_adapter_002/source_directional_002'
sys.path.insert(0,str(SOURCE/'tools'))
from rr_preload_diagnostic import RRFirstLandingPreload
from wave_reference import WaveContactReference,WaveConfig
import directional_contract as contract
import launch_directional_physics_spark as host
spec=importlib.util.spec_from_file_location('frozen_parent_wave',PARENT/'tools/wave_reference.py');parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
RAW=HERE.parent/'reference_directional_results_002/run/left_strafe'
TRACE=np.load(RAW/'trace.npz');NAMES=list(TRACE['joint_names'])
FIELDS=['time_s','position_world_m','quaternion_world_xyzw','rotation_world_from_body','velocity_body_mps','gyro_body_rad_s','joint_position_rad','joint_target_rad','reference_point_world_m','reference_point_velocity_world_mps','contact_point_world_m','contact_point_valid','distal_contact','shaft_contact','coxa_contact','femur_contact','base_contact','executable_target_velocity_rad_s','soft_joint_pos_limits_rad','terminated','truncated']
def snapshot(k):return {x:TRACE[x][k].copy() for x in FIELDS}
def controller_at(k=199):
 c=WaveContactReference(NAMES);c.reset(snapshot(199))
 for idx in range(200,k+1):
  r=c.step(snapshot(idx-1),[0,.005,0]);assert r['valid'][0],(idx,r['failure_reason'])
 return c

def helper():
 h=RRFirstLandingPreload();h.arm(5,1,2.,.3,[0,0,0],[0,0,.0005],[0,0,0]);return h

class CurveTests(unittest.TestCase):
 def test_exact_endpoint_and_C2_at_both_boundaries(self):
  h=helper()
  for time,expected in [(1.99,0),(2.,0),(2.3,-.0005),(2.31,-.0005)]:
   p,v,a=h.sample(time);np.testing.assert_array_equal(p,[0,0,expected]);np.testing.assert_allclose(v,0,atol=1e-15);np.testing.assert_allclose(a,0,atol=1e-12)
  eps=1e-7
  for center in [2.,2.3]:
   p0,v0,a0=h.sample(center)
   for time in [center-eps,center+eps]:
    p,v,a=h.sample(time);np.testing.assert_allclose(p,p0,atol=1e-12);np.testing.assert_allclose(v,v0,atol=1e-10);np.testing.assert_allclose(a,a0,atol=2e-7)
 def test_only_RR_first_landing_once_and_reset_is_fresh(self):
  h=RRFirstLandingPreload();self.assertFalse(h.arm(0,0,2,.3,[0]*3,[0]*3,[0]*3));self.assertFalse(h.started)
  self.assertTrue(h.arm(5,1,2,.3,[0]*3,[0]*3,[0]*3));self.assertFalse(h.arm(5,7,8,.3,[1]*3,[1]*3,[1]*3));self.assertEqual(h.start_s,2)
  self.assertFalse(RRFirstLandingPreload().started)
 def test_nofinite_or_wrong_hold_or_unbounded_endpoint(self):
  for t,d,a,m,e in [(float('nan'),.3,[0]*3,[0]*3,[0]*3),(2,.4,[0]*3,[0]*3,[0]*3),(2,.3,[0,0,float('inf')],[0]*3,[0]*3),(2,.3,[0]*3,[0]*3,[0,0,.012])]:
   with self.assertRaises(ValueError):RRFirstLandingPreload().arm(5,1,t,d,a,m,e)
  with self.assertRaises(ValueError):RRFirstLandingPreload().arm(5,7,2,.3,[0]*3,[0]*3,[0]*3)
 def test_apply_preserves_measured_anchor_and_other_five_targets(self):
  h=helper();anchors=np.zeros((6,3));measured=np.zeros((6,3));measured[5,2]=.0005;oldmeasured=measured.copy();preload=anchors-measured
  for t in np.linspace(2.02,2.30,15):h.require_contact(True,measured[5]);h.apply(t,anchors,measured,preload)
  np.testing.assert_array_equal(measured,oldmeasured);np.testing.assert_array_equal(anchors[:5],0);np.testing.assert_array_equal(anchors[5],[0,0,-.0005]);np.testing.assert_array_equal(preload,anchors-measured);self.assertTrue(h.completed)
 def test_lost_contact_region_and_late_clock_fail_before_emission(self):
  h=helper()
  with self.assertRaisesRegex(ValueError,'lost measured'):h.require_contact(False,[0,0,0])
  with self.assertRaisesRegex(ValueError,'12mm'):h.require_contact(True,[.013,0,0])
  with self.assertRaises(ValueError):h.apply(float('nan'),np.zeros((6,3)),np.zeros((6,3)),np.zeros((6,3)))
  with self.assertRaises(ValueError):h.apply(2.34,np.zeros((6,3)),np.zeros((6,3)),np.zeros((6,3)))

class RuntimeTests(unittest.TestCase):
 def test_actual_prefix_old_targets_before_trigger_then_one_RR_only(self):
  old=parent.WaveContactReference(NAMES);new=WaveContactReference(NAMES);ro=old.reset(snapshot(199));rn=new.reset(snapshot(199));np.testing.assert_array_equal(ro['q_ref'],rn['q_ref'])
  rr=[NAMES.index(name) for name in new.g.names[5]];others=[i for i in range(18) if i not in rr];changes=[]
  for k in range(200,505):
   m=snapshot(k-1);a=old.step(m,[0,.005,0]);b=new.step(m,[0,.005,0]);self.assertTrue(a['valid'][0]);self.assertTrue(b['valid'][0],(k,b['failure_reason']))
   np.testing.assert_array_equal(a['q_ref'][:,others],b['q_ref'][:,others])
   if k<420:np.testing.assert_array_equal(a['q_ref'],b['q_ref'])
   np.testing.assert_array_equal(a['state']['measured_anchors_world_m'],b['state']['measured_anchors_world_m'])
   for key in ['current_leg','liftoffs','confirmed_touchdowns','hold_until_s','mode']:
    self.assertEqual(a['state'][key],b['state'][key],(k,key))
   changes.append(float(np.max(np.abs(a['q_ref']-b['q_ref']))))
  h=new.rr_preload_diagnostic;self.assertTrue(h.completed);self.assertEqual(h.armed_touchdown_count,2);self.assertAlmostEqual(h.start_s,8.40);self.assertAlmostEqual(h.end_s,8.70);self.assertGreater(max(changes),0)
  # The real rejected row is not relabeled by the changed target-only prefix.
  rejected=new.step(snapshot(504),[0,.005,0]);self.assertFalse(rejected['valid'][0]);self.assertIn('Fewer than five',rejected['failure_reason']);self.assertIsNone(rejected['q_ref'])
 def test_exact_frozen_residual_core_zero_action_emits_each_new_reference(self):
  import torch
  from reference_residual import ReferenceResidualTarget,ResidualConfig
  c=WaveContactReference(NAMES);seed=snapshot(199);r=c.reset(seed);lo=seed['soft_joint_pos_limits_rad'][0,:,0];hi=seed['soft_joint_pos_limits_rad'][0,:,1]
  core=ReferenceResidualTarget(NAMES,dict(zip(NAMES,lo)),dict(zip(NAMES,hi)),1,ResidualConfig('formal_004',.02,.25,2.,8.),dtype=torch.float64)
  core.reset(r['q_ref'],r['q_ref'])
  for k in range(200,505):
   r=c.step(snapshot(k-1),[0,.005,0]);self.assertTrue(r['valid'][0])
   emitted=core.step(r['q_ref'],np.zeros((1,18)),reference_valid=r['valid'],analytic_reference_velocity=r['v_ref'],analytic_reference_acceleration=r['a_ref'])
   np.testing.assert_array_equal(emitted['target_position_rad'].numpy(),r['q_ref']);np.testing.assert_array_equal(emitted['residual_position_rad'].numpy(),0)
 def test_support_interruption_latches_and_keeps_last_executable_target(self):
  c=controller_at(421);previous=c.q.copy();m=snapshot(421);m['distal_contact'][0,5]=False
  # Five other feet remain, so the narrow active correction explicitly requires RR.
  self.assertEqual(int(m['distal_contact'][0].sum()),5)
  r=c.step(m,[0,.005,0]);self.assertFalse(r['valid'][0]);self.assertIn('RR preload correction lost',r['failure_reason']);self.assertEqual(c.rr_preload_diagnostic.status,'rejected');self.assertIsNone(r['q_ref']);np.testing.assert_array_equal(c.q,previous)
  again=c.step(snapshot(421),[0,.005,0]);self.assertFalse(again['valid'][0]);np.testing.assert_array_equal(c.q,previous)
 def test_parent_nonfoot_and_four_support_fail_unchanged(self):
  for kind in ['nonfoot','support']:
   c=controller_at(421);m=snapshot(421)
   if kind=='nonfoot':m['shaft_contact'][0,3]=True
   else:m['distal_contact'][0,[3,4]]=False
   r=c.step(m,[0,.005,0]);self.assertFalse(r['valid'][0]);self.assertIsNone(r['q_ref']);self.assertEqual(c.rr_preload_diagnostic.status,'rejected')
 def test_stop_finishes_correction_without_new_liftoff_then_quiet(self):
  c=controller_at(421);m=snapshot(421);liftoffs=c.liftoffs
  # Synthetic fixed six-support measured fixture; no physical stop claim.
  prior=c.q.copy();changed=False
  for _ in range(600):
   m['time_s'][0]=c.time;r=c.step(m,[0,0,0]);self.assertTrue(r['valid'][0],r['failure_reason']);changed|=bool(np.any(c.q!=prior));prior=c.q.copy()
   self.assertEqual(c.liftoffs,liftoffs)
   if c.rr_preload_diagnostic.active:self.assertNotEqual(r['state']['mode'],'reference_quiet_hold')
  self.assertTrue(changed);self.assertTrue(c.rr_preload_diagnostic.completed);self.assertEqual(r['state']['mode'],'reference_quiet_hold');np.testing.assert_array_equal(r['v_ref'],0);np.testing.assert_array_equal(r['a_ref'],0)
 def test_reset_clears_one_shot_and_preserves_incoming_target(self):
  c=controller_at(450);self.assertTrue(c.rr_preload_diagnostic.completed);r=c.reset(snapshot(199));self.assertFalse(c.rr_preload_diagnostic.started);np.testing.assert_array_equal(r['q_ref'][0],snapshot(199)['joint_target_rad'][0])
 def test_wrong_direction_rejected_before_reference_motion(self):
  for command in [[.005,0,0],[0,0,.015],[0,-.005,0],[.0035,-.0035,-.01]]:
   c=controller_at();previous=c.q.copy();r=c.step(snapshot(199),command)
   self.assertFalse(r['valid'][0]);self.assertIn('only exact left_strafe',r['failure_reason']);self.assertIsNone(r['q_ref']);np.testing.assert_array_equal(c.q,previous)
 def test_configuration_cadence_not_changed(self):
  from dataclasses import replace
  with self.assertRaises(ValueError):WaveContactReference(NAMES,replace(WaveConfig(),contact_hold_s=.32))

class ScopeTests(unittest.TestCase):
 def test_only_left_strafe_and_standing_RO_host(self):
  self.assertEqual(contract.CASES,{'left_strafe':(0.,.005,0.)});self.assertFalse(contract.PROTOCOL['PPO_permitted'])
  for phase in ['reverse','left_turn','forward_right_arc','train']:
   with self.assertRaises(ValueError):host.command(Path('/source'),Path('/out'),'owned',phase)
  for phase in ['standing','left_strafe']:
   cmd=host.command(Path('/source'),Path('/out'),'owned',phase);self.assertIn('/source:/workspace/hexapod:ro',cmd);self.assertIn('/out/inputs/study:/study:ro',cmd)
 def test_physical_loop_gates_and_owned_supervisor_unchanged(self):
  for file in ['run_directional_physics.py','directional_metrics.py','screen_metrics.py','reference_residual.py','reference_residual_env.py','physics_telemetry.py','physics_substeps.py']:
   self.assertEqual((SOURCE/'tools'/file).read_bytes(),(PARENT/'tools'/file).read_bytes(),file)
  for name in ['configure_comparison','readback_comparison','configure_iteration_comparison','iteration_scene_readback','validate_iteration_readback']:
   oldsolver=ast.parse((PARENT/'tools/solver_comparison.py').read_text());newsolver=ast.parse((SOURCE/'tools/solver_comparison.py').read_text())
   a=next(n for n in oldsolver.body if isinstance(n,ast.FunctionDef) and n.name==name);b=next(n for n in newsolver.body if isinstance(n,ast.FunctionDef) and n.name==name);self.assertEqual(ast.dump(a),ast.dump(b),name)
  old=ast.parse((PARENT/'tools/launch_directional_physics_spark.py').read_text());new=ast.parse((SOURCE/'tools/launch_directional_physics_spark.py').read_text())
  for name in ['owned_container','run_owned','main','check_source']:
   a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name);b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name);self.assertEqual(ast.dump(a),ast.dump(b),name)
 def test_fresh_source32quiet_receipt_required(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'receipt.json';args=SimpleNamespace(mode='directional',case='left_strafe',num_envs=1,steps=2400,admission=p);identity={'source':'new-preload'}
   good={'mode':'standing','identity':identity,'status':'completed','gate':{'passed':True,'num_envs':32,'control_steps':1000,'all_replica_quiet':{'passed':True}}}
   with patch.object(contract,'standing_preflight',return_value=(identity,{})),patch.object(contract,'check_wave'):
    p.write_text(json.dumps(good));self.assertEqual(contract.preflight(args,Path(td))[0]['case'],'left_strafe')
    for field in ['source','quiet','count']:
     b=copy.deepcopy(good)
     if field=='source':b['identity']={'source':'old-directional002'}
     elif field=='quiet':b['gate']['all_replica_quiet']['passed']=False
     else:b['gate']['num_envs']=1
     p.write_text(json.dumps(b))
     with self.assertRaises(ValueError):contract.preflight(args,Path(td))
if __name__=='__main__':unittest.main()
