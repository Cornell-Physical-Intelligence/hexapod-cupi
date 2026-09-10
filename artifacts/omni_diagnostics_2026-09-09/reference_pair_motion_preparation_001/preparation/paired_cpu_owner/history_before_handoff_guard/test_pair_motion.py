import copy,json,sys,unittest
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'oracle'))
from pair_motion import PairContactReference,PairMotionConfig
from ideal_fixture import IdealMeasuredFixture
from run_baseline import evaluate
from state_packet import pack,FragmentHistory
from checkpoint import export_controller,restore_controller,export_motor_target,restore_motor_target
from reference_residual import ReferenceResidualTarget,ResidualConfig

def core_for(f):
 limits=f.snapshot()['soft_joint_pos_limits_rad'][0]
 c=ReferenceResidualTarget(f.names,dict(zip(f.names,limits[:,0])),dict(zip(f.names,limits[:,1])),1,ResidualConfig('formal_004',.02,.25,2.,8.));q=f.snapshot()['joint_target_rad'];c.reset(q,q);return c

def force_contact(row,i,value):
 row['distal_contact'][0,i]=value;row['contact_point_valid'][0,i]=value
 row['contact_point_world_m'][0,i]=row['reference_point_world_m'][0,i] if value else np.nan
 row['normal_force_world_n'][0,i]=[0,0,10.] if value else 0
 row['reaction_force_world_n'][0,i]=row['normal_force_world_n'][0,i]

class PairMotionTests(unittest.TestCase):
 def test_full_forward_stop_and_independent_feet(self):
  r,rows,g,f=evaluate();self.assertIsNone(r['failure']);self.assertEqual(r['controls'],2200);self.assertEqual(r['zero_residual_lag'],0)
  self.assertGreaterEqual(r['completed_pairs'],9);self.assertEqual(r['touchdowns'],2*r['completed_pairs']);self.assertEqual(r['lifts_after_stop'],0);self.assertLess(r['vmax'],1.75);self.assertLess(r['amax'],6.)
  self.assertEqual(r['final_mode'],'reference_quiet_hold');self.assertGreaterEqual(44-r['quiet_time']-2,10)
  self.assertEqual({i for row in rows for i,f in enumerate(row['state']['foot_cycles']) if f and f['flight_seen']},set(range(6)))
 def test_one_leg_never_flight_cannot_borrow_other_qualification(self):
  def mutate(k,g,f,row):
   if g.active_pair is not None:force_contact(row,g.active_pair[1],True)
  r,rows,g,f=evaluate(duration=5,mutate=mutate);self.assertIsNotNone(r['failure']);self.assertIn('apex',r['failure']['reason']);self.assertEqual(g.completed_pairs,0);self.assertEqual(g.touchdowns,0)
  self.assertTrue(g.foot_cycles[1].flight_seen);self.assertFalse(g.foot_cycles[4].flight_seen)
 def test_unqualified_blip_does_not_count_as_flight_or_touchdown(self):
  observations=[]
  def mutate(k,g,f,row):
   if k in (101,102):force_contact(row,1,False)
   elif k in (103,104):force_contact(row,1,True)
   if k==104:observations.append(copy.deepcopy(g.foot_cycles[1]))
  r,rows,g,f=evaluate(duration=4.5,mutate=mutate);self.assertIsNone(r['failure']);f=observations[0];self.assertFalse(f.flight_seen);self.assertEqual(f.flight_count,0);self.assertEqual(f.unqualified_contact_returns,1)
 def test_one_delayed_contact_holds_pair_until_both_confirm(self):
  def mutate(k,g,f,row):
   if 3.96<=k*.02<4.18:force_contact(row,4,False)
  r,rows,g,f=evaluate(duration=5,mutate=mutate);self.assertIsNone(r['failure'])
  independent=[row for row in rows if row['state']['active_pair']==['lm','rm'] and row['state']['foot_cycles'][1]['current_leg'] is None and row['state']['foot_cycles'][4]['current_leg'] is not None]
  self.assertTrue(independent);self.assertTrue(all(row['state']['completed_pairs']==0 for row in independent));self.assertEqual(g.completed_pairs,1)
 def test_missing_partner_touchdown_stops_then_rejects(self):
  def mutate(k,g,f,row):
   if k*.02>=3.8:force_contact(row,4,False)
  r,rows,g,f=evaluate(duration=6,mutate=mutate);self.assertIsNotNone(r['failure']);self.assertIn('touchdown missing',r['failure']['reason']);self.assertEqual(g.completed_pairs,0);self.assertEqual(g.liftoffs,2)
  self.assertTrue(any(row['command_derating_factor']==0 and not row['admitted_target_command'].any() for row in rows))
 def test_retained_support_loss_and_nonfoot_fail_closed(self):
  for kind in ('support','nonfoot','terminal'):
   def mutate(k,g,f,row):
    if k==130:
     if kind=='support':force_contact(row,0,False)
     elif kind=='nonfoot':row['shaft_contact'][0,0]=True
     else:row['terminated'][0]=True
   r,rows,g,f=evaluate(duration=4,mutate=mutate);self.assertIsNotNone(r['failure']);self.assertIsNone(g.step(f.snapshot(),[.01,0,0])['q_ref'])
 def test_stop_during_flight_finishes_only_current_pair(self):
  r,rows,g,f=evaluate(stop_at=3.1,duration=20);self.assertIsNone(r['failure']);self.assertEqual(g.completed_pairs,1);self.assertEqual(g.liftoffs,2);self.assertEqual(r['lifts_after_stop'],0);self.assertEqual(g.mode,'reference_quiet_hold')
 def test_named_runtime_permutation_and_reset_preload(self):
  f=IdealMeasuredFixture();r,_,_,_=evaluate(duration=6,names=tuple(reversed(f.names)));self.assertIsNone(r['failure']);self.assertEqual(r['zero_residual_lag'],0)
  f=IdealMeasuredFixture(preload=.01);g=PairContactReference(f.names);q=f.snapshot()['joint_target_rad'].copy();out=g.reset(f.snapshot());np.testing.assert_array_equal(out['q_ref'],q);self.assertGreater(abs(out['state']['initial_joint_preload_rad']).max(),.009)
 def test_fixed_deflection_fixture_contact_loss_is_rejected_not_hidden(self):
  # Constant joint deflection is not a physics model: it can lift planted toes
  # slightly above the ideal sharp plane as stance angles change. Retain its
  # rejection rather than enlarging contact tolerance to make it pass.
  r,_,_,_=evaluate(duration=6,preload=.01);self.assertIsNotNone(r['failure']);self.assertIn('retained support',r['failure']['reason'])
 def test_all_contact_and_target_thresholds_fixed(self):
  from dataclasses import replace
  for name,value in [('maximum_touchdown_error_m',.02),('minimum_measured_lift_m',.001),('max_touchdown_delay_s',2.),('reference_acceleration_rad_s2',10.)]:
   with self.assertRaises(ValueError):PairContactReference(IdealMeasuredFixture().names,replace(PairMotionConfig(),**{name:value}))
 def test_json_interruption_replays_core_state_and_history_exactly(self):
  f=IdealMeasuredFixture();g=PairContactReference(f.names);out=g.reset(f.snapshot());c=core_for(f);hist=FragmentHistory();hist.reset(pack(out),1)
  for k in range(170):
   out=g.step(f.snapshot(),[.01,0,0]);c.step(out['q_ref'],np.zeros((1,18)),reference_valid=out['valid']);f.advance(out);hist.append(pack(out),1)
  checkpoint=json.loads(json.dumps(export_controller(g,f.snapshot()),allow_nan=False));motor=json.loads(json.dumps(export_motor_target(c),allow_nan=False));history=json.loads(json.dumps(hist.checkpoint(),allow_nan=False))
  clone=PairContactReference(f.names);packet=restore_controller(clone,checkpoint,f.snapshot());cc=core_for(f);restore_motor_target(cc,motor,clone);hh=FragmentHistory();hh.restore(history,pack(packet),1)
  for k in range(140):
   a=g.step(f.snapshot(),[.01,0,0]);b=clone.step(f.snapshot(),[.01,0,0]);self.assertTrue(a['valid'][0]);self.assertTrue(b['valid'][0])
   for key in ('q_ref','v_ref','a_ref'):np.testing.assert_array_equal(a[key],b[key])
   ca=c.step(a['q_ref'],np.zeros((1,18)),reference_valid=a['valid']);cb=cc.step(b['q_ref'],np.zeros((1,18)),reference_valid=b['valid'])
   np.testing.assert_array_equal(ca['target_position_rad'].numpy(),cb['target_position_rad'].numpy());np.testing.assert_array_equal(hist.append(pack(a),1),hh.append(pack(b),1));f.advance(a)
  bad=copy.deepcopy(checkpoint);bad['payload']['state']['time_s']+=.02
  with self.assertRaises(ValueError):restore_controller(PairContactReference(f.names),bad,f.snapshot())
 def test_history_new_epoch_clears_every_slot(self):
  f=IdealMeasuredFixture();g=PairContactReference(f.names);out=g.reset(f.snapshot());history=FragmentHistory();history.reset(pack(out),1)
  out=g.step(f.snapshot(),[.01,0,0]);f.advance(out);history.append(pack(out),1)
  f2=IdealMeasuredFixture();reset=g.reset(f2.snapshot());p=pack(reset)
  with self.assertRaises(ValueError):history.append(p,2)
  frames=history.reset(p,2);np.testing.assert_array_equal(frames,np.repeat(p['values'][None],3,axis=0));self.assertEqual(history.steps,0)
  with self.assertRaises(ValueError):history.append(p,2)
 def test_packet_has_two_foot_states_and_rejects_old_wave_version(self):
  r,rows,g,f=evaluate(duration=3);p=pack(rows[-1]);self.assertTrue(np.isfinite(p['values']).all());self.assertNotIn(p['schema']['features'],(846,849));self.assertEqual(p['schema']['fields'].count('foot.lm.flight_seen'),1);self.assertEqual(p['schema']['fields'].count('foot.rm.flight_seen'),1)
  bad=copy.deepcopy(rows[-1]);bad['state']['version']='wave005'
  with self.assertRaises(ValueError):pack(bad)
if __name__=='__main__':unittest.main()
