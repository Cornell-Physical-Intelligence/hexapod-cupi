"""Additional read-only CPU adversarial pair contact/continuation tests."""
from pathlib import Path
import copy,json,sys,unittest,os
import numpy as np
OWNER=Path(os.environ.get('PAIR_MOTION_OWNER',str(Path(__file__).resolve().parents[1]/'reference_pair_motion_001'))).resolve()
sys.path.insert(0,str(OWNER));sys.path.insert(0,str(OWNER/'oracle'))
from pair_motion import PairContactReference
from ideal_fixture import IdealMeasuredFixture
from run_baseline import evaluate
from test_pair_motion import force_contact,core_for
from checkpoint import export_controller,restore_controller,export_motor_target,restore_motor_target
from state_packet import pack,FragmentHistory

class IndependentTests(unittest.TestCase):
 def test_qualified_early_obstacle_contact_is_not_ignored(self):
  injected=[]
  def mutate(k,g,f,row):
   if g.active_pair:
    foot=g.foot_cycles[g.active_pair[0]]
    if foot.flight_seen and g.time<foot.swing.t0+.5*foot.swing.duration:
     force_contact(row,foot.leg,True);injected.append(k)
  report,rows,g,f=evaluate(duration=5,mutate=mutate)
  self.assertTrue(injected);self.assertIn('passed apex',report['failure']['reason']);self.assertEqual(g.completed_pairs,0);self.assertEqual(g.touchdowns,0)
  state=copy.deepcopy(vars(g));bad=g.step(f.snapshot(),[.01,0,0]);self.assertFalse(bad['valid'][0]);self.assertIsNone(bad['q_ref']);self.assertEqual(g.liftoffs,state['liftoffs']);self.assertEqual(g.touchdowns,state['touchdowns'])
 def test_landing_contact_reacquisition_remains_bounded(self):
  injected=[]
  def mutate(k,g,f,row):
   if g.active_pair:
    foot=g.foot_cycles[g.active_pair[0]]
    if foot.landing is not None:force_contact(row,foot.leg,False);injected.append(k)
  report,rows,g,f=evaluate(duration=6,mutate=mutate)
  self.assertEqual(len(injected),6);self.assertIn('100ms',report['failure']['reason']);self.assertEqual(g.completed_pairs,0)
 def test_asymmetric_confirmed_partner_checkpoint_and_history(self):
  f=IdealMeasuredFixture();g=PairContactReference(f.names);out=g.reset(f.snapshot());core=core_for(f);history=FragmentHistory();history.reset(pack(out),1)
  clone=None
  for k in range(280):
   row=f.snapshot()
   if 3.96<=k*.02<4.18:force_contact(row,4,False)
   cmd=[.01,0,0] if k>=100 else [0,0,0]
   out=g.step(row,cmd);self.assertTrue(out['valid'][0]);a=core.step(out['q_ref'],np.zeros((1,18)),reference_valid=out['valid']);h=history.append(pack(out),1)
   if clone is not None:
    other=clone.step(row,cmd);self.assertTrue(other['valid'][0])
    for key in ('q_ref','v_ref','a_ref'):np.testing.assert_array_equal(out[key],other[key])
    b=cc.step(other['q_ref'],np.zeros((1,18)),reference_valid=other['valid']);np.testing.assert_array_equal(a['target_position_rad'].numpy(),b['target_position_rad'].numpy());np.testing.assert_array_equal(h,hh.append(pack(other),1))
   f.advance(out)
   if clone is None and g.active_pair and g.foot_cycles[1].current_leg is None and g.foot_cycles[4].current_leg is not None:
    controller=json.loads(json.dumps(export_controller(g,f.snapshot()),allow_nan=False));target=json.loads(json.dumps(export_motor_target(core),allow_nan=False));hist=json.loads(json.dumps(history.checkpoint(),allow_nan=False))
    clone=PairContactReference(f.names);restored=restore_controller(clone,controller,f.snapshot());cc=core_for(f);restore_motor_target(cc,target,clone);hh=FragmentHistory();hh.restore(hist,pack(restored),1)
  self.assertIsNotNone(clone);self.assertGreaterEqual(g.completed_pairs,1);self.assertEqual(clone.completed_pairs,g.completed_pairs)
if __name__=='__main__':unittest.main()
