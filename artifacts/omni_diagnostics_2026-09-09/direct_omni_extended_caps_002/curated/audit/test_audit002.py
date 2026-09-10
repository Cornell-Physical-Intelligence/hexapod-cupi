"""Bounded auditor aggregation tests; synthetic receipts are never runtime evidence."""
import ast,copy,json,unittest
from pathlib import Path
from types import SimpleNamespace
H=Path(__file__).resolve().parent
TREE=ast.parse((H/'audit_remote_002.py').read_text())

def function(name,namespace):
 node=next(n for n in ast.walk(TREE) if isinstance(n,ast.FunctionDef) and n.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_audit_function','exec'),namespace)
 return namespace[name]

class AuditTests(unittest.TestCase):
 def setup_campaign(self):
  phases=('standing','initial_constant','initial_stop','train','final_constant','final_stop');R=Path('/synthetic_not_remote');digest='a'*64
  accepted={p:{'phase':p} for p in phases};accepted['train']['optimizer_diagnostics']={'schema':'direct315_actor_gradients_v1','updates':500,'minibatches':10000,'sparse_actor_gradient_rows':14};accepted['final_stop']['quiet_failed_env_ids']=list(range(48))
  identity={'source':'synthetic'};c={'identity':identity,'status':'completed','terminal_inputs_unchanged':True,'bounded_campaign_complete':True,'allocation':'extended','branch':'caps','planned_phases':list(phases),'accepted_phases':accepted,'PPO_updates_completed':500,'Stage2_complete':False,'host_freeze_sha256':digest,'automatic_continuation':False,'observed_training':{'observed_updates_completed':500,'training_receipt_complete':True,'training_receipt_sha256':digest}}
  receipt={'complete':True,'updates_completed':500,'audit':{'controls':12000,'replicas':1024},'final_checkpoint_sha256':digest,'decision_checkpoints':{},'extended_readback':{},'wall_seconds':1.0,'reload':{'passed':True}}
  values={str(R/'campaign.json'):c,str(R/'train/training_receipt.json'):receipt}
  for p in phases:values[str(R/(p+'_accepted.json'))]=copy.deepcopy(accepted[p]);values[str(R/(p+'_immutable.sha256.json'))]={p:digest}
  calls=[]
  def validate(directory,phase,identity,expected_checkpoint_sha256):
   calls.append((phase,expected_checkpoint_sha256));return accepted[phase]
  h=SimpleNamespace(CONTRACT=SimpleNamespace(validate_result=validate),tree_hashes=lambda path:{path.name:digest})
  env={'R':R,'PHASES':phases,'HOST_FREEZE':digest,'out':{'_host':h,'inputs':{'identity':identity}},'read':lambda p:values[str(p)],'sha':lambda p:digest}
  return function('campaign',env),c,receipt,values,calls
 def test_complete500_invokes_every_native_gate_keeps48failures(self):
  run,c,receipt,values,calls=self.setup_campaign();result=run()
  self.assertEqual([x[0] for x in calls],c['planned_phases']);self.assertEqual(result['updates'],500);self.assertEqual(result['actual_transitions'],12288000)
  self.assertEqual(result['quiet_failed_env_ids'],list(range(48)));self.assertTrue(result['acquisition_is_not_physical_acceptance'])
 def test_incomplete_wrong_counts_source_and_phase_reject(self):
  changes=[lambda c,r:c.update(status='failed'),lambda c,r:c.update(terminal_inputs_unchanged=False),lambda c,r:c.update(identity={'different':'source'}),lambda c,r:c.update(PPO_updates_completed=50),lambda c,r:c.update(branch='quiet_priority'),lambda c,r:c['accepted_phases'].pop('initial_stop'),lambda c,r:r.update(complete=False),lambda c,r:r['audit'].update(controls=11999),lambda c,r:r['audit'].update(replicas=32)]
  for change in changes:
   run,c,r,values,calls=self.setup_campaign();change(c,r)
   with self.assertRaises(AssertionError):run()
 def test_changed_accepted_or_immutable_alias_reject(self):
  for suffix in ('_accepted.json','_immutable.sha256.json'):
   run,c,r,values,calls=self.setup_campaign();values['/synthetic_not_remote/initial_stop'+suffix]={'changed':True}
   with self.assertRaises(AssertionError):run()
 def test_failure_capture_does_not_prevent_remaining_inventory(self):
  env={'out':{'errors':[],'checks':{}}};check=function('check',env)
  def missing():raise FileNotFoundError('partial result missing')
  self.assertIsNone(check('campaign',missing));self.assertEqual(check('inventory',lambda:{'raw':'retained'}),{'raw':'retained'})
  self.assertFalse(env['out']['checks']['campaign']);self.assertTrue(env['out']['checks']['inventory']);self.assertIn('partial result missing',env['out']['errors'][0])
 def test_failed_job_keeps_both_owned_absence_results(self):
  phase='train';calls=[];job={'status':'failed','exit_code':1,'cleanup_checked':True,'container_id':'exact-id','container_name':'exact-name'}
  def call(argv):calls.append(argv);return {'exit_code':1,'stderr':'Error: No such object'}
  env={'phase':phase,'R':Path('/synthetic_not_remote'),'out':{'jobs':{},'owned_absence':{}},'read':lambda p:job,'call':call,'sha':lambda p:'a'*64}
  with self.assertRaises(AssertionError):function('job',env)()
  self.assertEqual(set(env['out']['owned_absence']),{'exact-id','exact-name'});self.assertEqual(len(calls),2);self.assertEqual(env['out']['jobs']['train']['status'],'failed')
 def test_auditor_has_no_dispatch_or_large_fetch_path(self):
  calls=[n for n in ast.walk(TREE) if isinstance(n,ast.Call)]
  attrs={n.func.attr for n in calls if isinstance(n.func,ast.Attribute)}
  self.assertFalse(attrs & {'write_text','write_bytes','mkdir','unlink','remove','main'})
  literals={n.value for n in ast.walk(TREE) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
  self.assertFalse(literals & {'systemd-run','stop','start','docker run','scp'})

if __name__=='__main__':unittest.main()
