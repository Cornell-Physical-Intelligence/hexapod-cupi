import contextlib,hashlib,io,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import acquisition_plan as planner
import terminal_cpu_analysis as executor

class PlanTests(unittest.TestCase):
 def row(self,size):return {'sha256':'a'*64,'bytes':size}
 def test_large_training_and_autosaves_are_explicit_not_replayable_locally(self):
  r=planner.plan({'run/train/training_trace.npz':self.row(793383810),'run/train/policy/model_1847.pt':self.row(4379765),'run/train/policy/decision_500.pt':self.row(4379765)},1<<30)
  self.assertEqual([x['disposition'] for x in r['inventory_rows']],['local_required','remote_only_ordinary_autosave','remote_only_required_training_raw'])
  self.assertTrue(r['local_training_replay_blocked_by_omitted_raw']);self.assertFalse(r['local_training_replay_verified']);self.assertFalse(r['remote_full_replay_performed']);self.assertFalse(r['admission'])
 def test_large_eval_trace_kept_with_chunks_without_disk_discount(self):
  r=planner.plan({'run/final_stop/stop_trace.npz':self.row(150000000)},200000000)
  self.assertEqual(r['selected_local_bytes'],150000000);self.assertFalse(r['fits_local_budget'])
  self.assertTrue(r['inventory_rows'][0]['publication_encoding']['required_storage_is_not_reduced'])
 def test_small_required_training_kept_but_no_replay_claim(self):
  r=planner.plan({'train/training_trace.npz':self.row(100)},1<<30)
  self.assertFalse(r['local_training_replay_blocked_by_omitted_raw']);self.assertFalse(r['local_training_replay_verified']);self.assertFalse(r['remote_full_replay_performed'])
 def test_unsafe_or_bad_inventory_rejects(self):
  for row in [{'../escape':self.row(3)},{'/absolute':self.row(3)},{'train/raw':{'bytes':-1,'sha256':'a'*64}},{'train/raw':{'bytes':1,'sha256':'x'}}]:
   with self.subTest(row=row),self.assertRaises(ValueError):planner.plan(row,1<<30)

class ExecutorTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)
  self.campaign=self.p/'campaign';self.campaign.mkdir();self.cold=self.p/'cold';self.cold.mkdir();self.a=self.p/'analyzer';self.a.mkdir();self.out=self.p/'result'
  self.c=self.campaign/'campaign.json';self.c.write_text(json.dumps({'status':'completed','allocation':'extended','branch':'caps','identity':{'source_manifest_sha256':executor.SOURCE}}))
  (self.campaign/'raw').write_bytes(b'preserve');(self.cold/'raw').write_bytes(b'cold')
  (self.a/'analyze.py').write_text('pass\n');m=self.a/'FREEZE_SHA256.json';m.write_text(json.dumps({'analyze.py':executor.sha(self.a/'analyze.py')}));self.freeze=executor.sha(m)
  self.audit=self.p/'root_audit.json';self.audit.write_text('{}')
  self.argv=['test','--campaign',str(self.campaign),'--cold',str(self.cold),'--analyzer',str(self.a),'--output',str(self.out),'--terminal-audit',str(self.audit),'--campaign-sha256',executor.sha(self.c),'--terminal-audit-sha256',executor.sha(self.audit)]
 def run_case(self,fn):
  with patch.object(sys,'argv',self.argv),patch.object(executor,'ANALYZER_FREEZE',self.freeze),patch.object(executor.subprocess,'run',side_effect=fn),contextlib.redirect_stdout(io.StringIO()):return executor.main()
 def test_process_completed_can_still_have_unverified_evidence(self):
  def fake(*a,**k):
   (self.out/'analysis').mkdir();(self.out/'analysis/report.json').write_text(json.dumps({'evidence_verified':False,'errors':['missing raw']}));(self.out/'analysis/REPORT.md').write_text('forensic');(self.out/'analysis/INPUTS_SHA256.json').write_text('{}');return subprocess.CompletedProcess(a,0)
  self.assertEqual(self.run_case(fake),0)
  r=json.loads((self.out/'execution_receipt.json').read_text());self.assertFalse(r['analyzer_evidence_verified']);self.assertTrue(r['complete_raw_trees_unchanged']);self.assertFalse(r['Stage2_complete'])
 def test_timeout_preserves_failure_receipt_and_inputs(self):
  def fake(*a,**k):raise subprocess.TimeoutExpired(a[0],900)
  self.assertEqual(self.run_case(fake),1);r=json.loads((self.out/'execution_receipt.json').read_text());self.assertIn('TimeoutExpired',r['executor_error']);self.assertEqual((self.campaign/'raw').read_bytes(),b'preserve')
 def test_changed_input_fails_execution_receipt(self):
  def fake(*a,**k):(self.campaign/'raw').write_bytes(b'changed');return subprocess.CompletedProcess(a,0)
  self.assertEqual(self.run_case(fake),1);self.assertFalse(json.loads((self.out/'execution_receipt.json').read_text())['complete_raw_trees_unchanged'])
 def test_active_campaign_rejects_before_any_output_or_process(self):
  d=json.loads(self.c.read_text());d['status']='running';self.c.write_text(json.dumps(d));self.argv[self.argv.index('--campaign-sha256')+1]=executor.sha(self.c)
  with self.assertRaisesRegex(ValueError,'terminal'):self.run_case(lambda *a,**k:self.fail('must not run'))
  self.assertFalse(self.out.exists())

if __name__=='__main__':unittest.main()
