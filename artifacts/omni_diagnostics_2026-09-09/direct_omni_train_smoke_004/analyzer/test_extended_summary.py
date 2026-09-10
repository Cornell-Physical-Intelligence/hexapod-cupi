"""No torch/simulator imports: extended manifests and preserved old summaries."""
import ast,copy,importlib.util,json,tempfile,unittest
from pathlib import Path
import analyze,optimizer_summary
from test_optimizer_summary import fixture,NATIVE
H=Path(__file__).resolve().parent

def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=load('old_summary',H.parent/'direct_omni_train_analysis_003/optimizer_summary.py')
config=load('native005_config',NATIVE/'direct_config.py')

def extended_fixture():
 r=fixture(500);r['selection']=config.selection('train','extended','caps',None,500)
 g=copy.deepcopy(r['optimizer_updates'][0]['minibatches'][0]['gradient'])
 for u in (100,250,500):
  for mb in (1,20):r['optimizer_updates'][u-1]['minibatches'][mb-1]['gradient']=copy.deepcopy(g)
 return r

class ExtendedSummary(unittest.TestCase):
 def test_old_smoke50_summary_and_formatter_exact(self):
  for n,allocation in ((2,'smoke'),(50,'pilot')):
   r=fixture(n);r['selection']['allocation']=allocation
   self.assertEqual(old.summarize(r),optimizer_summary.summarize(r))
   self.assertEqual(old.human(old.summarize(r)),optimizer_summary.human(optimizer_summary.summarize(r)))
 def test_all500_counts_milestones_lr_and_native_validator_agree(self):
  r=extended_fixture();result=optimizer_summary.summarize(r)
  self.assertEqual(result['errors'],[]);self.assertEqual(result['minibatches_retained'],10000)
  self.assertEqual([(x['update'],x['minibatch']) for x in result['sparse_gradients']],[(u,m) for u in [1,10,25,50,100,250,500] for m in (1,20)])
  self.assertEqual(result['loss_windows']['final_update_ids'],list(range(491,501)))
  self.assertEqual(result['learning_rate_at_existing_floor_count'],9500)
  self.assertEqual(result['learning_rate_after']['max'],1.5e-5)
  tree=ast.parse((NATIVE/'direct_contract.py').read_text());scope={'math':__import__('math'),'gradient_updates':config.gradient_updates}
  exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('finite','validate_optimizer_diagnostics')],type_ignores=[]),'<native005 validator>','exec'),scope)
  expected=scope['validate_optimizer_diagnostics'](r,r['selection'])
  self.assertEqual(result['sparse_gradient_rows_retained'],expected['sparse_actor_gradient_rows'])
  self.assertFalse(result['admission'])
 def test_failed_prefix_missing_and_bad_gradients_stay_forensic(self):
  r=extended_fixture();r['complete']=False;r['updates_completed']=99;r['optimizer_updates']=r['optimizer_updates'][:99]
  result=optimizer_summary.summarize(r);self.assertEqual(result['updates_retained'],99);self.assertFalse(result['producer_reported_complete']);self.assertFalse(result['admission'])
  r=extended_fixture();r['optimizer_updates'][499]['minibatches'][19]['gradient']=None
  result=optimizer_summary.summarize(r);self.assertTrue(result['errors']);self.assertEqual(result['sparse_gradient_rows_retained'],13)
  r['optimizer_updates'][499]['losses']=None;result=optimizer_summary.summarize(r)
  json.dumps(result,allow_nan=False);optimizer_summary.human(result)
 def test_complete_curated_decisions_readback_and_omissions_explicit(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'train/policy').mkdir(parents=True);r=extended_fixture();r.update(initial_runner_iteration=1847,final_runner_iteration=2346)
   readback={**config.extended_protocol(),'ordinary_autosaves':500,'decision_checkpoints':7,'first_native_iteration':1847,'last_native_iteration':2346,'actual_controls':12000,'actual_transitions':12288000,'optimizer_minibatches':10000,'sparse_actor_gradient_rows':14}
   r['extended_readback']=readback;r['decision_checkpoints']={}
   for u in config.decision_updates(r['selection']):
    f=p/'train/policy'/f'decision_{u:03d}.pt';f.write_bytes(f'synthetic checkpoint {u}'.encode())
    r['decision_checkpoints'][str(u)]={'file':f.name,'sha256':analyze.sha(f),'completed_updates':u,'native_iteration':1847+u-1}
   (p/'train/training_receipt.json').write_text(json.dumps(r))
   host={'status':'completed','accepted_phases':{'train':{'receipt_sha256':analyze.sha(p/'train/training_receipt.json'),'updates_completed':500,'optimizer_diagnostics':{'schema':'direct315_actor_gradients_v1','updates':500,'minibatches':10000,'sparse_actor_gradient_rows':14}}}}
   inputs=analyze.Inputs();out=analyze.extended_receipt_review(inputs,p,r,r['selection'],host)
   self.assertEqual(out['selected_transitions'],12288000);self.assertIn('not read',out['ordinary_autosaves_scope']);self.assertFalse(out['admission']);self.assertEqual(len(inputs.hashes),8)
   for key in ('extended_readback','decision_checkpoints'):
    bad=copy.deepcopy(r);bad[key]=None
    with self.assertRaises(ValueError):analyze.extended_receipt_review(analyze.Inputs(),p,bad,r['selection'],host)
   bad=copy.deepcopy(r);bad['decision_checkpoints']['250']['native_iteration']+=1
   with self.assertRaises(ValueError):analyze.extended_receipt_review(analyze.Inputs(),p,bad,r['selection'],host)
   (p/'train/policy/decision_500.pt').write_bytes(b'wrong')
   with self.assertRaises(ValueError):analyze.extended_receipt_review(analyze.Inputs(),p,r,r['selection'],host)
 def test_extended_campaign_phase_binding_and_bad_budget_forensic(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);run=base/'run';cold=base/'cold';(run/'jobs').mkdir(parents=True);cold.mkdir()
   phases=['standing','initial_constant','initial_stop','train','final_constant','final_stop']
   sel=config.selection('train','extended','caps',None,500)
   identity={'schema':analyze.SCHEMA,'source_manifest_sha256':analyze.SOURCE,'plan_sha256':analyze.PLAN,'checkpoint_sha256':analyze.ORIGINAL,'actor_width':315,'critic_width':318,'selection':sel,**{k:'a'*64 for k in ('smoke_campaign_sha256','smoke_state_sha256','smoke_receipt_sha256')}}
   campaign={'identity':identity,'allocation':'extended','branch':'caps','status':'failed','terminal_inputs_unchanged':True,'planned_phases':phases,'accepted_phases':{}}
   for phase in phases:
    (run/phase).mkdir();state={'status':'failed','plan_sha256':analyze.PLAN,'urdf_sha256':'e916b1ebbb2720c90f23974bbffa5da120b32d2e5409fe419fb1492b8556746c','runtime_binding':{'source_manifest_sha256':analyze.SOURCE,'runtime_tree_sha256':'abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280'}}
    (run/phase/'state.json').write_text(json.dumps(state));(run/'jobs'/f'{phase}.json').write_text(json.dumps({'status':'failed','cleanup_checked':True,'exit_code':1}))
   receipt={'selection':sel,'optimizer_diagnostics_schema':'direct315_actor_gradients_v1','optimizer_updates':[],'updates_completed':0,'complete':False,'reload':None}
   (run/'train/training_receipt.json').write_text(json.dumps(receipt));(run/'campaign.json').write_text(json.dumps(campaign))
   (cold/'campaign.json').write_text(json.dumps({'identity':{'source_manifest_sha256':'4671eab012aaf79ce49769cfdc1667142237a4b4a45966fd0399a9caa900c51b'},'status':'completed','terminal_inputs_unchanged':True}))
   result=analyze.analyze(run,cold,base/'correct')
   self.assertEqual(len(result['phase_receipts']),6);self.assertIn('extended_allocation',result)
   self.assertFalse(any(x.startswith(('campaign identity:', 'phase allocation:', 'receipt selection:')) for x in result['errors']))
   self.assertFalse(result['evidence_verified']);self.assertFalse(result['Stage2_complete'])
   campaign['identity']['selection']['updates']=50;(run/'campaign.json').write_text(json.dumps(campaign))
   result=analyze.analyze(run,cold,base/'wrong_budget')
   self.assertTrue(any(x.startswith('campaign identity:') for x in result['errors']))
 def test_incomplete_extended_keeps_progress_without_demanding_final(self):
  r={'selection':config.selection('train','extended','caps',None,500),'complete':False,'updates_completed':123}
  out=analyze.extended_receipt_review(analyze.Inputs(),Path('/nonexistent'),r,r['selection'],{'status':'failed'})
  self.assertEqual(out['producer_updates_completed'],123);self.assertFalse(out['admission']);self.assertIn('incomplete',out['status'])

if __name__=='__main__':unittest.main()
