"""Strict result-contract counterexamples, explicitly synthetic CPU protocol data."""
from pathlib import Path
import copy,json,tempfile,unittest
from canonical_direct_ppo.native_contract import validate_result
from canonical_direct_ppo.binding_contract import sha
from canonical_direct_ppo.smoke_config import SCHEMA
from canonical_direct_ppo import source005_solver_diagnostics as diag
from test_solver_fixture import write_native_records,write_member_fixture


def write(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))


class ResultChecks(unittest.TestCase):
 def fixture(self,d):
  identity={'schema':SCHEMA,'CPU_PROTOCOL_FIXTURE_ONLY':True,'policy_lineage':{'CPU_ONLY':True},'solver_recipe':{'position_iterations':32,'velocity_iterations':0},'solver_diagnostics':diag.declaration(32,8384)}
  checks=['sdk_source_bound','usd_identity','native_scene','native_materials_offsets','all_native_inertias_limits_no_drives','all_native_sdf_paths','all_reset_coordinate_frames','fresh_neutral_prefix_admitted','learner_2_updates_strict_reload','neutral_solver_recipe_readback','solver_recipe_readback','legacy_friction_observed']
  state={'schema':SCHEMA,'identity':identity,'status':'completed','inputs_unchanged':True,'errors':[],'native_error_events':[],'explicit_steps_completed':8384,
   'checks':dict.fromkeys(checks,True),'quality_admitted':False,'Stage2_complete':False}
  write(d/'native_errors.json',[])
  for name in ['control_trace.npz','contacts.jsonl','initial_reset.json','native_readback.json','sdf_readback.json','entry_composition.json','substeps_000.npz','learner/policy_control_trace.pt','learner/decision_001.pt','learner/decision_002.pt']:
   p=d/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(b'CPU PROTOCOL FIXTURE ONLY')
  write_native_records(d,32,final=True)
  for name in ['control_trace.npz','substeps_000.npz']:write_member_fixture(d/name)
  write(d/'session.json',{'steps':8384,'captured_steps':8384,'controls':1048,'reset_count':1,'failure':None,'all_rows_recorded':True,'substep_files':['substeps_000.npz'],'solver_diagnostics':diag.declaration(32,8384)})
  write(d/'neutral_prefix/standing_report.json',{'all_pass':True,'num_envs':32,'replicas':[{'pass':True}]*32})
  write_native_records(d/'neutral_prefix',32,final=False)
  for name in ['control_trace.npz','substeps_000.npz']:write_member_fixture(d/'neutral_prefix'/name)
  (d/'neutral_prefix/contacts.jsonl').write_text('CPU PREFIX FIXTURE ONLY\n')
  write(d/'neutral_prefix/session.json',{'steps':8000,'captured_steps':8000,'controls':1000,'reset_count':1,'failure':None,'all_rows_recorded':True,'substep_files':['substeps_000.npz'],'solver_diagnostics':diag.declaration(32,8000)})
  write(d/'neutral_prefix/SHA256.json',{p.name:sha(p)for p in(d/'neutral_prefix').iterdir()})
  updates=[]
  for i in [1,2]:
   digest=sha(d/'learner'/f'decision_{i:03d}.pt')
   reload=dict.fromkeys(['passed','strict_model_normalizer_optimizer','deterministic_action_exact','runtime_state_exact','learning_rate_exact'],True)
   reload.update(optimizer_entries=17,checkpoint_sha256=digest)
   updates.append({'completed_update':i,'optimizer_minibatches':20,'learning_rate':.0003,'losses':{'surrogate':-.01,'value':.1},'reload':reload,'checkpoint_sha256':digest})
  learner={'status':'completed','lineage':identity['policy_lineage'],'updates_completed':2,'controls_attempted':48,'controls_completed':48,'optimizer_steps_completed':40,'transitions':1536,'errors':[],'quality_admitted':False,'updates':updates,
   'native_audit':{'failure':None,'policy_substeps_verified':384,'policy_controls':48,'policy_attempts':48,'requested_saturation_counts':[[0]*18 for _ in range(32)]},
   'outputs':{p.name:sha(p)for p in (d/'learner').iterdir()}}
  write(d/'learner/state.json',learner)
  self.seal(d,state);return identity,state,learner
 def seal(self,d,state):
  state['outputs']={str(p.relative_to(d)):sha(p)for p in d.rglob('*')if p.is_file()and p!=d/'state.json'and p!=d/'native_errors.json'}
  write(d/'state.json',state)
 def test_positive_explicit_integration_only(self):
  with tempfile.TemporaryDirectory()as td:
   d=Path(td);identity,state,learner=self.fixture(d)
   result=validate_result(d,identity);self.assertEqual(result['updates_completed'],2);self.assertFalse(result['quality_admitted'])
 def test_each_reload_and_optimizer_claim_must_be_true(self):
  mutations=[lambda r:r['updates'][0]['reload'].update(strict_model_normalizer_optimizer=False),
    lambda r:r['updates'][0].update(completed_update=2),lambda r:r['updates'][0].update(optimizer_minibatches=19),
    lambda r:r['updates'][0].update(learning_rate=0),lambda r:r['updates'][0]['reload'].update(checkpoint_sha256='a'*64),
    lambda r:r['native_audit']['requested_saturation_counts'][3].__setitem__(5,2),
    lambda r:r['native_audit'].update(requested_saturation_counts=[[0]*18]*31),lambda r:r.update(lineage={'OLD_ACTOR':True})]
  for change in mutations:
   with self.subTest(change=change),tempfile.TemporaryDirectory()as td:
    d=Path(td);identity,state,learner=self.fixture(d);change(learner);write(d/'learner/state.json',learner);self.seal(d,state)
    with self.assertRaises(ValueError):validate_result(d,identity)
 def test_late_error_and_missing_raw_seal_rejected(self):
  with tempfile.TemporaryDirectory()as td:
   d=Path(td);identity,state,learner=self.fixture(d);write(d/'native_errors.json',[{'error':'post-close'}])
   with self.assertRaisesRegex(ValueError,'late native'):validate_result(d,identity)
   write(d/'native_errors.json',[]);state['outputs'].pop('learner/policy_control_trace.pt');write(d/'state.json',state)
   with self.assertRaisesRegex(ValueError,'not sealed'):validate_result(d,identity)

if __name__=='__main__':unittest.main()
