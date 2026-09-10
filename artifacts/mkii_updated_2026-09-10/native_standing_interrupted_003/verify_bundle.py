"""Verify publication hashes and interruption evidence, without native imports or execution."""
from pathlib import Path
import hashlib,json,sys
sys.dont_write_bytecode=True
from stream_raw import verify_raw
r=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def inventory(p,omit=None):
 assert not any(q.is_symlink()for q in p.rglob('*')),'Unsafe symlink'
 return {q.relative_to(p).as_posix():sha(q)for q in p.rglob('*')if q.is_file()and q!=p/str(omit)and '__pycache__'not in q.parts}
expected=read(r/'BUNDLE_SHA256.json');actual=inventory(r,'BUNDLE_SHA256.json');assert actual==expected,'Published payload changed'
assert all((r/name).stat().st_size<=48<<20 for name in actual),'Git-safe48MiB bound exceeded'
provenance=read(r/'PROVENANCE.json')
for name,part in provenance['components'].items():
 p=r/name
 if 'freeze_sha256'in part:
  assert sha(p/'FREEZE_SHA256.json')==part['freeze_sha256'];m=read(p/'FREEZE_SHA256.json');assert len(m)==part['payloads']and inventory(p,'FREEZE_SHA256.json')==m,name
 elif 'snapshot'in part:assert inventory(p)==part['snapshot'],name
 else:raise ValueError('Unbound publication component')
a=read(r/'terminal/audit.json');result=read(r/'RESULT.json');campaign=read(r/'terminal/run/campaign.json');job=read(r/'terminal/run/jobs/standing.json');state=read(r/'terminal/run/standing/state.json')
assert a['audit_verified']and a['errors']==[]and a['raw_inventory_stable']
assert not a['standing_completed']and not a['raw_acquisition_completed']and not a['native_validation']['attempted']
assert a['unit']['InvocationID']==result['invocation']=='ff270c50458f47e5940dd93c4b3aa845'
assert a['unit']['MainPID']=='0'and a['unit']['ExecMainStatus']=='1'and a['unit']['ActiveState']=='failed'
assert campaign['status']==job['status']=='failed'
assert campaign['error']==job['error']==result['host_error']=="RuntimeError('Unrelated CUDA process appeared; yielding this owned job')"
competitors=[{'process':'3758066, /home/orionh/ithaca-reconstruction/env/bin/python','cgroup':'0::/user.slice/user-1000.slice/session-c10825.scope'}]
assert job['competitors']==result['recorded_competitors']==competitors
assert state['status']=='running'and state['explicit_steps_completed']==0 and state['checks']=={}and state['errors']==[]
assert result['outcome']=='operational_foreign_cuda_interruption'and result['native_state_unfinalized']
assert result['complete_final_native_row_count']is None and not result['physical_rejection_claimed']and not result['standing_pass_claimed']
assert result['source_optimized_in_this_run']and result['num_envs']==campaign['identity']['num_envs']==1
assert campaign['terminal_inputs_unchanged']and job['cleanup_checked']and a['restoration']['passed']
assert len(a['owned_absence']['identifiers'])==2 and all(x['absent']for x in a['owned_absence']['identifiers'].values())
for kind in ('source','host','guard','ownership_supervisor','asset'):
 assert a['input_'+kind]==a['final_input_'+kind]and a['input_'+kind]['passed']
for name in ('source','host','guard'):assert a['input_'+name]['manifest_sha256']==provenance['components'][name]['freeze_sha256']
for data in (campaign,state,result):assert data['physical_admission']is False and data['training_allowed']is False
assert job['no_policy_loaded']and not result['training_started']and not result['Stage2_complete']
assert len(a['raw_inventory'])==31 and a['raw_total_bytes']==144818179
raw=verify_raw(r);assert raw['logical_raw_files']==31 and raw['original_raw_bytes_verified']==144818179 and raw['stored_raw_bytes_verified']==23788742
print(json.dumps({'passed':True,'payloads':len(actual),'raw':raw,'outcome':result['outcome'],'standing_admitted':False,'training_started':False,'no_native_or_GPU_execution':True},sort_keys=True))
