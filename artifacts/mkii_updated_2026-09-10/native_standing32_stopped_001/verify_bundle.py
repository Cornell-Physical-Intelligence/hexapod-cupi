from pathlib import Path
import hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
r=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def inventory(p,omit):
 assert not any(q.is_symlink()for q in p.rglob('*'))
 return {q.relative_to(p).as_posix():sha(q)for q in p.rglob('*')if q.is_file()and q!=p/omit and '__pycache__'not in q.parts}
expected=read(r/'BUNDLE_SHA256.json');actual=inventory(r,'BUNDLE_SHA256.json');assert actual==expected,'Published payload changed'
assert all((r/name).stat().st_size<=48<<20 for name in actual),'Git-safe48MiB part bound exceeded'
provenance=read(r/'PROVENANCE.json')
for name,component in provenance['components'].items():
 if 'freeze_sha256'in component:
  p=r/name;assert sha(p/'FREEZE_SHA256.json')==component['freeze_sha256'];assert inventory(p,'FREEZE_SHA256.json')==read(p/'FREEZE_SHA256.json'),name
a=read(r/'terminal/audit.json');result=read(r/'RESULT.json');campaign=read(r/'terminal/run/campaign.json');job=read(r/'terminal/run/jobs/standing.json');state=read(r/'terminal/run/standing/state.json')
assert a['audit_verified']and not a['standing_completed']and not a['raw_acquisition_completed']and a['raw_inventory_stable']and a['errors']==[]
assert a['unit']['InvocationID']==result['invocation']=='ce49acca09a14d77b081fc5cb6c41622'
assert a['unit']['MainPID']=='0'and a['unit']['ExecMainStatus']=='1'and a['unit']['ActiveState']=='failed'
assert campaign['status']==job['status']=='stopped'and campaign['error']==job['error']=="InterruptedError('Stop requested')"
assert state['status']=='running'and state['explicit_steps_completed']==0 and state['checks']=={}
assert result['outcome']=='operational_throughput_stop'and not result['physics_rejection_claimed']and not result['complete_final_clock_claimed']
assert result['complete_final_native_row_count']is None and not result['source_optimized_in_this_run']
assert campaign['terminal_inputs_unchanged']and job['cleanup_checked']and a['restoration']['passed']
assert len(a['owned_absence']['identifiers'])==2 and all(x['absent']for x in a['owned_absence']['identifiers'].values())
for kind in ('source','host','guard','ownership_supervisor','asset'):
 assert a['input_'+kind]==a['final_input_'+kind]and a['input_'+kind]['passed']
for name in ('source','host','guard'):
 assert a['input_'+name]['manifest_sha256']==provenance['components'][name]['freeze_sha256']
for data in (campaign,state,result):
 assert data['physical_admission']is False and data['training_allowed']is False
assert read(r/'root_checks/STOP_REQUEST.json')['verified_exact_owner']['InvocationID']==result['invocation']
assert read(r/'root_checks/STOP_DECISION.json')['invocation']==result['invocation']
spec=importlib.util.spec_from_file_location('_published_streaming_raw',r/'stream_raw.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
verified=module.verify_raw(r)
print(json.dumps({'passed':True,'payloads':len(actual),'raw':verified,'outcome':'operational throughput stop; no completed standing32 or physics rejection','no_native_or_GPU_execution':True},sort_keys=True))
