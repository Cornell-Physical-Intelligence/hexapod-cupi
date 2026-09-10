"""Read-only terminal audit. Exit zero means authentic audited evidence, not native success."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib.util,json,math,re,subprocess,sys,time
sys.dont_write_bytecode=True
BASE=Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910')
RUN=BASE/'native_inspection_001'; PAUSE=BASE/'forecast_pause_001'
UNIT='hexapod-canonical-native-inspection-001-20260910.service'
INV='87a75ef9d130496a9e040b86b14367e9'
SOURCE=BASE/'inspection_source_001'; HOST=BASE/'inspection_host_001'; GUARD=BASE/'inspection_guard_001'; ASSET=BASE/'asset_001'
SUPERVISOR=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_physics_source_009')
PINS=[('source',SOURCE,'FREEZE_SHA256.json','c59ebd53bb221fd17b46d9fbc31e3df7240271bfd13d6f7269d7307876c7bfaa',21),
 ('host',HOST,'FREEZE_SHA256.json','2d21500ae7273ecf70576b577772c38844915a05f0fe67b8e2c213137d9994a1',17),
 ('guard',GUARD,'FREEZE_SHA256.json','b352f7e0d42b462cfc125fd2f1130128e3150094795ce2666d6233a56c3d0fb2',24),
 ('ownership_supervisor',SUPERVISOR,'campaign_source_hashes.json','04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e',926)]

def require(ok,message):
 if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def inventory(root):
 require(root.is_dir() and not root.is_symlink(),'Missing or symbolic inventory root: '+str(root))
 require(not any(p.is_symlink() for p in root.rglob('*')),'Symbolic inventory payload')
 return {p.relative_to(root).as_posix():{'sha256':sha(p),'size_bytes':p.stat().st_size} for p in sorted(root.rglob('*')) if p.is_file()}
def checked(root,manifest,bound,count):
 require(sha(root/manifest)==bound,'Wrong immutable manifest: '+str(root))
 actual=inventory(root);actual.pop(manifest)
 require({k:v['sha256'] for k,v in actual.items()}==read(root/manifest),'Changed or unlisted immutable payload: '+str(root))
 require(len(actual)==count,'Unexpected immutable payload count')
 return {'passed':True,'payloads':count,'manifest_sha256':bound}
def call(command):return subprocess.check_output(command,text=True,timeout=30)
def terminal_unit(unit,rows):
 require(unit.get('MainPID')=='0' and unit.get('ActiveState') in ('inactive','failed'),'Owner is not terminal')
 require(unit.get('InvocationID') in ('',INV),'Wrong current invocation')
 exact=[r for r in rows if r.get('_SYSTEMD_INVOCATION_ID')==INV or r.get('USER_INVOCATION_ID')==INV]
 require(bool(exact),'Expected invocation lacks exact journal evidence')
 return exact

def owned_absence(job):
 require(job.get('phase')=='inspection' and job.get('cleanup_checked') is True,'Owned cleanup not verified')
 name=job.get('container_name');identifier=job.get('container_id')
 require(isinstance(name,str) and re.fullmatch('hexapod-reference-physics-[0-9a-f]{32}',name),'Malformed owned name')
 require(identifier is None or isinstance(identifier,str) and re.fullmatch('[0-9a-f]{64}',identifier),'Malformed recorded ID')
 absent={}
 for key in [name]+([identifier] if identifier else []):
  r=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',key],text=True,capture_output=True,timeout=20)
  require(r.returncode!=0 and any(s in r.stderr.lower() for s in ('no such object','no such container')),'Owned absence unknown: '+key)
  absent[key]={'absent':True,'returncode':r.returncode,'stderr':r.stderr.strip()}
 return {'identifiers':absent,'recorded_id_available':identifier is not None,'unavailable_id_not_claimed_absent':identifier is None}

def expected_launch_command():
 return ['systemd-run','--user','--unit='+UNIT,'--property=RuntimeMaxSec=720',
  '--property=TimeoutStopSec=180','--property=KillMode=process','--property=Environment=PYTHONDONTWRITEBYTECODE=1',
  '--property=ExecStopPost=/usr/bin/python3 '+str(PAUSE/'resume_forecasting.py'),'--property=Environment=PYTHONUNBUFFERED=1',
  '/usr/bin/python3',str(HOST/'launch_inspection_spark.py'),'--source',str(SOURCE),'--asset',str(ASSET),
  '--supervisor-source',str(SUPERVISOR),'--output',str(RUN)]

def restoration(pause,restored,launch):
 require(pause.get('unit')==UNIT and pause.get('output')==str(RUN) and pause.get('source')==str(SOURCE),'Pause identity differs')
 require(pause.get('coordination_sha256')=='35af5100e39980f55a3f5e19654e4550b3c3fb6da114b64fbe0d143817242cf3','Wrong coordination identity')
 timers={k for k,v in pause['units'].items() if k.endswith('.timer') and 'ActiveState=active' in v}
 require(isinstance(restored.get('timers'),list) and len(restored['timers'])==len(timers) and set(restored['timers'])==timers,'Restored timer set differs')
 t=restored.get('restored_unix');require(isinstance(t,(float,int)) and not isinstance(t,bool) and math.isfinite(t) and t>=pause['created_unix'],'Invalid restoration time')
 require(launch.get('native_source_freeze_sha256')==PINS[0][3],'Wrong dispatched source freeze')
 require(launch.get('command')==expected_launch_command(),'Exact dispatch command differs')
 return {'passed':True,'timers':sorted(timers),'restored_unix':t,'recorded_owned_cleanup':restored.get('owned_cleanup_checked')}

def classify(campaign,job,state,unit,native_validation):
 require(campaign.get('status') in ('completed','failed','stopped'),'Campaign is not terminal')
 require(campaign.get('planned_phases')==['inspection'],'Unexpected allocated phase')
 require(campaign.get('training_allowed') is False and campaign.get('physical_admission') is False,'Inspection overclaims admission')
 if campaign['status']=='completed':
  require(job.get('status')=='completed' and job.get('exit_code')==0 and job.get('container_id') is not None,'Completed campaign has incomplete job')
  require(unit.get('ExecMainStatus')=='0' and unit.get('Result')=='success','Completed campaign has failed owner')
  require(native_validation.get('passed') is True and state.get('status')=='completed','Completed campaign fails native post-exit validation')
  require(campaign.get('inspection')==native_validation.get('receipt'),'Native completed receipt mismatch')
  require(campaign.get('terminal_inputs_unchanged') is True and campaign.get('post_exit_original_inputs_reverified') is True and campaign.get('post_exit_all_inspection_payloads_inventoried') is True,'Missing completed post-exit integrity')
  return 'authentic_completed_inspection'
 require(bool(campaign.get('error') or campaign.get('integrity_error') or job.get('error') or state.get('errors')),'Failed campaign lacks recorded failure evidence')
 return 'authentic_terminal_failure'

def audit():
 result={'schema':'canonical_native_terminal_audit_v1','expected_invocation':INV,'unit_name':UNIT,'checked_unix':time.time(),'read_only':True,'audit_verified':False,'inspection_completed':False,'physical_admission':False,'training_allowed':False,'errors':[]}
 def attempt(label,fn):
  try:v=fn();result[label]=v;return v
  except Exception as e:result['errors'].append({'check':label,'error':repr(e)});return None
 unit=attempt('unit',lambda:dict(line.split('=',1) for line in call(['systemctl','--user','show',UNIT,'-p','ActiveState','-p','SubState','-p','Result','-p','InvocationID','-p','ExecMainStatus','-p','MainPID']).splitlines() if '=' in line))
 journal=attempt('journal',lambda:[json.loads(s) for s in call(['journalctl','--user','-u',UNIT,'--no-pager','-o','json']).splitlines() if s.strip()])
 if unit is None or journal is None:return result
 if attempt('exact_invocation_journal',lambda:terminal_unit(unit,journal)) is None:return result
 checks={}
 for name,root,manifest,bound,count in PINS:checks[name]=attempt('input_'+name,lambda root=root,manifest=manifest,bound=bound,count=count:checked(root,manifest,bound,count))
 if all(checks.values()):
  def assetcheck():
   actual=inventory(ASSET);require(len(actual)==9 and {k:v['sha256'] for k,v in actual.items()}==read(SOURCE/'ASSET_SHA256.json'),'Canonical asset bytes/inventory differ');return {'passed':True,'payloads':9,'manifest_sha256':sha(SOURCE/'ASSET_SHA256.json')}
  attempt('input_asset',assetcheck)
 campaign=attempt('campaign',lambda:read(RUN/'campaign.json'))
 job=attempt('job',lambda:read(RUN/'jobs/inspection.json'))
 state=attempt('native_state',lambda:read(RUN/'inspection/state.json')) if (RUN/'inspection/state.json').is_file() else None
 result['native_state_available']=state is not None
 if job is not None:attempt('owned_absence',lambda:owned_absence(job))
 def phase_inventory():
  actual={p.name for p in (RUN/'jobs').glob('*.json') if not p.name.endswith('_contact_data_audit.json')};require(actual=={'inspection.json'},'Unexpected or missing allocated jobs');return sorted(actual)
 attempt('phase_inventory',phase_inventory)
 attempt('restoration',lambda:restoration(read(PAUSE/'pause.json'),read(PAUSE/'restored.json'),read(PAUSE/'launch.json')))
 native_validation={'attempted':False,'passed':False};result['native_validation']=native_validation
 if all(checks.values()) and result.get('input_asset'):
  def hostidentity():
   s=importlib.util.spec_from_file_location('_audit_canonical_host',HOST/'launch_inspection_spark.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
   args=SimpleNamespace(source=SOURCE,asset=ASSET,supervisor_source=SUPERVISOR,output=RUN,host_freeze_sha256=PINS[1][3])
   identity=h.verify_inputs(args)
   if campaign is not None:require(campaign.get('identity')==identity and campaign.get('host_freeze_sha256')==PINS[1][3] and campaign.get('source_freeze_sha256')==PINS[0][3],'Host campaign input identity mismatch')
   if state is not None and state.get('status')=='completed':
    native_validation['attempted']=True
    try:native_validation.update(passed=True,receipt=h.CONTRACT.validate_result(RUN/'inspection',identity))
    except Exception as e:native_validation.update(passed=False,error=repr(e))
   return identity
  attempt('identity',hostidentity)
 if campaign is not None and job is not None:
  outcome=attempt('terminal_outcome',lambda:classify(campaign,job,state or {},unit,native_validation))
  result['inspection_completed']=outcome=='authentic_completed_inspection'
  if result['inspection_completed']:
   def seal():
    p=RUN/'inspection_immutable.sha256.json';require(sha(p)==campaign.get('inspection_payload_manifest_sha256'),'Wrong post-exit manifest');require(read(p)=={k:v['sha256'] for k,v in inventory(RUN/'inspection').items()},'Post-exit inspection payload changed');return {'passed':True,'manifest_sha256':sha(p)}
   attempt('post_exit_seal',seal)
 def raw():
  data={'run/'+k:v for k,v in inventory(RUN).items()};data.update({'forecast_pause/'+k:v for k,v in inventory(PAUSE).items()});return data
 first=attempt('raw_inventory',raw);second=attempt('raw_inventory_second_pass',raw)
 if first is not None and second is not None:
  if first!=second:result['errors'].append({'check':'raw_inventory_stability','error':'Terminal raw bytes changed between full hash passes'})
  result['raw_inventory_stable']=first==second;result['raw_payloads']=len(first);result['raw_total_bytes']=sum(v['size_bytes'] for v in first.values())
 result.pop('raw_inventory_second_pass',None)
 result['audit_verified']=not result['errors']
 if not result['audit_verified']:result['inspection_completed']=False
 return result

if __name__=='__main__':
 result=audit();print(json.dumps(result,indent=2,allow_nan=False));raise SystemExit(0 if result['audit_verified'] else 1)
