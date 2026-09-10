"""Read-only terminal audit. Exit zero means authentic audited evidence, not native success."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib.util,json,math,re,subprocess,sys,time
sys.dont_write_bytecode=True
BASE=Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910')
RUN=BASE/'native_standing_005'; PAUSE=BASE/'forecast_pause_013'
UNIT='hexapod-canonical-native-standing-005-20260910.service'
INV='f70b9d4b929244699a5725e018b38ba9'
SOURCE=BASE/'standing_source_004'; HOST=BASE/'standing_host_006'; GUARD=BASE/'standing_guard_007'; ASSET=BASE/'asset_001'
ADMISSION=BASE/'native_actuation_001/actuation'
RESERVATION=BASE/'exclusive_reservation_001'
SUPERVISOR=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_physics_source_009')
PINS=[('source',SOURCE,'FREEZE_SHA256.json','037013353a30c5f3751235634aafdb3f204b676837962b14df72c3051c0a2786',95),
 ('host',HOST,'FREEZE_SHA256.json','51a0dbdc5d46a6b4f4a24b4e26c5091e7c588ae8d31785a264f95b3ffc042ed6',55),
 ('guard',GUARD,'FREEZE_SHA256.json','dc47526018efb5336dfa42fbb145366d9fbc53d47b59a657d25b8bd4784c3b8e',58),
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
 require(job.get('phase')=='standing' and job.get('cleanup_checked') is True,'Owned cleanup not verified')
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
 return ['systemd-run','--user','--unit='+UNIT,'--property=RuntimeMaxSec=1320',
  '--property=TimeoutStopSec=180','--property=KillMode=process','--property=Environment=PYTHONDONTWRITEBYTECODE=1',
  '--property=ExecStopPost=/usr/bin/python3 '+str(PAUSE/'resume_forecasting.py'),'--property=Environment=PYTHONUNBUFFERED=1',
  '/usr/bin/python3',str(HOST/'launch_standing_spark.py'),'--source',str(SOURCE),'--asset',str(ASSET),'--admission',str(ADMISSION),
  '--supervisor-source',str(SUPERVISOR),'--output',str(RUN),'--num-envs','1']

def restoration(pause,restored,launch):
 require(pause.get('unit')==UNIT and pause.get('output')==str(RUN) and pause.get('source')==str(SOURCE) and pause.get('admission')==str(ADMISSION),'Pause identity differs')
 require(pause.get('coordination_sha256')=='649ccda1bdef20bc967ae78f68a992cc8001f4ed7009f78cab5133cc67b29d8f','Wrong coordination identity')
 timers={k for k,v in pause['units'].items() if k.endswith('.timer') and 'ActiveState=active' in v}
 require(isinstance(restored.get('timers'),list) and len(restored['timers'])==len(timers) and set(restored['timers'])==timers,'Restored timer set differs')
 t=restored.get('restored_unix');require(isinstance(t,(float,int)) and not isinstance(t,bool) and math.isfinite(t) and t>=pause['created_unix'],'Invalid restoration time')
 require(pause.get('reservation_path')==str(RESERVATION) and pause.get('restoration_scope')=='per_job_snapshot_only' and pause.get('persistent_reservation_release_attempted')is False,'Missing reservation-aware pause metadata')
 require(restored.get('scope')=='per_job_snapshot_only' and restored.get('persistent_reservation_release_attempted')is False and restored.get('reservation_path')==str(RESERVATION),'Per-job restoration incorrectly claims persistent reservation release')
 require(launch.get('native_source_freeze_sha256')==PINS[0][3],'Wrong dispatched source freeze')
 require(launch.get('command')==expected_launch_command(),'Exact dispatch command differs')
 return {'passed':True,'timers':sorted(timers),'restored_unix':t,'recorded_owned_cleanup':restored.get('owned_cleanup_checked'),'scope':restored.get('scope'),'persistent_reservation_release_attempted':False,'reservation_path':str(RESERVATION)}

def classify(campaign,job,state,unit,native_validation):
 require(campaign.get('status') in ('completed','failed','stopped'),'Campaign is not terminal')
 require(campaign.get('planned_phases')==['standing'],'Unexpected allocated phase')
 require(campaign.get('training_allowed') is False and campaign.get('physical_admission') is False,'Actuation overclaims admission')
 if campaign['status']=='completed':
  require(job.get('status')=='completed' and job.get('exit_code')==0 and job.get('container_id') is not None,'Completed campaign has incomplete job')
  require(unit.get('ExecMainStatus')=='0' and unit.get('Result')=='success','Completed campaign has failed owner')
  require(native_validation.get('passed') is True and state.get('status')=='completed','Completed campaign fails native post-exit validation')
  require(campaign.get('standing')==native_validation.get('receipt'),'Native completed receipt mismatch')
  require(campaign.get('terminal_inputs_unchanged') is True and campaign.get('post_exit_original_inputs_reverified') is True and campaign.get('post_exit_all_standing_payloads_inventoried') is True,'Missing completed post-exit integrity')
  return 'authentic_completed_standing'
 require(bool(campaign.get('error') or campaign.get('integrity_error') or job.get('error') or state.get('errors')),'Failed campaign lacks recorded failure evidence')
 return 'authentic_terminal_failure'

def reservation_check():
 spec=importlib.util.spec_from_file_location('_audited_reservation_guard',GUARD/'launch_guarded_remote.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
 receipt=g.verify_reservation()
 return {'verified':True,'guard_verification':receipt,'files':{name:{'sha256':sha(Path(name)),'size_bytes':Path(name).stat().st_size}for name in g.RESERVATION_PINS},'persistent_release_attempted':False}

def solver_check(record,identity,state):
 require(identity.get('solver_recipe')=={'position_iterations':32,'velocity_iterations':4},'Wrong declared solver recipe')
 expected_roots={'/Robot'}
 require(identity.get('num_envs')==1,'Unexpected replica allocation')
 require(state.get('checks',{}).get('solver_recipe_readback')is True,'Missing completed solver check')
 for phase in ('before','after_authoring','after_reset','after_controlled_steps'):
  values=record.get(phase,{})
  require(set(values)==expected_roots,'Incomplete solver root/phase:'+phase)
  expected=(32,1,[False,False])if phase=='before'else(32,4,[True,True])
  require(all((v.get('position_iterations'),v.get('velocity_iterations'),v.get('authored'))==expected for v in values.values()),'Wrong solver readback:'+phase)
 return {'passed':True,'roots':sorted(expected_roots),'before':[32,1],'after':[32,4],'scope':'Authored USD attribute readback only; no independent backend solver introspection'}

def deadline_check(job,proof):
 expected={k:v for k,v in proof.items() if k not in ('original_function_source','adapted_function_source','_original_module_source')}
 require(job.get('deadline_seconds')==1200 and job.get('app_ready_deadline_seconds')==90,'Wrong operational deadline receipt')
 require(job.get('supervisor_runtime_adapter')==expected,'Wrong exact deadline adapter receipt')
 return {'passed':True,'phase_seconds':1200,'AppReady_seconds':90,'adapter':expected}

def audit():
 result={'schema':'canonical_native_standing_terminal_audit_v1','expected_invocation':INV,'unit_name':UNIT,'checked_unix':time.time(),'read_only':True,'audit_verified':False,'standing_completed':False,'physical_admission':False,'training_allowed':False,'errors':[]}
 def attempt(label,fn):
  try:v=fn();result[label]=v;return v
  except Exception as e:result['errors'].append({'check':label,'error':repr(e)});return None
 if attempt('invocation_binding',lambda:require(re.fullmatch('[0-9a-f]{32}',INV)is not None,'Actual native5 invocation remains pending')or True)is None:return result
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
  def admissioncheck():
   actual=inventory(ADMISSION);require(len(actual)==19 and {k:v['sha256'] for k,v in actual.items()}==read(SOURCE/'ACTUATION_SHA256.json'),'Original coordinate/effort admission changed');require(sha(SOURCE/'ACTUATION_SHA256.json')=='1ccd8f4ffaa2eadc1fd9c6202af8da965b0e9b95f39c585b68a3a64313ce2728','Wrong coordinate/effort map');return actual
  attempt('actuation_admission_inventory',admissioncheck)
  attempt('exclusive_reservation',reservation_check)
 campaign=attempt('campaign',lambda:read(RUN/'campaign.json'))
 job=attempt('job',lambda:read(RUN/'jobs/standing.json'))
 state=attempt('native_state',lambda:read(RUN/'standing/state.json')) if (RUN/'standing/state.json').is_file() else None
 result['native_state_available']=state is not None
 result['raw_acquisition_completed']=bool(state is not None and state.get('status')=='completed' and state.get('explicit_steps_completed')==8000)
 if (RUN/'standing/standing_report.json').is_file():attempt('standing_report',lambda:read(RUN/'standing/standing_report.json'))
 if (RUN/'standing/solver_readback.json').is_file():attempt('solver_readback',lambda:read(RUN/'standing/solver_readback.json'))
 if job is not None:attempt('owned_absence',lambda:owned_absence(job))
 def phase_inventory():
  actual={p.name for p in (RUN/'jobs').glob('*.json') if not p.name.endswith('_contact_data_audit.json')};require(actual=={'standing.json'},'Unexpected or missing allocated jobs');return sorted(actual)
 attempt('phase_inventory',phase_inventory)
 attempt('restoration',lambda:restoration(read(PAUSE/'pause.json'),read(PAUSE/'restored.json'),read(PAUSE/'launch.json')))
 native_validation={'attempted':False,'passed':False};result['native_validation']=native_validation
 if all(checks.values()) and result.get('input_asset'):
  def hostidentity():
   s=importlib.util.spec_from_file_location('_audit_canonical_host',HOST/'launch_standing_spark.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
   args=SimpleNamespace(source=SOURCE,asset=ASSET,admission=ADMISSION,supervisor_source=SUPERVISOR,output=RUN,host_freeze_sha256=PINS[1][3],num_envs=1,standing_one=None)
   identity=h.verify_inputs(args)
   if job is not None:attempt('operational_deadline',lambda:deadline_check(job,h.load_deadline_adapter().inspect_supervisor(SUPERVISOR/'tools/launch_reference_physics_spark.py')))
   if state is not None and state.get('status')=='completed':attempt('solver_verification',lambda:solver_check(result.get('solver_readback',{}),identity,state))
   if campaign is not None:require(campaign.get('identity')==identity and campaign.get('host_freeze_sha256')==PINS[1][3] and campaign.get('source_freeze_sha256')==PINS[0][3],'Host campaign input identity mismatch')
   if state is not None and state.get('status')=='completed':
    native_validation['attempted']=True
    try:native_validation.update(passed=True,receipt=h.CONTRACT.validate_result(RUN/'standing',identity))
    except Exception as e:native_validation.update(passed=False,error=repr(e))
   return identity
  attempt('identity',hostidentity)
 if campaign is not None and job is not None:
  outcome=attempt('terminal_outcome',lambda:classify(campaign,job,state or {},unit,native_validation))
  result['standing_completed']=outcome=='authentic_completed_standing'
  if result['standing_completed']:
   def seal():
    p=RUN/'standing_immutable.sha256.json';require(sha(p)==campaign.get('standing_payload_manifest_sha256'),'Wrong post-exit manifest');require(read(p)=={k:v['sha256'] for k,v in inventory(RUN/'standing').items()},'Post-exit standing payload changed');return {'passed':True,'manifest_sha256':sha(p)}
   attempt('post_exit_seal',seal)
 def raw():
  data={'run/'+k:v for k,v in inventory(RUN).items()};data.update({'forecast_pause/'+k:v for k,v in inventory(PAUSE).items()});return data
 first=attempt('raw_inventory',raw);second=attempt('raw_inventory_second_pass',raw)
 if first is not None and second is not None:
  if first!=second:result['errors'].append({'check':'raw_inventory_stability','error':'Terminal raw bytes changed between full hash passes'})
  result['raw_inventory_stable']=first==second;result['raw_payloads']=len(first);result['raw_total_bytes']=sum(v['size_bytes'] for v in first.values())
 result.pop('raw_inventory_second_pass',None)
 for name,root,manifest,bound,count in PINS:attempt('final_input_'+name,lambda root=root,manifest=manifest,bound=bound,count=count:checked(root,manifest,bound,count))
 if all(checks.values()):
  attempt('final_input_asset',assetcheck)
  final_admission=attempt('final_actuation_admission_inventory',admissioncheck)
  if final_admission is not None and final_admission!=result.get('actuation_admission_inventory'):result['errors'].append({'check':'actuation_stability','error':'Original coordinate/effort admission changed between audit passes'})
  result.pop('final_actuation_admission_inventory',None)
  final_reservation=attempt('final_exclusive_reservation',reservation_check)
  if final_reservation is not None and final_reservation!=result.get('exclusive_reservation'):result['errors'].append({'check':'reservation_stability','error':'Persistent reservation changed between audit passes'})
  result.pop('final_exclusive_reservation',None)

 result['audit_verified']=not result['errors']
 if not result['audit_verified']:result['standing_completed']=False
 return result

if __name__=='__main__':
 result=audit();print(json.dumps(result,indent=2,allow_nan=False));raise SystemExit(0 if result['audit_verified'] else 1)
