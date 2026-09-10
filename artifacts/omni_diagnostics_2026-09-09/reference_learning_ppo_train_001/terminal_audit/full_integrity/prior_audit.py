"""Read-only exact learning admission terminal audit; outputs JSON to stdout.

Run only after the admission unit is terminal. No container/service signals,
GPU calls, simulator imports, source writes, or numerical admission decisions.
"""
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import hashlib, importlib.util, json, math, subprocess, sys, time
sys.dont_write_bytecode=True
BASE=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
RUN=BASE/'reference_learning_ppo_001'; PAUSE=BASE/'forecast_pause_051'
UNIT='hexapod-learning-ppo-admission-001-20260910.service'
INVOCATION='d161f77036f346d2b1abe244d95b2dbe'
PHASES=('standing','calibration','learning_recovery_32')
SPECS=[
 ('source','reference_physics_source_009','campaign_source_hashes.json','04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e',926),
 ('consumer','reference_moving_ppo_source_003','FREEZE_SHA256.json','fd9bef87dda976f9b229e7541408d24e674245ca2c51231dfce9075b81d38334',51),
 ('host','reference_learning_ppo_host_001','FREEZE_SHA256.json','45f623d298c00136ca18d159261d8df601bd42840ebae467fa94bf38283e6792',3),
 ('guard','reference_learning_ppo_guard_001','FREEZE_SHA256.json','fc623f16e5a559da3940852892bc3eeef39b6143ddf3587873dcc07480cafc78',11),
 ('bridge','reference_device_smoke_adapter_001','FREEZE_SHA256.json','be4870a8f7ca0857ba50ffd3a0c92ac5925b9a816767d66d826af7b59957aa0e',18),
 ('observation','reference_policy_observation_005_001','FREEZE_SHA256.json','22402b0079b0b5ac0acdd4e1d7e819fe206b20ee1883077f7de6eaca55808a63',160),
]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for data in iter(lambda:f.read(8<<20),b''):h.update(data)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(test,message):
 if not test:raise ValueError(message)
def relative(name):
 p=PurePosixPath(name)
 require(not p.is_absolute() and name and '..' not in p.parts and str(p)==name,'Unsafe payload path')
 return p

def tree(root,manifest,bound,count):
 root=Path(root);require(sha(root/manifest)==bound,'Wrong manifest: '+str(root))
 m=read(root/manifest);require(len(m)==count,'Wrong count: '+str(root))
 require(not root.is_symlink() and not any(p.is_symlink() for p in root.rglob('*')),'Symbolic source substitution')
 for f,h in m.items():relative(f);require(sha(root/f)==h,'Changed payload: '+str(root/f))
 actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
 require(actual==set(m)|{manifest},'Missing/unlisted source payload')
 return {'path':str(root),'manifest':manifest,'manifest_sha256':bound,'files':count,'unchanged':True}

def terminal(fields,journal=None):
 require(fields.get('ActiveState') in ('failed','inactive'),'Unit is not terminal')
 require(fields.get('SubState') in ('failed','dead'),'Unit terminal substate is unknown')
 if fields.get('InvocationID')!=INVOCATION:
  require(fields.get('InvocationID')=='' and isinstance(journal,list) and len(journal)>=2,'Wrong or unavailable invocation identity')
  matching=[r for r in journal if r.get('USER_UNIT')==UNIT]
  require({r.get('USER_INVOCATION_ID') for r in matching}=={INVOCATION},'Historical journal contains another or unknown invocation')
  require(any(r.get('MESSAGE','').startswith('Started '+UNIT) for r in matching),'Exact historical start missing')
  require(any('Consumed ' in r.get('MESSAGE','') for r in matching),'Exact historical terminal accounting missing')
 require(fields.get('Result') in ('success','exit-code','signal','timeout'),'Unexpected terminal result')
 require('ExecMainStatus' in fields,'Missing main status')

def absence(result,token):
 require(result.returncode==1 and any(s in result.stderr.lower() for s in ('no such object','no such container')),
         'Container absence is unknown: '+token)
 return {'returncode':result.returncode,'stderr':result.stderr.strip(),'absent':True}

def restoration(pause,restored):
 require(pause['unit']==UNIT and pause['output']==str(RUN),'Wrong pause owner')
 timestamp=restored.get('restored_unix')
 require(type(timestamp) in (int,float) and math.isfinite(timestamp) and timestamp>=pause['created_unix'],'Invalid restoration timestamp')
 expected=[name for name,state in pause['units'].items() if name.endswith('.timer') and 'ActiveState=active' in state]
 require(restored.get('timers')==expected,'Restored timer list differs from authorized prior state')
 require(isinstance(restored.get('owned_cleanup_checked'),list),'Restoration cleanup receipt missing')

def main():
 unittext=subprocess.check_output(['systemctl','--user','show',UNIT,'-p','ActiveState','-p','SubState','-p','Result','-p','ExecMainStatus','-p','InvocationID'],text=True,timeout=20)
 unit=dict(line.split('=',1) for line in unittext.splitlines())
 journaltext=subprocess.check_output(['journalctl','--user','-u',UNIT,'--no-pager','-o','json'],text=True,timeout=20)
 journal=[json.loads(line) for line in journaltext.splitlines() if line.strip()]
 terminal(unit,journal)
 campaign=read(RUN/'campaign.json')
 require(campaign.get('status') in ('completed','failed','stopped'),'Campaign is not terminal')
 require(campaign.get('PPO_updates_completed',0)==0,'This is not the admission-only allocation')
 initial_campaign_sha=sha(RUN/'campaign.json')
 require(not (RUN/'learning_campaign.json').exists(),'Later learning allocation already exists; use admission-specific immutable snapshot')
 inputs={k:tree(BASE/folder,manifest,h,count) for k,folder,manifest,h,count in SPECS}
 assets=read(RUN/'inputs/study_before.sha256.json');require(len(assets)==550,'Wrong asset count')
 root=RUN/'inputs/study'
 require(not root.is_symlink() and not any(p.is_symlink() for p in root.rglob('*')),'Symbolic assets')
 for name,h in assets.items():relative(name);require(sha(root/name)==h,'Changed asset: '+name)
 require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(assets),'Asset tree inventory changed')
 # Exact reviewed host read-only proof verifier, after its whole source tree is verified.
 spec=importlib.util.spec_from_file_location('_audit_bound_learning_host',BASE/'reference_learning_ppo_host_001/launch_learning_ppo_spark.py')
 host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
 args=SimpleNamespace(source=BASE/'reference_physics_source_009',run=BASE/'reference_physics_009',
      device_run=BASE/'reference_device_smoke_001',bridge=BASE/'reference_device_smoke_adapter_001',
      consumer=BASE/'reference_moving_ppo_source_003',observation=BASE/'reference_policy_observation_005_001',
      output=RUN,phase_group='admission',decision_receipt=None)
 standing_present=(RUN/'standing/admission.json').is_file()
 identity=host.validate_inputs(args,phase='learning_recovery_32',require_standing=standing_present)
 jobs={};absent={};phases={}
 for path in sorted((RUN/'jobs').glob('*.json')):
  j=read(path)
  if 'container_name' not in j:continue
  require(path.stem in PHASES,'Undeclared job')
  require(j.get('phase')==path.stem and j.get('cleanup_checked') is True,'Job cleanup/phase not verified')
  for token in (j.get('container_name'),j.get('container_id')):
   if token is None:continue
   r=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',token],text=True,capture_output=True,timeout=20)
   absent[token]=absence(r,token)
  require(j.get('container_name'),'Missing allocated name')
  jobs[path.stem]={k:j.get(k) for k in ('status','phase','container_name','container_id','cleanup_checked')}
 require(set(jobs) and set(jobs).issubset(PHASES),'No recorded job or undeclared allocation')
 if campaign['status']=='completed':
  require(set(jobs)==set(PHASES) and all(j['container_id'] for j in jobs.values()),'Completed admission lacks all three job IDs')
 for name in PHASES:
  p=RUN/name/'state.json'
  if p.exists():
   state=read(p);phases[name]={'sha256':sha(p),'status':state.get('status'),'failure':state.get('failure'),
      'PPO_updates_completed':state.get('PPO_updates_completed',0),'policy_training_started':state.get('policy_training_started',False)}
   require(not phases[name]['policy_training_started'] and phases[name]['PPO_updates_completed']==0,'Unexpected learning in admission')
 restored=read(PAUSE/'restored.json');pause=read(PAUSE/'pause.json')
 restoration(pause,restored)
 raw={};sizes={}
 for prefix,folder in [('run',RUN),('forecast_pause',PAUSE)]:
  require(not any(p.is_symlink() for p in folder.rglob('*')),'Symbolic raw evidence')
  for p in sorted(folder.rglob('*')):
   if p.is_file() and not (prefix=='run' and p.is_relative_to(root)):
    name=prefix+'/'+p.relative_to(folder).as_posix();raw[name]=sha(p);sizes[name]=p.stat().st_size
 for name in ('reference_learning_ppo_preflight_001.py','reference_learning_ppo_preflight_001.json'):
  raw['preflight/'+name]=sha(BASE/name);sizes['preflight/'+name]=(BASE/name).stat().st_size
 require(initial_campaign_sha==sha(RUN/'campaign.json'),'Campaign changed during audit')
 print(json.dumps({'verified_unix':time.time(),'unit':unit,'campaign_status':campaign['status'],
    'campaign_sha256':initial_campaign_sha,'historical_invocation_journal':journal,'journal_sha256':hashlib.sha256(journaltext.encode()).hexdigest(),'PPO_updates_completed':0,'input_maps':inputs,
    'assets_files':550,'assets_unchanged':True,'host_readonly_proof_identity':identity,
    'raw_payloads':raw,'raw_sizes_bytes':sizes,'jobs':jobs,'phases':phases,
    'owned_names_and_ids_absent':absent,'unallocated_phases':[p for p in PHASES if p not in jobs],
    'pause_restoration':restored,'audit_actions_read_only':True,
    'scope':'Terminal identity/integrity/cleanup only; numerical recovery and root allocation review remain separate.'},indent=2))

if __name__=='__main__':main()
