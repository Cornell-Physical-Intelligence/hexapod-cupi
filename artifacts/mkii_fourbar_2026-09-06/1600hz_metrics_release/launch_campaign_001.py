import datetime,hashlib,importlib.machinery,importlib.util,json,os,subprocess,time
from pathlib import Path
base=Path('/home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1')
source=base/'source'
wrapper=base/'host'/'run_campaign_per_job_lock.py'
expected_wrapper=(base/'host'/'wrapper.sha256').read_text().strip()
assert hashlib.sha256(wrapper.read_bytes()).hexdigest()==expected_wrapper
meta=json.loads((base/'staging.json').read_text())
assert meta['source_commit']=='c2af43ca0f384a4c2c7ab8f1d627f309dc78a683'
loader=importlib.machinery.SourceFileLoader('hexapod_launch_campaign',str(source/'isaaclab/deploy/run-mkii-fourbar-campaign'))
spec=importlib.util.spec_from_loader(loader.name,loader);campaign=importlib.util.module_from_spec(spec);loader.exec_module(campaign)
assert campaign.identity(source)['sha256']==meta['functional_sha256']=='c33b4591e99f287579181469dcc132f659d1980200708475442dbe3ffcb7ecfc'
assert not (base/'launch_001.json').exists()
assert not (base/'campaign_001').exists()
preflight=campaign.resource_ready()
shared=campaign.host.COORDINATION;before=shared.read_bytes()
campaign.host.require_coordination_none(campaign.host.read_coordination_control())
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
note=(f'\n## Hexapod 1600 Hz campaign — {now}\n\nUser instruction remains: take full training priority. Frozen source {meta["source_commit"]}, functional identity {meta["functional_sha256"]}, at {source}. Current actual work: one bounded validation-to-PPO campaign at {base}/campaign_001. All 935 CPU tests and GitHub CI passed. New physics recipe requires fresh 1x100 probe, complete32x1000standing+2400driven nominal/refined, then64x3scratch and512x1000full PPO only after unchanged admission gates. Each actual phase holds /opt/wx/gpu.lock and the project lock only for its own job; no separate persistent reservation. GPU may be briefly idle between phases; hexapod priority remains active. The five-minute app progress automation stays removed.\n')
# Append only after checking original bytes. Control line is not changed.
if shared.read_bytes()!=before:raise RuntimeError('Coordination changed during preparation')
with shared.open('a') as f:f.write(note)
campaign.host.require_coordination_none(campaign.host.read_coordination_control())
argv=['/usr/bin/python3',str(wrapper),'--source-dir',str(source),'--source-commit',meta['source_commit'],'--asset-model','mkii_fourbar_v5','--environment-layout','coincident_flat_origin_v1','--output-root',str(base/'campaign_001'),'--wait-seconds','0','--phase-timeout-seconds','7200','--full-timeout-seconds','21600']
with (base/'host_001.log').open('x') as log:
 proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
start_ticks=int(Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
record={'utc':now,'pid':proc.pid,'start_ticks':start_ticks,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'argv':argv,'source_commit':meta['source_commit'],'functional_sha256':meta['functional_sha256'],'wrapper_sha256':expected_wrapper,'preflight':preflight,'persistent_reservation':False,'per_job_lock':'/opt/wx/gpu.lock','coordination_before_sha256':hashlib.sha256(before).hexdigest(),'coordination_after_sha256':hashlib.sha256(shared.read_bytes()).hexdigest()}
for _ in range(100):
 matches=[p for p in (base/'campaign_001').glob('fourbar-campaign-*/campaign.json') if json.loads(p.read_text()).get('pid')==proc.pid]
 if len(matches)==1:record['campaign']=str(matches[0]);break
 if proc.poll() is not None:break
 time.sleep(.1)
if proc.poll() is not None:record['early_exit_code']=proc.returncode
(base/'launch_001.json').write_text(json.dumps(record,indent=2)+'\n')
recovery_path=base.parent/'mkii_motion_recovery_v1/recovery.json'
recovery=json.loads(recovery_path.read_text());recovery.update(state='running_1600hz_campaign',last_update_utc=now,active_job=record.get('campaign'),latest_campaign_launch=str(base/'launch_001.json'),physical_ppo_started=False,next_step='Fresh1600Hz nominal/refined admission followed by scratch/full PPO and learned policy video; no prior failed report is reused.')
recovery_path.write_text(json.dumps(recovery,indent=2)+'\n')
print(json.dumps(record))
