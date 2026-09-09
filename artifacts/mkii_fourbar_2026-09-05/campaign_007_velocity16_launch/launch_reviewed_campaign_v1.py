#!/usr/bin/env python3
"""Launch one bounded campaign after independent standing-pair review."""
import datetime
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

base=Path('/home/orionh/HEXAPOD_runs/mkii_placement_convergence_v1')
source=base/'source'
loader=importlib.machinery.SourceFileLoader('reviewed_campaign',str(source/'isaaclab/deploy/run-mkii-fourbar-campaign'))
spec=importlib.util.spec_from_loader(loader.name,loader)
campaign=importlib.util.module_from_spec(spec);loader.exec_module(campaign)
contract=campaign.identity(source)
if contract['sha256']!='1fcab03b2c9f810a931e3d65fd057312d6dfb3b92003ca30058ccd821ff965e6':
 raise ValueError('Staged source changed')
preflight=campaign.resource_ready()
pair_path=Path('/home/orionh/HEXAPOD_runs/mkii_final_velocity16_v1/standing_pair_20260905T215119Z/pair.json')
pair=json.loads(pair_path.read_text())
if pair.get('state')!='standing_pair_complete_requires_independent_comparison':
 raise ValueError('Standing pair is incomplete')
reports=[json.loads(Path(p['report']).read_text()) for p in pair['phases']]
if len(reports)!=2 or [r.get('solver_multiplier') for r in reports]!=[1,2]: raise ValueError('Wrong standing pair')
for r in reports:
 if r.get('contract',{}).get('sha256')!='d3442002687f4ff7b34bd2e24134a8e3265a87e05221d3cbf33f2630c0b52d36' or r.get('pass') is not True or r.get('errors')!=[] or r.get('num_envs')!=32 or r.get('steps_completed')!=600 or r.get('driven_steps')!=0:
  raise ValueError('Standing workload failed')
from mkii_asset_binding import solver_runtime_equivalent
from qualify_mkii_fourbar import reset_root_positions
if not solver_runtime_equivalent(*(r['runtime_manifest'] for r in reports)) or reset_root_positions(reports[0])!=reset_root_positions(reports[1]):
 raise ValueError('Standing placements or runtime differ')
for key,absolute,relative in [('mean_height_m',.001,0.),('max_applied_nm',.05,.05),('max_demand_nm',.05,.05)]:
 a,b=(r['windows']['settled'][key] for r in reports)
 if abs(a-b)>max(absolute,relative*abs(b)): raise ValueError('Standing convergence failed: '+key)
now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
launch_file=base/f'campaign_launch_{now}.json'
old_guard=Path('/home/orionh/HEXAPOD_runs/mkii_fourbar_diagnostics_v1/priority_20260905T1616')
old=json.loads((old_guard/'status.json').read_text())
reservation=base/f'priority_{now}'
reservation.mkdir()
script=Path('/home/orionh/HEXAPOD_runs/mkii_fourbar_diagnostics_v1/reserve_workflow.py')
with (reservation/'guard.log').open('w') as log:
 guard=subprocess.Popen(['/usr/bin/python3',str(script),str(reservation)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(50):
 if (reservation/'status.json').exists(): break
 time.sleep(.1)
else: raise RuntimeError('New guard did not publish startup')
if old.get('state')=='reserved':
 pid=old['pid']
 actual=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
 if str(script).encode() not in actual or str(old_guard).encode() not in actual:
  raise RuntimeError('Prior guard PID identity differs; no signal sent')
 os.kill(pid,signal.SIGTERM)
for _ in range(100):
 state=json.loads((reservation/'status.json').read_text())
 if state['state']=='reserved': break
 time.sleep(.1)
else:
 (reservation/'release').touch()
 raise RuntimeError('New bounded priority guard did not acquire lock')
campaign.resource_ready()
with campaign.host.COORDINATION.open('a') as note:
 note.write(f'\n{now} Hexapod: matched 32-world standing comparison completed with unchanged physical bounds. Starting fresh probe, full 32-world nominal/refined standing+driven validation, scratch64x3 and full512x1000 PPO only if all admission/checkpoint gates pass. Frozen fd34f66 at {source}; 898 CPU tests and CI passed. State: {base}/campaigns/fourbar-campaign-*/campaign.json. Priority guard {guard.pid} replaces1465763 without interrupting GPU work, maximum ten hours and release on selected campaign completion/failure/pause. Existing sharing handshake retained.\n')
argv=['/usr/bin/python3',str(source/'isaaclab/deploy/run-mkii-fourbar-campaign'),
 '--source-dir',str(source),'--source-commit','fd34f661bd672df9f68e6d8568548ad092ef32cf',
 '--asset-model','mkii_fourbar_v5','--output-root',str(base/'campaigns'),
 '--wait-seconds','0','--phase-timeout-seconds','7200','--full-timeout-seconds','21600']
with (base/f'campaign_launch_{now}.log').open('w') as log:
 proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
launch={'pid':proc.pid,'argv':argv,'utc':now,'source_contract_sha256':contract['sha256'],
 'standing_pair':str(pair_path),'standing_pair_sha256':hashlib.sha256(pair_path.read_bytes()).hexdigest(),
 'standing_report_sha256':[hashlib.sha256(Path(p['report']).read_bytes()).hexdigest() for p in pair['phases']],
 'bootstrap_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'preflight':preflight,'guard_pid':guard.pid,'reservation':str(reservation),
 'guard_expires_utc':state['expires_utc']}
launch_file.write_text(json.dumps(launch,indent=2)+'\n')
for _ in range(150):
 matches=list((base/'campaigns').glob('fourbar-campaign-*/campaign.json'))
 matches=[p for p in matches if json.loads(p.read_text()).get('pid')==proc.pid]
 if len(matches)==1:
  (reservation/'campaign_path.txt').write_text(str(matches[0])+'\n')
  launch['campaign']=str(matches[0]);break
 if proc.poll() is not None: break
 time.sleep(.1)
if 'campaign' not in launch:
 launch['campaign_selection_pending']=True
launch_file.write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps(launch))
