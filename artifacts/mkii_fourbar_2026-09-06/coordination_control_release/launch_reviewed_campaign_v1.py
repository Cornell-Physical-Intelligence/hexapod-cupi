#!/usr/bin/env python3
"""Start one fresh, source-bound campaign under the explicit exclusive lease."""
import datetime
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

base=Path('/home/orionh/HEXAPOD_runs/mkii_coordination_control_v1')
source=base/'source'
expected_commit='1239159c185cd504c359bb20e98bde9986acbbb4'
expected_identity='c06e56ed68508f944363bfe6564a13d0760066e4a338164b664214edb695fb5a'
meta=json.loads((base/'staging.json').read_text())
if meta['commit']!=expected_commit or meta['functional_sha256']!=expected_identity:
 raise ValueError('Staged release differs')
loader=importlib.machinery.SourceFileLoader('exclusive_campaign',str(source/'isaaclab/deploy/run-mkii-fourbar-campaign'))
spec=importlib.util.spec_from_loader(loader.name,loader)
campaign=importlib.util.module_from_spec(spec);loader.exec_module(campaign)
contract=campaign.identity(source)
if contract['sha256']!=expected_identity:raise ValueError('Frozen source changed')
preflight=campaign.resource_ready()
reservation=base/'priority_20260906T020649Z'
guard=json.loads((reservation/'status.json').read_text())
args=Path(f"/proc/{guard['pid']}/cmdline").read_bytes().split(b'\0')
if guard['state']!='reserved' or guard['pid']!=1711343 or str(reservation).encode() not in args:
 raise ValueError('Expected exclusive reservation is not held')
if (reservation/'campaign_path.txt').exists():raise ValueError('Reservation already has a campaign; inspect instead of duplicating')
now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
shared=campaign.host.COORDINATION
before=shared.read_bytes()
campaign.host.require_coordination_none(campaign.host.read_coordination_control())
text=before.decode()+f'\n## Exclusive hexapod campaign restart — {now}\n\nFrozen source1239159 at {source}; 906 CPU tests and independent coordination review passed. User explicitly grants full Spark takeover. New campaign: fresh1x100 probe, full32x1000standing+2400driven nominal/refined, scratch64x3 and separate full512x1000 resume only after all admission/checkpoint gates pass. The prior interrupted run is preserved and supplies no full admission. State: {base}/campaigns/fourbar-campaign-*/campaign.json. Guard1711343 remains exclusive, maximum12:06:50UTC or earlier campaign termination/release. New canonical_share_status_v2 supervisor records prose-note changes without pausing; explicit or invalid control still yields. No other GPU work is authorized alongside this campaign.\n'
temporary=shared.with_name(f'.hexapod-campaign-note-{now}.tmp')
with temporary.open('x') as stream:stream.write(text)
os.chmod(temporary,shared.stat().st_mode & 0o777)
if shared.read_bytes()!=before:
 temporary.unlink();raise ValueError('Coordination changed before publication')
os.replace(temporary,shared)
campaign.host.require_coordination_none(campaign.host.read_coordination_control())
argv=['/usr/bin/python3',str(source/'isaaclab/deploy/run-mkii-fourbar-campaign'),
 '--source-dir',str(source),'--source-commit',expected_commit,'--asset-model','mkii_fourbar_v5',
 '--output-root',str(base/'campaigns'),'--wait-seconds','0',
 '--phase-timeout-seconds','7200','--full-timeout-seconds','21600']
with (base/f'campaign_launch_{now}.log').open('w') as log:
 proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'utc':now,'pid':proc.pid,'argv':argv,'source_commit':expected_commit,'source_contract_sha256':expected_identity,
 'preflight':preflight,'guard':guard,'reservation':str(reservation),
 'shared_before_sha256':hashlib.sha256(before).hexdigest(),
 'shared_after_sha256':hashlib.sha256(shared.read_bytes()).hexdigest(),
 'bootstrap_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
path=base/f'campaign_launch_{now}.json';path.write_text(json.dumps(record,indent=2)+'\n')
for _ in range(150):
 matches=[p for p in (base/'campaigns').glob('fourbar-campaign-*/campaign.json') if json.loads(p.read_text()).get('pid')==proc.pid]
 if len(matches)==1:
  record['campaign']=str(matches[0]);(reservation/'campaign_path.txt').write_text(str(matches[0])+'\n');break
 if proc.poll() is not None:break
 time.sleep(.1)
if 'campaign' not in record:record['campaign_selection_pending']=True
path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
