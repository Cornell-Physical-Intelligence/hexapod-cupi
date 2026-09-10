"""Read-only terminal binding fetch; emits bytes/hashes and exact absence evidence."""
from pathlib import Path
import base64,hashlib,json,subprocess,time
BASE=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
names=['direct_omni_train_smoke_002/campaign.json',*[f'direct_omni_train_smoke_002/jobs/{p}.json' for p in ['standing','train','final_constant','final_stop']],'forecast_pause_055/pause.json','forecast_pause_055/restored.json']
files={n:(BASE/n).read_bytes() for n in names}
c=json.loads(files[names[0]]);assert c['status']=='completed' and c['terminal_inputs_unchanged'] is True
unit='hexapod-direct-omni-train-smoke-002-20260910.service';inv='6cd738f30a0b4f499026de5cbf8a5bb6'
status=subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','-p','SubState','-p','InvocationID','-p','Result'],text=True)
x=dict(line.split('=',1) for line in status.splitlines());assert x['ActiveState'] in ['inactive','failed'];assert x.get('InvocationID') in ['',inv]
absent={}
for name in names[1:5]:
 j=json.loads(files[name]);assert j['status']=='completed' and j['cleanup_checked'] is True
 for token in [j['container_name'],j['container_id']]:
  p=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',token],text=True,capture_output=True,timeout=20)
  assert p.returncode and any(s in p.stderr.lower() for s in ['no such object','no such container'])
  absent[token]={'exit_code':p.returncode,'stderr':p.stderr.strip()}
assert all((BASE/n).read_bytes()==v for n,v in files.items())
print(json.dumps({'checked_unix':time.time(),'unit':x,'expected_invocation':inv,'container_absence':absent,'files':{n:{'sha256':hashlib.sha256(v).hexdigest(),'base64':base64.b64encode(v).decode()} for n,v in files.items()}},indent=2))
