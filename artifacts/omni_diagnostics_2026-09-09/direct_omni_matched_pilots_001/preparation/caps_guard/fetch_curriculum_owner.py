"""Read-only exact terminal receipt capture; no training-quality requirement."""
from pathlib import Path
import base64,hashlib,json,subprocess,time
BASE=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909');root=BASE/'direct_omni_train_pilot_curriculum_001'
unit='hexapod-direct-omni-train-pilot-curriculum-001-20260910.service';inv='d9ffdd9882ec48dc8417d36a4245b2c4'
phases=('standing','initial_constant','initial_stop','train','final_constant','final_stop')
fields=subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','-p','SubState','-p','InvocationID','-p','Result'],text=True);status=dict(x.split('=',1) for x in fields.splitlines())
assert status['ActiveState'] in ['inactive','failed'] and status.get('InvocationID') in ['',inv]
if not status.get('InvocationID'):
 journal=[json.loads(x) for x in subprocess.check_output(['journalctl','--user','-u',unit,'--no-pager','-o','json'],text=True).splitlines() if x.strip()]
 rows=[x for x in journal if x.get('USER_UNIT')==unit]
 assert {x.get('USER_INVOCATION_ID') for x in rows}=={inv}
 assert any(x.get('MESSAGE','').startswith('Started '+unit) for x in rows)
else:rows=[]
actual={p.stem for p in (root/'jobs').glob('*.json') if not p.name.endswith('_contact_data_audit.json')};attempted=[x for x in phases if x in actual];assert set(attempted)==actual and tuple(attempted)==phases[:len(attempted)] and attempted
names=['direct_omni_train_pilot_curriculum_001/campaign.json',*[f'direct_omni_train_pilot_curriculum_001/jobs/{p}.json' for p in attempted],'forecast_pause_056/pause.json','forecast_pause_056/restored.json']
files={n:(BASE/n).read_bytes() for n in names};campaign=json.loads(files[names[0]]);assert campaign['status'] in ['completed','failed'] and campaign['terminal_inputs_unchanged'] is True
assert (campaign['allocation'],campaign['branch'])==('pilot','curriculum')
absences={}
for name in names[1:1+len(attempted)]:
 j=json.loads(files[name]);assert j['status'] in ['completed','failed'] and j['cleanup_checked'] is True
 for token in [j['container_name'],j['container_id']]:
  p=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',token],text=True,capture_output=True,timeout=20)
  assert p.returncode and any(x in p.stderr.lower() for x in ['no such object','no such container'])
  absences[token]={'exit_code':p.returncode,'stderr':p.stderr.strip()}
assert all((BASE/n).read_bytes()==v for n,v in files.items())
print(json.dumps({'checked_unix':time.time(),'read_only':True,'unit':status,'expected_invocation':inv,'historical_invocation_journal':rows,'attempted_phases':attempted,'absences':absences,'quality_gate_applied':False,'files':{n:{'sha256':hashlib.sha256(v).hexdigest(),'base64':base64.b64encode(v).decode()} for n,v in files.items()}},indent=2))
