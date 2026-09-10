"""Compact read-only status; autosave presence is not admitted completed updates."""
from pathlib import Path
import json,re,subprocess,time
R=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909/direct_omni_train_extended_caps_002')
UNIT='hexapod-direct-omni-train-extended-caps-002-20260910.service'
INV='ca9455cefb554c7691e7f8b2ed3ac939'
q=subprocess.run(['systemctl','--user','show',UNIT,'-p','ActiveState','-p','SubState','-p','InvocationID','-p','MainPID','-p','ExecMainStatus'],capture_output=True,text=True,timeout=20)
fields=dict(x.split('=',1) for x in q.stdout.splitlines() if '=' in x)
out={'observed_unix':time.time(),'read_only':True,'expected_invocation':INV,'owner_returncode':q.returncode,'owner':fields,'owner_stderr':q.stderr,'jobs':{}}
if (R/'campaign.json').is_file():
 c=json.loads((R/'campaign.json').read_text());out['campaign']={k:c.get(k) for k in ['status','last_completed_phase','PPO_updates_completed','error','observed_training','terminal_inputs_unchanged']}
for phase in ('standing','initial_constant','initial_stop','train','final_constant','final_stop'):
 p=R/'jobs'/(phase+'.json')
 if p.is_file():
  j=json.loads(p.read_text());out['jobs'][phase]={k:j.get(k) for k in ('status','error','exit_code')}
p=R/'logs/train.log'
if p.is_file():
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-16000));tail=f.read().decode(errors='replace')
 lines=re.sub(r'\x1b\[[0-9;]*m','',tail).splitlines()
 out['train_markers']=[s.strip() for s in lines if any(k in s for k in ('Learning iteration','Total timesteps','Computation:','Collection time','Learning time','Mean total reward','Traceback','Error','RuntimeError','APP_READY'))][-14:]
models=[int(x.stem.removeprefix('model_')) for x in (R/'train/policy').glob('model_*.pt')]
out['autosave_presence_only']={'files':len(models),'highest_native_iteration':max(models) if models else None,'not_completed_receipt':True}
print(json.dumps(out,indent=2))
