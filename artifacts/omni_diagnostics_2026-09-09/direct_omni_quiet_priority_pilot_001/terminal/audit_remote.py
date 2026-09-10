from pathlib import Path
import subprocess,json,hashlib,time
B=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909');R=B/'direct_omni_train_pilot_quiet_priority_001';P=B/'forecast_pause_060'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def call(a):
 q=subprocess.run(a,capture_output=True,text=True,timeout=20);return {'exit_code':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
owner=call(['systemctl','--user','show','hexapod-direct-omni-train-pilot-quiet-priority-001-20260910.service','-p','ActiveState','-p','SubState','-p','InvocationID','-p','MainPID','-p','ExecMainStatus'])
f=dict(x.split('=',1) for x in owner['stdout'].splitlines() if '=' in x);assert f['ActiveState']=='inactive' and f['ExecMainStatus']=='0';assert f.get('InvocationID','') in ('','721f1886478a4e518c7e0b741256413f')
phases=('standing','initial_constant','initial_stop','train','final_constant','final_stop');c=json.loads((R/'campaign.json').read_text());assert c['status']=='completed' and c['PPO_updates_completed']==50 and c['bounded_campaign_complete'] is True and c['terminal_inputs_unchanged'] is True and c['Stage2_complete'] is False and set(c['accepted_phases'])==set(phases)
cleanup=[]
for phase in phases:
 j=json.loads((R/'jobs'/f'{phase}.json').read_text());assert j['status']=='completed' and j['exit_code']==0 and j['cleanup_checked'] is True
 for key in ('container_id','container_name'):
  q=call(['docker','inspect',j[key]]);assert q['exit_code']!=0 and any(t in q['stderr'].lower() for t in ('no such object','no such container'));cleanup.append({'phase':phase,'identifier':j[key],'absent':True})
restored=json.loads((P/'restored.json').read_text());assert restored['restored_unix']>0
inv={};excluded=[];total=0;selectedbytes=0
for label,root in [('run',R),('pause',P)]:
 assert not root.is_symlink() and not any(p.is_symlink() for p in root.rglob('*'))
 for p in sorted(root.rglob('*')):
  if p.is_file():
   k=label+'/'+str(p.relative_to(root));v={'sha256':sha(p),'bytes':p.stat().st_size};inv[k]=v;total+=v['bytes']
   if p.parent.name=='policy' and p.name.startswith('model_') and p.suffix=='.pt':excluded.append(k)
   else:selectedbytes+=v['bytes']
pins={str(p.relative_to(B)):sha(p) for p in [R/'campaign.json',*(R/'jobs'/f'{x}.json' for x in phases),P/'pause.json',P/'restored.json']}
print(json.dumps({'observed_unix':time.time(),'owner':owner,'campaign':c,'cleanup':cleanup,'restored':restored,'pins':pins,'inventory':inv,'files':len(inv),'bytes':total,'remote_only_intermediate_autosaves':excluded,'selected_files':len(inv)-len(excluded),'selected_bytes':selectedbytes,'gpu':call(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])}))
