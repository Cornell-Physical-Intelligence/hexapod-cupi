"""Collect selected receipts after exact owned cleanup; raw data stays on Spark."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys,tarfile
root=Path('/home/orionh/HEXAPOD_runs/restart_20260914/standing_translation_001')
arm=sys.argv[1];assert arm in ('origin','xy14_4')
output=root/(arm+'_001');phase=output/'standing';unit='hexapod-placement-'+arm.replace('_','-')+'-001-20260914.service'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
unit_state=subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus','-p','InvocationID'],text=True)
fields=dict(l.split('=',1) for l in unit_state.splitlines() if '=' in l)
assert fields['MainPID']=='0' and fields['ActiveState'] in ('inactive','failed'), fields
assert not subprocess.check_output(['docker','ps','-q'],text=True).strip()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
guard=json.loads((output/'guard.json').read_text())
assert guard.get('terminal_inputs_unchanged') is True and guard.get('owned_cleanup',{}).get('cleanup_checked') is True, guard
assert guard.get('terminal_resources')=={'cuda_processes':'','active_containers':''}
assert json.loads((output/'guard_cleanup.json').read_text()).get('cleanup_checked') is True
result=root/('results_'+arm+'_001');result.mkdir(exist_ok=False)
state=json.loads((phase/'state.json').read_text())
assert state['status']=='completed' and state['explicit_steps_completed']==8000 and state['inputs_unchanged'] is True
analysis=root/'analysis';analysis_hashes={p.relative_to(analysis).as_posix():sha(p) for p in analysis.rglob('*') if p.is_file()}
p=subprocess.run(['/usr/bin/python3','-B','-S',str(analysis/'readout.py'),'--phase',str(phase)],text=True,capture_output=True,timeout=120)
(result/'ANALYSIS.stdout.json').write_text(p.stdout);(result/'ANALYSIS.stderr.txt').write_text(p.stderr)
assert p.returncode==0, p.stderr
assert analysis_hashes=={p.relative_to(analysis).as_posix():sha(p) for p in analysis.rglob('*') if p.is_file()}
(result/'ANALYSIS_SOURCE_SHA256.json').write_text(json.dumps(analysis_hashes,indent=2)+'\n')
full={p.relative_to(output).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in output.rglob('*') if p.is_file()}
selected=[]
for rel in sorted(full):
 # Structured receipts only; contacts, mesh snapshots, native logs and raw NPZ remain remote.
 if rel.endswith('.json') and (len(Path(rel).parts)==1 or (len(Path(rel).parts)==2 and Path(rel).parts[0] in ('standing','jobs'))):
  destination=result/'receipts'/rel;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(output/rel,destination);selected.append(rel)
(result/'FULL_REMOTE_INVENTORY.json').write_text(json.dumps({'remote_output':str(output),'files':full,'selected':selected,'raw_data_deleted':False},indent=2)+'\n')
readout=json.loads(p.stdout)
report=readout['original_gate_report'];replica=report['replicas'][0]
record={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unit_state':fields,'guard_status':guard['status'],'native_status':state['status'],'standing_gate_pass':report['all_pass'],'raw_steps':8000,'physical':replica['physical'],'failed_physical_bounds':replica['failed_physical_bounds'],'quiet':replica['quiet'],'per_toe':readout['per_toe'],'events':readout['events'],'contact_slots':readout['contact_slots'],'training_allowed':False,'standing_admission':False,'batch_admission':False,'stage2_complete':False,'gpu_after':[],'owned_containers_after':[],'raw_output_preserved':str(output)}
(result/'SUMMARY.json').write_text(json.dumps(record,indent=2)+'\n')
archive=root/(result.name+'.tar.gz')
assert not archive.exists()
with tarfile.open(archive,'w:gz') as tf:tf.add(result,arcname=result.name)
print(json.dumps({'summary':record,'archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size},indent=2))
