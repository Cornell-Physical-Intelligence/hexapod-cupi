"""Offline selected-receipt verification; no SSH, GPU or native execution."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
result=read(R/'RESULTS.json')
assert result['standing_qualified'] is False and result['training_allowed'] is False
checked=0
for arm,gate,events,missing in [('origin',True,0,0),('xy14_4',False,17,5),('xy14_4_repeat',False,17,5)]:
 p=R/('results_'+arm+'_001');inventory=read(p/'FULL_REMOTE_INVENTORY.json')
 for relative in inventory['selected']:
  f=p/'receipts'/relative;assert f.is_file() and sha(f)==inventory['files'][relative]['sha256'];checked+=1
 s=read(p/'receipts/standing/state.json');identity=s['identity']
 assert identity['urdf_sha256']==result['urdf_sha256']
 assert identity['inspector_freeze_sha256']==result['diagnostic_source_sha256']
 assert s['status']=='completed' and s['inputs_unchanged'] is True and s['explicit_steps_completed']==8000
 assert all(identity[k] is False for k in ('standing_admission','batch_admission','stage2_complete','training_allowed','physics_admitted','physical_admission'))
 session=read(p/'receipts/standing/session.json');assert session['steps']==8000 and session['controls']==1000 and session['all_rows_recorded'] is True
 for name in s['outputs']:
  if 'standing/'+name in inventory['files']:assert s['outputs'][name]==inventory['files']['standing/'+name]['sha256']
 report=read(p/'receipts/standing/standing_report.json');assert report['all_pass'] is gate
 assert report['replicas'][0]['physical']['post_settle_missing_six_toe_substeps']==missing
 assert report['replicas'][0]['quiet']['pass'] is gate
 analysis=read(p/'ANALYSIS.stdout.json');assert len(analysis['events'])==events and analysis['steps']==8000
 assert sum(e['post_settle'] for e in analysis['events'])==missing
 for name,digest in analysis['consumed_inputs'].items():assert digest==inventory['files']['standing/'+name]['sha256']
 assert all(e['exact128_inactive_zero_tuples'] is True for e in analysis['events'])
 guard=read(p/'receipts/guard.json');assert guard['terminal_inputs_unchanged'] is True
 assert guard['status']==('completed' if gate else 'failed') and guard['owned_cleanup']['cleanup_checked'] is True
 assert guard['terminal_resources']=={'cuda_processes':'','active_containers':''}
 assert guard['terminal_reservation']['reservation_released'] is False
 assert read(p/'receipts/guard_cleanup.json')['cleanup_checked'] is True
for name in ('TRANSLATED_REPEAT_TRACE_COMPARISON.json','ORIGIN_PRIOR_RECORDED_TRACE_COMPARISON.json'):
 data=read(R/name);data=data.get('comparisons',data)
 assert len(data)==12 and all(v['matches'] for v in data.values())
 for value in data.values():
  fields=[v for k,v in value.items() if k.endswith('sha256')]
  assert len(fields)==2 and fields[0]==fields[1] and len(fields[0])==64
first=read(R/'results_xy14_4_001/FULL_REMOTE_INVENTORY.json')['files']
repeat=read(R/'results_xy14_4_repeat_001/FULL_REMOTE_INVENTORY.json')['files']
for name,value in read(R/'TRANSLATED_REPEAT_TRACE_COMPARISON.json').items():
 assert first[name]['sha256']==value['first_sha256']==repeat[name]['sha256']==value['repeat_sha256']
origin=read(R/'results_origin_001/FULL_REMOTE_INVENTORY.json')['files']
for name,value in read(R/'ORIGIN_PRIOR_RECORDED_TRACE_COMPARISON.json')['comparisons'].items():
 assert origin['standing/'+name]['sha256']==value['new_sha256']==value['prior_recorded_sha256']
print(json.dumps({'selected_receipts_verified':checked,'completed_native_runs':3,'gates':[True,False,False],'repeat_trace_hashes_verified':24,'training_allowed':False,'scope':'Saved selected bytes and recorded remote hashes only; not live telemetry'},indent=2))
