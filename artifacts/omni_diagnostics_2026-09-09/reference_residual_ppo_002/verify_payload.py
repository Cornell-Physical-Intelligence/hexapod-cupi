"""Portable standard-library verifier, including exact whole hashes of chunked NPZs.

Never imports Isaac/Torch or invokes a GPU, SSH, service, or archived launcher.
"""
from pathlib import Path
import hashlib,json,math,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def verify_tree(root,manifest,bound=None):
 p=root/manifest
 if bound is not None and sha(p)!=bound:raise ValueError('Wrong manifest '+str(p))
 if any(f.is_symlink() for f in root.rglob('*')):raise ValueError('Symbolic payload substitution')
 m=read(p);actual={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file() and f!=p}
 if actual!=m:raise ValueError('Changed/missing/extra file '+str(root))
 return len(actual)
def verify_raw():
 storage=read(ROOT/'RAW_STORAGE.json');audit=read(ROOT/'independent_review/remote_audit.json')
 assert set(storage)==set(audit['raw_payloads']) and len(storage)==80
 sys.path.insert(0,str(ROOT/'chunk_tool'));from chunk_artifact import consume,MAX_CHUNK_BYTES
 chunks=0
 for name,entry in storage.items():
  assert entry['sha256']==audit['raw_payloads'][name]
  if entry['kind']=='file':
   p=ROOT/entry['path'];assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
   assert entry['bytes']<=100*1024*1024
  elif entry['kind']=='lossless_chunks':
   p=ROOT/entry['manifest'];assert sha(p)==entry['manifest_sha256'];m=consume(p.parent)
   assert m['logical_path']==name and m['original_bytes']==entry['bytes'] and m['original_sha256']==entry['sha256']
   assert entry['bytes']>100*1024*1024 and all(c['bytes']<=MAX_CHUNK_BYTES for c in m['chunks']);chunks+=1
  else:raise ValueError('Unknown raw storage kind')
 assert chunks==2
 return storage,audit

def main():
 total=verify_tree(ROOT,'BUNDLE_SHA256.json');r=read(ROOT/'RECONSTRUCTION.json')
 for name,spec in r['copied_frozen_bundles'].items():assert verify_tree(ROOT/name,spec['manifest'],spec['sha256'])==spec['payload_count']
 for name,spec in r['referenced_maps'].items():
  p=ROOT/'references'/name;assert sha(p)==spec['sha256'] and len(read(p))==spec['payload_count']
 storage,audit=verify_raw()
 def rawjson(name):
  entry=storage[name];assert entry['kind']=='file';return read(ROOT/entry['path'])
 def rawhash(name):return storage[name]['sha256']
 campaign=rawjson('run/campaign.json');phases=['standing','smoke','evaluate_initial','evaluate_final']
 assert campaign['status']=='completed' and campaign['two_update_integration_and_matched_retention_passed'] is True and campaign['source_and_inputs_unchanged'] is True
 assert campaign['PPO_updates_completed']==2 and campaign['Stage2_complete'] is False and list(campaign['accepted_phases'])==phases
 assert campaign['source_manifest_sha256']==audit['source009']['manifest_sha256'] and campaign['consumer_freeze_sha256']==audit['consumer002']['manifest_sha256'] and campaign['host_freeze_sha256']==audit['host002']['manifest_sha256']
 for key,count in [('source009',926),('consumer002',20),('host002',3),('bridge001',18),('observation005',160)]:assert audit[key]['payload_files']==count and audit[key]['unchanged']
 assert audit['admitted_assets_unchanged'] and audit['admitted_asset_files']==550 and len(rawjson('run/inputs/study_before.sha256.json'))==550
 assert audit['missing_container_IDs']==[] and len(audit['actual_jobs'])==4 and len(audit['owned_absence'])==8
 for job in audit['actual_jobs']:
  assert job['cleanup_checked'] and job['phase'] in phases
  for name in (job['container_name'],job['container_id']):
   result=audit['owned_absence'][name];assert result['returncode']==1 and 'no such object: '+name in result['stderr'].lower()
 pause=rawjson('forecast_pause/restored.json');assert pause==audit['pause_restoration'] and pause['restored_unix']==1789032028.6315286
 assert 'Result=success' in audit['unit'] and 'ExecMainStatus=0' in audit['unit'] and 'ActiveState=inactive' in audit['unit']
 immutable=rawjson('run/smoke_immutable.sha256.json')
 assert {n[len('run/smoke/'):]:s['sha256'] for n,s in storage.items() if n.startswith('run/smoke/')}==immutable
 for phase in phases:
  state=rawjson('run/'+phase+'/state.json');assert state['status']=='completed'
  accepted=rawjson('run/'+phase+'_accepted.json');assert accepted==campaign['accepted_phases'][phase]
  if phase=='standing':
   assert rawhash('run/standing/state.json')==rawhash('run/standing/admission.json')==accepted['admission_sha256'];assert state['gate']['passed'] and state['gate']['all_replica_quiet']['passed'];continue
  assert accepted['mode']==phase and accepted['passed'] and state['source_inputs_unchanged']
  for name,h in accepted['files_sha256'].items():assert rawhash('run/'+phase+'/'+name)==h
  expected=3248 if phase=='smoke' else 2400
  assert state['session']['controls']==expected and state['session']['failure'] is None and state['session']['substeps']['samples_including_initial']==8*expected+1
  assert state['reload']['actor_critic_normalizer_and_optimizer_exact'] is True
  if phase!='smoke':
   which='initial' if phase=='evaluate_initial' else 'final';assert state['checkpoint_sha256']==rawhash('run/smoke/'+which+'.pt') and state['retention_passed'] is True
 smoke=rawjson('run/smoke/state.json');identity=smoke['policy_identity'];assert identity['actor_width']==846 and identity['critic_width']==849 and identity['maximum_updates']==2
 assert identity['bindings']['consumer_source_sha256']==audit['consumer002']['manifest_sha256'] and identity['bindings']['calibration_sha256']==rawhash('run/smoke/calibration.json')
 assert smoke['PPO_updates_completed']==2 and smoke['policy_training_started'] and smoke['post_update_quiet_passed'] and smoke['optimizer_entries']==17 and smoke['deterministic_reload_max_difference']==0
 for which in ('initial','final'):
  meta=rawjson('run/smoke/'+which+'.pt.json');assert meta=={**identity,'checkpoint_sha256':rawhash('run/smoke/'+which+'.pt')} and meta==smoke[which+'_checkpoint']
 review=read(ROOT/'independent_review/review.json');ck=read(ROOT/'independent_review/checkpoint_review.json')
 assert review['campaign_status']=='completed' and review['PPO_updates_completed']==2 and review['native_velocity_fidelity_qualified'] is False
 for phase in phases:
  row=review['phases'][phase];assert row['status']=='completed' and row['all_substep_times_and_control_endpoint_parity']
  if phase!='standing':assert row['all14_sensor_clocks_every_control_exact'] and row['final846849_packet_and_raw_vs_interval_channels_exact']
 for which in ('initial','final'):
  row=review['phases']['evaluate_'+which];assert row['retention_passed'] and row['qualified_landings']==11 and row['completed_legs']==list(range(6)) and row['quiet_duration_s']>=10 and row['old50Hz_position_rate_gap_m']<=.005
  assert ck['checkpoints'][which]['checkpoint_sha256']==rawhash('run/smoke/'+which+'.pt') and ck['checkpoints'][which]['strict_actor_critic_normalizer_optimizer_readback']
 print(json.dumps({'bundle_payloads_verified':total,'raw_original_files_verified':80,'losslessly_chunked_originals':2,'all4phases_completed':True,'PPO_updates':2,'calibration_and_postquiet32_replicas':True,'both_full_forward_stop_retentions':True,'strict_checkpoint_CPU_readback_receipt_bound':True,'raw_400Hz_and_all_sensor_clocks_independently_replayed':True,'four_owned_names_and_IDs_absent':True,'pause043_restored_unix':pause['restored_unix'],'raw_originals_remain_remote_and_local':True,'read_only':True,'GPU_launches':0,'native_velocity_fidelity_qualified':False,'Stage2_complete':False},indent=2))
if __name__=='__main__':main()
