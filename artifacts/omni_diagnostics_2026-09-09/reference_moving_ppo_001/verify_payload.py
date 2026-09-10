"""Portable stdlib integrity/result verifier; no Isaac, Torch, SSH or services."""
from pathlib import Path
import hashlib,json,sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def tree(root,manifest,bound=None):
 f=root/manifest
 if bound is not None and sha(f)!=bound:raise ValueError('Manifest mismatch')
 m=read(f);m=m.get('files',m)
 if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('Symlink substitution')
 actual={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p!=f}
 if actual!=m:raise ValueError('Changed, missing or extra payloads: '+str(root))
 return len(m)
def main():
 total=tree(H,'BUNDLE_SHA256.json');recon=read(H/'RECONSTRUCTION.json')
 for name,s in recon['copied_frozen_bundles'].items():assert tree(H/name,s['manifest'],s['sha256'])==s['payload_count']
 for name,s in recon['referenced_maps'].items():assert sha(H/'references'/name)==s['sha256'] and len(read(H/'references'/name))==s['payload_count']
 audit=read(H/'independent_review/remote_audit.json');storage=read(H/'RAW_STORAGE.json');assert set(storage)==set(audit['raw_payloads']) and len(storage)==56
 sys.path.insert(0,str(H/'chunk_tool'));from chunk_artifact import consume
 chunks=0
 for logical,s in storage.items():
  assert not Path(logical).is_absolute() and '..' not in Path(logical).parts and s['sha256']==audit['raw_payloads'][logical] and s['bytes']==audit['raw_sizes_bytes'][logical]
  if s['kind']=='file':
   p=H/s['path'];assert p.stat().st_size==s['bytes'] and sha(p)==s['sha256'] and s['bytes']<=100*1024*1024
  else:
   assert s['kind']=='lossless_chunks' and s['bytes']>100*1024*1024;manifest=H/s['manifest'];assert sha(manifest)==s['manifest_sha256'];m=consume(manifest.parent)
   assert m['logical_path']==logical and m['original_bytes']==s['bytes'] and m['original_sha256']==s['sha256'] and all(c['bytes']<=64*1024*1024 for c in m['chunks']);chunks+=1
 assert chunks==2
 def raw(name):
  s=storage[name];assert s['kind']=='file';return read(H/s['path'])
 campaign=raw('run/campaign.json');assert campaign['status']=='failed' and campaign['PPO_updates_completed']==0 and campaign['Stage2_complete'] is False and campaign['source_and_inputs_unchanged']
 assert list(campaign['accepted_phases'])==['standing','calibration'] and campaign['last_completed_phase']=='calibration'
 for key,count in [('source',926),('consumer',35),('host',3),('guard',2),('bridge',18),('observation',160)]:assert audit['input_maps'][key]['files']==count and audit['input_maps'][key]['unchanged']
 assert audit['assets_files']==550 and audit['assets_unchanged'] and len(raw('run/inputs/study_before.sha256.json'))==550
 assert set(audit['jobs'])=={'standing','calibration','profile_32'} and len(audit['owned_names_and_ids_absent'])==6
 for job in audit['jobs'].values():
  assert job['cleanup_checked']
  for token in (job['container_name'],job['container_id']):
   result=audit['owned_names_and_ids_absent'][token];assert result['returncode']==1 and any(x in result['stderr'].lower() for x in ('no such object','no such container'))
 restored=raw('forecast_pause/restored.json');assert restored==audit['pause_restoration'] and restored['restored_unix']==1789046082.2900512
 assert 'InvocationID=53fa4c8afc7940e0931a77c18d310557' in audit['unit'] and 'ExecMainStatus=1' in audit['unit'] and 'ActiveState=failed' in audit['unit']
 for phase in ('standing','calibration'):
  accepted=raw('run/'+phase+'_accepted.json');assert accepted==campaign['accepted_phases'][phase]
  mapping={name[len('run/'+phase+'/'):]:s['sha256'] for name,s in storage.items() if name.startswith('run/'+phase+'/')}
  assert mapping==raw('run/'+phase+'_immutable.sha256.json')
  if phase=='standing':assert accepted['admission_sha256']==mapping['state.json']==mapping['admission.json']
  else:assert accepted['files_sha256']==mapping and accepted['passed'] and accepted['mode']=='calibrate' and accepted['phase']=='calibration'
 profile=raw('run/profile_32/state.json');assert profile['status']=='rejected' and profile['session']['controls']==224 and profile['PPO_updates_completed']==0 and not profile['policy_training_started']
 assert 'Inplace update to inference tensor' in profile['error'] and 'reset_buf.copy_' in profile['traceback']
 initial=storage['run/calibration/initial.pt']['sha256'];meta=raw('run/calibration/initial.pt.json');assert meta['checkpoint_sha256']==initial and profile['input_checkpoint_sha256']==initial and raw('run/calibration/state.json')['initial_checkpoint']==meta
 r=read(H/'independent_review/REVIEW.json');assert r['raw_payloads_verified']==56 and r['PPO_updates_completed']==0 and r['standing']['passed_all32'] and r['calibration']['passed']
 assert r['profile']['retained_contacts_excluding_LF']==4 and r['profile']['final_RR_force_N']<1 and r['profile']['attempted_reset_receipt_incomplete'] and r['profile']['complete_pre_event223_control_audit']['completed_controls']==223
 assert read(H/'independent_review/full_profile_primitive_review.json')['integrity_passed'] is False
 print(json.dumps({'bundle_payloads_verified':total,'raw_original_files_verified':56,'losslessly_chunked_originals':chunks,'standing_and_calibrations_pass':True,'profile_controls':224,'profile_rejected_and_reset_incomplete':True,'PPO_updates':0,'all3_owned_names_and_IDs_absent':True,'pause049_restored_unix':restored['restored_unix'],'source_and_inputs_unchanged':True,'physical_or_Stage2_admission':False,'read_only':True},indent=2))
if __name__=='__main__':main()
