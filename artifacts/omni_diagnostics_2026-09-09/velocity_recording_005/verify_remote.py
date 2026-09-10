from pathlib import Path
import json,hashlib,subprocess,time
base=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tree(p):return {str(x.relative_to(p)):sha(x) for x in sorted(p.rglob('*')) if x.is_file()}
src=base/'omni_velocity_source_003';source_map=json.loads((src/'campaign_source_hashes.json').read_text());actual=tree(src);actual.pop('campaign_source_hashes.json');assert actual==source_map
pilot=base/'omni_velocity_pilot_003';assert tree(pilot/'inputs')==json.loads((pilot/'inputs_before.sha256.json').read_text())
assert sha(pilot/'train/policy/final.pt')=='88f8e60a7b1536229f51524cdc646f06d53e4f33cabb74aa89ebde73d06f7247'
report={'verified_unix':time.time(),'source_files':len(source_map),'source_unchanged':True,'source_manifest_sha256':sha(src/'campaign_source_hashes.json'),'pilot_inputs_unchanged':True,'checkpoint_sha256':sha(pilot/'train/policy/final.pt'),'attempts':{}}
for idx,pause in [(4,23),(5,24),(6,25)]:
 out=base/f'omni_velocity_recording_{idx:03d}';pa=base/f'forecast_pause_{pause:03d}'
 c=json.loads((out/'campaign.json').read_text());assert c['status']=='failed'
 j=json.loads((out/'jobs/recording.json').read_text());identity=j['container_id'];name=j['container_name']
 checks={}
 for target in (identity,name):
  r=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',target],capture_output=True,text=True,timeout=15)
  assert r.returncode!=0 and 'no such' in r.stderr.lower(),(target,r.stdout,r.stderr)
  checks[target]={'returncode':r.returncode,'stderr':r.stderr.strip()}
 assert (pa/'restored.json').is_file()
 hashes={**{'run/'+k:v for k,v in tree(out).items()},**{'forecast_pause/'+k:v for k,v in tree(pa).items()}}
 report['attempts'][str(idx)]={'raw_payloads':hashes,'owned_containers_absent':checks,'unit':subprocess.check_output(['systemctl','--user','show',f'hexapod-omni-velocity-recording-{idx:03d}-20260910.service','-p','ActiveState','-p','SubState','-p','Result'],text=True),'pause_restoration':json.loads((pa/'restored.json').read_text())}
print(json.dumps(report,indent=2))
