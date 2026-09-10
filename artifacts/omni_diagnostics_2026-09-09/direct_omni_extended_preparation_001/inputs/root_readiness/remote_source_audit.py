from pathlib import Path
import hashlib,json
B=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(name,manifest,pin):
 p=B/name;m=p/manifest;assert sha(m)==pin
 assert not p.is_symlink() and not any(x.is_symlink() for x in p.rglob('*'))
 actual={str(x.relative_to(p)):sha(x) for x in p.rglob('*') if x.is_file() and x!=m}; expected=json.loads(m.read_text()); assert actual==expected
 return {'directory':str(p),'manifest_sha256':pin,'files':len(actual),'inventory':expected}
source=verify('direct_omni_train_source_004','campaign_source_hashes.json','aa70659d33243f1e28e9c71b1a7f9d33341642db770fd7356192573a26fe904e')
native=verify('direct_omni_train_preparation_004','FREEZE_SHA256.json','0bdd2abf3811e68ac59e4fbc325eb5d2bb96afd2183eb84c4d37faeb86ff9e9f')
host=verify('direct_omni_train_host_004','FREEZE_SHA256.json','b55282c968d64dca218baf10278339ff2013c23e28dc7cbe9a0d0b8dfd05d51f')
guard=verify('direct_omni_train_smoke_guard_005','FREEZE_SHA256.json','16c70dafa3168e1e0de3b4619512a6bb27ea64711e3aa20deb4d819a75eebae7')
cold=verify('direct_omni_cold_source_001','campaign_source_hashes.json','4671eab012aaf79ce49769cfdc1667142237a4b4a45966fd0399a9caa900c51b')
p=B/'direct_omni_train_source_004'; plan=p/'robot/hexapod_mkii_length_study/training_plan.json';origin=p/'source_origin.json'
assert sha(plan)=='b2286828148c89fb34624d38beaf84b389584f7bda4d2648512e017da4b50acb';assert sha(origin)=='850aa811e702dcfd0284862c0c27f475e82a5335aab06b96e0ea21ea51e6f8b7'
orig=json.loads(origin.read_text());overlay=orig['runtime_overlays']
added=set(source['inventory'])-set(cold['inventory']);changed={x for x in cold['inventory'] if source['inventory'].get(x)!=cold['inventory'][x]}
assert added=={'tools/'+x for x in overlay};assert changed=={'tools/train_length_study.py','robot/hexapod_mkii_length_study/training_plan.json','source_origin.json'}
assert all(source['inventory']['tools/'+n]==h==native['inventory'][n] for n,h in overlay.items())
print(json.dumps({'source':source,'native':native,'host':host,'guard':guard,'cold_parent_sha256':cold['manifest_sha256'],'added':sorted(added),'changed':sorted(changed),'plan':json.loads(plan.read_text()),'origin':orig,'source_build_verified':{'source_manifest_sha256':source['manifest_sha256'],'files':source['files'],'plan_sha256':sha(plan),'origin_sha256':sha(origin)}}))
