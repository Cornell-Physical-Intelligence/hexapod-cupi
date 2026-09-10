from pathlib import Path
import hashlib,json
B=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(name,manifest,pin):
 p=B/name;m=p/manifest;assert sha(m)==pin
 assert not p.is_symlink() and not any(x.is_symlink() for x in p.rglob('*'))
 actual={str(x.relative_to(p)):sha(x) for x in p.rglob('*') if x.is_file() and x!=m}; expected=json.loads(m.read_text()); assert actual==expected
 return {'directory':str(p),'manifest_sha256':pin,'files':len(actual),'inventory':expected}
source=verify('direct_omni_train_source_003','campaign_source_hashes.json','ef962070bd770d03553eb824fb6caccada7d6a06ac26d2f180bad068f0a9cc62')
native=verify('direct_omni_train_preparation_003','FREEZE_SHA256.json','1ab678beeba6bf74978c210c4e38f13655e25728fe66e3007e3b1db5e19b6b20')
host=verify('direct_omni_train_host_003','FREEZE_SHA256.json','df815208f340a9c8c0c2ce8c9fcfad7fd28791f745961ab714f96c0508454efc')
guard=verify('direct_omni_train_smoke_guard_004','FREEZE_SHA256.json','d7ecf21be4f9d38d9d5ba50d1d83b844b51ae82656c8e46efe51956d0630becb')
cold=verify('direct_omni_cold_source_001','campaign_source_hashes.json','4671eab012aaf79ce49769cfdc1667142237a4b4a45966fd0399a9caa900c51b')
p=B/'direct_omni_train_source_003'; plan=p/'robot/hexapod_mkii_length_study/training_plan.json';origin=p/'source_origin.json'
assert sha(plan)=='eee25089a65ec112f5eeed9c70f6310ef019ca5f077618c736fcfaa859739fc2';assert sha(origin)=='3b89ac6451e85020381994dce073b2b61af9caa08c79c12c887aff0cb41fb0c2'
orig=json.loads(origin.read_text());overlay=orig['runtime_overlays']
added=set(source['inventory'])-set(cold['inventory']);changed={x for x in cold['inventory'] if source['inventory'].get(x)!=cold['inventory'][x]}
assert added=={'tools/'+x for x in overlay};assert changed=={'tools/train_length_study.py','robot/hexapod_mkii_length_study/training_plan.json','source_origin.json'}
assert all(source['inventory']['tools/'+n]==h==native['inventory'][n] for n,h in overlay.items())
print(json.dumps({'source':source,'native':native,'host':host,'guard':guard,'cold_parent_sha256':cold['manifest_sha256'],'added':sorted(added),'changed':sorted(changed),'plan':json.loads(plan.read_text()),'origin':orig,'source_build_verified':{'source_manifest_sha256':source['manifest_sha256'],'files':source['files'],'plan_sha256':sha(plan),'origin_sha256':sha(origin)}}))
