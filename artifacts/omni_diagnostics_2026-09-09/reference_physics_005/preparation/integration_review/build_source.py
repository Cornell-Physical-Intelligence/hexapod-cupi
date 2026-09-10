"""Build immutable standing-only solver comparison from exact004."""
from pathlib import Path
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'reference_physics_adapter_004/source_004'
DEST=HERE/'source_005'
PARENT='a433e529d29d5360c828b406a3dfd769e078d69fc9deaf03f4fe110eb6fa7a63'
RUNTIME=['solver_comparison.py','reference_physics_env.py','screen_contract.py','launch_reference_physics_spark.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if DEST.exists():raise FileExistsError('New immutable source required')
    if sha(BASE/'campaign_source_hashes.json')!=PARENT:raise ValueError('Wrong immutable source004')
    old=json.loads((BASE/'campaign_source_hashes.json').read_text())
    if {str(p.relative_to(BASE)) for p in BASE.rglob('*') if p.is_file()}!=set(old)|{'campaign_source_hashes.json'}:raise ValueError('Parent extras or missing files')
    for name,digest in old.items():
        if sha(BASE/name)!=digest:raise ValueError('Changed parent '+name)
    DEST.mkdir()
    for name in old:
        target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/name,target)
    overlays={}
    for name in RUNTIME:
        shutil.copy2(HERE/name,DEST/'tools'/name);overlays['tools/'+name]=sha(DEST/'tools'/name)
    origin=json.loads((BASE/'source_origin.json').read_text())
    origin['successor_of']={'manifest_sha256':PARENT,'source_origin_sha256':sha(BASE/'source_origin.json')}
    origin['source005_overlays']=overlays
    origin['source005_scope']='Isolated32x1000standing comparison only; no wave, policy or solver default adoption'
    origin['source005_only_physics_change']={'sim.physics.enable_external_forces_every_iteration':{'from':False,'to':True}}
    origin['source005_existing_metrics_and_gates_changed']=False
    origin['source005_observer_and_targets_unchanged']=True
    (DEST/'source_origin.json').write_text(json.dumps(origin,indent=2,sort_keys=True)+'\n')
    mapping={str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
    (DEST/'campaign_source_hashes.json').write_text(json.dumps(mapping,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'files':len(mapping),'manifest_sha256':sha(DEST/'campaign_source_hashes.json'),'source':str(DEST)},indent=2))
if __name__=='__main__':main()
