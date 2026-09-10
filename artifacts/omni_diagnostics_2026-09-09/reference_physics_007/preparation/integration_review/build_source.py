"""Build immutable standing-only iteration readback correction from exact006."""
from pathlib import Path
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'reference_physics_adapter_006/source_006'
DEST=HERE/'source_007'
PARENT='90d7cf551ccf52f7cc2d224b81547c1bad0ad6191c2db56389bb2d54de0ff193'
RUNTIME=['solver_comparison.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if DEST.exists():raise FileExistsError('New immutable source required')
    if sha(BASE/'campaign_source_hashes.json')!=PARENT:raise ValueError('Wrong immutable source006')
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
    origin['source007_overlays']=overlays
    origin['source007_scope']='Introspection-only correction of006; same32x1000standing, no wave, policy or solver default adoption'
    origin['source007_physics_change_from006']=None
    origin['source007_actual006_readback_sha256']=sha(HERE/'inputs/actual006_failed_readback.json')
    origin['source007_existing_metrics_and_gates_changed']=False
    origin['source007_external_forces_every_iteration_remains_true']=True
    origin['source007_observer_and_targets_unchanged']=True
    (DEST/'source_origin.json').write_text(json.dumps(origin,indent=2,sort_keys=True)+'\n')
    mapping={str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
    (DEST/'campaign_source_hashes.json').write_text(json.dumps(mapping,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'files':len(mapping),'manifest_sha256':sha(DEST/'campaign_source_hashes.json'),'source':str(DEST)},indent=2))
if __name__=='__main__':main()
