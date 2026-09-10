"""Fresh source002 differs from001 only by the bounded canonical-target startup."""
from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'reference_physics_adapter_001/source_001'
DEST=HERE/'source_002'
EXPECTED='a13f1534f9f802871a2cacbc0be603eae2d80738905381da40691f7cd4beff7c'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if DEST.exists():raise FileExistsError('New immutable source version required')
    if sha(BASE/'campaign_source_hashes.json')!=EXPECTED:raise ValueError('Wrong parent source')
    files=json.loads((BASE/'campaign_source_hashes.json').read_text())
    actual={str(p.relative_to(BASE)) for p in BASE.rglob('*') if p.is_file()}
    if actual!=set(files)|{'campaign_source_hashes.json'}:raise ValueError('Parent extra/missing files')
    for name,digest in files.items():
        if sha(BASE/name)!=digest:raise ValueError('Changed parent '+name)
    DEST.mkdir()
    for name in files:
        target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(BASE/name,target)
    overlays={}
    for name in ('canonical_stance_startup.py','run_reference_physics.py','screen_contract.py'):
        target=DEST/'tools'/name;shutil.copy2(HERE/name,target);overlays['tools/'+name]=sha(target)
    origin=json.loads((DEST/'source_origin.json').read_text())
    origin['successor_of']={'manifest_sha256':EXPECTED,'source_origin_sha256':sha(BASE/'source_origin.json')}
    origin['source002_overlays']=overlays
    origin['source002_reason']='Source001 held32randomized reset targets forever; preserve physical reset and smoothly bring joint targets to exact nominal C stance over2s, then hold2s before unchanged gates.'
    origin['source002_gates_changed']=False
    (DEST/'source_origin.json').write_text(json.dumps(origin,indent=2,sort_keys=True)+'\n')
    result={str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
    (DEST/'campaign_source_hashes.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'source':str(DEST),'files':len(result),'manifest_sha256':sha(DEST/'campaign_source_hashes.json')},indent=2))


if __name__=='__main__':main()
