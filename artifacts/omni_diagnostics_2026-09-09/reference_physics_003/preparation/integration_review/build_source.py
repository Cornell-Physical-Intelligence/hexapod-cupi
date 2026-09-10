"""Freeze landing-only reference003 from exact924-file reference002."""
from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'reference_physics_adapter_002/source_002'
WAVE=HERE.parent/'omni_reference_wave_002'
DEST=HERE/'source_003'
PARENT='34390c6162f462baa4c531f1d5aff346ffbc98d4d0266a68ea797f5e5b992ee0'
EXPECTED_RUNTIME={'wave_reference.py','serial_geometry.py','wave_math.py',
                  'geometry/candidate_c_reference.json','geometry/f050_t060.urdf'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DEST.exists():raise FileExistsError('Fresh source version required')
    if sha(BASE/'campaign_source_hashes.json')!=PARENT:raise ValueError('Wrong source002 parent')
    old=json.loads((BASE/'campaign_source_hashes.json').read_text())
    if {str(p.relative_to(BASE)) for p in BASE.rglob('*') if p.is_file()}!=set(old)|{'campaign_source_hashes.json'}:
        raise ValueError('Parent contains extra/missing files')
    for name,digest in old.items():
        if sha(BASE/name)!=digest:raise ValueError('Changed parent '+name)
    frozen=json.loads((WAVE/'FREEZE_SHA256.json').read_text())
    if set(frozen['runtime_files'])!=EXPECTED_RUNTIME:raise ValueError('Runtime dependency list changed')
    for name,digest in frozen['files'].items():
        if sha(WAVE/name)!=digest:raise ValueError('Changed owner freeze '+name)
    DEST.mkdir()
    for name in old:
        target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/name,target)
    overlays={}
    for name in frozen['runtime_files']:
        target=DEST/'tools'/name
        shutil.copy2(WAVE/name,target);overlays['tools/'+name]=sha(target)
    # Geometry is immutable; only the contact-aware reference algorithm changes.
    changed=[name for name,digest in overlays.items() if digest!=old[name]]
    if changed!=['tools/wave_reference.py']:raise ValueError('Unexpected runtime delta '+str(changed))
    origin=json.loads((BASE/'source_origin.json').read_text())
    origin['successor_of']={'manifest_sha256':PARENT,'source_origin_sha256':sha(BASE/'source_origin.json')}
    origin['wave_owner_freeze_sha256']=sha(WAVE/'FREEZE_SHA256.json')
    origin['source003_overlays']=overlays
    origin['source003_reason']='Preserve admitted canonical startup and physics; new bounded C2 landing after measured flight/descent and returned contact, with three stable confirmations after endpoint.'
    origin['source003_physical_acceptance_gates_changed']=False
    origin['source003_cpu_landing_report_sha256']=sha(WAVE/'report.json')
    origin['source003_target_planar_overshoot_then_return_m']=json.loads((WAVE/'report.json').read_text())['new_landing_reference']['max_planar_overshoot_then_return_m']
    (DEST/'source_origin.json').write_text(json.dumps(origin,indent=2,sort_keys=True)+'\n')
    mapping={str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
    (DEST/'campaign_source_hashes.json').write_text(json.dumps(mapping,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'source':str(DEST),'files':len(mapping),'manifest_sha256':sha(DEST/'campaign_source_hashes.json')},indent=2))


if __name__=='__main__':main()
