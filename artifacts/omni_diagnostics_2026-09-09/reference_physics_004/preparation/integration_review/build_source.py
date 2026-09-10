"""Freeze reference004: reviewed7mm lift and read-only eight-substep telemetry."""
from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'reference_physics_adapter_003/source_003'
WAVE=HERE.parent/'omni_reference_wave_003'
DEST=HERE/'source_004'
PARENT='7c75f0372abcea9eace3a280a2c204164179b8c433fc9d6280f60dd189737e24'
EXPECTED_WAVE_FREEZE='8d7ced506abfb7875ab35befacb91b2e5694acf4ca8071077a81f37f61766221'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DEST.exists():raise FileExistsError('New immutable source required')
    if sha(BASE/'campaign_source_hashes.json')!=PARENT:raise ValueError('Wrong source003 parent')
    old=json.loads((BASE/'campaign_source_hashes.json').read_text())
    if {str(p.relative_to(BASE)) for p in BASE.rglob('*') if p.is_file()}!=set(old)|{'campaign_source_hashes.json'}:
        raise ValueError('Parent contains extra/missing files')
    for name,digest in old.items():
        if sha(BASE/name)!=digest:raise ValueError('Changed parent '+name)
    if sha(WAVE/'FREEZE_SHA256.json')!=EXPECTED_WAVE_FREEZE:raise ValueError('Unexpected wave003 freeze')
    owner=json.loads((WAVE/'FREEZE_SHA256.json').read_text())
    for name,digest in owner['files'].items():
        if sha(WAVE/name)!=digest:raise ValueError('Changed owner file '+name)
    for name in owner['runtime_files']:
        if name!='wave_reference.py' and sha(WAVE/name)!=old['tools/'+name]:raise ValueError('Geometry/dependency changed')
    previous=(BASE/'tools/wave_reference.py').read_text()
    expected=previous.replace('lift_m: float = .005','lift_m: float = .007')
    if expected==previous or (WAVE/'wave_reference.py').read_text()!=expected:raise ValueError('Only5mm to7mm lift is allowed')
    DEST.mkdir()
    for name in old:
        target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/name,target)
    overlays={}
    for origin,name in [(WAVE/'wave_reference.py','tools/wave_reference.py'),
                        (HERE/'physics_substeps.py','tools/physics_substeps.py'),
                        (HERE/'run_reference_physics.py','tools/run_reference_physics.py')]:
        shutil.copy2(origin,DEST/name);overlays[name]=sha(DEST/name)
    origin=json.loads((BASE/'source_origin.json').read_text())
    origin['successor_of']={'manifest_sha256':PARENT,'source_origin_sha256':sha(BASE/'source_origin.json')}
    origin['wave_owner_freeze_sha256']=EXPECTED_WAVE_FREEZE
    origin['source004_overlays']=overlays
    origin['source004_reason']='RF physical lift1.445mm failed unchanged2mm gate; planned lift5mm to7mm. Add read-only all-eight-substep position/velocity/torque evidence to diagnose control-rate integral discrepancy.'
    origin['source004_physics_or_acceptance_gates_changed']=False
    origin['source004_control_rate_metrics_preserved']=True
    origin['source004_substep_metrics_are_diagnostic_not_substituted_for_gate']=True
    origin['source004_sdk_contract_sha256']=sha(HERE/'installed_sdk_contract.json')
    origin['source004_velocity_kernel_sha256']=sha(HERE/'installed_velocity_kernel.json')
    (DEST/'source_origin.json').write_text(json.dumps(origin,indent=2,sort_keys=True)+'\n')
    result={str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
    (DEST/'campaign_source_hashes.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'source':str(DEST),'files':len(result),'manifest_sha256':sha(DEST/'campaign_source_hashes.json')},indent=2))


if __name__=='__main__':main()
