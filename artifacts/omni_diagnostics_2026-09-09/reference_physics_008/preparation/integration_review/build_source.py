"""Freeze a new bounded full-C standing→wave trial from exact007 plus timing004."""
from pathlib import Path
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'reference_physics_adapter_007/source_007'
DEST=HERE/'source_008'
WAVE=HERE.parent/'omni_reference_wave_004'
PARENT='666571ed6e37a73b857179323573ae20cdef0eee53aa25908108bd4e29465465'
WAVE_FREEZE='dedd5a2deb21703fe1d8713fcc2c276339a58db3f14b9ab85fb65a3f2ee7e834'
RUNTIME=['solver_comparison.py','screen_contract.py','launch_reference_physics_spark.py','screen_metrics.py','run_reference_physics.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if DEST.exists():raise FileExistsError('New immutable source required')
    if sha(BASE/'campaign_source_hashes.json')!=PARENT:raise ValueError('Wrong immutable source007')
    old=json.loads((BASE/'campaign_source_hashes.json').read_text())
    if {str(p.relative_to(BASE)) for p in BASE.rglob('*') if p.is_file()}!=set(old)|{'campaign_source_hashes.json'}:raise ValueError('Parent extras or missing files')
    for name,digest in old.items():
        if sha(BASE/name)!=digest:raise ValueError('Changed parent '+name)
    if sha(WAVE/'FREEZE_SHA256.json')!=WAVE_FREEZE:raise ValueError('Wrong reviewed wave004 freeze')
    wave_map=json.loads((WAVE/'FREEZE_SHA256.json').read_text());wave_map=wave_map.get('files',wave_map)
    for name,digest in wave_map.items():
        if sha(WAVE/name)!=digest:raise ValueError('Changed wave owner '+name)
    for name in ['wave_math.py','serial_geometry.py','geometry/candidate_c_reference.json','geometry/f050_t060.urdf']:
        if sha(BASE/'tools'/name)!=sha(WAVE/name):raise ValueError('Unexpected wave geometry/math delta '+name)
    DEST.mkdir()
    for name in old:
        target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/name,target)
    overlays={}
    for name in RUNTIME:
        shutil.copy2(HERE/name,DEST/'tools'/name);overlays['tools/'+name]=sha(DEST/'tools'/name)
    shutil.copy2(WAVE/'wave_reference.py',DEST/'tools/wave_reference.py')
    overlays['tools/wave_reference.py']=sha(DEST/'tools/wave_reference.py')
    origin=json.loads((BASE/'source_origin.json').read_text())
    origin['successor_of']={'manifest_sha256':PARENT,'source_origin_sha256':sha(BASE/'source_origin.json')}
    origin['source008_overlays']=overlays
    origin['source008_scope']='Fresh32x1000standing including unchanged all32quiet gates, then1x2400 bounded zero-residual wave; no PPO or production adoption'
    origin['source008_physics_change_from007']=None
    origin['source008_wave_owner_freeze_sha256']=WAVE_FREEZE
    origin['source008_wave_schema_change']='AdvancedHorizontalSwing horizontal_duration_fraction=.8 and horizontal_coefficients; incompatible actor state is not reused'
    origin['source008_existing_wave_metrics_and_gates_changed']=False
    origin['source008_standing_admission_strengthening']='Every fresh replica must also pass the existing unchanged QUIET_GATES before wave'
    origin['source008_observer_startup_targets_and_actuators_unchanged']=True
    origin['source008_combined_changes_vs_last_wave004_no_causal_attribution']=['TGS externalforcesTrue and velocityiterations1','horizontal swing completion80percent']
    (DEST/'source_origin.json').write_text(json.dumps(origin,indent=2,sort_keys=True)+'\n')
    mapping={str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
    (DEST/'campaign_source_hashes.json').write_text(json.dumps(mapping,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'files':len(mapping),'manifest_sha256':sha(DEST/'campaign_source_hashes.json'),'source':str(DEST)},indent=2))
if __name__=='__main__':main()
