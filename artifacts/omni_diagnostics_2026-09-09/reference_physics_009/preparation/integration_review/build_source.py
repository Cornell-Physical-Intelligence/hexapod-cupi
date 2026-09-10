"""Freeze exact008 plus independently reviewed qualified-liftoff/angle observer."""
from pathlib import Path
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'reference_physics_adapter_008_final/source_008'
DEST=HERE/'source_009'
WAVE=HERE.parent/'omni_reference_wave_005'
OBSERVER=HERE.parent/'reference_joint_substep_observer_001'
PARENT='a73075ebdf5d6c986728b96117495b8de85adc8db62de1c41ffbe49eb6176d66'
WAVE_FREEZE='5b6c076cb03426ac8e24cc6df7e8684e48862aa78a495bd61ae49773a7826814'
OBSERVER_FREEZE='1c3a736017afeb810261a734558c7a4a904dd9010d0fdae09d68ef32efb80415'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verified_owner(path,digest):
    if sha(path/'FREEZE_SHA256.json')!=digest:raise ValueError('Wrong owner freeze '+str(path))
    mapping=json.loads((path/'FREEZE_SHA256.json').read_text());mapping=mapping.get('files',mapping)
    for name,h in mapping.items():
        if sha(path/name)!=h:raise ValueError('Changed owner '+str(path/name))
def main():
    if DEST.exists():raise FileExistsError('New immutable source required')
    if sha(BASE/'campaign_source_hashes.json')!=PARENT:raise ValueError('Wrong immutable source008')
    old=json.loads((BASE/'campaign_source_hashes.json').read_text())
    if {str(p.relative_to(BASE)) for p in BASE.rglob('*') if p.is_file()}!=set(old)|{'campaign_source_hashes.json'}:raise ValueError('Parent extras or missing files')
    for name,h in old.items():
        if sha(BASE/name)!=h:raise ValueError('Changed parent '+name)
    verified_owner(WAVE,WAVE_FREEZE);verified_owner(OBSERVER,OBSERVER_FREEZE)
    for name in ['wave_math.py','serial_geometry.py','geometry/candidate_c_reference.json','geometry/f050_t060.urdf']:
        if sha(BASE/'tools'/name)!=sha(WAVE/name):raise ValueError('Unexpected geometry/math delta '+name)
    DEST.mkdir()
    for name in old:
        target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/name,target)
    overlays={}
    for name,parent in [('solver_comparison.py',HERE),('wave_reference.py',WAVE),('physics_substeps.py',OBSERVER)]:
        shutil.copy2(parent/name,DEST/'tools'/name);overlays['tools/'+name]=sha(DEST/'tools'/name)
    origin=json.loads((BASE/'source_origin.json').read_text())
    origin['successor_of']={'manifest_sha256':PARENT,'source_origin_sha256':sha(BASE/'source_origin.json')}
    origin['source009_overlays']=overlays
    origin['source009_scope']='Fresh32x1000standing+unchanged all32quiet, then1x2400 qualified-liftoff wave; read-only400Hz joint angles; noPPO/productionadoption'
    origin['source009_physics_change_from008']=None
    origin['source009_wave_owner_freeze_sha256']=WAVE_FREEZE
    origin['source009_joint_angle_observer_freeze_sha256']=OBSERVER_FREEZE
    origin['source009_original_entry_metrics_host_and_gates_unchanged']=True
    origin['source009_future_actor_state_schema_not_claimed_compatible']=True
    (DEST/'source_origin.json').write_text(json.dumps(origin,indent=2,sort_keys=True)+'\n')
    mapping={str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
    (DEST/'campaign_source_hashes.json').write_text(json.dumps(mapping,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'files':len(mapping),'manifest_sha256':sha(DEST/'campaign_source_hashes.json'),'source':str(DEST)},indent=2))
if __name__=='__main__':main()
