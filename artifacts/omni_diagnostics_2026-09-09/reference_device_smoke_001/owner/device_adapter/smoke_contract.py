"""Fail-closed inputs for a short actualIsaac device bridge; no policy admission."""
from pathlib import Path
import hashlib,json
SOURCE='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
CAMPAIGN='cdcaa80c1156509d60e336042172dc6651755767e08ecbb66ef31d8e788bcb26'
OBSERVATION='22402b0079b0b5ac0acdd4e1d7e819fe206b20ee1883077f7de6eaca55808a63'
ADMISSION='6007537cf6dd7af31b0078b2ab765349b861e862370c97ca40ca6f1793c85202'
WAVE='dc42d919b57e548e03bf3c0208ab5af6e71f77beb493ba600adbdb1c6b5476c9'
STUDY='99fd81ee260bd517e6590fe56b8bb46715827fef1a295553f154524a1978ce00'
RUNTIME='abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280'
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def tree(path):
    if any(p.is_symlink() for p in path.rglob('*')):raise ValueError('No symbolic source/asset substitutions')
    return {str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file()}
def verify_tree(root,manifest,expected):
    if sha(root/manifest)!=expected:raise ValueError('Wrong frozen manifest: '+str(root))
    wanted=read(root/manifest);actual=tree(root);actual.pop(manifest)
    if wanted!=actual:raise ValueError('Frozen tree changed: '+str(root))
def verify(args):
    if args.num_envs not in (1,32) or args.controls!=264:raise ValueError('Only1/32 replicas and200startup+64hold controls are admitted for bridge smoke')
    verify_tree(args.source_root,'campaign_source_hashes.json',SOURCE)
    verify_tree(args.observation_bundle,'FREEZE_SHA256.json',OBSERVATION)
    adapter_manifest=HERE/'FREEZE_SHA256.json'
    adapter_sha=sha(adapter_manifest)
    verify_tree(HERE,'FREEZE_SHA256.json',adapter_sha)
    for relative,expected in [('campaign.json',CAMPAIGN),('standing/admission.json',ADMISSION),('wave/state.json',WAVE),('inputs/study_before.sha256.json',STUDY)]:
        if sha(args.run/relative)!=expected:raise ValueError('Wrong completed009 evidence: '+relative)
    campaign=read(args.run/'campaign.json')
    if campaign['status']!='completed' or campaign['bounded_wave_physics_passed'] is not True:raise ValueError('Completed009 scalar physics proof required')
    assets=read(args.run/'inputs/study_before.sha256.json')
    if len(assets)!=550 or tree(args.package)!=assets:raise ValueError('Complete550-file admitted asset required')
    return {'schema':'actual_device_bridge_smoke_v1','source_manifest_sha256':SOURCE,'observation_freeze_sha256':OBSERVATION,
            'adapter_freeze_sha256':adapter_sha,'actor_width':846,'critic_width':849,'replicas':args.num_envs,'controls':264,'startup_controls':200,'hold_controls':64,
            'dt_s':.02,'physical_dt_s':.0025,'source009_admitted_identity':campaign['identity'],
            'not_a_fresh_full_standing_admission':True,'PPO_permitted':False,'stage2_complete':False,'native_velocity_fidelity_qualified':False}
def save(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
