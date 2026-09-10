"""Independent frozen bridge/source contract checks, without importing Isaac."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
BRIDGE=ROOT/'tmp/reference_device_smoke_001';OUT=Path(__file__).resolve().parent
EXPECTED='be4870a8f7ca0857ba50ffd3a0c92ac5925b9a816767d66d826af7b59957aa0e'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify():
    if sha(BRIDGE/'FREEZE_SHA256.json')!=EXPECTED:raise ValueError('Wrong reviewed bridge freeze')
    wanted=json.loads((BRIDGE/'FREEZE_SHA256.json').read_text())
    actual={str(p.relative_to(BRIDGE)):sha(p) for p in BRIDGE.rglob('*') if p.is_file() and p.name!='FREEZE_SHA256.json'}
    if actual!=wanted:raise ValueError('Bridge files changed or unlisted files added')
    source=json.loads((BRIDGE/'sensor_source_contract.json').read_text())
    for module,data in source['installed_modules'].items():
        if sha(BRIDGE/data['local_source'])!=data['sha256']:raise ValueError('Sensor API evidence mismatch')
    # These are exact copied installed-source equations, not inferred APIs.
    kernel=(BRIDGE/'installed_sensor_source/sensor_kernels.py').read_text()
    for text in ['new_timestamp = timestamp[env] + dt','timestamp_last_update[env] = timestamp[env]','is_outdated[env] = False']:
        if text not in kernel:raise ValueError('Clock recurrence or cache contract changed')
    entry=(BRIDGE/'run_device_smoke.py').read_text()
    assert entry.index('app=AppLauncher(args).app')<entry.index('        import torch')
    assert 'with physics_recorder,capture_before_reset(env,capture)' in entry
    assert 'physics_recorder.begin_control(step)' in entry and 'physics_recorder.end_control' in entry
    assert 'wave.step(last[\'measurement\'],commands)' in entry
    assert 'zero=torch.zeros((args.num_envs,18)' in entry
    assert 'range(264)' in entry and 'if step==200:' in entry
    from types import SimpleNamespace
    sys.path.insert(0,str(BRIDGE));from smoke_contract import verify as preflight
    src=ROOT/'tmp/reference_physics_adapter_009/source_009';run=ROOT/'tmp/reference_physics_results_009/run';obs=ROOT/'tmp/reference_policy_observation_005_001'
    checks=[]
    for count in (1,32):
        args=SimpleNamespace(source_root=src,run=run,observation_bundle=obs,package=src/'robot/hexapod_mkii_length_study',num_envs=count,controls=264)
        checks.append(preflight(args))
    return {'bridge_freeze_sha256':EXPECTED,'files':len(wanted),'all_files_match':True,'source_api_files_checked':source['installed_modules'],'read_only_preflight_1_and_32':checks,'source009_substep_observer_sha256':sha(src/'tools/physics_substeps.py'),'sensor_lazy_update_review':'read installed float32 Warp timestamp/cached-update arrays after ordinary .data; no manual sensor update/reset','target_scope':'200 canonical startup/settle controls then64 zero-command hold controls; exact named default target verified','actual_installed_GPU_API_executed':False,'PPO_or_physical_admission':False}
if __name__=='__main__':
    value=verify();(OUT/'verification.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
