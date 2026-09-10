"""Portable, read-only terminal payload verification; standard library only.

Never imports Isaac, Torch, an app launcher, or the GPU dispatch guard. It does
not reconstruct omitted source/observation trees or launch a new allocation.
"""
from pathlib import Path
import hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def verify_tree(p,manifest,expected=None):
    path=p/manifest
    if expected is not None and sha(path)!=expected:raise ValueError('Wrong bound manifest: '+str(path))
    if any(f.is_symlink() for f in p.rglob('*')):raise ValueError('Symlink payload substitution')
    inventory=read(path);actual={str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file() and f!=path}
    if inventory!=actual:raise ValueError('Missing, extra or changed payload: '+str(p))
    return len(inventory)

def main():
    total=verify_tree(ROOT,'BUNDLE_SHA256.json');reconstruction=read(ROOT/'RECONSTRUCTION.json')
    frozen={}
    for name,entry in reconstruction['copied_frozen_bundles'].items():
        frozen[name]=verify_tree(ROOT/name,'FREEZE_SHA256.json',entry['freeze_sha256'])
        if frozen[name]!=entry['payload_files']:raise ValueError('Copied manifest count mismatch')
    for name,entry in reconstruction['referenced_maps'].items():
        path=ROOT/'references'/name
        if sha(path)!=entry['sha256'] or len(read(path))!=entry['files_in_map']:raise ValueError('Referenced map changed')
    audit=read(ROOT/'actual_review/remote_audit.json');raw=ROOT/'actual_review/raw'
    for name,digest in audit['raw_payloads'].items():
        if sha(raw/name)!=digest:raise ValueError('Raw payload differs from independent remote audit')
    if len(audit['raw_payloads'])!=38 or audit['assets_verified']!=550:raise ValueError('Incomplete raw/assets receipt')
    host_root=ROOT/'owner/host002';sys.path.insert(0,str(host_root))
    if 'numeric_evidence' in sys.modules:raise ValueError('Refuse a preloaded numeric evidence implementation')
    spec=importlib.util.spec_from_file_location('_bound_terminal_host002',host_root/'launch_device_smoke_spark.py');host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
    run=raw/'reference_device_smoke_001';campaign=read(run/'campaign.json')
    if campaign['status']!='completed' or campaign['both_bridge_phases_passed'] is not True or campaign['source_and_inputs_unchanged'] is not True:raise ValueError('Completed bridge receipt missing')
    results={}
    for phase in ('replicas_1','replicas_32'):
        result=host.validate_result(run/phase,campaign['identities'][phase])
        if result!=read(run/(phase+'_accepted.json')):raise ValueError('Raw numeric acceptance differs from original accepted receipt')
        results[phase]=result['passed']
    print(json.dumps({'bundle_files_verified':total,'copied_frozen_payloads':frozen,'raw_payloads_verified':38,'original_host002_numeric_gates_rechecked':results,'full_source009_and_observation005_trees_referenced_not_duplicated':True,'read_only':True,'GPU_launches':0,'PPO_admitted':False,'full_quiet_admitted':False,'velocity_fidelity_qualified':False},indent=2))
if __name__=='__main__':main()
