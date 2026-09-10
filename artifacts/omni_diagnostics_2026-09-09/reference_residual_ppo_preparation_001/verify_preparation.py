"""Portable hash/scope verification only; no simulator, learner or remote access."""
from pathlib import Path
import argparse,hashlib,json
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def tree(root):
    if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('Symbolic payload substitution')
    return {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
def verify(repository=None):
    actual=tree(HERE);actual.pop('BUNDLE_SHA256.json')
    if actual!=read(HERE/'BUNDLE_SHA256.json'):raise ValueError('Primary inventory differs')
    reconstruction=read(HERE/'RECONSTRUCTION.json')
    for relative,record in reconstruction['copied_frozen_bundles'].items():
        root=HERE/relative;manifest=root/'FREEZE_SHA256.json'
        if sha(manifest)!=record['freeze_sha256']:raise ValueError('Copied freeze differs: '+relative)
        wanted=read(manifest);observed=tree(root);observed.pop('FREEZE_SHA256.json')
        if len(wanted)!=record['payload_files'] or wanted!=observed:raise ValueError('Copied payload differs: '+relative)
    local=read(HERE/'preflight/local_consumer.json');spark=read(HERE/'preflight/spark_host.json')
    if local!=spark or local['standing_admission_sha256'] is not None or local['Stage2_complete'] is not False:
        raise ValueError('Preflights differ or inflate scope')
    if local['consumer_freeze_sha256']!=reconstruction['copied_frozen_bundles']['consumer']['freeze_sha256']:
        raise ValueError('Preflight consumer identity differs')
    lineage=read(HERE/'REVIEW_LINEAGE.json')
    if not lineage['all_existing_plan_values_identical'] or not lineage['independently_reviewed_session_runner_helper_and_CPU_regression_bytes_unchanged']:
        raise ValueError('Review lineage is not preserved')
    current=read(HERE/'consumer/plan.json');old=dict(current)
    for key in lineage['added_declarations']:old.pop(key)
    if hashlib.sha256((json.dumps(old,indent=2)+'\n').encode()).hexdigest()!=lineage['prior_independent_plan_sha256']:
        raise ValueError('Prior reviewed plan cannot be reconstructed exactly')
    if sha(HERE/'consumer/plan.json')!=lineage['final_consumer_plan_sha256']:raise ValueError('Final plan differs')
    referenced=0
    if repository is not None:
        for key,record in reconstruction['referenced_published_inputs'].items():
            if sha(repository/record['repository_path'])!=record['sha256']:raise ValueError('Published reference differs: '+key)
            referenced+=1
    return {'passed':True,'preparation_payloads':len(actual),'copied_frozen_bundles':4,'published_reference_hashes_checked':referenced,
        'actual_policy_GPU_admission':False,'CPU_physics_is_synthetic':True,'Stage2_complete':False}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repository',type=Path);args=parser.parse_args()
    print(json.dumps(verify(args.repository),indent=2))
