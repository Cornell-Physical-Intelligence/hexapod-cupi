"""Read-only binding/lifecycle validation for actual evaluation015."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
A=HERE.parent
inputs={}
def read(path):
    inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())
def check():
    base=A/'results_evaluate_015'
    state=read(base/'standing/state.json');identity=read(base/'standing/identity.json')
    old=read(A/'results_evaluate_014/standing/identity.json')
    launch=read(base/'launch_binding.json');dispatch=read(A/'DISPATCH_evaluate_015.json')
    freeze=read(A/'source_018/FREEZE_SHA256.json')
    native=read(base/'standing/native/native_readback.json')
    previous_native=read(A/'results_evaluate_014/standing/native/native_readback.json')
    errors=read(base/'standing/native/native_errors.json')
    contact=read(base/'jobs/standing_contact_data_audit.json');cleanup=read(base/'cleanup.json')
    summary=read(base/'standing/evaluation/summary.json');transfer=read(A/'VERIFIED_TRANSFER_evaluate_015.json')
    digest=inputs[str(A/'source_018/FREEZE_SHA256.json')]
    assert digest==launch['source_freeze_sha256']==dispatch['source_freeze_sha256']==state['runtime_binding']['runtime_tree_sha256']
    assert identity['source_files']==freeze==state['identity']['source_files']
    checkpoint=A/'results_train_008/standing/learner/checkpoint_000320.pt'
    checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest();inputs[str(checkpoint)]=checkpoint_hash
    assert checkpoint_hash==state['input_checkpoint_sha256']=='7a694caeab00bc6c23a7724b99c88e6de8cfdbfd88e60d97288df4440c8f0a2a'
    assert checkpoint_hash in launch['input_files'].values()
    unchanged=['physics_source_files','physics_config','model_sha256','usd_sha256','geometry_sha256','geometry_extrema_sha256','prior_metadata_sha256']
    for key in unchanged:assert identity[key]==old[key],key
    assert native['config']['num_envs']==1 and native['sdf_shapes']==153
    assert len(native['native_body_names'])==19 and len(native['native_joint_names'])==18
    assert len(native['masses'])==1 and len(native['masses'][0])==19
    for key in ['native_joint_names','native_body_names','canonical_joint_names','masses','limits','native_max_velocity','recipe_readbacks']:
        assert native[key]==previous_native[key],key
    assert abs(native['mass_per_replica_kg'][0]-7.466088235225788)<1e-6
    assert native['implicit_drive_and_armature_zero']
    assert state['status']=='completed' and state['errors']==[] and errors==[]
    assert contact['passed'] and contact['incomplete_data_warning_count']==0
    assert cleanup['cleanup_checked'] and all(i['absent'] for i in cleanup['inspections'])
    assert not cleanup['reservation_released']
    assert transfer['all_files_verified'] and transfer['file_count']==71
    passes=[]
    for i in range(3):
        report=read(base/f'standing/evaluation/batch_{i:03d}/report.json')
        assert report['acquisition_complete'] and report['failure'] is None
        passes.append(report['results'][0]['pass'])
    assert passes==[False,False,False]
    assert len(summary['required_cases'])==len(summary['missing_cases'])==96
    assert not summary['stage2_complete'] and not summary['diagnostic_screens_passed']
    return {'schema':'evaluate015_binding_review_v1','checks_passed':True,'source_freeze_sha256':digest,
        'checkpoint_sha256':checkpoint_hash,'same_physics_identity_as014':True,
        'exact_19body18joint153shape_single_robot':True,'native_errors_empty':True,
        'contact_completeness_passed':True,'owned_container_absent':True,
        'original_case_pass_flags':passes,'missing_full_stage2_cases':96,'stage2_complete':False,
        'input_sha256':inputs}
if __name__=='__main__':
    result=check()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in inputs.items())
    result['inputs_unchanged']=True
    with (HERE/'BINDING_REVIEW.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))
