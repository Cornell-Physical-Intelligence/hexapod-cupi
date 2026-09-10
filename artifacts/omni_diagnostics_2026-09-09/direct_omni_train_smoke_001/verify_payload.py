"""Portable, read-only verification of complete historical evidence; no simulator imports."""
from pathlib import Path
import hashlib,json,sys
sys.dont_write_bytecode=True

SOURCE='37b4b6d5d75f6d697a122fc2a1903f8f625f6a67d1727e1257e1533300e44da6'
PLAN='9fb7d0b01480523972b4f1c17e717756d7dcf1c54b9caa436245671937de538c'
CHECKPOINT='4aaf556613e72a80332381c09309cc0ab52a03c6d55527bc76e8a7000a29e1f2'
INVOCATION='1de80a43928f4d7497846dc762048d86'

def require(condition,message):
    if not condition:raise ValueError(message)
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()
def read(path):return json.loads(path.read_text())
def inventory(root):return {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}

def derive(root):
    raw=root/'raw';audit=read(root/'audit/terminal_audit.json');local=read(root/'audit/local_raw_map.json')
    measured={p.relative_to(raw).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in raw.rglob('*') if p.is_file()}
    require(measured==local,'Raw inventory differs from captured local map')
    require(len(measured)==30 and sum(x['bytes'] for x in measured.values())==13549925,'Wrong raw file count or bytes')
    require({k:v['sha256'] for k,v in measured.items()}==audit['raw_payloads'],'Raw differs from remote audit')
    require({k:v['bytes'] for k,v in measured.items()}==audit['raw_sizes_bytes'],'Raw sizes differ from remote audit')
    campaign=read(raw/'run/campaign.json');receipt=read(raw/'run/train/training_receipt.json')
    require(campaign['status']=='failed' and campaign['terminal_inputs_unchanged'] is True,'Changed campaign verdict')
    require(campaign['PPO_updates_completed']==0 and set(campaign['accepted_phases'])=={'standing'},'Changed host acceptance')
    require(receipt['updates_completed']==2 and receipt['complete'] is False,'Changed actual update/finalizer result')
    require([x['completed_update'] for x in receipt['optimizer_updates']]==[1,2],'Actual update sequence differs')
    require(receipt['audit']['controls']==48 and receipt['audit']['replicas']==32,'Wrong actual learning allocation')
    require(campaign['observed_training']['observed_updates_completed']==2,'Actual updates hidden by accepted-update count')
    require('obs_normalizer._std' in receipt['error'] and 'Inplace update to inference tensor outside InferenceMode' in receipt['error'],'Wrong failure')
    require(receipt['error']==read(raw/'run/train/failure.json')['error']==audit['failure'],'Failure evidence mismatch')
    require(sha(raw/'run/train/policy/final.pt')==CHECKPOINT==audit['saved_final_checkpoint_sha256'],'Wrong preserved final checkpoint')
    require(sha(raw/'run/campaign.json')==audit['campaign_sha256'],'Wrong campaign receipt')
    require(sha(raw/'run/train/training_receipt.json')==audit['training_receipt_sha256'],'Wrong training receipt')
    require(campaign['identity']==audit['identity'],'Source/selection differs across audit and run')
    require(campaign['identity']['source_manifest_sha256']==SOURCE and campaign['identity']['plan_sha256']==PLAN,'Wrong native lineage')
    require(not (raw/'run/final_constant').exists() and not (raw/'run/final_stop').exists(),'Unexpected evaluation results')
    standing=read(raw/'run/standing/admission.json')
    require(standing['gate']['passed'] is True and standing['gate']['num_envs']==32 and standing['gate']['control_steps']==1000,'Standing admission differs')
    require(inventory(raw/'run/standing')==read(raw/'run/standing_immutable.sha256.json'),'Standing changed after admission')
    require(audit['all_input_trees_verified'] is True and audit['standing_unchanged'] is True,'Missing original input audit')
    require(audit['expected_invocation']==INVOCATION and audit['unit']['ActiveState'] in ('failed','inactive'),'Wrong terminal ownership')
    require(audit['actual_completed_updates']==2 and audit['accepted_updates']==0 and audit['campaign_completed'] is False,'Audit verdict differs')
    tokens=[]
    for phase,status in [('standing','completed'),('train','failed')]:
        job=read(raw/'run/jobs'/(phase+'.json'))
        require(job==audit['jobs'][phase] and job['status']==status and job['cleanup_checked'] is True,'Wrong job/cleanup receipt')
        tokens.extend([job['container_name'],job['container_id']])
    require(set(tokens)==set(audit['owned_names_IDs_absent']) and len(tokens)==4,'Wrong owned name/ID inventory')
    require(all(audit['owned_names_IDs_absent'][x]['absent'] is True for x in tokens),'Owned container absence unproved')
    restored=read(raw/'forecast_pause/restored.json');pause=read(raw/'forecast_pause/pause.json')
    require(restored==audit['restoration'] and restored['restored_unix']>=pause['created_unix'],'Wrong restoration receipt')
    require(audit['halo_archive_unchanged'] is True and audit['halo_restarted'] is False,'Recorded unrelated workload history differs')
    pins={
        'native_source_sha256.json':(SOURCE,598),
        'native_contract_sha256.json':('9e591b93ae99ca3a76b7f7500ea20cc778da76c7f6b9b2c86fd22d31aceab3d7',21),
        'host_sha256.json':('332fad29e0a0f88ba07012700f2a21c3a4f0453f3d71224da84fcc477e48394f',8),
        'guard_sha256.json':('fb457055503fbc2167f3e87f2916d5153ff3b33e672c96d5c973cd3118f15d7b',14),
        'supervisor_source_sha256.json':('04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e',926)}
    for name,(bound,count) in pins.items():
        require(sha(root/'inputs'/name)==bound and len(read(root/'inputs'/name))==count,'Changed input manifest: '+name)
    legacy=read(root/'inputs/legacy_runtime_sha256.json')
    normalized={k.removeprefix('isaaclab/hexapod_rl/'):v for k,v in legacy.items()}
    require(len(legacy)==16 and hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280','Wrong canonical legacy16 manifest')
    return {'kind':'actual_failed_native002_caps_smoke','campaign_status':'failed','standing_passed':True,
        'actual_PPO_updates':2,'host_accepted_training_updates':0,'actual_controls':48,'replicas':32,
        'final_constant_evaluation_ran':False,'final_stop_evaluation_ran':False,
        'failure_stage':'strict final checkpoint reload','failure_buffer':'obs_normalizer._std',
        'final_checkpoint_sha256':CHECKPOINT,'source_manifest_sha256':SOURCE,'plan_sha256':PLAN,
        'raw_payloads':30,'raw_bytes':13549925,'owned_names_absent':2,'owned_recorded_IDs_absent':2,
        'forecast_pause':'054','restored_unix':restored['restored_unix'],'root_terminal_audit_replayed_offline':True,
        'remote_input_checks_are_preserved_historical_evidence':True,'full_input_source_bytes_embedded':False,
        'quality_gain_established':False,'Stage2_complete':False,'checkpoint_qualified':False}

def verify(root):
    require(not any(p.is_symlink() for p in root.rglob('*')),'Symbolic payload')
    expected=read(root/'BUNDLE_SHA256.json');actual=inventory(root);actual.pop('BUNDLE_SHA256.json')
    require(actual==expected,'Missing, changed or unlisted bundle payload')
    result=derive(root);require(result==read(root/'RESULT.json'),'Derived summary differs')
    return {'passed':True,'payloads':len(actual),'bundle_sha256':sha(root/'BUNDLE_SHA256.json'),**result}

if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parent),indent=2))
