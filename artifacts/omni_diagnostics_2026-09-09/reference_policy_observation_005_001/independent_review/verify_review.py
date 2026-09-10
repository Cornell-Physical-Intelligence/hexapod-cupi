"""Read-only independent identities/schema inspection; no GPU or source edits."""
from pathlib import Path
import ast,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
TENSOR=ROOT/'tmp/reference_tensor_wave_005_001'
OBS=ROOT/'tmp/reference_policy_observation_005_001'
OUT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check_freeze(path,expected):
    if sha(path/'FREEZE_SHA256.json')!=expected:raise ValueError('Unexpected freeze identity')
    wanted=json.loads((path/'FREEZE_SHA256.json').read_text());wanted=wanted.get('files',wanted)
    actual={str(p.relative_to(path)):sha(p) for p in sorted(path.rglob('*')) if p.is_file() and p!=path/'FREEZE_SHA256.json'}
    if wanted!=actual:raise ValueError('Missing, modified, or unlisted source bytes')
    return dict(path=str(path.relative_to(ROOT)),freeze_sha256=expected,files=len(wanted),all_bytes_match=True)

def main():
    tensor=check_freeze(TENSOR,'5f46b6f99c172bb037004e758f4f3715d41da42725abd10bfd313ee2b84fe286')
    obs_expected='22402b0079b0b5ac0acdd4e1d7e819fe206b20ee1883077f7de6eaca55808a63'
    observation=check_freeze(OBS,obs_expected)
    copied=json.loads((OBS/'reference/FREEZE_SHA256.json').read_text())
    assert sha(OBS/'reference/FREEZE_SHA256.json')==tensor['freeze_sha256']
    assert all(sha(OBS/'reference'/k)==sha(TENSOR/k)==v for k,v in copied.items())
    tc=json.loads((TENSOR/'source_contract.json').read_text());oc=json.loads((OBS/'source_contract.json').read_text())
    assert tc['scalar_controller_sha256']==oc['scalar_reference_sha256']=='8bacf0c61f7fdc6d43d9f9ebdb5adc643b026495bb303ec3c18157d121c78893'
    raw=ROOT/'tmp/reference_physics_results_009/run/wave'
    assert sha(TENSOR/'inputs/actual009_trace.npz')==sha(raw/'trace.npz')
    assert sha(TENSOR/'inputs/actual009_reference_states.json')==sha(raw/'reference_states.json')
    static={}
    for path in [TENSOR/'batch_wave.py',OBS/'observation.py',OBS/'device_pack.py']:
        tree=ast.parse(path.read_text())
        prohibited=[{'line':n.lineno,'attribute':n.attr} for n in ast.walk(tree) if isinstance(n,ast.Attribute) and n.attr in ('cpu','numpy','item','tolist')]
        if prohibited:raise ValueError('Unexpected host tensor transfer')
        static[str(path.relative_to(ROOT))]={'sha256':sha(path),'prohibited_transfer_attributes':prohibited}
    sys.path.insert(0,str(OBS))
    import torch
    torch.set_num_threads(1)
    from fixtures import Rig
    rig=Rig(2);encoded=rig.builder.build(rig.packet());spec=encoded['schema']
    assert bool(encoded['valid'].all()) and encoded['policy'].shape==(2,846) and encoded['critic'].shape==(2,849)
    cursor=0
    for field in spec['fields']:
        assert field['start']==cursor and field['stop']>field['start'];cursor=field['stop']
    assert cursor==846
    fields={f['name']:f for f in spec['fields']}
    required=['reference.unqualified_return_time','reference.unqualified_lift','reference.unqualified_return_valid','reference.raw_force_free_samples','reference.raw_force_free_runs','reference.unqualified_contact_returns','reference.last_unqualified_run_samples','reference.mode','joint_position_interval_average_rate','joint_position_interval_rate_valid']
    assert all(key in fields for key in required)
    assert fields['reference.mode']['stop']-fields['reference.mode']['start']==9
    assert spec['schema_sha256']==json.loads((OBS/'SPEC.json').read_text())['schema_sha256']
    assert rig.builder.history_valid.tolist()==[[False]*4+[True]]*2
    assert not encoded['interval_rate_valid'].any() and torch.isnan(encoded['interval_joint_rate_rad_s']).all()
    result={'tensor':tensor,'observation':observation,'copied_tensor_reference_files_match':len(copied),'exact009_raw_evidence_matches':True,'static_dynamic_host_transfer_scan':static,'actor_width':846,'critic_width':849,'schema_sha256':spec['schema_sha256'],'schema_fields_contiguous':True,'new_qualified_flight_fields_encoded':required,'initial_history_valid':rig.builder.history_valid.tolist(),'initial_interval_rate_invalid':True,'physical_or_GPU_admission':False}
    check_freeze(TENSOR,tensor['freeze_sha256']);check_freeze(OBS,obs_expected)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
