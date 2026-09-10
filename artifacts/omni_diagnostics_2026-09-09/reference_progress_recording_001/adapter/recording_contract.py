"""CPU contracts for a fresh native recording of admitted source009 physics."""
import ast,hashlib,json
from pathlib import Path
SOURCE_MANIFEST_SHA256='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
CAMPAIGN_SHA256='cdcaa80c1156509d60e336042172dc6651755767e08ecbb66ef31d8e788bcb26'
ADMISSION_SHA256='6007537cf6dd7af31b0078b2ab765349b861e862370c97ca40ca6f1793c85202'
WAVE_STATE_SHA256='dc42d919b57e548e03bf3c0208ab5af6e71f77beb493ba600adbdb1c6b5476c9'
STUDY_TREE_SHA256='99fd81ee260bd517e6590fe56b8bb46715827fef1a295553f154524a1978ce00'
LABEL='STEPPING REFERENCE — NOT PPO / STAGE2 INCOMPLETE'
STEPS=2400;DT=.02;FPS=25

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def tree_hashes(path):
    path=Path(path);files=sorted(path.rglob('*'))
    if any(p.is_symlink() for p in files):raise ValueError('No asset/source symlink substitution')
    return {str(p.relative_to(path)):digest(p) for p in files if p.is_file()}
def verify_inputs(args):
    source=args.source_root.resolve();mapping=source/'campaign_source_hashes.json'
    if digest(mapping)!=SOURCE_MANIFEST_SHA256:raise ValueError('Exact source009 manifest required')
    expected=json.loads(mapping.read_text());actual=tree_hashes(source);actual.pop('campaign_source_hashes.json',None)
    if actual!=expected:raise ValueError('Frozen source009 tree changed')
    for path,sha in [(args.campaign,CAMPAIGN_SHA256),(args.admission,ADMISSION_SHA256),(args.study_tree_receipt,STUDY_TREE_SHA256)]:
        if digest(path)!=sha:raise ValueError('Exact completed009 evidence required: '+str(path))
    root=args.campaign.resolve().parent
    if args.admission.resolve()!=root/'standing/admission.json' or args.study_tree_receipt.resolve()!=root/'inputs/study_before.sha256.json':raise ValueError('Standing and asset receipts must belong to the completed009 campaign')
    if digest(root/'wave/state.json')!=WAVE_STATE_SHA256:raise ValueError('Completed bounded009 wave receipt mismatch')
    campaign=json.loads(args.campaign.read_text());state=json.loads((root/'wave/state.json').read_text())
    if campaign.get('status')!='completed' or not campaign.get('bounded_wave_physics_passed') or campaign.get('source_manifest_sha256')!=SOURCE_MANIFEST_SHA256 or not campaign.get('source_unchanged') or not campaign.get('admitted_asset_unchanged') or state.get('status')!='completed' or not state.get('gate',{}).get('passed'):raise ValueError('Completed exact-source009 admission required')
    assets=json.loads(args.study_tree_receipt.read_text())
    if len(assets)!=550 or tree_hashes(args.package)!=assets:raise ValueError('Complete550-file admitted package mismatch')
    if digest(args.geometry_reference)!=campaign['identity']['geometry_reference_sha256']:raise ValueError('Exact admitted geometry reference required')
    return {'source_manifest_sha256':SOURCE_MANIFEST_SHA256,'campaign_sha256':CAMPAIGN_SHA256,'standing_admission_sha256':ADMISSION_SHA256,'original_wave_state_sha256':WAVE_STATE_SHA256,'study_tree_sha256':STUDY_TREE_SHA256,'asset_files':550,'identity':campaign['identity']}
def load_frozen_functions(source,namespace):
    """Compile only unchanged function definitions; never execute source CLI."""
    file=Path(source)/'tools/run_reference_physics.py';tree=ast.parse(file.read_text())
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('main','serializable')]
    if [f.name for f in functions]!=['serializable','main']:raise ValueError('Unexpected frozen entrypoint function layout')
    module=ast.Module(body=functions,type_ignores=[])
    exec(compile(module,str(file),'exec'),namespace)
    return {f.name:hashlib.sha256(ast.dump(f,include_attributes=False).encode()).hexdigest() for f in functions}
def external_hashes(root):return {p.name:digest(p) for p in sorted(Path(root).glob('*.py'))}
def segment(step):
    if step<100:return 'CANONICAL TARGET STARTUP',0.
    if step<200:return 'QUIET SETTLE',0.
    if step<1400:return 'FORWARD',.005
    return 'STOP AND QUIET HOLD',0.
