"""Stdlib contract: acquire canonical SDF evidence, never admit contact or training."""
from pathlib import Path
import importlib.util
_spec=importlib.util.spec_from_file_location('_canonical_sdf_query_parent_contract',Path(__file__).resolve().with_name('inspection_contract.py'))
parent=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(parent)
read=parent.read;sha=parent.sha;check_map=parent.check_map
DT=parent.DT;STEPS=parent.STEPS
SCHEMA='canonical_native_sdf_query_v1'
PARENT_FREEZE='1d636f6b6909171e366be7b08b3590c0a65a8a9206e57b76792d3f4bcd166cd8'
PROBE_FREEZE='e1109fc90dc3953bcb064fd0ab80158503ff0d48a3457453d504c5aa0e23e367'
PHASE='query'


def verify_admission(directory):
    own=Path(__file__).resolve().parent;directory=Path(directory).resolve()
    expected=read(own/'PHASE_A_SHA256.json');check_map(directory,expected)
    if {str(p.relative_to(directory))for p in directory.rglob('*')if p.is_file()}!=set(expected):raise ValueError('Changed phase-A inventory')
    state=read(directory/'state.json');parent.validate_result(directory,state['identity'])
    if state['identity']['inspector_freeze_sha256']!=PARENT_FREEZE:raise ValueError('Wrong canonical import admission')
    return {'phase_a_state_sha256':sha(directory/'state.json'),'phase_a_manifest_sha256':sha(own/'PHASE_A_SHA256.json')}


def verify_inputs(args):
    own=Path(__file__).resolve().parent
    freeze=read(own/'FREEZE_SHA256.json');check_map(own,freeze)
    actual={str(p.relative_to(own))for p in own.rglob('*')if p.is_file()and '__pycache__'not in p.parts and p!=own/'FREEZE_SHA256.json'}
    if actual!=set(freeze):raise ValueError('Changed query source inventory')
    if sha(own/'probe/FREEZE_SHA256.json')!=PROBE_FREEZE:raise ValueError('Changed query addon lineage')
    if sha(own/'ASSET_SHA256.json')!=parent.ASSET_MAP_SHA256:raise ValueError('Changed canonical asset map')
    asset=Path(args.asset).resolve();expected=read(own/'ASSET_SHA256.json');check_map(asset,expected)
    if {str(p.relative_to(asset))for p in asset.rglob('*')if p.is_file()}!=set(expected):raise ValueError('Changed canonical asset inventory')
    admission=Path(args.admission).resolve();prior=verify_admission(admission)
    output=Path(args.output).resolve()
    if any(output.is_relative_to(p)for p in [own,asset,admission]):raise ValueError('Output overlaps immutable inputs')
    return {'schema':SCHEMA,'inspector_freeze_sha256':sha(own/'FREEZE_SHA256.json'),
            'runtime_binding':{'runtime_tree_sha256':sha(own/'FREEZE_SHA256.json'),'scope':'canonical_native_sdf_query_only'},
            'asset_manifest_sha256':parent.ASSET_MAP_SHA256,'urdf_sha256':expected['source/source.urdf'],
            'model_sha256':expected['source/model.json'],'usd_sha256':expected['robot.usda'],
            'steps':STEPS,'dt':DT,'gravity':[0.,0.,0.],'ground':False,'query_calls':12,'points_per_call':140,
            'sdf_inspection_contract':'canonical_native_sdf_initialization_v2','probe_freeze_sha256':PROBE_FREEZE,
            'query_schema':'canonical_sdf_distance_diagnostic_v1',
            'physics_admitted':False,'physical_admission':False,'training_allowed':False,**prior}


def validate_result(directory,identity):
    directory=Path(directory);s=read(directory/'state.json')
    if s.get('schema')!=SCHEMA or s.get('identity')!=identity:raise ValueError('Wrong query identity')
    if s.get('status')!='completed'or s.get('inputs_unchanged')is not True:raise ValueError('Failed/incomplete query acquisition')
    if s.get('errors')!=[]or s.get('native_error_events')!=[]or read(directory/'native_errors.json')!=[]:raise ValueError('Native or late runtime error')
    if any(s.get(k)is not False for k in ['physics_admitted','physical_admission','training_allowed']):raise ValueError('Query cannot admit physics or learning')
    required={'usd_identity','native_identity','native_frames','native_scene','native_sdf_paths','no_drive_gains','finite_samples','sdk_source_bound','sdf_query_acquisition'}
    if not required.issubset(s.get('checks',{}))or not all(v is True for v in s['checks'].values())or s.get('explicit_steps_completed')!=STEPS:raise ValueError('Missing native import/query checks')
    outputs=s.get('outputs',{});check_map(directory,outputs)
    expected={'usd_readback.json','native_readback.json','samples.json','sdf_readback.json','resolved_stage.usda','runtime_api.json','query/state.json','query/report.json','query/before.json','query/after.json','query/frame_hypotheses.json'}
    expected|={f'query/{kind}_{k}.{suffix}'for k in range(6)for kind,suffix in [('query','npz'),('query','json'),('view','json')]}
    if not expected.issubset(outputs):raise ValueError('Missing sealed query raw evidence')
    q=read(directory/'query/state.json')
    if s.get('sdf_query_state_sha256')!=sha(directory/'query/state.json')or q.get('status')!='completed'or q.get('errors')!=[]or q.get('inputs_unchanged')is not True or q.get('queries_completed')!=12:raise ValueError('Failed or changed addon result')
    if q['identity'].get('probe_freeze_sha256')!=PROBE_FREEZE:raise ValueError('Wrong actual addon source')
    check_map(directory/'query',q['outputs'])
    actual_query={p.name for p in (directory/'query').iterdir()if p.is_file()and p.name!='state.json'}
    if actual_query!=set(q['outputs']):raise ValueError('Extra or missing query output')
    before=read(directory/'query/before.json');after=read(directory/'query/after.json');prefix=read(directory/'samples.json')[-1]
    if before!=after or before['explicit_counter']!=prefix['explicit_step_counter']:raise ValueError('Query changed observed state/clock')
    for key,parent_key in [('link_pose','link_pose_xyzw'),('link_com_velocity','link_com_velocity'),('joint_position','joint_position'),('joint_velocity','joint_velocity_sdk')]:
        if before[key]!=prefix[parent_key]:raise ValueError('Query boundary differs from import prefix:'+key)
    report=read(directory/'query/report.json')
    if any(report.get(k)is not False for k in ['standing_admitted','contact_admitted','training_allowed']):raise ValueError('Distance query cannot admit support')
    diagnostics={k:report.get(k)for k in ['declared_semantics_supported','geometry_accuracy_within_proposed_bounds']}
    if any(type(v)is not bool for v in diagnostics.values())or diagnostics!=s.get('query_diagnostics'):raise ValueError('Malformed query diagnostic verdicts')
    return {'phase':PHASE,'status':'completed','scope':'sdf_query_acquisition_only','physics_admitted':False,
            'physical_admission':False,'training_allowed':False,'state_sha256':sha(directory/'state.json'),
            'query_state_sha256':sha(directory/'query/state.json'),'explicit_steps':STEPS,'query_calls':12,**diagnostics}
