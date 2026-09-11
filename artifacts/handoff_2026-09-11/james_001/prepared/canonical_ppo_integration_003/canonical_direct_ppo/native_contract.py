"""Standard-library source/admission contract for the new canonical two-update smoke."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json,math
from .binding_contract import sha,read,verify_directory,verify_admissions,bind_consumer_lineage
from .smoke_config import SCHEMA,protocol
from .native_entry_adapter import STANDING_FREEZE
from . import source005_solver_diagnostics as diagnostics
from .solver_evidence import validate_solver,validate_diagnostic_members


def verify_inputs(args):
    own=Path(__file__).resolve().parent.parent
    own_digest=sha(own/'FREEZE_SHA256.json');verify_directory(own,own_digest)
    bindings=read(args.bindings)
    lineage=bind_consumer_lineage(verify_admissions(args.standing_source,args.standing_one,args.standing32,bindings))
    if lineage['standing_source_freeze_sha256']!=STANDING_FREEZE:raise ValueError('Native composition requires exact reviewed standing source')
    root=Path(args.standing_source).resolve()
    spec=importlib.util.spec_from_file_location('_canonical_native_smoke_contract',root/'standing_contract.py')
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    if sha(root/'solver_diagnostics.py')!=sha(own/'canonical_direct_ppo/source005_solver_diagnostics.py'):raise ValueError('Consumer diagnostic protocol differs from exact standing source')
    underlying=native.verify_inputs(SimpleNamespace(asset=Path(args.asset),admission=Path(args.admission),output=Path(args.output),num_envs=32,standing_one=Path(args.standing_one)))
    one=read(Path(args.standing_one)/'state.json')['identity'];many=read(Path(args.standing32)/'state.json')['identity']
    for field in ['actuation_state_sha256','actuation_inventory_sha256','asset_manifest_sha256']:
        if underlying[field]!=one[field]or underlying[field]!=many[field]:raise ValueError('Fresh setup differs from exact standing inputs:'+field)
    output=Path(args.output).resolve()
    for path in [own,args.standing_source,args.standing_one,args.standing32,args.asset,args.admission,args.bindings]:
        source=Path(path).resolve()
        if output==source or output.is_relative_to(source)or source.is_relative_to(output):raise ValueError('Output overlaps immutable input')
    return {'schema':SCHEMA,'phase':'canonical_ppo_smoke','runtime_binding':{'runtime_tree_sha256':own_digest,'scope':'fresh_canonical_405_408_two_update_smoke'},
        'policy_lineage':lineage,'native_identity':underlying,'bindings_sha256':sha(args.bindings),'protocol':protocol(),
        'fresh_neutral_controls':1000,'fresh_neutral_substeps':8000,'policy_controls':48,'policy_substeps':384,
        'total_controls':1048,'total_substeps':8384,'solver_recipe':{'position_iterations':32,'velocity_iterations':0},'solver_diagnostics':diagnostics.declaration(32,8384),'training_integration_only':True,'quality_admitted':False,'Stage2_complete':False}


def finish(output,state,args,identity,app):
    # Same pre-close measurement sealing/late-log separation as canonical native source.
    from .native_entry_adapter import exact_helper_imports
    with exact_helper_imports(args.standing_source):
        from native_support import save
        try:
            try:
                state['inputs_unchanged']=verify_inputs(args)==identity
                if not state['inputs_unchanged']:raise ValueError('Terminal input identity changed')
            except BaseException as error:
                state['status']='failed';state['inputs_unchanged']=False;state.setdefault('errors',[]).append(repr(error))
            state['outputs']={str(p.relative_to(output)):sha(p)for p in Path(output).rglob('*')if p.is_file()
                and 'isaac_logs'not in p.relative_to(output).parts and p!=Path(output)/'state.json'
                and p.name not in ['failure.json','native_errors.json']and not p.name.endswith('.part')}
            state['unsealed_lifecycle_logs']=['isaac_logs/**','native_errors.json']
            state['log_inventory_owner']='Host inventories all logs after process exit; no pre-close complete-log claim.'
            save(Path(output)/'state.json',state)
            if state['status']!='completed':save(Path(output)/'failure.json',{'status':'failed','errors':state.get('errors',[])})
        finally:
            if app is not None:app.close()


def validate_result(directory,identity):
    d=Path(directory);state=read(d/'state.json')
    if state.get('schema')!=SCHEMA or state.get('identity')!=identity or identity.get('schema')!=SCHEMA:raise ValueError('Wrong canonical PPO result identity')
    if state.get('status')!='completed'or state.get('inputs_unchanged')is not True or state.get('errors')!=[]:raise ValueError('Incomplete/failed canonical PPO smoke')
    if state.get('native_error_events')!=[]or read(d/'native_errors.json')!=[]:raise ValueError('Actual late native error')
    if state.get('explicit_steps_completed')!=8384:raise ValueError('Incomplete exact native steps')
    if not state.get('checks')or not all(v is True for v in state['checks'].values()):raise ValueError('Native setup checks rejected')
    checks={'sdk_source_bound','usd_identity','native_scene','native_materials_offsets','all_native_inertias_limits_no_drives',
        'all_native_sdf_paths','all_reset_coordinate_frames','fresh_neutral_prefix_admitted','learner_2_updates_strict_reload','neutral_solver_recipe_readback','solver_recipe_readback','legacy_friction_observed'}
    if not checks.issubset(state['checks']):raise ValueError('Incomplete native setup/admission checks')
    if state.get('quality_admitted')is not False or state.get('Stage2_complete')is not False:raise ValueError('Integration cannot claim walking/quiet quality')
    required={'session.json','control_trace.npz','contacts.jsonl','initial_reset.json','native_readback.json','sdf_readback.json',
        'entry_composition.json','solver_readback.json','legacy_friction_readback.json','contact_view.json',
        'neutral_prefix/SHA256.json','neutral_prefix/standing_report.json','neutral_prefix/session.json','neutral_prefix/solver_readback.json','neutral_prefix/legacy_friction_readback.json','neutral_prefix/native_readback.json','neutral_prefix/contact_view.json','neutral_prefix/contacts.jsonl','neutral_prefix/control_trace.npz',
        'learner/state.json','learner/policy_control_trace.pt','learner/decision_001.pt','learner/decision_002.pt'}
    if not required.issubset(state.get('outputs',{})):raise ValueError('Required raw evidence was not sealed')
    for name,digest in state['outputs'].items():
        p=d/name
        if p.is_symlink()or not p.resolve().is_relative_to(d.resolve())or sha(p)!=digest:raise ValueError('Changed measured output:'+name)
    if identity.get('solver_recipe')!={'position_iterations':32,'velocity_iterations':0}or identity.get('solver_diagnostics')!=diagnostics.declaration(32,8384):raise ValueError('Wrong solver/diagnostic composition identity')
    solver=read(d/'solver_readback.json');neutral_solver=read(d/'neutral_prefix/solver_readback.json')
    validate_solver(solver,32,final=True);validate_solver(neutral_solver,32,final=False)
    if any(solver[k]!=neutral_solver[k]for k in ['before','after_authoring','after_reset','after_neutral_steps']):raise ValueError('Solver neutral prefix was not preserved')
    session=read(d/'session.json');learner=read(d/'learner/state.json');prefix=read(d/'neutral_prefix/standing_report.json')
    neutral_session=read(d/'neutral_prefix/session.json')
    for where,current,steps in [(d,session,8384),(d/'neutral_prefix',neutral_session,8000)]:
        if current.get('solver_diagnostics')!=diagnostics.declaration(32,steps):raise ValueError('Incomplete diagnostic channel declaration')
        diagnostics.validate_legacy(read(where/'legacy_friction_readback.json'),32,read(where/'native_readback.json')['joint_names'])
        validate_diagnostic_members(where,current)
    for name in ['legacy_friction_readback.json','native_readback.json','contact_view.json']:
        if sha(d/name)!=sha(d/'neutral_prefix'/name):raise ValueError('Changed native prefix readback:'+name)
    if neutral_session.get('steps')!=8000 or neutral_session.get('captured_steps')!=8000 or neutral_session.get('controls')!=1000 or neutral_session.get('reset_count')!=1 or neutral_session.get('failure')is not None or neutral_session.get('all_rows_recorded')is not True:raise ValueError('Incomplete neutral clock/reset declaration')
    if not session.get('substep_files')or any(n not in state['outputs']for n in session['substep_files']):raise ValueError('Missing native substeps')
    prefix_map=read(d/'neutral_prefix/SHA256.json')
    if not {'session.json','standing_report.json','solver_readback.json','legacy_friction_readback.json','native_readback.json','contact_view.json','contacts.jsonl','control_trace.npz'}.issubset(prefix_map)or not neutral_session.get('substep_files')or any(name not in prefix_map for name in neutral_session['substep_files']):raise ValueError('Incomplete immutable neutral prefix inventory')
    for name,digest in prefix_map.items():
        if state['outputs'].get('neutral_prefix/'+name)!=digest:raise ValueError('Changed immutable neutral prefix')
    for name,digest in learner.get('outputs',{}).items():
        if state['outputs'].get('learner/'+name)!=digest:raise ValueError('Changed learner sealed outputs')
    if learner.get('lineage')!=identity['policy_lineage']:raise ValueError('Actual learner lineage differs')
    if session.get('steps')!=8384 or session.get('captured_steps')!=8384 or session.get('controls')!=1048 or session.get('reset_count')!=1 or session.get('failure')is not None or session.get('all_rows_recorded')is not True:raise ValueError('Native clock/reset/prefix incomplete')
    if prefix.get('all_pass')is not True or prefix.get('num_envs')!=32 or len(prefix.get('replicas',[]))!=32 or any(r.get('pass')is not True for r in prefix['replicas']):raise ValueError('Fresh neutral prefix failed')
    if learner.get('status')!='completed'or learner.get('updates_completed')!=2 or learner.get('controls_attempted')!=48 or learner.get('controls_completed')!=48 or learner.get('optimizer_steps_completed')!=40 or learner.get('transitions')!=1536:raise ValueError('Incomplete actual learner updates')
    if learner.get('errors')!=[]or learner.get('quality_admitted')is not False or len(learner.get('updates',[]))!=2:raise ValueError('Learner error/quality mismatch')
    for i,row in enumerate(learner['updates'],1):
        reload=row.get('reload',{});actual=sha(d/'learner'/f'decision_{i:03d}.pt')
        if row.get('completed_update')!=i or row.get('optimizer_minibatches')!=20:raise ValueError('False completed update/minibatch count')
        lr=row.get('learning_rate');losses=row.get('losses')
        if type(lr)not in(int,float)or not math.isfinite(lr)or lr<=0 or not isinstance(losses,dict)or not losses or any(type(v)not in(int,float)or not math.isfinite(v)for v in losses.values()):raise ValueError('Invalid learning-rate/loss diagnostics')
        for key in ['passed','strict_model_normalizer_optimizer','deterministic_action_exact','runtime_state_exact','learning_rate_exact']:
            if reload.get(key)is not True:raise ValueError('Strict checkpoint reload incomplete:'+key)
        if type(reload.get('optimizer_entries'))is not int or reload['optimizer_entries']<=0:raise ValueError('Missing actual Adam state')
        if row.get('checkpoint_sha256')!=actual or reload.get('checkpoint_sha256')!=actual:raise ValueError('Checkpoint/reload hash differs')
    audit=learner['native_audit']
    if audit.get('failure')is not None or audit.get('policy_substeps_verified')!=384 or audit.get('policy_controls')!=48 or audit.get('policy_attempts')!=48:raise ValueError('Incomplete per-substep policy audit')
    counts=audit.get('requested_saturation_counts')
    if not isinstance(counts,list)or len(counts)!=32 or any(not isinstance(row,list)or len(row)!=18 or any(type(v)is not int or not 0<=v<=384*.005 for v in row)for row in counts):raise ValueError('Full400Hz requested saturation contradicts declared bound')
    return {'phase':'canonical_ppo_smoke','status':'completed','updates_completed':2,'controls':48,'transitions':1536,
            'quality_admitted':False,'Stage2_complete':False,'state_sha256':sha(d/'state.json')}
