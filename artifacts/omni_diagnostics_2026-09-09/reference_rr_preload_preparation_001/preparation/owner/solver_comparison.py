"""Explicit fresh full-C standing→wave trial with source008 solver settings.

Retains TGS16/1 and external-force timingTrue; binds qualified liftoff and
actual400Hz joint-angle observation. Existing quiet/physical gates remain; no production/PPO.
"""
BASELINE_SOURCE_SHA='a73075ebdf5d6c986728b96117495b8de85adc8db62de1c41ffbe49eb6176d66'
PROTOCOL={
    'name':'reference_rr_preload001_retained_reference009_solver',
    'parent_solver_protocol_name':'reference009_fresh_standing_then_qualified_liftoff_wave_with_joint_angle_observer',
    'scope':'New full-C zero-residual bounded reference trial; not production adoption or PPO',
    'baseline_source_manifest_sha256':BASELINE_SOURCE_SHA,
    'baseline_run':'reference_physics_008',
    'physics_change_from008':None,
    'retained_solver':{'solver':'TGS','position_iterations':16,'velocity_iterations':1,
        'external_forces_every_iteration':True,'stabilization':False},
    'previous_source_admission_reusable':False,
    'native_tensor_iteration_getter_status':'Unverified/unavailable in inspected source; no guessed getter or internal iteration-count claim',
    'phase_contract':{'standing':{'num_envs':32,'steps':1000},'wave':{'num_envs':1,'steps':2400},
        'wave_requires_fresh_exact_source_standing_admission':True,
        'all32_standing_must_pass_existing_quiet_gates':True},
    'wave_dependency':{'owner':'reference_rr_preload_diagnostic_001',
        'base_owner':'omni_reference_wave_005',
        'base_owner_freeze_sha256':'5b6c076cb03426ac8e24cc6df7e8684e48862aa78a495bd61ae49773a7826814',
        'base_wave_reference_py_sha256':'8bacf0c61f7fdc6d43d9f9ebdb5adc643b026495bb303ec3c18157d121c78893',
        'wave_reference_py_sha256':'dbc0b046a9a6b46fae43ac8a3f01ce9162860a6ccb9deac54193f342671255e4',
        'post_landing_diagnostic':'RR only first measured landing,0.5mm downward C2 over existing0.3s hold; all physics settings unchanged',
        'swing_duration_s':2.0,'lift_m':.007,'horizontal_duration_fraction':.8,
        'serialized_swing_fields':['horizontal_duration_fraction','horizontal_coefficients'],
        'qualified_liftoff_schema':'unloading until two consecutive force-free samples and actual2mm lift; fail at original apex if unqualified',
        'actor_checkpoint_compatible':False},
    'standing_only':False,'production_default_adopted':False,'PPO_permitted':False,
    'legacy_50Hz_metric_and_5mm_gate_unchanged':True,
    'all_existing_physical_contact_torque_and_quiet_gates_retained':True,
    'comparison_with_last_wave008':{
        'interventions':['Explicit unloading versus qualified flight before bounded landing',
                         'Read-only actual400Hz joint-angle observation and endpoint parity'],
        'solver_geometry_targets_and_lift_timing_unchanged':True},
    'joint_angle_observer':{'owner_freeze_sha256':'1c3a736017afeb810261a734558c7a4a904dd9010d0fdae09d68ef32efb80415',
        'runtime_sha256':'8d52e0f56e3671a18c194b3f5d623cd95ca9187c5d58289001a74868c8d0c091',
        'new_raw_field':'joint_position_rad','joint_names_stable':True,
        'original_physics_updates_only':True,'control_endpoint_q_and_qdot_exact':True},
    'predeclared_measurement':'Retain original50Hz gates and independent400Hz pose/velocity/torque; add actualjointangle increments versus reportedrates without substituting metrics.'}



def configure_comparison(cfg):
    physics=cfg.sim.physics
    before=physics.to_dict()
    if before.get('solver_type')!=1 or before.get('enable_stabilization') is not False:
        raise ValueError('Comparison requires unchanged TGS and stabilization disabled')
    if before.get('enable_external_forces_every_iteration') is not False:
        raise ValueError('Parent external-force option must be exactlyFalse')
    physics.enable_external_forces_every_iteration=True
    after=physics.to_dict()
    if set(after)!=set(before) or [k for k in before if before[k]!=after[k]]!=['enable_external_forces_every_iteration']:
        raise ValueError('Only the declared external-force option may change')
    return {'protocol':PROTOCOL,'constructor_default_physics_config':before,
            'inherited005_physics_config':after,
            'note':'Setting external-force timingTrue reproduces005; it is not the new physical intervention.'}


def readback_comparison(env,receipt):
    """Read authored actual scene attributes after construction; no physics writes."""
    expected={'physxScene:solverType':'TGS','physxScene:enableStabilization':False,
              'physxScene:enableExternalForcesEveryIteration':True}
    prim=env.sim.stage.GetPrimAtPath(env.cfg.sim.physics_prim_path)
    if not prim.IsValid():raise RuntimeError('Missing actual physics scene')
    values={}
    for name,value in expected.items():
        attr=prim.GetAttribute(name)
        if not attr.IsValid():raise RuntimeError('Missing solver attribute '+name)
        values[name]=attr.Get()
        if values[name]!=value:raise RuntimeError('Actual solver readback differs: '+name)
    return {**receipt,'scene_path':env.cfg.sim.physics_prim_path,'actual_scene_attributes':values,
            'actual_iteration_settings':iteration_scene_readback(env),
            'readback_scope':'Authored resolved PhysX scene attributes after environment initialization; native solver internals not instrumented'}


def configure_iteration_comparison(cfg):
    properties=cfg.robot.spawn.articulation_props
    before={'position':properties.solver_position_iteration_count,'velocity':properties.solver_velocity_iteration_count}
    if before!={'position':16,'velocity':4}:raise ValueError('Exact source005 articulation16/4 required')
    if cfg.sim.physics.enable_external_forces_every_iteration is not True:raise ValueError('Source005 external-force setting must remainTrue')
    properties.solver_velocity_iteration_count=1
    return {'constructor_articulation':before,'retained007_settings':{'position':properties.solver_position_iteration_count,'velocity':properties.solver_velocity_iteration_count}}


def iteration_scene_readback(env):
    """Read actual authored/fallback attributes on every cloned robot body/root."""
    rows=[]
    for prim in env.sim.stage.Traverse():
        path=str(prim.GetPath())
        if not path.startswith('/World/envs/env_') or '/Robot' not in path:continue
        schemas=set(prim.GetAppliedSchemas())
        for schema,kind,namespace in [('PhysicsArticulationRootAPI','articulation','physxArticulation'),('PhysicsRigidBodyAPI','body','physxRigidBody')]:
            if schema not in schemas:continue
            row={'path':path,'kind':kind}
            for key,suffix in [('position','solverPositionIterationCount'),('velocity','solverVelocityIterationCount')]:
                attr=prim.GetAttribute(namespace+':'+suffix)
                row[key]=attr.Get() if attr.IsValid() else None
                row[key+'_authored']=bool(attr.IsValid() and attr.HasAuthoredValueOpinion())
            rows.append(row)
    result={'rows':rows,'articulation_count':sum(x['kind']=='articulation' for x in rows),
            'body_count':sum(x['kind']=='body' for x in rows),
            'scope':'Resolved authored/fallback USD iteration attributes; no claim of internal native per-island counters'}
    try:
        validate_iteration_readback(rows,env.num_envs)
    except Exception as error:
        error.iteration_readback=result
        raise
    return result


def validate_iteration_readback(rows,num_envs):
    roots=[r for r in rows if r['kind']=='articulation'];bodies=[r for r in rows if r['kind']=='body']
    if len(roots)!=num_envs or len(bodies)!=19*num_envs:raise RuntimeError('Actual32 articulations/608body iteration readback required')
    for row in roots:
        if row['position']!=16 or row['velocity']!=1:raise RuntimeError('Actual articulation is not16/1: '+repr(row))
        if row.get('position_authored') is not True or row.get('velocity_authored') is not True:
            raise RuntimeError('Explicitly authored articulation16/1 required: '+repr(row))
    for row in bodies:
        owners=[root for root in roots if row['path']==root['path'] or row['path'].startswith(root['path']+'/')]
        if len(owners)!=1:raise RuntimeError('Rigid body is not within one proven articulation: '+row['path'])
        row['articulation_owner_path']=owners[0]['path']
        row['solver_iteration_authority']='articulation_root_not_individual_link'
        absent=[]
        for key,bound in [('position',16),('velocity',1)]:
            value=row[key]
            if value is None and row.get(key+'_authored') is False:
                absent.append(key)
            elif type(value) is not int or not 0<=value<=bound:
                raise RuntimeError('Conflicting/invalid present per-link iteration metadata: '+repr(row))
        row['absent_unauthored_link_attributes']=absent
        row['metadata_scope']='Absent link metadata is not a zero/unknown articulation count; root16/1 remains mandatory.'
