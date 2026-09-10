"""Explicit fresh full-C standing→wave trial with source007 solver settings.

Retains TGS16/1 and external-force timingTrue; binds reviewed horizontal swing
completion at80percent. Existing quiet/physical gates remain; no production/PPO.
"""
BASELINE_SOURCE_SHA='666571ed6e37a73b857179323573ae20cdef0eee53aa25908108bd4e29465465'
PROTOCOL={
    'name':'reference008_fresh_standing_then_wave_TGS16_1_externalforces_true_horizontal80',
    'scope':'New full-C zero-residual bounded reference trial; not production adoption or PPO',
    'baseline_source_manifest_sha256':BASELINE_SOURCE_SHA,
    'baseline_run':'reference_physics_007',
    'physics_change_from007':None,
    'retained_solver':{'solver':'TGS','position_iterations':16,'velocity_iterations':1,
        'external_forces_every_iteration':True,'stabilization':False},
    'source007_standing_only_admission_reusable':False,
    'native_tensor_iteration_getter_status':'Unverified/unavailable in inspected source; no guessed getter or internal iteration-count claim',
    'phase_contract':{'standing':{'num_envs':32,'steps':1000},'wave':{'num_envs':1,'steps':2400},
        'wave_requires_fresh_exact_source_standing_admission':True,
        'all32_standing_must_pass_existing_quiet_gates':True},
    'wave_dependency':{'owner':'omni_reference_wave_004',
        'owner_freeze_sha256':'dedd5a2deb21703fe1d8713fcc2c276339a58db3f14b9ab85fb65a3f2ee7e834',
        'wave_reference_py_sha256':'a32b22eba03ff6fdd6b87d612c96173c519c7d2b212cb367a9b1f8f28fcd697d',
        'swing_duration_s':2.0,'lift_m':.007,'horizontal_duration_fraction':.8,
        'serialized_swing_fields':['horizontal_duration_fraction','horizontal_coefficients'],
        'actor_checkpoint_compatible':False},
    'standing_only':False,'production_default_adopted':False,'PPO_permitted':False,
    'legacy_50Hz_metric_and_5mm_gate_unchanged':True,
    'all_existing_physical_contact_torque_and_quiet_gates_retained':True,
    'comparison_with_last_wave004':{
        'interventions':['TGS external-force timingFalse toTrue and velocity iterations4 to1',
                         'Wave horizontal completion100percent to80percent of the2s swing'],
        'no_causal_attribution_to_single_intervention':True},
    'predeclared_measurement':'Retain original50Hz progress/quiet criteria and independent400Hz pose,velocity,torque diagnostics; no substituted metric.'}


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
