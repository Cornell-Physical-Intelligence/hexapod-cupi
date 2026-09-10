"""Explicit standing-only velocity-iteration diagnostic, not a production solver.

Relative to source005, external-force timing staysTrue and articulation velocity
iterations alone change4 to1; original standing and quiet gates are preserved.
"""
BASELINE_SOURCE_SHA='c557b71636c8b1e10883ab8a6b2b49f40b661988f9b09faeaefe4821a72844b4'
PROTOCOL={
    'name':'reference007_standing_TGS_velocity_iterations_one_readback_correction',
    'baseline_source_manifest_sha256':BASELINE_SOURCE_SHA,
    'baseline_run':'reference_physics_005',
    'only_physics_change':'robot.spawn.articulation_props.solver_velocity_iteration_count',
    'baseline_value':4,'comparison_value':1,
    'same_physics_as_source006':True,
    'native_tensor_iteration_getter_status':'Unverified/unavailable in inspected source; no guessed getter or internal iteration-count claim',
    'source006_retry_reason':'Articulation links have no independent solver iteration counts; absent per-link attributes are not unknown root settings',
    'inherited_external_forces_every_iteration':True,
    'unchanged_position_iterations':16,
    'standing_only':True,'num_envs':32,'control_steps':1000,
    'production_default_adopted':False,'automatic_wave_or_PPO':False,
    'legacy_50Hz_metric_and_5mm_gate_unchanged':True,
    'predeclared_comparison':{
        'interval_control_boundaries':[200,1000],
        'duration_s':16.0,
        'all32envs_individually':True,
        'frames':['root_link','root_COM'],
        'velocity_integrals':['400Hz_left','400Hz_right','400Hz_trapezoid','50Hz_last_substep_right'],
        'report':['world_pose_displacement','integrated_velocity','3D_difference_norm','forward_displacement',
                  'per_joint_400Hz_computed_and_applied_torque_extrema','postsettle_contact_and_reset_gates',
                  'unchanged_50Hz_quiet_metrics_all32','joint_position_delta_vs_reported_velocity_bias_all32x18'],
        '5mm_difference_reported_as_diagnostic_not_new_standing_gate':True,
        'retain_every_existing_standing_gate':True,
        'interpretation':'Compare every matched env to005; retain004 context; no metric chosen for a favorable result and no default adoption.'}}


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
    return {'baseline005':before,'comparison006':{'position':properties.solver_position_iteration_count,'velocity':properties.solver_velocity_iteration_count}}


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
