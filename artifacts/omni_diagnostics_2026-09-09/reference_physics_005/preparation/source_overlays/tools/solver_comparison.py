"""Explicit one-option standing diagnostic; not an adopted production solver.

Only enable_external_forces_every_iteration changes from source004.
"""
BASELINE_SOURCE_SHA='a433e529d29d5360c828b406a3dfd769e078d69fc9deaf03f4fe110eb6fa7a63'
PROTOCOL={
    'name':'reference005_standing_TGS_external_forces_each_iteration',
    'baseline_source_manifest_sha256':BASELINE_SOURCE_SHA,
    'baseline_run':'reference_physics_004',
    'only_physics_change':'sim.physics.enable_external_forces_every_iteration',
    'baseline_value':False,'comparison_value':True,
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
                  'per_joint_400Hz_computed_and_applied_torque_extrema','postsettle_contact_and_reset_gates'],
        '5mm_difference_reported_as_diagnostic_not_new_standing_gate':True,
        'retain_every_existing_standing_gate':True,
        'interpretation':'Compare every matched env to004; no metric chosen for a favorable result and no default adoption.'}}


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
    return {'protocol':PROTOCOL,'physics_config_before':before,'physics_config_after':after}


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
            'readback_scope':'Authored resolved PhysX scene attributes after environment initialization; native solver internals not instrumented'}
