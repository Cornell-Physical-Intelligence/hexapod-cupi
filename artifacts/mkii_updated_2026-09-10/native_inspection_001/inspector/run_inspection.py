#!/usr/bin/env python3
"""One canonical zero-G import inspection. This is not a robot task or controller."""
from pathlib import Path
import argparse
import importlib
import inspect
import json
import math
import sys
import time
import traceback

from inspection_contract import DT, STEPS, SCHEMA, read, sha, verify_inputs


def save(path, value):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.part')
    # Preserve malformed numeric evidence as explicit JSON objects; validation still rejects it.
    def safe(v):
        if isinstance(v,float) and not math.isfinite(v):return {'nonfinite':repr(v)}
        if isinstance(v,dict):return {k:safe(x)for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [safe(x)for x in v]
        return v
    tmp.write_text(json.dumps(safe(value),indent=2,sort_keys=True,allow_nan=False)+'\n')
    tmp.replace(path)


def to_list(a):
    # Raw Warp getters return arrays; this copies their current values before stepping.
    if hasattr(a,'numpy'): a=a.numpy()
    return a.tolist()


def sdk_readback(output):
    expected=read(Path(__file__).parent/'SDK_READBACK.json')['entries']
    rows={}
    for rel,binding in expected.items():
        module='.'.join(rel.split('/')[1:])[:-3]
        obj=importlib.import_module(module)
        path=Path(inspect.getsourcefile(obj)).resolve()
        rows[module]={'path':str(path),'sha256':sha(path),'expected':binding['sha256']}
        save(output/'sdk_readback.json',rows)
        if rows[module]['sha256']!=binding['sha256']:
            raise ValueError(f'Installed SDK source mismatch: {module}')
    return rows


def collect_native(view, output):
    native={'count':view.count,'fixed_base':view.shared_metatype.fixed_base,
            'body_names':list(view.shared_metatype.link_names),'joint_names':list(view.shared_metatype.dof_names)}
    save(output/'native_readback.json',native)
    for key,getter in [('masses','get_masses'),('coms','get_coms'),('inertias','get_inertias'),
                       ('limits','get_dof_limits'),('stiffness','get_dof_stiffnesses'),
                       ('damping','get_dof_dampings'),('armature','get_dof_armatures'),
                       ('friction_properties','get_dof_friction_properties'),
                       ('native_max_effort','get_dof_max_forces'),('native_max_velocity','get_dof_max_velocities')]:
        try:native[key]=to_list(getattr(view,getter)())
        except Exception as e:
            native['failed_getter']={'name':getter,'error':repr(e)}
            save(output/'native_readback.json',native)
            raise
        save(output/'native_readback.json',native)
    return native


def snapshot(view, sim, explicit_step):
    sim.physics_sim_view.update_articulations_kinematic()
    return {'explicit_step':explicit_step, 'explicit_step_counter':sim.get_physics_step_count(),
            'link_pose_xyzw':to_list(view.get_link_transforms()),
            'link_com_velocity':to_list(view.get_link_velocities()),
            'joint_position':to_list(view.get_dof_positions()),
            'joint_velocity_sdk':to_list(view.get_dof_velocities())}


def finish(output, state, args, identity, app):
    """Seal all evidence before SimulationApp.close can terminate the process."""
    try:
        try:
            state['inputs_unchanged']=verify_inputs(args)==identity
            if not state['inputs_unchanged']:raise ValueError('Terminal identity changed')
        except Exception as e:
            state['inputs_unchanged']=False
            state['status']='failed';state.setdefault('errors',[]).append(repr(e))
        state['outputs']={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()
                          and 'isaac_logs' not in p.relative_to(output).parts
                          and p.name not in {'state.json','failure.json','native_errors.json'} and not p.name.endswith('.part')}
        state['unsealed_lifecycle_logs']=['isaac_logs/**','native_errors.json']
        state['log_inventory_owner']='Host inventories all logs after process exit; they are not falsely sealed before close.'
        save(output/'state.json',state)
        if state['status']!='completed':save(output/'failure.json',{'status':'failed','errors':state.get('errors',[])})
    finally:
        if app is not None: app.close()


def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--asset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    # Avoid importing Isaac in stdlib preflight, including full late launcher flags.
    pre,_=parser.parse_known_args(argv)
    identity=verify_inputs(pre)
    if pre.preflight_only:
        print(json.dumps(identity,sort_keys=True));return 0
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args(argv)
    if args.device!='cuda:0' or not args.headless:raise ValueError('Fixed native GPU headless inspection required')
    args.output.mkdir(parents=True,exist_ok=False)
    output=args.output; app=None; rows=[]
    state={'schema':SCHEMA,'identity':identity,'runtime_binding':identity['runtime_binding'],
           'status':'running','physics_admitted':False,'physical_admission':False,'training_allowed':False,
           'explicit_steps_completed':0,'checks':{},'errors':[]}
    save(output/'state.json',state)
    start=time.monotonic()
    try:
        app=AppLauncher(args).app
        print('REFERENCE_SCREEN_APP_READY',flush=True)
        import numpy as np
        from isaaclab.sim import SimulationContext, SimulationCfg
        from isaaclab_physx.physics import PhysxCfg
        import omni.physx
        import omni.kit.app
        from pxr import PhysxSchema,UsdPhysics
        from inspect_core import usd_readback,validate_usd,validate_native,validate_sdf,validate_samples,validate_native_frames,validate_scene
        api=sdk_readback(output)
        # Native schema availability and loaded extension paths are recorded, not inferred offline.
        extmanager=omni.kit.app.get_app().get_extension_manager()
        from isaacsim.core.version import get_version
        native_version=list(get_version())
        if str(native_version[0])!='6.0.1':raise ValueError(f'Wrong native Isaac Sim version: {native_version}')
        runtime={'sdk':api,'isaac_sim_version':native_version,
                 'physx_schema_python':str(PhysxSchema.__file__),
                 'physx_extension':extmanager.get_enabled_extension_id('omni.physx'),
                 'tensor_extension':extmanager.get_enabled_extension_id('omni.physics.tensors'),
                 'tensor_api_source':inspect.getsourcefile(importlib.import_module('omni.physics.tensors.impl.api')),
                 'warmup_scope':'SDK reset contains two update_simulation(dt,0) calls and play/app pump; explicit counter does not certify total warmup steps.'}
        runtime['tensor_api_sha256']=sha(runtime['tensor_api_source'])
        save(output/'runtime_api.json',runtime)
        state['checks']['sdk_source_bound']=True
        cfg=SimulationCfg(dt=DT,gravity=(0.,0.,0.),device=args.device,render_interval=8,
                          physics=PhysxCfg(),visualizer_cfgs=[],log_dir=str(output/'isaac_logs'))
        sim=SimulationContext(cfg)
        sim.stage.DefinePrim('/Robot','Xform').GetReferences().AddReference(str(args.asset.resolve()/'robot.usda'))
        model=read(args.asset/'source/model.json')
        usd=usd_readback(sim.stage)
        save(output/'usd_readback.json',usd)
        state['checks']['usd_identity']=validate_usd(usd,model)
        sim.stage.GetRootLayer().Export(str(output/'resolved_stage.usda'))
        # JSON-safe config values; class fields use their qualified repr, all numeric values retained.
        def clean(x):
            if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
            if isinstance(x,(list,tuple)):return [clean(v)for v in x]
            return x if x is None or isinstance(x,(str,int,float,bool)) else repr(x)
        save(output/'simulation_config.json',clean(cfg.to_dict()))
        errors=[]
        save(output/'native_errors.json',errors)
        def on_error(e):
            errors.append({'type':int(e.type),'payload':repr(e.payload)})
            save(output/'native_errors.json',errors)
        error_subscription=omni.physx.get_physx_interface().get_error_event_stream().create_subscription_to_pop(on_error)
        cooking=omni.physx.get_physx_cooking_interface()
        initstart=time.monotonic()
        sim.reset()
        scene_prim=sim.stage.GetPrimAtPath(cfg.physics_prim_path)
        scene_api=UsdPhysics.Scene(scene_prim)
        scene={'gravity_magnitude':scene_api.GetGravityMagnitudeAttr().Get(),
               'gravity_direction':list(scene_api.GetGravityDirectionAttr().Get()),
               'time_steps_per_second':scene_prim.GetAttribute('physxScene:timeStepsPerSecond').Get(),
               'manager_dt':sim.get_physics_dt(),
               'root_self_collisions':sim.stage.GetPrimAtPath('/Robot/body').GetAttribute('physxArticulation:enabledSelfCollisions').Get(),
               'attributes':{str(a.GetName()):clean(a.Get())for a in scene_prim.GetAttributes()}}
        save(output/'native_scene.json',scene)
        state['checks']['native_scene']=validate_scene(scene)
        # Required native getters themselves fail closed; no fallback to authored attributes.
        view=sim.physics_sim_view.create_articulation_view('/Robot/body')
        rows.append(snapshot(view,sim,0));save(output/'samples.json',rows)
        native=collect_native(view,output)
        state['checks']['native_identity']=validate_native(native,model)
        state['checks']['native_frames']=validate_native_frames(rows[0],native['body_names'],native['joint_names'],model)
        state['checks']['no_drive_gains']=True
        paths=[x['path']for x in usd['colliders']]
        sdfview=sim.physics_sim_view.create_sdf_shape_view(paths,1)
        sdf={'count':sdfview.count,'valid':bool(sdfview.check()),'paths':list(sdfview.object_paths),
             'pending_cooking_tasks':cooking.get_num_collision_tasks(),
             'scope':'Actual native SDF shape existence, not contact accuracy or geometric distance validation'}
        save(output/'sdf_readback.json',sdf)
        state['checks']['native_sdf_paths']=validate_sdf(sdf,usd)
        state['initialization_wall_s']=time.monotonic()-initstart
        for n in range(1,STEPS+1):
            sim.step(render=False)
            rows.append(snapshot(view,sim,n));save(output/'samples.json',rows)
            state['explicit_steps_completed']=n
            validate_native_frames(rows[-1],native['body_names'],native['joint_names'],model)
            if errors:raise RuntimeError('Native PhysX error event; raw event retained')
        state['checks']['finite_samples']=validate_samples(rows)
        sim.stage.GetRootLayer().Export(str(output/'resolved_after_stage.usda'))
        state['native_error_events']=errors
        import torch
        state['memory_bytes']={'allocated':torch.cuda.memory_allocated(),
                               'reserved':torch.cuda.memory_reserved(),
                               'peak_allocated':torch.cuda.max_memory_allocated(),
                               'free_total_at_end':list(torch.cuda.mem_get_info())}
        positions=np.asarray([r['joint_position'][0]for r in rows])
        sdk_rates=np.asarray([r['joint_velocity_sdk'][0]for r in rows])
        roots=np.asarray([r['link_pose_xyzw'][0][native['body_names'].index('body')][:3]for r in rows])
        save(output/'motion_diagnostic.json',{'scope':'Post-warmup observation, not passive stability or torque admission',
             'joint_names':native['joint_names'], 'joint_position_range_rad':np.ptp(positions,axis=0).tolist(),
             'interval_angle_rate_rad_s':(np.diff(positions,axis=0)/DT).tolist(),
             'reported_sdk_joint_rate_rad_s':sdk_rates.tolist(),
             'root_position_delta_m':(roots-roots[0]).tolist(),
             'warmup_motion_unobserved':True})
        state['wall_s']=time.monotonic()-start
        state['status']='completed'
        print('CANONICAL_NATIVE_INSPECTION_COMPLETED steps=8 physical_admission=false',flush=True)
    except BaseException as e:
        state['status']='failed';state['errors'].append(repr(e))
        (output/'traceback.txt').write_text(traceback.format_exc())
        print(traceback.format_exc(),flush=True)
    finally:
        finish(output,state,args,identity,app)
    return 0 if state['status']=='completed' else 1


if __name__=='__main__':sys.exit(main())
