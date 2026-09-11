#!/usr/bin/env python3
"""Canonical native provisional standing; one robot, then same-source32. No policy."""
from pathlib import Path
import argparse,json,sys,time,traceback
from standing_contract import verify_inputs,SCHEMA,DT,CONTROLS
from native_support import save,finish,sdk_readback,tensor_provider,collect_native,snapshot,legacy_cooking_counter

def main(argv=None):
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--asset',type=Path,required=True);p.add_argument('--admission',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--num-envs',type=int,choices=[1,32],required=True);p.add_argument('--standing-one',type=Path);p.add_argument('--preflight-only',action='store_true')
 a,_=p.parse_known_args(argv);identity=verify_inputs(a)
 if a.preflight_only:print(json.dumps(identity,sort_keys=True));return 0
 from isaaclab.app import AppLauncher
 AppLauncher.add_app_launcher_args(p);a=p.parse_args(argv)
 if a.device!='cuda:0'or not a.headless:raise ValueError('Declared native GPU headless allocation required')
 a.output.mkdir(parents=True,exist_ok=False);out=a.output;app=None;session=None;errors=[]
 state={'schema':SCHEMA,'identity':identity,'runtime_binding':identity['runtime_binding'],'status':'running','physical_admission':False,'physics_admitted':False,'training_allowed':False,'checks':{},'errors':[],'explicit_steps_completed':0}
 save(out/'state.json',state);started=time.monotonic()
 try:
  app=AppLauncher(a).app;print('REFERENCE_SCREEN_APP_READY',flush=True)
  import numpy as np
  from isaaclab.sim import SimulationContext,SimulationCfg
  from isaaclab_physx.physics import PhysxCfg
  import omni.physics.tensors as tensors
  import omni.physx
  from pxr import UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Gf
  from isaacsim.core.version import get_version
  from inspect_core import usd_readback,validate_usd,validate_native,validate_native_frames,validate_sdf
  from standing_math import Geometry
  from standing_session import NativeStandingSession,memory
  version=list(get_version())
  if str(version[0])!='6.0.1':raise ValueError('Wrong native Isaac version')
  save(out/'runtime_api.json',{'isaac_sim_version':version,'sdk':sdk_readback(out),**tensor_provider(tensors),'warmup_scope':'SDK reset includes native warmup/app-pump; only subsequent controlled steps have exact400Hz command evidence.'})
  state['checks']['sdk_source_bound']=True;save(out/'memory_before_scene.json',memory())
  cfg=SimulationCfg(dt=DT,gravity=(0.,0.,-9.81),device=a.device,render_interval=8,physics=PhysxCfg(enable_external_forces_every_iteration=True),visualizer_cfgs=[],log_dir=str(out/'isaac_logs'))
  sim=SimulationContext(cfg);stage=sim.stage
  roots=['/Robot']+[f'/Robot_{i:03d}'for i in range(1,a.num_envs)]
  for i,root in enumerate(roots):
   prim=stage.DefinePrim(root,'Xform');prim.GetReferences().AddReference(str(a.asset.resolve()/'robot.usda'))
   if i:UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d((i%8)*2.,(i//8)*2.,0.))
  model=json.loads((a.asset/'source/model.json').read_text());usd=usd_readback(stage);save(out/'usd_readback.json',usd);state['checks']['usd_identity']=validate_usd(usd,model)
  from solver_recipe import configure as configure_solver,verify as verify_solver
  solver_record=configure_solver(stage,roots,PhysxSchema);save(out/'solver_readback.json',solver_record)
  # Static plane and material are explicit new environmental assumptions; asset meshes unchanged.
  floor=UsdGeom.Mesh.Define(stage,'/Ground');floor.CreatePointsAttr([(-40.,-40.,0.),(40.,-40.,0.),(40.,40.,0.),(-40.,40.,0.)]);floor.CreateFaceVertexCountsAttr([3,3]);floor.CreateFaceVertexIndicesAttr([0,1,2,0,2,3]);UsdPhysics.CollisionAPI.Apply(floor.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(floor.GetPrim()).CreateApproximationAttr('none')
  mat=UsdShade.Material.Define(stage,'/StandingMaterial');material=UsdPhysics.MaterialAPI.Apply(mat.GetPrim());material.CreateStaticFrictionAttr(1.);material.CreateDynamicFrictionAttr(1.);material.CreateRestitutionAttr(0.)
  pm=PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim());pm.CreateFrictionCombineModeAttr('multiply');pm.CreateRestitutionCombineModeAttr('multiply')
  UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(mat,materialPurpose='physics')
  for prim in stage.Traverse():
   if prim.HasAPI(UsdPhysics.RigidBodyAPI)and str(prim.GetPath()).startswith('/Robot'):
    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
   if prim.HasAPI(UsdPhysics.CollisionAPI)and str(prim.GetPath()).startswith('/Robot'):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat,materialPurpose='physics')
  stage.GetRootLayer().Export(str(out/'resolved_stage.usda'))
  save(out/'native_errors.json',errors)
  def on_error(e):errors.append({'type':int(e.type),'payload':repr(e.payload)});save(out/'native_errors.json',errors)
  error_subscription=omni.physx.get_physx_interface().get_error_event_stream().create_subscription_to_pop(on_error)
  t=time.monotonic();sim.reset();state['initialization_wall_s']=time.monotonic()-t
  solver_record['after_reset']=verify_solver(stage,roots,PhysxSchema);save(out/'solver_readback.json',solver_record)
  scene_prim=stage.GetPrimAtPath(cfg.physics_prim_path);scene=UsdPhysics.Scene(scene_prim)
  actual={'gravity_magnitude':scene.GetGravityMagnitudeAttr().Get(),'gravity_direction':list(scene.GetGravityDirectionAttr().Get()),'dt_s':sim.get_physics_dt(),'time_steps_per_second':scene_prim.GetAttribute('physxScene:timeStepsPerSecond').Get(),'external_forces_every_iteration':scene_prim.GetAttribute('physxScene:enableExternalForcesEveryIteration').Get(),'root_self_collisions':{r:stage.GetPrimAtPath(r+'/body').GetAttribute('physxArticulation:enabledSelfCollisions').Get()for r in roots},'floor_points_m':[[float(v)for v in p]for p in floor.GetPointsAttr().Get()],'floor_collision_enabled':UsdPhysics.CollisionAPI(floor.GetPrim()).GetCollisionEnabledAttr().Get(),'floor_approximation':UsdPhysics.MeshCollisionAPI(floor.GetPrim()).GetApproximationAttr().Get(),'material_static_dynamic_restitution':[material.GetStaticFrictionAttr().Get(),material.GetDynamicFrictionAttr().Get(),material.GetRestitutionAttr().Get()],'material_combine_modes':[pm.GetFrictionCombineModeAttr().Get(),pm.GetRestitutionCombineModeAttr().Get()],'material_assumption':'Provisional flat material; not calibrated hardware friction.'}
  save(out/'native_scene.json',actual)
  if abs(actual['gravity_magnitude']-9.81)>1e-5 or actual['gravity_direction']!=[0.,0.,-1.]or actual['dt_s']!=DT or actual['time_steps_per_second']!=400 or actual['external_forces_every_iteration']is not True or not all(v is True for v in actual['root_self_collisions'].values()):raise ValueError('Native standing scene differs')
  if actual['floor_points_m']!=[[-40.,-40.,0.],[40.,-40.,0.],[40.,40.,0.],[-40.,40.,0.]]or actual['floor_collision_enabled']is not True or actual['floor_approximation']!='none' or actual['material_static_dynamic_restitution']!=[1.,1.,0.]or actual['material_combine_modes']!=['multiply','multiply']:raise ValueError('Floor/material authored readback differs')
  state['checks']['native_scene']=True
  view=sim.physics_sim_view.create_articulation_view('/Robot*/body');native=collect_native(view,out)
  from solver_diagnostics import collect_legacy_friction
  collect_legacy_friction(view,native,out,save);state['checks']['legacy_friction_observed']=True
  if view.count!=a.num_envs or set(view.prim_paths)!={r+'/body'for r in roots}:raise ValueError('Wrong native replica roots')
  for e in range(a.num_envs):
   one={k:v if k in ['body_names','joint_names','fixed_base']else 1 if k=='count'else[v[e]]for k,v in native.items()};validate_native(one,model)
  material_readback={}
  for key,getter in [('materials','get_material_properties'),('contact_offsets','get_contact_offsets'),('rest_offsets','get_rest_offsets')]:
   try:material_readback[key]=np.array(getattr(view,getter)().numpy(),copy=True).tolist()
   except Exception as e:material_readback['failed_getter']={'name':getter,'error':repr(e)};save(out/'native_materials.json',material_readback);raise
   save(out/'native_materials.json',material_readback)
  materials=np.asarray(material_readback['materials']);offsets=np.asarray(material_readback['contact_offsets']);rests=np.asarray(material_readback['rest_offsets'])
  if materials.shape!=(a.num_envs,153,3)or not np.allclose(materials,[1.,1.,0.],atol=1e-7,rtol=0)or offsets.shape!=(a.num_envs,153)or not np.allclose(offsets,.001,atol=1e-9,rtol=0)or not np.all(rests==0):raise ValueError('Native material/contact skin differs')
  state['checks']['native_materials_offsets']=True
  state['checks']['all_native_inertias_limits_no_drives']=True
  warm=snapshot(view,sim,0);save(out/'warmup_native_state.json',warm)
  sdfview=sim.physics_sim_view.create_sdf_shape_view('/Robot*/*/collisions/part_*',1);paths=list(sdfview.object_paths)
  expected={r+c['path'][len('/Robot'):]for r in roots for c in usd['colliders']}
  sdf={'count':sdfview.count,'valid':bool(sdfview.check()),'paths':paths,'expected_count':153*a.num_envs,'initialization_barrier':{'sim_reset_returned':True,'physics_view_valid':bool(sim.physics_sim_view.check()),'articulation_view_valid':bool(view.check())},'legacy_task_counter':legacy_cooking_counter(omni.physx),'scope':'Native SDF presence; no new distance-fidelity claim'};save(out/'sdf_readback.json',sdf)
  if sdf['count']!=len(expected)or len(paths)!=len(expected)or set(paths)!=expected or not sdf['valid']or not all(sdf['initialization_barrier'].values()):raise ValueError('Missing/additional native SDF shapes')
  if sdf['legacy_task_counter']['available']and sdf['legacy_task_counter']['pending']!=0:raise ValueError('Legacy cooking counter not idle')
  state['checks']['all_native_sdf_paths']=True
  geo=json.loads((Path(__file__).parent/'geometry/geometry.json').read_text())
  with np.load(Path(__file__).parent/'geometry/geometry_extrema.npz')as z:geometry=Geometry(geo,{k:z[k]for k in z.files},native['body_names'])
  contact=sim.physics_sim_view.create_rigid_contact_view('/Robot*/*',['/Ground'],max_contact_data_count=1024*a.num_envs)
  session=NativeStandingSession(view,contact,sim,model,native,geometry,out,save,a.device);session.reset_canonical()
  for e in range(a.num_envs):
   row=session.observe();validate_native_frames({'link_pose_xyzw':row['link_pose_xyzw'][e:e+1],'joint_position':row['joint_position_rad'][e:e+1]},native['body_names'],native['joint_names'],model)
  state['checks']['all_reset_coordinate_frames']=True
  for control in range(CONTROLS):
   session.step_control(np.zeros((a.num_envs,18),np.float32));state['explicit_steps_completed']=session.count
   if errors:raise RuntimeError('Native error event')
   if (control+1)%100==0:print(f'CANONICAL_STANDING control={control+1} replicas={a.num_envs}',flush=True)
  session.close();session=None
  solver_record['after_controlled_steps']=verify_solver(stage,roots,PhysxSchema);save(out/'solver_readback.json',solver_record)
  state['checks']['solver_recipe_readback']=True
  from standing_score import score
  report=score(out);save(out/'standing_report.json',report)
  state['standing_pass']=report['all_pass'];state['status']='completed';state['native_error_events']=errors;state['wall_s']=time.monotonic()-started
  print(f'CANONICAL_STANDING_ACQUISITION_COMPLETED standing_pass={report["all_pass"]} training_allowed=false',flush=True)
 except BaseException as e:
  state['status']='failed';state['errors'].append(repr(e));(out/'traceback.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
 finally:
  if session is not None:
   state['explicit_steps_completed']=session.count
   try:session.close()
   except BaseException as e:state['status']='failed';state['errors'].append('Session close:'+repr(e))
  state['native_error_events']=errors
  finish(out,state,a,identity,app)
 return 0 if state['status']=='completed'else 1
if __name__=='__main__':sys.exit(main())
