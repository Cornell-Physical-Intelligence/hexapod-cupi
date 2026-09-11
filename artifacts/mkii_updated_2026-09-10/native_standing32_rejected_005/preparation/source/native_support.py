"""Byte-preserved helper function bodies from canonical actuation002; standing contract bound explicitly."""
from pathlib import Path
import argparse,importlib,inspect,json,math,sys,time,traceback
from standing_contract import read,sha,verify_inputs
TENSOR_API_SHA='b913fdb1a3c0cf04b62aa01d6ceaf7d96cfe577a510d0e02b71485018661c903'
TENSOR_PACKAGE_SHA='377af3ebe0786768209c0582f5032db37d712e1a45c6f58f00e5b5284158afd4'

def tensor_provider(tensors):
    """Resolve the installed public factory; never assume a private package layout."""
    source=inspect.getsourcefile(tensors.create_simulation_view)
    if source is None:raise ValueError('Tensor factory has no inspectable source')
    package=Path(tensors.__file__).resolve()
    result={'tensor_api_source':str(Path(source).resolve()),'tensor_api_sha256':sha(source),
            'tensor_package_source':str(package),'tensor_package_sha256':sha(package)}
    if result['tensor_api_sha256']!=TENSOR_API_SHA or result['tensor_package_sha256']!=TENSOR_PACKAGE_SHA:
        raise ValueError('Actual native tensor provider differs from extracted110.1.13 source')
    return result

def create_sdf_view(sim_view):
    # Actual110.1.13 compiled backend requires a string, despite Python docs accepting lists.
    return sim_view.create_sdf_shape_view('/Robot/*/collisions/part_*',1)

def legacy_cooking_counter(physx):
    """Optional diagnostic only. An absent API never means zero pending tasks."""
    factory=getattr(physx,'get_physx_cooking_interface',None)
    if not callable(factory):return {'available':False,'pending':None,'reason':'legacy cooking interface unavailable'}
    cooking=factory()
    getter=getattr(cooking,'get_num_collision_tasks',None)
    if not callable(getter):return {'available':False,'pending':None,'reason':'legacy task getter unavailable'}
    value=getter()
    if type(value) is not int or value<0:raise ValueError('Invalid legacy cooking task count')
    return {'available':True,'pending':value,'reason':None}

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
