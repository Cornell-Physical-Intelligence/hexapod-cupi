"""Read-only addon to an already initialized canonical inspector. No launcher."""
from pathlib import Path
import hashlib
import inspect
import json
import time
import numpy as np
from mesh_oracle import frame_prediction
from score_probe import score, finite

API_SHA='b913fdb1a3c0cf04b62aa01d6ceaf7d96cfe577a510d0e02b71485018661c903'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path,obj):
    def safe(v):
        if isinstance(v,float) and not np.isfinite(v):return {'nonfinite':repr(v)}
        if isinstance(v,dict):return {str(k):safe(x)for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [safe(x)for x in v]
        return v
    path=Path(path);part=path.with_suffix(path.suffix+'.part');part.write_text(json.dumps(safe(obj),indent=2,allow_nan=False)+'\n');part.replace(path)


def copied(x):
    if hasattr(x,'numpy'):x=x.numpy()
    return np.array(x,copy=True)


def query_copy(view, encoded):
    # The native API returns the same persistent tensor on subsequent queries.
    return copied(view.get_sdf_and_gradients(encoded))


def query_pair(view, points, order, encode, destination):
    """Persist the first result before the second native call can fail."""
    destination=Path(destination)
    first_input=points[order][None];second_order=order[::-1];second_input=points[second_order][None]
    metadata={'input_order':order.tolist(),'input_points':first_input.tolist(),
         'repeat_input_order':second_order.tolist(),'repeat_input_points':second_input.tolist(),'completed_calls':0}
    save(destination.with_suffix('.json'),metadata)
    a=query_copy(view,encode(first_input))
    np.savez_compressed(destination,input_points=first_input,input_order=order,raw=a)
    finite(a,(1,len(points),4))
    metadata['completed_calls']=1;metadata['first_result_saved']=True;save(destination.with_suffix('.json'),metadata)
    b=query_copy(view,encode(second_input))
    np.savez_compressed(destination,input_points=first_input,input_order=order,raw=a,
                        repeat_input_points=second_input,repeat_input_order=second_order,repeat_raw=b)
    finite(b,(1,len(points),4))
    metadata['completed_calls']=2;metadata['both_results_saved']=True;save(destination.with_suffix('.json'),metadata)
    return a[0][np.argsort(order)],b[0][np.argsort(second_order)]


def rotation(q):
    x,y,z,w=finite(q,(4,))
    if abs(np.dot(q,q)-1)>2e-5:raise ValueError('Native quaternion is not unit XYZW')
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def snapshot(sim,view):
    return {'explicit_counter':sim.get_physics_step_count(),'link_names':list(view.shared_metatype.link_names),
            'link_pose':copied(view.get_link_transforms()).tolist(),
            'link_com_velocity':copied(view.get_link_velocities()).tolist(),
            'joint_position':copied(view.get_dof_positions()).tolist(),
            'joint_velocity':copied(view.get_dof_velocities()).tolist()}


def verify_inputs(fixture_path,asset):
    own=Path(__file__).resolve().parent
    freeze=json.loads((own/'FREEZE_SHA256.json').read_text())
    for name,digest in freeze.items():
        if sha(own/name)!=digest:raise ValueError('Changed probe source: '+name)
    actual={x.relative_to(own).as_posix()for x in own.rglob('*')if x.is_file()and '__pycache__'not in x.parts and x!=own/'FREEZE_SHA256.json'}
    if actual!=set(freeze):raise ValueError('Unexpected probe payload')
    fixture_path=Path(fixture_path).resolve();asset=Path(asset).resolve()
    if sha(fixture_path)!=freeze['fixture/fixture.json']:raise ValueError('Changed fixture')
    fixture=json.loads(fixture_path.read_text())
    if fixture['schema']!='canonical_tibia_sdf_fixture_v1':raise ValueError('Wrong fixture schema')
    for name,digest in fixture['asset_files'].items():
        if sha(asset/name)!=digest:raise ValueError('Changed canonical asset: '+name)
    mesh=fixture_path.parent/'mesh.npz'
    if sha(mesh)!=fixture['mesh_npz_sha256']:raise ValueError('Changed source mesh oracle')
    return fixture,mesh,{'probe_freeze_sha256':sha(own/'FREEZE_SHA256.json'),'fixture_sha256':sha(fixture_path),
                         'mesh_npz_sha256':sha(mesh),'asset_files':fixture['asset_files']}


def run_probe(sim,articulation_view,fixture_path,asset_root,output):
    """Caller owns initialization, locks, deadlines and shutdown; never advance it."""
    from pxr import UsdGeom
    import warp as wp
    fixture,mesh_path,identity=verify_inputs(fixture_path,asset_root)
    output=Path(output).resolve();own=Path(__file__).resolve().parent
    if output.is_relative_to(own) or output.is_relative_to(Path(asset_root).resolve()):raise ValueError('Output overlaps immutable input')
    output.mkdir(parents=True,exist_ok=False)
    state={'status':'running','identity':identity,'queries_completed':0,'errors':[],
           'standing_admitted':False,'contact_admitted':False,'training_allowed':False}
    save(output/'state.json',state)
    raw=[];repeat=[];world=[];link_predictions=[];poses=[]
    points=np.asarray(fixture['points'],dtype=np.float32);anchors=np.asarray(fixture['semantic_anchor_indices'])
    start=time.monotonic()
    try:
        if UsdGeom.GetStageMetersPerUnit(sim.stage)!=1.0 or str(UsdGeom.GetStageUpAxis(sim.stage))!='Z':raise ValueError('Wrong live stage units or up axis')
        before=snapshot(sim,articulation_view);save(output/'before.json',before)
        with np.load(mesh_path,allow_pickle=False)as m:v=m['vertices'];f=m['faces']
        for k,item in enumerate(fixture['shapes']):
            prim=sim.stage.GetPrimAtPath(item['path'])
            if not prim.IsValid() or prim.GetAttribute('hexapod:sourceMesh').Get()!='mesh_056_tibia.stl':raise ValueError('Wrong tibia collider')
            mesh=UsdGeom.Mesh(prim)
            if not mesh or not np.array_equal(np.asarray(mesh.GetPointsAttr().Get()),v) or not np.array_equal(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()),f.reshape(-1)) or not np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get())==3):raise ValueError('Live collider source triangles differ')
            # Bind constant collider-to-link transform from the actual composed stage.
            link=sim.stage.GetPrimAtPath('/Robot/'+item['link'])
            W=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
            L=np.asarray(UsdGeom.Xformable(link).ComputeLocalToWorldTransform(0)).T
            local=np.linalg.inv(L)@W
            if not np.allclose(local,item['shape_to_link'],atol=2e-7,rtol=0):raise ValueError('Shape-to-link transform differs')
            i=before['link_names'].index(item['link']);p=finite(before['link_pose'][0][i],(7,))
            live=np.eye(4);live[:3,:3]=rotation(p[3:]);live[:3,3]=p[:3];world_shape=live@local
            poses.append({'path':item['path'],'shape_to_world_from_live_link':world_shape.tolist()})
            world.append(frame_prediction(points[anchors],v,f,world_shape))
            link_predictions.append(frame_prediction(points[anchors],v,f,local))
            view=sim.physics_sim_view.create_sdf_shape_view(item['path'],len(points))
            if view.count!=1 or list(view.object_paths)!=[item['path']] or view.max_num_points!=len(points) or not view.check():raise ValueError('Wrong or invalid exact SDF view')
            provider=inspect.getsourcefile(view.get_sdf_and_gradients)
            if provider is None or sha(provider)!=API_SHA:raise ValueError('Unbound native110 query provider')
            save(output/f'view_{k}.json',{'count':int(view.count),'object_paths':list(view.object_paths),
                 'max_num_points':int(view.max_num_points),'check':bool(view.check()),'provider':provider,'provider_sha256':sha(provider)})
            # Distinct order per shape catches ignored point ordering and cached outputs.
            order=np.roll(np.arange(len(points)),k*17)
            encode=lambda values:wp.array(values,dtype=wp.float32,device=sim.physics_sim_view.device)
            a,b=query_pair(view,points,order,encode,output/f'query_{k}.npz')
            raw.append(a);repeat.append(b)
            state['queries_completed']+=2;save(output/'state.json',state)
        after=snapshot(sim,articulation_view);save(output/'after.json',after)
        if before!=after:raise ValueError('Observed state or explicit counter advanced during read-only probe')
        report=score(np.asarray(raw),np.asarray(repeat),fixture,np.asarray(world),np.asarray(link_predictions))
        report['no_observed_state_change']=True
        report['state_check_scope']='No advance call exists in addon; exact before/after counter, poses, velocities and q. This does not instrument hidden native clock activity.'
        save(output/'frame_hypotheses.json',{'shape_poses':poses,
              'expected_d_g_if_inputs_and_gradient_were_world':np.asarray(world).tolist(),
              'expected_d_g_if_inputs_and_gradient_were_link':np.asarray(link_predictions).tolist()})
        save(output/'report.json',report)
        state['status']='completed';state['declared_semantics_supported']=report['declared_semantics_supported']
        state['geometry_accuracy_within_proposed_bounds']=report['geometry_accuracy_within_proposed_bounds']
    except BaseException as e:
        state['status']='failed';state['errors'].append(repr(e))
        try:save(output/'after_failure.json',snapshot(sim,articulation_view))
        except Exception as later:state['errors'].append('post-failure snapshot: '+repr(later))
    finally:
        try:
            state['inputs_unchanged']=verify_inputs(fixture_path,asset_root)[2]==identity
            if not state['inputs_unchanged']:state['status']='failed';state['errors'].append('Terminal identity changed')
        except Exception as e:state['inputs_unchanged']=False;state['errors'].append(repr(e));state['status']='failed'
        state['wall_s']=time.monotonic()-start
        state['outputs']={x.name:sha(x)for x in output.iterdir()if x.is_file()and x.name!='state.json'and not x.name.endswith('.part')}
        save(output/'state.json',state)
    return state
