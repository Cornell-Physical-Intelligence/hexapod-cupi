"""Read-only CPU replay of frozen source016 and independent 128-root/contact audit."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent
A=HERE.parent
REPO=A.parents[2]
SOURCE=A/'source_016'
EXPECTED_FREEZE='e82077d7fbe9b8c5873b119c940d95b3db33189ed46a00d20c75e6c773fb64da'
EXPECTED_BINDING='40ca999173fc3ce93f40a43c8268c5ddd4fd450f8950e8c5bf0883845b98e7c8'
N=128


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)


def frozen_scorer():
    require(sha(SOURCE/'FREEZE_SHA256.json')==EXPECTED_FREEZE,'Source016 manifest changed')
    manifest=read(SOURCE/'FREEZE_SHA256.json')
    actual={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*')
            if p.is_file() and p.name!='FREEZE_SHA256.json'}
    require(actual==manifest,'Frozen source016 tree changed')
    cfg=ast.parse((SOURCE/'env_config.py').read_text())
    kd=ast.literal_eval(next(n.value for n in cfg.body if isinstance(n,ast.Assign)
                            and any(isinstance(t,ast.Name) and t.id=='KD' for t in n.targets)))
    legs=('lf','lm','lr','rf','rm','rr')
    names=tuple(f'{leg}_{joint}' for leg in legs for joint in ('coxa_yaw','femur_pitch','tibia_pitch'))
    namespace={'np':np,'json':json,'Path':Path,'KD':kd,'JOINT_NAMES':names}
    wanted={'_diagnostic_servo','_quiet_metrics','score_diagnostic'}
    nodes=[n for n in ast.parse((SOURCE/'env.py').read_text()).body
           if isinstance(n,ast.FunctionDef) and n.name in wanted or isinstance(n,ast.Assign)
           and any(isinstance(t,ast.Name) and t.id=='_QUIET_GATES' for t in n.targets)]
    require(len(nodes)==4,'Frozen replay extraction differs')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE/'env.py')+'::CPU_replay','exec'),namespace)
    return namespace['score_diagnostic'],names,manifest


def rotations(q):
    q=np.asarray(q,float)
    require(np.isfinite(q).all() and np.max(abs(np.sum(q*q,axis=-1)-1))<=2e-5,'Invalid native quaternion')
    x,y,z,w=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),axis=-1).reshape(*q.shape[:-1],3,3)


def segment_distances(previous,current,i,j):
    start=previous[i]-previous[j];delta=(current[i]-current[j])-start
    fraction=np.clip(-np.sum(start*delta,axis=-1)/np.maximum(np.sum(delta*delta,axis=-1),1e-30),0,1)
    return np.linalg.norm(start+fraction[:,None]*delta,axis=-1)


def contact_packet(packet,row,body_names,geometry):
    patches=packet['patches'];count=len(patches)
    require(count<1024*N,'Contact data capacity exhausted')
    indices=[p['buffer_index'] for p in patches]
    require(len(indices)==len(set(indices)) and all(type(k)is int and 0<=k<1024*N for k in indices),'Invalid contact indices')
    e=np.array([p['env'] for p in patches],int)
    require(all(type(p['env'])is int for p in patches) and np.all((e>=0)&(e<N)),'Contact replica index')
    b=np.array([body_names.index(p['body']) for p in patches],int)
    points=np.array([p['point_world_m'] for p in patches],np.float32).reshape(-1,3)
    normals=np.array([p['normal_world'] for p in patches],np.float32).reshape(-1,3)
    force=np.array([p['normal_force_n'] for p in patches],np.float32)
    separation=np.array([p['separation_m'] for p in patches],float)
    require(all(np.isfinite(x).all() for x in (points,normals,force,separation)),'Nonfinite patch')
    inactive=(force==0)&np.all(normals==0,axis=1)&(separation==0)
    require(np.array_equal(inactive,[p['inactive_zero_normal'] for p in patches]),'Inactive tuple declaration differs')
    require(np.all((abs(np.linalg.norm(normals,axis=1)-1)<=1e-3)|inactive),'Invalid patch normal')
    categories=np.array([0 if p['body']=='body' else 1 if p['body'].endswith('_coxa')
                         else 2 if p['body'].endswith('_femur') else 3 for p in patches],int)
    toes=categories==3
    if toes.any():
        positions=row['link_pose_xyzw'][e[toes],b[toes]]
        transforms=np.array([geometry[p['body']]['shape_to_link'] for p in patches if p['body'].endswith('_tibia')],float)
        local=np.einsum('bi,bij->bj',points[toes]-positions[:,:3],rotations(positions[:,3:]))
        local=np.einsum('bi,bij->bj',local-transforms[:,:3,3],transforms[:,:3,:3])
        shapes=[geometry[p['body']] for p in patches if p['body'].endswith('_tibia')]
        bounds=np.array([s['cap_bounds_m'] for s in shapes]);skin=np.array([s['contact_offset_m'] for s in shapes])
        lower=np.array([s['cap_lower_x_m'] for s in shapes])
        cap=(local[:,0]>=lower-1e-12)&(local[:,0]<=bounds[:,1,0]+skin)
        cap &= np.all(local[:,1:]>=bounds[:,0,1:]-skin[:,None],axis=1)&np.all(local[:,1:]<=bounds[:,1,1:]+skin[:,None],axis=1)
        recorded=np.array([p['shape_point_m'] for p in patches if p['body'].endswith('_tibia')])
        require(np.allclose(local,recorded,atol=2e-12,rtol=0),'Contact point frame reconstruction differs')
        categories[np.flatnonzero(toes)[cap]]=4
    labels=np.array(['body','coxa','femur','shaft','toe'])
    require(np.array_equal(labels[categories],[p['category'] for p in patches]),'Exact cap category differs')
    vectors=force[:,None]*normals
    feet=np.zeros((N,6,3));other=np.zeros((N,4,3));body_force=np.zeros((N,19,3))
    validtoe=categories==4
    legmap={leg:index for index,leg in enumerate(('lf','lm','lr','rf','rm','rr'))}
    leg=np.array([legmap.get(p['body'][:2],0) for p in patches],int)
    np.add.at(feet,(e[validtoe],leg[validtoe]),vectors[validtoe])
    np.add.at(other,(e[~validtoe],categories[~validtoe]),vectors[~validtoe])
    np.add.at(body_force,(e[~validtoe],b[~validtoe]),vectors[~validtoe])
    nonfoot=np.linalg.norm(body_force,axis=-1).max(-1)>1.
    nonfoot[e[(~validtoe)&(abs(force)>1.)]]=True
    require(np.array_equal(np.linalg.norm(feet,axis=-1)>1.,row['distal_contact']),'Six-toe force contact differs')
    require(np.array_equal(nonfoot,row['nonfoot_contact']),'Nonfoot contact differs')
    require(np.array_equal(feet,row['distal_force_world_n']) and np.array_equal(other,row['nonfoot_force_world_n']),'Contact force aggregation differs')
    return count


def audit(result,remote_layout=False):
    started=time.monotonic();scorer,names,source_files=frozen_scorer()
    preparation=A/'preparation_diagnostic_batch128_001'
    require(sha(preparation/'SHA256.json')=='9b49448b20e7676f21f6f50e8b3f6e9125cc185653b25f62b621fd3cd6b3cb1b','Preparation manifest changed')
    for name,digest in read(preparation/'SHA256.json').items():require(sha(preparation/name)==digest,'Preparation changed: '+name)
    translations=read(preparation/'input_translation.json')
    def input_path(remote):return Path(remote if remote_layout else translations[remote]['local_path'])
    for remote,translated in translations.items():
        require(sha(input_path(remote))==translated['sha256'],'Prepared input changed')
    result=Path(result).resolve();d=result/'standing/standing'
    require(sha(result/'launch_binding.json')==EXPECTED_BINDING,'Actual reviewed128 launch binding differs')
    binding=read(result/'launch_binding.json')
    state=read(result/'standing/state.json');native=read(result/'standing/native/native_readback.json')
    session=read(d/'session.json');initial=read(d/'initial_reset.json')
    prep=read(A/'preparation_diagnostic_batch128_001/PREPARATION.json')
    require(state['status']=='completed' and state['mode']=='diagnostic' and state['errors']==[],'Native lifecycle failed')
    require(state['runtime_binding']['runtime_tree_sha256']==EXPECTED_FREEZE,'Wrong native source freeze')
    require(state['identity']['source_files']==source_files,'Native source-file identity differs')
    local=read(A/'preparation_diagnostic_batch128_001/cpu_preflight.json')['identity']
    require(state['identity']==local,'Actual identity differs from exact prepared CPU identity')
    require(session['steps']==8000 and session['controls']==1000 and session['reset_count']==1
        and session['failure']is None and session['all_rows_recorded'] and session['contact_evidence_complete'],'Incomplete session')
    roots=[f'/Robot_{i:03d}' for i in range(N)]
    require(session['root_paths']==native['root_paths']==roots,'Root order/count')
    require(session['joint_names']==native['canonical_joint_names']==list(names),'Named joint order')
    require(session['body_names']==native['native_body_names'] and len(set(session['body_names']))==19,'Native body order/count')
    model=read(input_path(binding['asset']+'/source/model.json'))
    bodies={b['name']:b for b in model['links']};joints={j['name']:j for j in model['joints']}
    require(set(bodies)==set(native['native_body_names']) and set(joints)==set(native['native_joint_names']),'Canonical topology')
    expected_masses=np.array([bodies[b]['mass'] for b in native['native_body_names']])
    masses=np.asarray(native['masses']);limits=np.asarray(native['limits'])
    require(masses.shape==(N,19) and np.allclose(masses,expected_masses,atol=1e-6,rtol=0),'Native per-body masses')
    require(np.allclose(masses.sum(-1),7.466088235225788,atol=2e-6,rtol=0),'Native total masses')
    expected_limits=np.array([[joints[j]['lower'],joints[j]['upper']] for j in names])
    require(limits.shape==(N,18,2) and np.allclose(limits,expected_limits,atol=2e-6,rtol=0),'Named limits')
    require(np.asarray(native['native_max_velocity']).shape==(N,18)
        and np.allclose(native['native_max_velocity'],np.float32(2*np.pi*480/60),atol=2e-6,rtol=0),'Native velocity cap')
    require(native['sdf_shapes']==19584 and native['implicit_drive_and_armature_zero']is True,'Detailed shapes/drive')
    require(read(result/'standing/native/native_errors.json')==[],'Native error events')
    for phase in ('after_sdk_reset','after_controlled_steps'):
        r=native['recipe_readbacks'][phase]
        require(r['solver_attributes']=={root:[32,0] for root in roots},'Solver attributes')
        require(r['scene_attributes']=={'physxScene:timeStepsPerSecond':400,'physxScene:enableExternalForcesEveryIteration':True},'Scene settings')
        require(np.asarray(r['material_properties']).shape==(N,153,3) and np.allclose(r['material_properties'],[1,1,0],atol=1e-7,rtol=0),'Native material')
        require(np.asarray(r['contact_offsets']).shape==(N,153) and np.allclose(r['contact_offsets'],.001,atol=1e-9,rtol=0),'Contact offsets')
        require(np.asarray(r['rest_offsets']).shape==(N,153) and np.all(np.asarray(r['rest_offsets'])==0),'Rest offsets')
        require(r['native_error_events']==[] and r['backend_iteration_introspection']is False,'Native errors/introspection claim')
    recomputed=scorer(d);recorded=read(d/'standing_report.json')
    require(recomputed==recorded,'Independent frozen-scorer result differs from native report')
    require(recomputed['num_envs']==N and len(recomputed['replicas'])==N,'Missing replica score')
    print('Frozen original scorer recomputed for all128',flush=True)
    geometry_path=input_path(binding['geometry_source']+'/geometry/geometry.json')
    geometry={s['body']:s for s in read(geometry_path)['shapes']}
    radius=read(A/'preparation_diagnostic_batch128_001/layout_geometry.json')['articulation_reach']['radius_m']
    original=np.asarray(initial['post_reset']['root_pose_xyzw'],float)
    expected=np.zeros((N,7));expected[:,:3]=prep['expected_layout']['origins'];expected[:,2]=local['physics_config']['reset_root_height_m'];expected[:,6]=1
    require(original.shape==(N,7) and np.allclose(original,expected,atol=2e-6,rtol=0),'Initial root/frame readback')
    require(np.allclose(initial['post_reset']['joint_position_rad'],np.asarray(session['neutral_joint_position_rad']),atol=2e-6,rtol=0),'Initial joint reset')
    require(np.all(np.asarray(initial['post_reset']['joint_velocity_rad_s'])==0),'Initial joint velocity')
    i,j=np.triu_indices(N,1);previous=original[:,:3]
    min_sample={'gap_m':float('inf')};min_swept={'gap_m':float('inf')};min_floor={'clearance_m':float('inf')}
    max_excursion=np.zeros(N);min_gap_per_replica=np.full(N,np.inf);steps=0;patches=0
    def position_sample(position,sequence):
        nonlocal min_sample,min_swept,min_floor,previous,max_excursion,min_gap_per_replica
        require(position.shape==(N,3) and np.isfinite(position).all(),'Invalid root sample')
        gaps=np.linalg.norm(position[i]-position[j],axis=-1)-2*radius
        k=int(np.argmin(gaps))
        if gaps[k]<min_sample['gap_m']:min_sample={'gap_m':float(gaps[k]),'pair':[int(i[k]),int(j[k])],'sequence':sequence}
        np.minimum.at(min_gap_per_replica,i,gaps);np.minimum.at(min_gap_per_replica,j,gaps)
        swept=segment_distances(previous,position,i,j)-2*radius;k=int(np.argmin(swept))
        if swept[k]<min_swept['gap_m']:min_swept={'gap_m':float(swept[k]),'pair':[int(i[k]),int(j[k])],'ending_sequence':sequence}
        floor=40.-np.abs(position[:,:2])-radius;k=int(np.argmin(floor));env,axis=np.unravel_index(k,floor.shape)
        if floor[env,axis]<min_floor['clearance_m']:min_floor={'clearance_m':float(floor[env,axis]),'env':int(env),'axis':int(axis),'sequence':sequence}
        max_excursion=np.maximum(max_excursion,np.linalg.norm(position[:,:2]-original[:,:2],axis=-1));previous=position
    position_sample(previous,-1)
    with (d/'contacts.jsonl').open() as contacts:
        for name in session['substep_files']:
            with np.load(d/name,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
            count=len(raw['sequence'])
            require(raw['root_pose_xyzw'].shape==(count,N,7) and raw['link_pose_xyzw'].shape==(count,N,19,7),'Native body/root state dimensions')
            require(np.max(abs(np.sum(raw['root_pose_xyzw'][...,3:].astype(float)**2,axis=-1)-1))<=2e-5,'Native root quaternion norm')
            require(np.max(abs(np.sum(raw['link_pose_xyzw'][...,3:].astype(float)**2,axis=-1)-1))<=2e-5,'Native link quaternion norms')
            for offset,sequence in enumerate(raw['sequence']):
                require(int(sequence)==steps,'Additional raw sequence check')
                position_sample(np.asarray(raw['root_pose_xyzw'][offset,:,:3],float),steps)
                packet=json.loads(next(contacts));require(packet['sequence']==steps and packet['explicit_counter']==int(raw['explicit_counter'][offset]),'Contact packet ordering/counter')
                patches+=contact_packet(packet,{k:v[offset] for k,v in raw.items()},session['body_names'],geometry)
                steps+=1
            print('Independent contact/root audit steps',steps,flush=True)
        require(next(contacts,None)is None,'Unexpected trailing contact packet')
    require(steps==8000,'Incomplete independent trace')
    contact_audit=read(result/'jobs/standing_contact_data_audit.json')
    require(contact_audit['passed']is True and contact_audit['incomplete_data_warning_count']==0,'Native contact truncation warnings')
    job=read(result/'jobs/standing.json');cleanup=read(result/'cleanup.json')
    require(job['exit_code']==0 and job['cleanup_checked']is True and job['native_mode']=='diagnostic','Guard terminal lifecycle')
    require(cleanup['cleanup_checked']is True and cleanup['reservation_released']is False,'Owned cleanup/reservation')
    separation_ok=min_sample['gap_m']>0 and min_swept['gap_m']>0 and min_floor['clearance_m']>0
    all_pass=recomputed['all_pass'] and state['standing_gate_pass'] and separation_ok
    return {'schema':'independent_canonical_batch128_verification_v1','status':'completed','all_checks_passed':bool(all_pass),
        'recomputed_report':recomputed,'exact_recomputed_report':True,'replicas_verified':N,'controls':1000,'substeps':steps,
        'contact_packets_verified':steps,'contact_patches_reclassified':patches,'native_metadata_and_lifecycle_verified':True,
        'minimum_sampled_sphere_gap':min_sample,'minimum_linear_swept_sphere_gap':min_swept,'minimum_floor_envelope_clearance':min_floor,
        'radius_m':radius,'per_replica_minimum_sampled_sphere_gap_m':min_gap_per_replica.tolist(),
        'per_replica_maximum_root_planar_excursion_from_reset_m':max_excursion.tolist(),
        'separation_and_floor_checks_passed':bool(separation_ok),'source_freeze_sha256':EXPECTED_FREEZE,
        'wall_seconds':time.monotonic()-started,'stage2_complete':False,'physical_admission_created':False,
        'numpy_version':np.__version__,'python_version':sys.version,'remote_layout':remote_layout,
        'limitations':['Sampled roots plus linear sweeps are not continuous collision detection or solver independence.',
            'Exact standing scorer uses recorded mesh-clearance channels; all mesh vertices were not re-evaluated independently.',
            'Reported native attributes do not expose actual backend iteration counts.',
            'No contact insulation/filter is added. Every one of128 replicas is required; no fallback to a passing subset.']}


def self_test():
    i,j=np.array([0]),np.array([1])
    old=np.array([[-2.,0,0],[2.,0,0]]);new=-old
    require(np.linalg.norm(new[0]-new[1])==4 and segment_distances(old,new,i,j)[0]==0,'Sweep must detect between-sample crossing')
    stationary=segment_distances(old,old,i,j);require(stationary[0]==4,'Zero-motion sweep')
    require(np.array_equal(rotations(np.array([[0,0,0,1.]])),np.eye(3)[None]),'Identity frame')
    body_names=['body','lf_tibia']+[f'fixture_{i}' for i in range(17)]
    row={'link_pose_xyzw':np.zeros((N,19,7),np.float32),'distal_contact':np.zeros((N,6),bool),
         'distal_force_world_n':np.zeros((N,6,3)),'nonfoot_contact':np.zeros(N,bool),'nonfoot_force_world_n':np.zeros((N,4,3))}
    require(contact_packet({'patches':[]},row,body_names,{})==0,'Empty contact frame')
    row['link_pose_xyzw'][0,1]=[14,4,1,0,0,0,1]
    row['distal_contact'][0,0]=True;row['distal_force_world_n'][0,0,2]=2
    geometry={'lf_tibia':{'shape_to_link':np.eye(4).tolist(),'cap_bounds_m':[[.9,-.1,-.1],[1,.1,.1]],'cap_lower_x_m':.9,'contact_offset_m':.001}}
    packet={'patches':[{'buffer_index':4,'env':0,'body':'lf_tibia','point_world_m':[15,4,1],
        'normal_world':[0,0,1],'normal_force_n':2.,'separation_m':0.,'inactive_zero_normal':False,
        'category':'toe','shape_point_m':[1,0,0]}]}
    require(contact_packet(packet,row,body_names,geometry)==1,'Translated exact-toe contact fixture')
    packet['patches'][0]['category']='shaft'
    try:contact_packet(packet,row,body_names,geometry)
    except ValueError:pass
    else:raise ValueError('Misclassified patch was not rejected')
    print('CPU fixtures passed: crossing/stationary sweep, identity frame, empty contacts, translated toe, wrong category rejected')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--result',type=Path);parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--remote-layout',action='store_true',help='Use hash-pinned Spark paths from the existing input translation receipt')
    args=parser.parse_args()
    if args.self_test:self_test();sys.exit(0)
    if args.result is None:parser.error('--result required')
    require(not (HERE/'audit.json').exists(),'Immutable audit output already exists')
    print('Hashing immutable result files',flush=True)
    inputs={str(p.resolve()):sha(p) for p in args.result.rglob('*') if p.is_file()}
    inputs.update({str(p.resolve()):sha(p) for p in SOURCE.iterdir() if p.is_file()})
    preparation=A/'preparation_diagnostic_batch128_001'
    inputs.update({str(p.resolve()):sha(p) for p in preparation.iterdir() if p.is_file()})
    inputs.update({str(remote if args.remote_layout else row['local_path']):sha(remote if args.remote_layout else row['local_path'])
                   for remote,row in read(preparation/'input_translation.json').items()})
    try:report=audit(args.result,args.remote_layout)
    except BaseException as error:
        report={'status':'failed','error':repr(error),'traceback':traceback.format_exc(),'all_checks_passed':False,'stage2_complete':False,'physical_admission_created':False}
    report['input_sha256']=inputs;report['audit_script_sha256']=sha(__file__)
    report['inputs_unchanged']=all(sha(path)==value for path,value in inputs.items())
    if not report['inputs_unchanged']:report['all_checks_passed']=False
    with (HERE/'audit.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','recomputed_report')},indent=2))
    sys.exit(0 if report['all_checks_passed'] else 1)
