"""Read-only stdlib comparison of three pinned acquisitions; stdout only.

The launcher prepends the exact saved numeric reader and INPUT_PINS. No writes,
subprocesses, native imports or GPU APIs are used on the remote host.
"""
from pathlib import Path
import hashlib,json,math,time

BASE=Path('/home/orionh/HEXAPOD_runs/restart_20260914/standing_translation_001')
PATHS={'origin':BASE/'origin_001/standing','xy14_4':BASE/'xy14_4_001/standing',
       'batch23':Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910/native_standing32_005/standing')}
ENVS={'origin':0,'xy14_4':0,'batch23':23}
COUNTS={'origin':1,'xy14_4':1,'batch23':32}
XY={'origin':(0.,0.),'xy14_4':(14.,4.),'batch23':(14.,4.)}
LEGS=['lf','lm','lr','rf','rm','rr']
M=800

def require(value,message):
    if not value:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def flat(value):
    if isinstance(value,list):return [v for item in value for v in flat(item)]
    return [value]
def pose_normalized(row,xy):
    out=list(row)
    for k in range(0,len(out),7):out[k]-=xy[0];out[k+1]-=xy[1]
    return out
def difference(a,b,labels=None):
    require(len(a)==len(b),'Different row counts')
    first=None;peak=None;nonidentical=0
    for i,(ar,br)in enumerate(zip(a,b)):
        require(len(ar)==len(br),'Different row widths')
        for j,(x,y)in enumerate(zip(ar,br)):
            if x!=y:
                d=float(y)-float(x);item={'row':i,'component':labels[j]if labels else j,
                                         'a':x,'b':y,'b_minus_a':d}
                if first is None:first=item
                if peak is None or abs(d)>abs(peak['b_minus_a']):peak=item
                nonidentical+=1
    return {'first_exact_difference':first,'maximum_absolute_difference':peak,
            'unequal_scalars':nonidentical,'total_scalars':sum(map(len,a)),
            'equality':'Exact stored values; no tolerance applied'}

states={};sessions={};native={};resets={};used={};arrays={};metadata={};contact_prefixes={}
started=time.monotonic()
for label,path in PATHS.items():
    state=read(path/'state.json');states[label]=state
    require(sha(path/'state.json')==INPUT_PINS[label]['state_sha256'],'Wrong exact state:'+label)
    require(state['identity']['runtime_binding']['runtime_tree_sha256']==INPUT_PINS[label]['source_freeze_sha256'],'Wrong source:'+label)
    require(state['status']=='completed'and state['explicit_steps_completed']==8000,'Incomplete original acquisition')
    used[label]={'state.json':sha(path/'state.json')}
    def checked(name):
        actual=sha(path/name);require(state['outputs'].get(name)==actual,'Changed sealed input:'+label+':'+name)
        used[label][name]=actual;return path/name
    sessions[label]=read(checked('session.json'));native[label]=read(checked('native_readback.json'))
    resets[label]=read(checked('initial_reset.json'))
    metadata[label]={'scene':read(checked('native_scene.json')),
                     'solver':read(checked('solver_readback.json')),
                     'native_materials':read(checked('native_materials.json')),
                     'warmup':read(checked('warmup_native_state.json'))}
    e=ENVS[label];n=COUNTS[label]
    require(sessions[label]['root_paths'][e]==('/Robot_023'if label=='batch23'else'/Robot'),'Wrong selected root')
    require(native[label]['body_names']==native['origin']['body_names']and native[label]['joint_names']==native['origin']['joint_names'],'Named order differs')
    chunk=checked('substeps_000.npz');fields={}
    specs={**{k:((18,),'float')for k in ['joint_position_rad','joint_velocity_rad_s','pre_joint_position_rad','pre_joint_velocity_rad_s','computed_torque_nm','applied_torque_nm','interval_angle_rate_rad_s','joint_target_rad']},
           'distal_force_world_n':((6,3),'float'),'distal_contact':((6,),'bool'),
           'root_pose_xyzw':((7,),'float'),'link_pose_xyzw':((19,7),'float'),
           'root_com_velocity':((6,),'float'),'minimum_mesh_floor_m':((),'float'),
           'minimum_non_toe_floor_m':((),'float')}
    for key,(tail,kind)in specs.items():
        values=numeric(chunk,key,(M,n,*tail),kind);width=math.prod(tail)
        rows=[list(values[(i*n+e)*width:(i*n+e+1)*width])for i in range(M)]
        if key.endswith('pose_xyzw'):rows=[pose_normalized(row,XY[label])for row in rows]
        fields[key]=rows
    seq=numeric(chunk,'sequence',(M,),'int');clock=numeric(chunk,'time_s',(M,));counter=numeric(chunk,'explicit_counter',(M,),'int')
    require(list(seq)==list(range(M))and all(clock[i]==(i+1)*.0025 for i in range(M)),'Clock mismatch')
    require(all(counter[i]==resets[label]['counter_after']+i+1 for i in range(M)),'Counter mismatch')
    fields['distal_force_norm_n']=[[math.sqrt(sum(row[k*3+j]**2 for j in range(3)))for k in range(6)]for row in fields['distal_force_world_n']]
    arrays[label]=fields
    # Bounded first45 rows only: full4.5GB batch contact stream is not rehashed.
    contacts=path/'contacts.jsonl';before=contacts.stat();prefix_hash=hashlib.sha256();rows=[]
    with contacts.open('rb')as stream:
        for i in range(45):
            line=stream.readline();require(line and len(line)<32<<20,'Invalid bounded contact row')
            prefix_hash.update(line);row=json.loads(line);require(row['sequence']==i,'Contact prefix sequence differs')
            selected=[p for p in row['patches']if p['env']==e]
            bodies={}
            for leg in LEGS:
                patches=[p for p in selected if p['body']==leg+'_tibia']
                bodies[leg]={'count':len(patches),'inactive':sum(p['inactive_zero_normal']for p in patches),
                    'toe_count':sum(p['category']=='toe'for p in patches),
                    'forces':[p['normal_force_n']for p in patches],
                    'separations_m':[p['separation_m']for p in patches]}
            rows.append({'sequence':i,'tibia_patches':bodies})
        prefix_bytes=stream.tell()
    after=contacts.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Contact stream changed while reading prefix')
    contact_prefixes[label]={'rows':rows,'first45_raw_bytes':prefix_bytes,'prefix_sha256':prefix_hash.hexdigest(),
                            'file_size_bytes':before.st_size,'full_file_expected_sha256':state['outputs']['contacts.jsonl'],
                            'full_file_rehashed':False,'mtime_ns':before.st_mtime_ns}

# Compute each tibia's exact stored mesh-extrema lower Z for only the early45steps.
# This is diagnostic geometry arithmetic, not a replacement for native contacts.
geometry=BASE/'source/geometry/geometry_extrema.npz'
geometry_sha=sha(geometry);require(geometry_sha=='268c7c072e3d84febed10000f31b4e99baf55bea461a713e42fb1137cda1e2ce','Geometry changed')
clouds={leg:list(numeric(geometry,'all__'+leg+'_tibia',(3367,3)))for leg in LEGS}
for label,fields in arrays.items():
    rows=[]
    for poses in fields['link_pose_xyzw'][:45]:
        lows=[]
        for leg in LEGS:
            j=native[label]['body_names'].index(leg+'_tibia');p=poses[j*7:(j+1)*7]
            x,y,z,w=p[3:];axis=[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]
            points=clouds[leg];lows.append(min(points[i]*axis[0]+points[i+1]*axis[1]+points[i+2]*axis[2]+p[2]for i in range(0,len(points),3)))
        rows.append(lows)
    fields['tibia_mesh_lowest_z_m_first45']=rows

comparisons={}
for a,b in [('origin','xy14_4'),('xy14_4','batch23'),('origin','batch23')]:
    result={}
    for key in arrays[a]:
        labels=native[a]['joint_names']if len(arrays[a][key][0])==18 and 'force_world'not in key else LEGS if len(arrays[a][key][0])==6 and key.startswith(('distal','tibia'))else None
        result[key]=difference(arrays[a][key],arrays[b][key],labels)
        for item in (result[key]['first_exact_difference'],result[key]['maximum_absolute_difference']):
            if item:item.update(sequence=item['row'],time_s=(item['row']+1)*.0025)
    reset_compare={}
    for phase in ('pre_reset','post_reset'):
        reset_compare[phase]={}
        for key in resets[a][phase]:
            av=flat(resets[a][phase][key][ENVS[a]]);bv=flat(resets[b][phase][key][ENVS[b]])
            if key.endswith('pose_xyzw'):av=pose_normalized(av,XY[a]);bv=pose_normalized(bv,XY[b])
            reset_compare[phase][key]=difference([av],[bv])
    reset_compare['requested_root_a']=resets[a]['requested_root'][ENVS[a]]
    reset_compare['requested_root_b']=resets[b]['requested_root'][ENVS[b]]
    comparisons[a+'_vs_'+b]={'first800':result,'reset':reset_compare}

early={label:[{'sequence':i,'time_s':(i+1)*.0025,
              'joint_position_rad':fields['joint_position_rad'][i],
              'joint_velocity_rad_s':fields['joint_velocity_rad_s'][i],
              'applied_torque_nm':fields['applied_torque_nm'][i],
              'toe_force_norm_n':fields['distal_force_norm_n'][i],
              'tibia_mesh_lowest_z_m':fields['tibia_mesh_lowest_z_m_first45'][i],
              'minimum_mesh_floor_m':fields['minimum_mesh_floor_m'][i][0],
              'minimum_non_toe_floor_m':fields['minimum_non_toe_floor_m'][i][0]}
             for i in range(45)]for label,fields in arrays.items()}
for label,path in PATHS.items():
    for name,expected in used[label].items():require(sha(path/name)==expected,'Input changed during analysis')
require(sha(geometry)==geometry_sha,'Geometry changed during analysis')
print(json.dumps({'schema':'canonical_placement_early_comparison_v1','input_pins':INPUT_PINS,
                  'scope':{'arrays_steps':800,'geometry_steps':45,'contact_prefix_steps':45,
                           'no_new_physics':True,'remote_mutations':False,'full_contact_files_rehashed':False},
                  'native_orders':{'bodies':native['origin']['body_names'],'joints':native['origin']['joint_names'],'legs':LEGS},
                  'comparisons':comparisons,'early_rows':early,'contact_prefixes':contact_prefixes,
                  'metadata':metadata,'consumed_rehashed_files':used,'geometry_sha256':geometry_sha,
                  'wall_seconds':time.monotonic()-started},indent=2,allow_nan=False))
