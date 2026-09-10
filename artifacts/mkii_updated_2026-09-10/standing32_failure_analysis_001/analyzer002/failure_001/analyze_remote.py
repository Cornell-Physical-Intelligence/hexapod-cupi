"""Bounded read-only stdlib analysis of original standing32_002; stdout only."""
from pathlib import Path
import argparse,array,hashlib,json,math,re,struct,sys
sys.dont_write_bytecode=True
from numeric_evidence import numeric
LEGS=['lf','lm','lr','rf','rm','rr'];DT=.0025;N=32;J=18
AUDIT_SHA='c5e6697742a61d53f82c42324f45d7c8cdd54a77625fc5b0b955232ab0e080f5'
SOURCE_FREEZE='e922ff1312b5106f957b384874ca5eb6ed4a9ed6566c0ef9a0602247a49670f0'
INV='1a38495bda9f4fefa4e5585574aabc56'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def norm(v):return math.sqrt(sum(x*x for x in v))
def shape_point(point,pose,T):
 x,y,z,w=pose[3:];require(abs(x*x+y*y+z*z+w*w-1)<2e-5,'Invalid pose quaternion')
 R=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
 local=[sum((point[k]-pose[k])*R[k][j]for k in range(3))for j in range(3)]
 return [sum((local[k]-T[k][3])*T[k][j]for k in range(3))for j in range(3)]
def summarize_report(report,roots):
 out=[]
 for r,root in zip(report['replicas'],roots):
  i=0 if root=='/Robot'else int(root.rsplit('_',1)[1]);q=r['quiet']
  out.append({'env':r['env'],'root_path':root,'authored_index':i,'grid_xy_m':[(i%8)*2.,(i//8)*2.],'pass':r['pass'],'missing_support_substeps':r['physical']['post_settle_missing_six_toe_substeps'],'quiet_failed_bounds':q['failed_bounds'],'failed_physical_bounds':r['failed_physical_bounds'],'failed_rate_joints':{k:v for k,v in q['joints'].items()if v['velocity_rms_rad_s']>.03},'max_q_range_rad':q['max_joint_position_range_rad'],'planar_excursion_m':q['max_planar_excursion_m']})
 return out

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--run',type=Path,required=True);p.add_argument('--single',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);a=p.parse_args()
 audit=read(a.audit);require(sha(a.audit)==AUDIT_SHA,'Wrong original audited report');require(audit['audit_verified']and audit['expected_invocation']==INV,'Wrong terminal audit')
 require(sha(a.source/'FREEZE_SHA256.json')==SOURCE_FREEZE,'Wrong original source')
 source_map=read(a.source/'FREEZE_SHA256.json');require(len(source_map)==87,'Source size')
 for k,h in source_map.items():require(sha(a.source/k)==h,'Source changed:'+k)
 d=a.run/'standing';used={};single_used={}
 def checked(name):
  path=d/name;b=audit['raw_inventory']['run/standing/'+name];require(path.stat().st_size==b['size_bytes']and sha(path)==b['sha256'],'Raw changed:'+name);used[name]=b;return path
 def single(name):
  path=a.single/name;b=audit['standing_one_admission_inventory'][name];require(path.stat().st_size==b['size_bytes']and sha(path)==b['sha256'],'Standing1 changed:'+name);single_used[name]=b;return path
 session=read(checked('session.json'));report=read(checked('standing_report.json'));reset=read(checked('initial_reset.json'));contact=read(checked('contact_view.json'));native=read(checked('native_readback.json'));geometry={s['body']:s for s in read(a.source/'geometry/geometry.json')['shapes']};single_session=read(single('session.json'))
 names=session['joint_names'];bodies=session['body_names'];roots=session['root_paths'];require(len(roots)==32 and len(set(roots))==32 and len(names)==18 and len(bodies)==19,'Actual names shape')
 require(names==single_session['joint_names']and bodies==single_session['body_names'],'Standing1 named mapping differs')
 expected_paths={r+'/'+b for r in roots for b in bodies};require(len(contact['sensor_paths'])==608 and set(contact['sensor_paths'])==expected_paths,'Contact path coverage differs')
 require(contact['filter_paths']==[['/Ground'] for _ in range(608)],'Contact filter differs')
 result={'schema':'canonical_standing32_failure_analysis_v1','read_only':True,'CPU_only':True,'original_audit_sha256':AUDIT_SHA,'source_freeze_sha256':SOURCE_FREEZE,'native_standing_pass':False,'host_error':audit['job']['error'],'replicas':summarize_report(report,roots),'contact_paths_exact':True,'source_mapping_method':'Contact sensor paths map by exact root and body names, not array index order; raw classified patches omit original starts/counts so cannot independently prove native count/start ABI from JSONL alone.','limitations':['Observational comparison only: translations and batch context are confounded.','Interval-angle rate is diagnostic only; original SDKrate/support gates remain unchanged.','Joint-rate differences cannot by themselves identify an unobserved solver mechanism.'],'native_checks':audit['native_state']['checks']}
 require(roots[0]=='/Robot','Origin env index changed')
 requested=reset['requested_root'];post=reset['post_reset'];single_reset=read(single('initial_reset.json'))['post_reset'];e0=post['link_pose_xyzw'][0];reset_rows=[]
 for e,root in enumerate(roots):
  i=0 if root=='/Robot'else int(root.rsplit('_',1)[1]);grid=[(i%8)*2.,(i//8)*2.]
  require(requested[e][:2]==grid,'Authored path/reset grid mismatch');actual=post['root_pose_xyzw'][e]
  rooterr=max(abs(x-y)for x,y in zip(actual,requested[e]));localerr=max(abs((post['link_pose_xyzw'][e][b][axis]-actual[axis])-(e0[b][axis]-post['root_pose_xyzw'][0][axis]))for b in range(19)for axis in range(3));quaternionerr=max(abs(post['link_pose_xyzw'][e][b][axis]-e0[b][axis])for b in range(19)for axis in range(3,7))
  require(rooterr<=2e-6 and all(x==0 for x in post['joint_position_rad'][e]+post['joint_velocity_rad_s'][e]),'Reset differs')
  reset_rows.append({'env':e,'root':root,'grid_xy_m':grid,'root_reset_max_abs_error':rooterr,'translated_link_position_max_abs_delta_from_env0_m':localerr,'link_quaternion_max_delta_from_env0':quaternionerr})
 result['reset_comparison']=reset_rows;result['env0_reset_equal_single']={k:post[k][0]==single_reset[k][0]for k in post}
 result['native_replicas_equal_env0']={k:all(v[e]==v[0]for e in range(1,32))for k,v in native.items()if isinstance(v,list)and len(v)==32}
 # Read one compressed chunk/key at a time. No full contact or multi-chunk numeric accumulation.
 sq_sdk=[0.]*(N*J);sq_interval=[0.]*(N*J);sum_sdk=[0.]*(N*J);sum_interval=[0.]*(N*J);cross=[0.]*(N*J);qmin=[math.inf]*(N*J);qmax=[-math.inf]*(N*J)
 missing=[[]for _ in range(N)];leg_counts=[[0]*6 for _ in range(N)];force_min=[[math.inf]*6 for _ in range(N)];events=[];targets={};parity={k:{'equal':True,'max_abs_difference':0.}for k in ['joint_position_rad','joint_velocity_rad_s','root_pose_xyzw','distal_force_world_n','distal_contact']};postrows=0
 for chunk,file in enumerate(session['substep_files']):
  path=checked(file);sp=single(file);seq=numeric(path,'sequence',(800,),'int');ctrl=numeric(path,'control_index',(800,),'int');t=numeric(path,'time_s',(800,));q=numeric(path,'joint_position_rad',(800,N,J));sdk=numeric(path,'joint_velocity_rad_s',(800,N,J));interval=numeric(path,'interval_angle_rate_rad_s',(800,N,J));flags=numeric(path,'distal_contact',(800,N,6),'bool');forces=numeric(path,'distal_force_world_n',(800,N,6,3));rpose=numeric(path,'root_pose_xyzw',(800,N,7))
  for key,data,width,kind in [('joint_position_rad',q,18,'float'),('joint_velocity_rad_s',sdk,18,'float'),('root_pose_xyzw',rpose,7,'float'),('distal_force_world_n',forces,18,'float'),('distal_contact',flags,6,'bool')]:
   shape=(800,1,6,3)if key=='distal_force_world_n'else(800,1,width);orig=numeric(sp,key,shape,kind)
   delta=max(abs(data[(i*N)*width+j]-orig[i*width+j])for i in range(800)for j in range(width));parity[key]['equal']&=delta==0;parity[key]['max_abs_difference']=max(parity[key]['max_abs_difference'],delta)
  for i in range(800):
   sequence=chunk*800+i;require(seq[i]==sequence and ctrl[i]==sequence//8 and abs(t[i]-(sequence+1)*DT)<1e-12,'Raw sequence/time mismatch')
   if ctrl[i]<200:continue
   postrows+=1
   for e in range(N):
    off=(i*N+e)*J;s=e*J
    for j in range(J):
     x=sdk[off+j];y=interval[off+j];sq_sdk[s+j]+=x*x;sq_interval[s+j]+=y*y;sum_sdk[s+j]+=x;sum_interval[s+j]+=y;cross[s+j]+=x*y;qmin[s+j]=min(qmin[s+j],q[off+j]);qmax[s+j]=max(qmax[s+j],q[off+j])
    absent=[]
    for l in range(6):
     f=norm(forces[(i*N+e)*18+l*3:(i*N+e)*18+l*3+3]);force_min[e][l]=min(force_min[e][l],f);require(bool(flags[(i*N+e)*6+l])==(f>1.),'Support flags differ from exact stored resultant')
     if not flags[(i*N+e)*6+l]:absent.append(LEGS[l]);leg_counts[e][l]+=1
    if absent:
     missing[e].append(sequence);events.append({'env':e,'sequence':sequence,'time_s':t[i],'missing_legs':absent,'all_foot_force_norm_n':[norm(forces[(i*N+e)*18+l*3:(i*N+e)*18+l*3+3])for l in range(6)]})
     for near in range(max(0,sequence-1),min(8000,sequence+2)):
      targets.setdefault(near,{}).setdefault(e,set()).update(leg+'_tibia'for leg in absent)
 require(postrows==6400,'Postsettle sample count');require([len(m)for m in missing]==[r['physical']['post_settle_missing_six_toe_substeps']for r in report['replicas']],'Missing support count differs from original score')
 result['env0_substep_parity_to_single']=parity;result['support_events']=events;result['postsettle_force_minima_n']=[dict(env=e,feet=dict(zip(LEGS,force_min[e])),missing_counts=dict(zip(LEGS,leg_counts[e])))for e in range(N)]
 rate_rows=[]
 for e in range(N):
  joint={}
  for j,name in enumerate(names):
   k=e*J+j;mx=sum_sdk[k]/postrows;my=sum_interval[k]/postrows;vx=sq_sdk[k]/postrows-mx*mx;vy=sq_interval[k]/postrows-my*my
   joint[name]={'sdk_rms_400hz':math.sqrt(sq_sdk[k]/postrows),'interval_rms_400hz':math.sqrt(sq_interval[k]/postrows),'sdk_mean_400hz':mx,'interval_mean_400hz':my,'sdk_minus_interval_integral_16s_rad':(sum_sdk[k]-sum_interval[k])*DT,'pearson_sdk_interval':(cross[k]/postrows-mx*my)/math.sqrt(vx*vy)if vx>0 and vy>0 else None,'q_range_400hz_rad':qmax[k]-qmin[k]}
  rate_rows.append({'env':e,'joints':joint})
 result['rates_400hz_float64_accumulation']=rate_rows
 # Capture exact target joints and body poses around every missing-foot event.
 detail={}
 for chunk,file in enumerate(session['substep_files']):
  selected=[s for s in sorted(targets)if s//800==chunk]
  if not selected:continue
  path=d/file;q=numeric(path,'joint_position_rad',(800,N,J));dq=numeric(path,'joint_velocity_rad_s',(800,N,J));iv=numeric(path,'interval_angle_rate_rad_s',(800,N,J));pose=numeric(path,'link_pose_xyzw',(800,N,19,7));forces=numeric(path,'distal_force_world_n',(800,N,6,3))
  for s in selected:
   i=s%800;detail[s]=[]
   for e,bs in sorted(targets[s].items()):
    for body in sorted(bs):
     b=bodies.index(body);l=LEGS.index(body[:2]);js=[j for j,n in enumerate(names)if n.startswith(body[:2]+'_')]
     detail[s].append({'env':e,'body':body,'pose_xyzw':list(pose[((i*N+e)*19+b)*7:((i*N+e)*19+b)*7+7]),'stored_toe_force_world_n':list(forces[(i*N+e)*18+l*3:(i*N+e)*18+l*3+3]),'joints':{names[j]:{'q':q[(i*N+e)*18+j],'sdk_dq':dq[(i*N+e)*18+j],'interval_dq':iv[(i*N+e)*18+j]}for j in js}})
 # Hash every original JSONL byte; only parse selected rows and verify all row prefixes.
 cp=d/'contacts.jsonl';bound=audit['raw_inventory']['run/standing/contacts.jsonl'];h=hashlib.sha256();count=0;contact_samples=[]
 with cp.open('rb')as f:
  for sequence,line in enumerate(f):
   h.update(line);count+=1
   m=re.match(rb'\{"sequence": (\d+), "explicit_counter": (\d+), "patches": ',line[:120]);require(m is not None and int(m[1])==sequence,'JSONL sequence prefix differs')
   if sequence not in targets:continue
   row=json.loads(line);require(row['sequence']==sequence,'JSONL row mismatch')
   selected=[p for p in row['patches']if p['env']in targets[sequence]and p['body']in targets[sequence][p['env']]]
   for pch in selected:require(pch['env']in range(32)and roots[pch['env']]+'/'+pch['body']in expected_paths,'Patch named mapping differs')
   for entry in detail[sequence]:
    ps=[p for p in selected if p['env']==entry['env']and p['body']==entry['body']];agg=[0.,0.,0.]
    for pch in ps:
     if pch['category']=='toe':
      for axis in range(3):agg[axis]+=pch['normal_force_n']*pch['normal_world'][axis]
    require(agg==entry['stored_toe_force_world_n'],'Raw patch aggregate differs from NPZ')
    shape=geometry[entry['body']];T=shape['shape_to_link'];skin=shape['contact_offset_m'];bounds=shape['cap_bounds_m'];maxerr=0.;minedge=None
    for patch in ps:
     local=shape_point(patch['point_world_m'],entry['pose_xyzw'],T);maxerr=max(maxerr,max(abs(x-y)for x,y in zip(local,patch['shape_point_m'])))
     category='toe' if local[0]>=shape['cap_lower_x_m']-1e-12 and local[0]<=bounds[1][0]+skin and all(bounds[0][j]-skin<=local[j]<=bounds[1][j]+skin for j in (1,2)) else 'shaft'
     require(category==patch['category'],'Independent shape/category projection differs')
     if patch['normal_force_n']!=0:
      edge=min(local[0]-shape['cap_lower_x_m'],bounds[1][0]+skin-local[0],*(local[j]-(bounds[0][j]-skin)for j in (1,2)),*((bounds[1][j]+skin)-local[j]for j in (1,2)))
      minedge=edge if minedge is None else min(minedge,edge)
    require(maxerr<1e-10,'Independent world-to-shape transform differs')
    entry['independent_shape_projection_max_error_m']=maxerr;entry['minimum_active_patch_cap_boundary_signed_margin_m']=minedge
    entry['raw_patches']=ps;entry['raw_aggregate_exact']=True
   contact_samples.append({'sequence':sequence,'explicit_counter':row['explicit_counter'],'observations':detail[sequence]})
 require(count==8000 and cp.stat().st_size==bound['size_bytes']and h.hexdigest()==bound['sha256'],'Original JSONL size/hash/count mismatch');used['contacts.jsonl']=bound
 result['targeted_contact_samples']=contact_samples;result['contact_stream_original_sha256']=h.hexdigest();result['all_raw_inputs_used']=used;result['single_inputs_used']=single_used
 # Recheck all consumed originals after computation without any write to them.
 for name,b in used.items():require(sha(d/name)==b['sha256'],'Final raw changed:'+name)
 for name,b in single_used.items():require(sha(a.single/name)==b['sha256'],'Final standing1 changed:'+name)
 result['used_raw_reverified_after_analysis']=True;result['self_sha256']=sha(Path(__file__));result['helper_sha256']=sha(Path(__file__).with_name('numeric_evidence.py'))
 print(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__':main()
