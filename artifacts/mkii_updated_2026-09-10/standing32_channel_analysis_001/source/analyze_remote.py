"""Read-only stdlib analysis of exact standing32_005 channels; stdout only."""
import argparse,hashlib,json,math,re,struct,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
from numeric_evidence import numeric
N=32;J=18;B=19;ROWS=800;TOTAL=8000;DT=.0025
LEGS=['lf','lm','lr','rf','rm','rr']
AUDIT_SHA='8834ba3d90b713266f4a2dc270e07a7977ca154376862148fac8329147bad4cb'
SOURCE_SHA='c87f2d340c935cd947c7aed3f3dcd7c9a9965b6e19d6726546abb560b27ab131'
INV='c5397bd5a8dd4640899ad3dec2c30f47'
FLOOR='floor_contact_force_matrix_world_n';LINK='link_com_velocity_world'
def require(ok,message):
 if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def vec(data,offset,width):return list(data[offset:offset+width])
def norm(x):return math.sqrt(sum(v*v for v in x))
def delta(a,b):return [x-y for x,y in zip(a,b)]
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def products(p):
 f=p['normal_force_n'];v=p['normal_world']
 require(math.isfinite(f)and len(v)==3 and all(math.isfinite(x)for x in v),'Malformed force/normal')
 require(f32(f)==f and all(f32(x)==x for x in v),'Original contact inputs are not exact FP32')
 return [f32(f*x)for x in v]
def aggregate(patches,n):
 toe=[[[0.,0.,0.]for _ in LEGS]for _ in range(n)];body={};counts={};seen=set()
 for p in patches:
  e=p['env'];b=p['body'];k=p['buffer_index']
  require(type(e)is int and 0<=e<n and type(k)is int and k>=0 and k not in seen,'Patch mapping/duplicate buffer index')
  seen.add(k);v=products(p);key=(e,b);body.setdefault(key,[0.,0.,0.]);counts[key]=counts.get(key,0)+1
  for j in range(3):body[key][j]+=v[j]
  if p['category']=='toe':
   require(b in [l+'_tibia'for l in LEGS],'Toe on wrong body');l=LEGS.index(b[:2])
   for j in range(3):toe[e][l][j]+=v[j]
 return toe,body,counts

def sensor_indices(contact,roots,bodies):
 paths=contact['sensor_paths'];require(len(paths)==len(roots)*len(bodies)and len(set(paths))==len(paths),'Sensor path count/duplicates')
 require(set(paths)=={r+'/'+b for r in roots for b in bodies},'Sensor coverage differs')
 require(contact['filter_paths']==[['/Ground']for _ in paths],'Wrong sole ground filter')
 return {(e,b):paths.index(root+'/'+b)for e,root in enumerate(roots)for b in bodies}
def body_com_position(pose,com):
 x,y,z,w=pose[3:];require(abs(x*x+y*y+z*z+w*w-1)<2e-5,'Bad native quaternion')
 R=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
 return [pose[j]+sum(R[j][k]*com[k]for k in range(3))for j in range(3)]
def stats(x):
 return {'count':len(x),'minimum':min(x)if x else None,'maximum':max(x)if x else None,'mean':sum(x)/len(x)if x else None}
def prior_summary():
 p=Path(__file__).parent/'inputs/prior_source003_summary.json';require(sha(p)=='a04733c7edc6ee8afbe8492a297949a555263018fc9aa3e87f58b88e2c6cfb56','Published source003 extract changed');x=read(p)
 return {'input_sha256':sha(p),**x}

def analyze(run,source,audit_path):
 started=time.monotonic();audit=read(audit_path);require(sha(audit_path)==AUDIT_SHA,'Wrong exact terminal audit')
 require(audit['audit_verified']is True and audit['expected_invocation']==INV and audit['raw_acquisition_completed']is True,'Audit/acquisition/invocation differs')
 require(audit['native_state']['standing_pass']is False and audit['native_state']['explicit_steps_completed']==TOTAL,'Original rejected acquisition differs')
 require(sha(source/'FREEZE_SHA256.json')==SOURCE_SHA,'Source freeze differs');sm=read(source/'FREEZE_SHA256.json');require(len(sm)==109,'Source map count')
 for k,h in sm.items():require(sha(source/k)==h,'Source changed:'+k)
 phase=run/'standing';used={}
 def checked(name):
  p=phase/name;b=audit['raw_inventory']['run/standing/'+name]
  require(p.stat().st_size==b['size_bytes']and sha(p)==b['sha256'],'Raw hash/size differs:'+name);used[name]=b;return p
 session=read(checked('session.json'));report=read(checked('standing_report.json'));contact=read(checked('contact_view.json'));native=read(checked('native_readback.json'));reset=read(checked('initial_reset.json'));solver=read(checked('solver_readback.json'));legacy=read(checked('legacy_friction_readback.json'))
 require(session==audit['session']and report==audit['standing_report'],'Actual metadata differs from audited value')
 names=session['joint_names'];bodies=session['body_names'];roots=session['root_paths'];masses=native['masses'];coms=native['coms']
 require(len(roots)==N and len(set(roots))==N and len(names)==J and len(set(names))==J and len(bodies)==B and len(set(bodies))==B,'Native named dimensions')
 require(native['body_names']==bodies and native['joint_names']==names,'Native names differ')
 si=sensor_indices(contact,roots,bodies);capacity=contact['capacity'];require(type(capacity)is int and capacity>0,'Invalid contact capacity')
 require(session['steps']==TOTAL and session['captured_steps']==TOTAL and session['controls']==1000 and session['all_rows_recorded']is True,'Incomplete session')
 require(session['solver_diagnostics']['channels']=={LINK:[N,B,6],FLOOR:[N*B,1,3]},'Diagnostic declaration changed')
 files=session['substep_files'];require(files==[f'substeps_{k:03d}.npz'for k in range(10)],'Unexpected chunk allocation')
 summary=[]
 for e,r in enumerate(report['replicas']):
  require(r['env']==e,'Report environment order changed');index=0 if roots[e]=='/Robot'else int(roots[e].rsplit('_',1)[1])
  summary.append({'env':e,'root_path':roots[e],'grid_xy_m':[(index%8)*2.,(index//8)*2.],'combined_pass':r['pass'],'physical_failed_bounds':r['failed_physical_bounds'],'quiet_failed_bounds':r['quiet']['failed_bounds'],'missing_substeps':r['physical']['post_settle_missing_six_toe_substeps'],'quiet_joint_sdk_rms':{k:v['velocity_rms_rad_s']for k,v in r['quiet']['joints'].items()}})
 missing=[0]*N;legs=[[0]*6 for _ in range(N)];events=[];targets={};rates=[[0.,0.,0.,0.,0.]for _ in range(N*J)];samples=0;exact_angle_rows=0
 # Chunk pass 1 discovers every post-settle event and replays SDK/angle evidence.
 for c,name in enumerate(files):
  p=checked(name);seq=numeric(p,'sequence',(ROWS,),'int');clock=numeric(p,'time_s',(ROWS,));ctrl=numeric(p,'control_index',(ROWS,),'int');counter=numeric(p,'explicit_counter',(ROWS,),'int')
  flags=numeric(p,'distal_contact',(ROWS,N,6),'bool');force=numeric(p,'distal_force_world_n',(ROWS,N,6,3));sdk=numeric(p,'joint_velocity_rad_s',(ROWS,N,J));q=numeric(p,'joint_position_rad',(ROWS,N,J));pre=numeric(p,'pre_joint_position_rad',(ROWS,N,J));interval=numeric(p,'interval_angle_rate_rad_s',(ROWS,N,J))
  for i in range(ROWS):
   s=c*ROWS+i;require(seq[i]==s and ctrl[i]==s//8 and clock[i]==(s+1)*DT,'Exact clock/sequence mismatch')
   if c==0 and i==0:counter_offset=counter[i]-1
   require(counter[i]==counter_offset+s+1,'Actual counter continuity changed')
   if s<1600:continue
   samples+=1
   for e in range(N):
    offset=(i*N+e)*J
    for j in range(J):
     k=offset+j;x=sdk[k];y=interval[k];expected=(q[k]-pre[k])/DT
     require(y==expected,json.dumps({'error':'Angle-increment mismatch','sequence':s,'env':e,'joint':names[j],'stored':y,'from_pre_post':expected}))
     a=rates[e*J+j];a[0]+=x*x;a[1]+=y*y;a[2]+=x;a[3]+=y;a[4]+=x*y
    absent=[];forces=[]
    for l in range(6):
     v=vec(force,((i*N+e)*6+l)*3,3);f=norm(v);forces.append(f)
     require(bool(flags[(i*N+e)*6+l])==(f>1.),'Original strict >1 N support differs')
     if f<=1.:absent.append(LEGS[l]);legs[e][l]+=1
    if absent:
     missing[e]+=1;events.append({'env':e,'sequence':s,'explicit_counter':counter[i],'time_s':clock[i],'missing_legs':absent,'toe_force_norm_n':forces})
     for near in range(max(0,s-2),min(TOTAL,s+3)):targets.setdefault(near,set()).add(e)
   exact_angle_rows+=N*J
 require(samples==6400 and missing==[r['physical']['post_settle_missing_six_toe_substeps']for r in report['replicas']],'Original support score count mismatch')
 rate_summary=[]
 for e in range(N):
  js={}
  for j,name in enumerate(names):
   xx,yy,x,y,xy=rates[e*J+j];mx=x/samples;my=y/samples;vx=xx/samples-mx*mx;vy=yy/samples-my*my
   js[name]={'sdk_rms':math.sqrt(xx/samples),'interval_angle_rms':math.sqrt(yy/samples),'sdk_mean':mx,'interval_mean':my,'pearson':(xy/samples-mx*my)/math.sqrt(vx*vy)if vx>0 and vy>0 else None,'sdk_minus_interval_integral_rad':(x-y)*DT}
  rate_summary.append({'env':e,'joints':js})
 # Selected numeric rows retain all 32 patch/floor vectors for exact later replay;
 # kinematics are restricted to event environments plus two neighboring steps.
 detail={};row_vectors={};global_series=[]
 for c,name in enumerate(files):
  chosen=[s for s in sorted(targets)if s//ROWS==c]
  if not chosen:continue
  p=phase/name;q=numeric(p,'joint_position_rad',(ROWS,N,J));pre=numeric(p,'pre_joint_position_rad',(ROWS,N,J));sdk=numeric(p,'joint_velocity_rad_s',(ROWS,N,J));iv=numeric(p,'interval_angle_rate_rad_s',(ROWS,N,J));pose=numeric(p,'link_pose_xyzw',(ROWS,N,B,7));lv=numeric(p,LINK,(ROWS,N,B,6));floor=numeric(p,FLOOR,(ROWS,N*B,1,3));force=numeric(p,'distal_force_world_n',(ROWS,N,6,3));rvel=numeric(p,'root_com_velocity',(ROWS,N,6));torque=numeric(p,'applied_torque_nm',(ROWS,N,J))
  for s in chosen:
   i=s%ROWS;row_vectors[s]={'toe':vec(force,i*N*18,N*18),'floor':vec(floor,i*N*B*3,N*B*3)};detail[s]={}
   for e in sorted(targets[s]):
    mass=sum(masses[e]);require(len(masses[e])==B and all(x>0 and math.isfinite(x)for x in masses[e]),'Invalid named masses')
    linkposes=[vec(pose,((i*N+e)*B+b)*7,7)for b in range(B)];linkv=[vec(lv,((i*N+e)*B+b)*6,6)for b in range(B)]
    cpos=[body_com_position(linkposes[b],coms[e][b])for b in range(B)]
    whole_v=[sum(masses[e][b]*linkv[b][k]for b in range(B))/mass for k in range(3)];whole_p=[sum(masses[e][b]*cpos[b][k]for b in range(B))/mass for k in range(3)]
    fs={b:vec(floor,(i*N*B+si[e,b])*3,3)for b in bodies};toe={l:vec(force,((i*N+e)*6+j)*3,3)for j,l in enumerate(LEGS)}
    detail[s][e]={'env':e,'sequence':s,'time_s':(s+1)*DT,'root_path':roots[e],'mass_kg':mass,'all_joint_q_rad':vec(q,(i*N+e)*J,J),'all_joint_sdk_dq_rad_s':vec(sdk,(i*N+e)*J,J),'all_joint_interval_rate_rad_s':vec(iv,(i*N+e)*J,J),'all_joint_pre_q_rad':vec(pre,(i*N+e)*J,J),'applied_torque_nm':vec(torque,(i*N+e)*J,J),'root_body_com_velocity_world':vec(rvel,(i*N+e)*6,6),'whole_robot_com_velocity_world_m_s':whole_v,'whole_robot_com_position_world_m':whole_p,'body_link_com_velocity_world':linkv[bodies.index('body')],'body_link_com_position_world_m':cpos[bodies.index('body')],'toe_patch_vectors_world_n':toe,'floor_body_vectors_world_n':fs,'floor_total_world_n':[sum(v[k]for v in fs.values())for k in range(3)],'tibia_link_com_velocity_world':{l:linkv[bodies.index(l+'_tibia')]for l in LEGS},'tibia_pose_xyzw':{l:linkposes[bodies.index(l+'_tibia')]for l in LEGS}}
 # Full JSONL stream: hash all bytes; used-slot counts come from preserved records,
 # not inferred native starts/counts. Parse event neighborhoods only.
 cp=phase/'contacts.jsonl';bound=audit['raw_inventory']['run/standing/contacts.jsonl'];h=hashlib.sha256();capacity_rows=[];patch_replay_rows=0;max_floor_gap=0.;first_floor_gap=None;targeted_raw=[]
 with cp.open('rb')as f:
  for s,line in enumerate(f):
   require(s<TOTAL and len(line)<=32<<20,'Unexpected contact stream size');h.update(line)
   m=re.match(rb'\{"sequence": (\d+), "explicit_counter": (\d+), "patches": ',line[:120]);require(m is not None and int(m[1])==s and int(m[2])==counter_offset+s+1,'JSONL clock/sequence mismatch')
   slots=[int(x)for x in re.findall(rb'"buffer_index": (\d+)',line)]
   require(len(slots)==len(set(slots))and len(slots)<capacity and (not slots or max(slots)<capacity),'Recorded used-slot capacity/range violated')
   pressure={'sequence':s,'used_count':len(slots),'max_used_index':max(slots)if slots else None,'capacity':capacity,'fraction_used':len(slots)/capacity,'missing_environments':sum(1 for x in events if x['sequence']==s)}
   capacity_rows.append(pressure)
   if s not in targets:continue
   row=json.loads(line);require(len(row['patches'])==len(slots),'Regex count does not match complete selected row')
   require(all(type(p['env'])is int and (p['env'],p['body'])in si for p in row['patches']),'Patch references an unregistered sensor/body')
   toe,body,counts=aggregate(row['patches'],N);rv=row_vectors[s]
   for e in range(N):
    for l in range(6):
     expected=rv['toe'][(e*6+l)*3:(e*6+l)*3+3]
     require(toe[e][l]==expected,json.dumps({'error':'Exact FP32 product/FP64 sum mismatch','sequence':s,'env':e,'leg':LEGS[l],'replay':toe[e][l],'stored':expected,'delta':delta(toe[e][l],expected)},allow_nan=False))
    for b in bodies:
     expected=rv['floor'][si[e,b]*3:si[e,b]*3+3];pvec=body.get((e,b),[0.,0.,0.]);gap=norm(delta(expected,pvec));max_floor_gap=max(max_floor_gap,gap)
     if gap and first_floor_gap is None:first_floor_gap={'sequence':s,'env':e,'body':b,'floor_matrix_vector':expected,'all_body_patch_FP32_vector':pvec,'difference':delta(expected,pvec)}
   patch_replay_rows+=1
   selected=[]
   for e,entry in detail[s].items():
    ep=[p for p in row['patches']if p['env']==e]
    entry['used_patch_count_env']=len(ep);entry['inactive_zero_normal_count_env']=sum(p['inactive_zero_normal']for p in ep);entry['negative_force_count_env']=sum(p['normal_force_n']<0 for p in ep)
    entry['used_patch_count_by_body']={b:counts.get((e,b),0)for b in bodies};entry['all_body_patch_vectors_world_n']={b:body.get((e,b),[0.,0.,0.])for b in bodies}
    entry['patch_total_world_n']=[sum(v[k]for v in entry['all_body_patch_vectors_world_n'].values())for k in range(3)];entry['toe_patch_aggregation_exact']=True
    selected.append({'env':e,'patches':ep})
   targeted_raw.append({'sequence':s,'explicit_counter':int(m[2]),'global_pressure':pressure,'environments':selected})
 require(len(capacity_rows)==TOTAL and cp.stat().st_size==bound['size_bytes']and h.hexdigest()==bound['sha256'],'Contact full stream hash/size/row count mismatch');used['contacts.jsonl']=bound
 # Derivatives never substitute for the original SDK quiet or support gates.
 out_events=[]
 for event in events:
  s=event['sequence'];e=event['env'];window=[]
  for t in range(max(1,s-1),min(TOTAL,s+2)):
   now=detail[t][e];prev=detail[t-1][e]
   obs=dict(now);acc=[x/DT for x in delta(now['whole_robot_com_velocity_world_m_s'],prev['whole_robot_com_velocity_world_m_s'])]
   obs['whole_robot_com_acceleration_from_native_velocity_m_s2']=acc
   obs['body_link_com_acceleration_from_native_velocity_m_s2']=[x/DT for x in delta(now['body_link_com_velocity_world'][:3],prev['body_link_com_velocity_world'][:3])]
   obs['expected_contact_force_from_native_momentum_world_n']=[now['mass_kg']*(acc[k]+(9.81 if k==2 else 0))for k in range(3)]
   obs['floor_minus_momentum_force_n']=delta(now['floor_total_world_n'],obs['expected_contact_force_from_native_momentum_world_n'])
   obs['all_six_patch_vectors_exact_zero']=all(x==0 for v in now['toe_patch_vectors_world_n'].values()for x in v)
   obs['all_eighteen_sdk_dq_exact_zero']=all(x==0 for x in now['all_joint_sdk_dq_rad_s'])
   if t-2 in detail and e in detail[t-2]:
    prior=detail[t-2][e];obs['whole_robot_com_second_difference_acceleration_m_s2']=[(now['whole_robot_com_position_world_m'][k]-2*prev['whole_robot_com_position_world_m'][k]+prior['whole_robot_com_position_world_m'][k])/(DT*DT)for k in range(3)]
   # Matrix is all tibia contacts. It cannot turn shaft contact into valid toe support.
   obs['missing_leg_channel_comparison']={l:{'toe_patch_norm_n':norm(now['toe_patch_vectors_world_n'][l]),'tibia_floor_matrix_norm_n':norm(now['floor_body_vectors_world_n'][l+'_tibia']),'all_tibia_patch_norm_n':norm(now['all_body_patch_vectors_world_n'][l+'_tibia']),'tibia_raw_patch_count':now['used_patch_count_by_body'][l+'_tibia']}for l in event['missing_legs']}
   obs['global_pressure']=capacity_rows[t];window.append(obs)
  out_events.append({**event,'window':window})
 channel_summary={'missing_foot_events':0,'detailed_toe_low_tibia_matrix_above_1N':0,'detailed_toe_low_tibia_matrix_at_or_below_1N':0,'all_six_patch_vectors_zero_and_all18_sdk_dq_zero_event_env_rows':0}
 for event in out_events:
  center=next(x for x in event['window']if x['sequence']==event['sequence'])
  channel_summary['all_six_patch_vectors_zero_and_all18_sdk_dq_zero_event_env_rows']+=int(center['all_six_patch_vectors_exact_zero']and center['all_eighteen_sdk_dq_exact_zero'])
  for value in center['missing_leg_channel_comparison'].values():
   channel_summary['missing_foot_events']+=1
   channel_summary['detailed_toe_low_tibia_matrix_above_1N'if value['tibia_floor_matrix_norm_n']>1. else'detailed_toe_low_tibia_matrix_at_or_below_1N']+=1
 prev=prior_summary();prior_by={r['env']:r for r in prev['replicas']};comparison=[]
 for e,r in enumerate(summary):
  old=prior_by[e];comparison.append({'env':e,'grid_xy_m':r['grid_xy_m'],'current_missing_substeps':r['missing_substeps'],'source003_missing_substeps':old['missing_support_substeps'],'current_combined_pass':r['combined_pass'],'source003_combined_pass':old['pass'],'current_quiet_failed_bounds':r['quiet_failed_bounds'],'source003_quiet_failed_bounds':old['quiet_failed_bounds']})
 for name,b in used.items():require(sha(phase/name)==b['sha256'],'Consumed raw changed after analysis:'+name)
 for k,hsh in sm.items():require(sha(source/k)==hsh,'Source changed after analysis:'+k)
 result={'schema':'canonical_standing32_channel_failure_analysis_v1','CPU_only':True,'read_only':True,'original_standing_pass':False,'source_freeze_sha256':SOURCE_SHA,'audit_sha256':AUDIT_SHA,'invocation':INV,'source_payloads':len(sm),'native_report_unchanged':report,'solver_readback':solver,'legacy_friction_readback':legacy,'replicas':summary,'original_support_counts_exact':True,'missing_counts_by_leg':[dict(env=e,legs=dict(zip(LEGS,legs[e])))for e in range(N)],'events':out_events,'event_channel_summary_observations_only':channel_summary,'rates_postsettle400Hz_float64_accumulation':rate_summary,'exact_pre_post_angle_scalar_comparisons':exact_angle_rows,'patch_FP32_product_FP64_sum_exact_rows_all32':patch_replay_rows,'maximum_all_body_floor_minus_patch_vector_norm_n_in_selected_rows':max_floor_gap,'first_floor_patch_difference':first_floor_gap,'recorded_capacity':capacity,'used_slot_count_all8000_rows':capacity_rows,'pressure_summary_all':stats([x['used_count']for x in capacity_rows]),'pressure_summary_event_rows':stats([x['used_count']for x in capacity_rows if x['missing_environments']]),'targeted_raw_patch_rows':targeted_raw,'source003_comparison':comparison,'source003_prior_reference':{k:prev[k]for k in ['full_analysis_sha256','full_analysis_reference','scope','input_sha256']},'source003_rates_400hz':prev['rates_400hz_float64_accumulation'],'consumed_raw_inputs':used,'full_audit_raw_inventory_reference':AUDIT_SHA,'raw_and_source_reverified_after_analysis':True,'clock_offset':counter_offset,'joint_names':names,'body_names':bodies,'sensor_paths':contact['sensor_paths'],'diagnostic_channels':session['solver_diagnostics'],'seconds':time.monotonic()-started,'analyzer_sha256':sha(Path(__file__)),'numeric_reader_sha256':sha(Path(__file__).with_name('numeric_evidence.py')),'limitations':['Native floor matrix and detailed patches may share a backend; agreement cannot prove completeness or support.','Recorded patch count/index reconstructs used slots only; original native starts/counts arrays and global PhysX buffer occupancy were not exported.','Whole-robot native momentum acceleration and pose second differences are diagnostics, not independent physics truth; pose differencing amplifies FP32 grid quantization.','Force balance uses all 19 measured link masses/COM velocities and gravity, but does not isolate hidden constraints, cache/reporting or solver mechanisms.','Source003 comparison changes solver32/1 to32/0 and capture instrumentation, with batch state confounds; it lacks new force/link channels.','All original strict >1 N six-toe support and SDK quiet limits remain unchanged; no favorable diagnostic rate or channel replaces a failed gate.']}
 return result

def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);a=p.parse_args()
 print(json.dumps(analyze(a.run,a.source,a.audit),indent=2,allow_nan=False))
if __name__=='__main__':main()
