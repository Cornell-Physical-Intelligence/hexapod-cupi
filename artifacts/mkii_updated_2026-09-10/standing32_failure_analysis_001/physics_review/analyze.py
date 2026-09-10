"""Independent CPU source/one-row dynamics study; no native32 raw claim."""
from pathlib import Path
import argparse,gzip,hashlib,json,sys
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'report.json');a=p.parse_args();repo=a.repo.resolve();src=repo/'tmp/updated_native_standing_003';one=repo/'tmp/canonical_native_standing_terminal_004';audit=repo/'tmp/canonical_native_standing32_terminal_002/audit.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
source=read(src/'FREEZE_SHA256.json');assert sha(src/'FREEZE_SHA256.json')=='e922ff1312b5106f957b384874ca5eb6ed4a9ed6566c0ef9a0602247a49670f0'
for name,bound in source.items():assert sha(src/name)==bound,name
rawmap=read(one/'RAW_SHA256.json');inputs={str(src/'FREEZE_SHA256.json'):sha(src/'FREEZE_SHA256.json'),str(audit):sha(audit),str(one/'RAW_SHA256.json'):sha(one/'RAW_SHA256.json')}
def checked(rel):
 f=one/rel;assert sha(f)==rawmap[rel]['sha256'];inputs[str(f)]=sha(f);return f
session=read(checked('run/standing/session.json'));names=session['joint_names'];rows=[]
keys=['pre_joint_position_rad','pre_joint_velocity_rad_s','joint_position_rad','joint_velocity_rad_s','interval_angle_rate_rad_s','joint_target_rad','computed_torque_nm','applied_torque_nm','control_index','sequence','link_pose_xyzw','distal_force_world_n']
for name in session['substep_files']:
 with np.load(checked('run/standing/'+name))as z:rows.append({k:z[k].copy() for k in keys})
data={k:np.concatenate([r[k]for r in rows])for k in keys};post=data['control_index']>=200;assert post.sum()==6400 and len(post)==8000
joints=[];g=read(src/'servo_candidate.json');order=[g['joint_names'].index(n)for n in names];kp=np.array(g['stiffness_nm_per_rad'])[order];kd=np.array(g['damping_nm_s_per_rad'])[order]
for j,name in enumerate(names):
 q=data['joint_position_rad'][post,0,j].astype(float);dq=data['joint_velocity_rad_s'][post,0,j].astype(float);iq=data['interval_angle_rate_rad_s'][post,0,j]
 joints.append({'name':name,'sdk_rms_400hz':float(np.sqrt(np.mean(dq*dq))),'sdk_mean_400hz':float(dq.mean()),'interval_angle_rms_400hz':float(np.sqrt(np.mean(iq*iq))),'joint_range_rad_400hz':float(np.ptp(q)),'sdk_integral_minus_angle_increment_rad':float(np.sum(dq)*.0025-np.sum(iq)*.0025),'damping_from_sdk_rms_nm':float(kd[j]*np.sqrt(np.mean(dq*dq))),'damping_if_interval_rms_nm_NOT_applied':float(kd[j]*np.sqrt(np.mean(iq*iq))),'sdk_interval_correlation':float(np.corrcoef(dq,iq)[0,1])})
# Exact existing geometry transform on sampled actual1 patches; translation then
# float32 rounding mimics the public native output precision, not the solver.
sys.path.insert(0,str(src));from standing_math import Geometry,servo
np.testing.assert_allclose(data['interval_angle_rate_rad_s'],(data['joint_position_rad'].astype(float)-data['pre_joint_position_rad'].astype(float))/.0025,atol=1e-12,rtol=0)
a_raw,a_apply,_=servo(data['pre_joint_position_rad'].reshape(-1,18),data['pre_joint_velocity_rad_s'].reshape(-1,18),data['joint_target_rad'].reshape(-1,18),kp.astype(np.float32),kd.astype(np.float32))
assert np.array_equal(a_raw.reshape(8000,1,18),data['computed_torque_nm']) and np.array_equal(a_apply.reshape(8000,1,18),data['applied_torque_nm'])
with np.load(src/'geometry/geometry_extrema.npz')as z:geometry=Geometry(read(src/'geometry/geometry.json'),{k:z[k]for k in z.files},session['body_names'])
selected=set(range(0,8000,400))|{7999};pointcount=0;comparisoncount=0;flips=0;maxerr=0.;minmargin=float('inf');rawhash=hashlib.sha256()
with gzip.open(one/'run/standing/contacts.jsonl.gz','rb')as stream:
 for line in stream:
  rawhash.update(line);row=json.loads(line)
  if row['sequence']not in selected:continue
  poses=data['link_pose_xyzw'][row['sequence'],0]
  for patch in row['patches']:
   body=patch['body']
   if not body.endswith('_tibia'):continue
   pointcount+=1;point=np.asarray(patch['point_world_m'],np.float32);pose=poses[session['body_names'].index(body)].copy();ok,base=geometry.cap(body,point,pose)
   assert ('toe'if ok else'shaft')==patch['category']
   sh=geometry.shapes[body];b=np.asarray(sh['cap_bounds_m']);skin=sh['contact_offset_m'];margins=[base[0]-sh['cap_lower_x_m'],b[1,0]+skin-base[0],*(base[1:]-(b[0,1:]-skin)),*((b[1,1:]+skin)-base[1:])]
   if ok:minmargin=min(minmargin,min(margins))
   for ix in range(8):
    for iy in range(4):
     shift=np.array([2*ix,2*iy,0],np.float32);pt=(point+shift).astype(np.float32);ps=pose.copy();ps[:3]=(ps[:3]+shift).astype(np.float32);got,local=geometry.cap(body,pt,ps);maxerr=max(maxerr,float(np.max(abs(local-base))));flips+=got!=ok;comparisoncount+=1
assert rawhash.hexdigest()==rawmap['run/standing/contacts.jsonl']['sha256'];inputs[str(one/'run/standing/contacts.jsonl.gz')]=sha(one/'run/standing/contacts.jsonl.gz')
x=read(audit);summary=[]
for r in x['standing_report']['replicas']:
 q=r['quiet'];physical=r['physical'];summary.append({'env':r['env'],'pass':r['pass'],'quiet_failed':q['failed_bounds'],'missing_six_toe_substeps':physical['post_settle_missing_six_toe_substeps'],'sdk_worst_rms_50hz':q['max_joint_velocity_rms_rad_s'],'q_worst_range_50hz':q['max_joint_position_range_rad'],'requested_all_peak':physical['max_requested_all_substeps_nm'],'requested_saturation_400hz':physical['max_requested_saturation_fraction_400hz']})
result={'schema':'canonical_standing32_independent_source_dynamics_review_v1','inputs':inputs,'raw32_scope':'Only root-verified terminal audit embedded report; no independently loaded32 arrays. Sensor agent owns full32 numerical replay.','summary32':summary,'pass_count':sum(r['pass']for r in summary),'support_failed_count':sum(r['missing_six_toe_substeps']>0 for r in summary),'SDK_quiet_failed_count':sum(bool(r['quiet_failed']) for r in summary),'max_requested_all32_nm':max(r['requested_all_peak']for r in summary),'one_interval_reconstruction_all8000':True,'one_servo_replay_all8000':True,'one_postsettle_400hz':joints,'translation_float32_classifier_test':{'sample_sequences':sorted(selected),'actual_tibia_patches':pointcount,'comparisons':comparisoncount,'category_flips':int(flips),'worst_shape_coordinate_delta_m':maxerr,'minimum_sampled_cap_margin_m':float(minmargin),'scope':'Existing geometry/classifier math, actual1 points and poses translated+rounded to32 grid positions. Does not simulate native contact or prove solver translation invariance.'},'no_source_changes':True,'GPU_calls':0,'no_gate_waiver':True}
a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'pass_count':result['pass_count'],'support_failed_count':result['support_failed_count'],'SDK_quiet_failed_count':result['SDK_quiet_failed_count'],'translation':result['translation_float32_classifier_test'],'middle_tibia':[r for r in joints if r['name']in['lm_tibia_pitch','rm_tibia_pitch']]}))
