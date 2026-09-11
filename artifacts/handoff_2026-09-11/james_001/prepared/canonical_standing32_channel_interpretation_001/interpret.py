"""Read-only exact interpretation of the root's frozen actual CPU result."""
import argparse,collections,hashlib,json,math,statistics
from pathlib import Path
EXPECTED='304bccb643f1be8ab95a41d98c85c985e7248652a208e20ac51b71006ff60231'
SOURCE='c87f2d340c935cd947c7aed3f3dcd7c9a9965b6e19d6726546abb560b27ab131'
ANALYZER='8d4146b847224e08b1b1fded1e5832bc0e7dee56bd5f06ee6cb1e6be44cd8c28'
def norm(v):return math.sqrt(sum(x*x for x in v))
def stats(x):
 x=list(x)
 return {'count':len(x),'min':min(x),'median':statistics.median(x),'max':max(x),'mean':statistics.mean(x)}if x else {'count':0,'min':None,'median':None,'max':None,'mean':None}
def interpret(p):
 raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==EXPECTED
 A=json.loads(raw);assert A['source_freeze_sha256']==SOURCE and A['analyzer_sha256']==ANALYZER and A['raw_and_source_reverified_after_analysis']is True and A['original_standing_pass']is False
 rows=[];patchmap={(r['sequence'],e['env']):e['patches']for r in A['targeted_raw_patch_rows']for e in r['environments']};N=A['joint_names']
 for event in A['events']:
  win={x['sequence']:x for x in event['window']};s=event['sequence'];e=event['env']
  for l in event['missing_legs']:
   r={'env':e,'sequence':s,'leg':l,'time_s':event['time_s']};inds=[j for j,n in enumerate(N)if n.startswith(l+'_')];r['joint_names']=[N[j]for j in inds]
   for off,label in [(-1,'before'),(0,'at'),(1,'after')]:
    w=win[s+off];ps=[p for p in patchmap[s+off,e]if p['body']==l+'_tibia']
    r[label]={'patch_force_n':norm(w['toe_patch_vectors_world_n'][l]),'matrix_force_n':norm(w['floor_body_vectors_world_n'][l+'_tibia']),'patch_count':len(ps),'inactive_count':sum(p['inactive_zero_normal']for p in ps),'nonzero_count':sum(p['normal_force_n']!=0 for p in ps),'all_patch_normals_exactzero':all(all(v==0 for v in p['normal_world'])for p in ps),'all_patch_separation_exactzero':all(p['separation_m']==0 for p in ps),'point_range_world_m':[[min(p['point_world_m'][j]for p in ps),max(p['point_world_m'][j]for p in ps)]for j in range(3)]if ps else None,'allfloor_z_n':w['floor_total_world_n'][2],'momentum_z_n':w['expected_contact_force_from_native_momentum_world_n'][2],'force_balance_z_n':w['floor_minus_momentum_force_n'][2],'com_acc_z_m_s2':w['whole_robot_com_acceleration_from_native_velocity_m_s2'][2],'pose_acc_z_m_s2':w.get('whole_robot_com_second_difference_acceleration_m_s2',[None]*3)[2],'body_acc_z_m_s2':w['body_link_com_acceleration_from_native_velocity_m_s2'][2],'whole_com_vz_m_s':w['whole_robot_com_velocity_world_m_s'][2],'whole_com_z_m':w['whole_robot_com_position_world_m'][2],'joint_q_rad':[w['all_joint_q_rad'][j]for j in inds],'joint_sdk_rad_s':[w['all_joint_sdk_dq_rad_s'][j]for j in inds],'joint_interval_rad_s':[w['all_joint_interval_rate_rad_s'][j]for j in inds],'tibia_vz_m_s':w['tibia_link_com_velocity_world'][l][2],'tibia_z_m':w['tibia_pose_xyzw'][l][2],'separation_range_m':[min(p['separation_m']for p in ps),max(p['separation_m']for p in ps)]if ps else None,'applied_torque_nm':[w['applied_torque_nm'][j]for j in inds],'pressure_used':w['global_pressure']['used_count']}
   r['step_impulses']={}
   for first,last in [('before','at'),('at','after')]:
    x=r[first];y=r[last];r['step_impulses'][first+'_to_'+last]={'joint_q_delta_rad':[b-a for a,b in zip(x['joint_q_rad'],y['joint_q_rad'])],'joint_sdk_delta_rad_s':[b-a for a,b in zip(x['joint_sdk_rad_s'],y['joint_sdk_rad_s'])],'tibia_z_delta_m':y['tibia_z_m']-x['tibia_z_m'],'tibia_vz_delta_m_s':y['tibia_vz_m_s']-x['tibia_vz_m_s'],'whole_com_vz_delta_m_s':y['whole_com_vz_m_s']-x['whole_com_vz_m_s'],'whole_com_z_delta_m':y['whole_com_z_m']-x['whole_com_z_m']}
   rows.append(r)
 summary={'event_foot_count':len(rows),'leg_counts':dict(collections.Counter(r['leg']for r in rows)),'exact_zero_events':sum(r['at']['patch_force_n']==0 for r in rows),'nonzero_low_events':sum(0<r['at']['patch_force_n']<=1 for r in rows),'exact128_all_inactive_zero_tuple_events':sum(r['at']['patch_count']==128 and r['at']['inactive_count']==128 and r['at']['all_patch_normals_exactzero']and r['at']['all_patch_separation_exactzero']for r in rows),'all128_same_recorded_point_events':sum(all(lo==hi for lo,hi in r['at']['point_range_world_m'])for r in rows),'isolated_single_substep_events':sum(r['before']['patch_force_n']>1 and r['after']['patch_force_n']>1 for r in rows),'concurrent_event_rows':dict(collections.Counter(collections.Counter(r['sequence']for r in rows).values())),'capacity':A['recorded_capacity'],'all_run_max_used':A['pressure_summary_all']['maximum'],'event_max_used':A['pressure_summary_event_rows']['maximum'],'full_run_max_capacity_fraction':A['pressure_summary_all']['maximum']/A['recorded_capacity'],'event_max_capacity_fraction':A['pressure_summary_event_rows']['maximum']/A['recorded_capacity']}
 windows={}
 for label in ['before','at','after']:
  windows[label]={k:stats(r[label][k]for r in rows if r[label][k]is not None)for k in ['patch_force_n','matrix_force_n','patch_count','inactive_count','allfloor_z_n','momentum_z_n','force_balance_z_n','com_acc_z_m_s2','pose_acc_z_m_s2','body_acc_z_m_s2','tibia_vz_m_s','whole_com_vz_m_s']}
  windows[label]['joint_sdk_absolute_rad_s']=stats(abs(x)for r in rows for x in r[label]['joint_sdk_rad_s']);windows[label]['joint_interval_absolute_rad_s']=stats(abs(x)for r in rows for x in r[label]['joint_interval_rad_s'])
 impulses={}
 for label in ['before_to_at','at_to_after']:
  impulses[label]={k:stats(r['step_impulses'][label][k]for r in rows)for k in ['tibia_z_delta_m','tibia_vz_delta_m_s','whole_com_vz_delta_m_s','whole_com_z_delta_m']}
  impulses[label]['max_abs_leg_joint_q_step_rad']=max(abs(v)for r in rows for v in r['step_impulses'][label]['joint_q_delta_rad']);impulses[label]['max_abs_leg_joint_sdk_step_rad_s']=max(abs(v)for r in rows for v in r['step_impulses'][label]['joint_sdk_delta_rad_s'])
 grid=[]
 for dim in [0,1]:
  for pos in sorted({r['grid_xy_m'][dim]for r in A['replicas']}):
   selection=[r for r in A['replicas']if r['grid_xy_m'][dim]==pos];grid.append({'axis':'x'if dim==0 else'y','position_m':pos,'replicas':len(selection),'failed_support_replicas':sum(bool(r['missing_substeps'])for r in selection),'event_count':sum(r['missing_substeps']for r in selection),'quiet_failed_replicas':sum(bool(r['quiet_failed_bounds'])for r in selection)})
 prior=A['source003_comparison'];nowrates=A['rates_postsettle400Hz_float64_accumulation'];oldrates=A['source003_rates_400hz'];quiet=[]
 for r in A['replicas']:
  if not r['quiet_failed_bounds']:continue
  e=r['env'];quiet.append({'env':e,'grid_xy_m':r['grid_xy_m'],'original_50Hz_failing_joints':{j:v for j,v in r['quiet_joint_sdk_rms'].items()if v>.03},'all_400Hz_joints_with_sdk_rms_over_03':{j:v for j,v in nowrates[e]['joints'].items()if v['sdk_rms']>.03}})
 contrast={'prior_support_total':sum(x['source003_missing_substeps']for x in prior),'current_support_total':sum(x['current_missing_substeps']for x in prior),'prior_combined_pass_envs':[x['env']for x in prior if x['source003_combined_pass']],'current_combined_pass_envs':[x['env']for x in prior if x['current_combined_pass']],'prior_quiet_failed_envs':[x['env']for x in prior if x['source003_quiet_failed_bounds']],'current_quiet_failed_envs':[x['env']for x in prior if x['current_quiet_failed_bounds']],'original_source003_reference':A['source003_prior_reference']}
 assert hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED
 return {'input_sha256':EXPECTED,'source005_freeze_sha256':SOURCE,'analyzer_sha256':ANALYZER,'original_outcome':'standing32 rejected; PPO disabled','summary':summary,'window_statistics':windows,'impulse_statistics':impulses,'all32_spatial_rows':A['replicas'],'grid_axis_summary':grid,'quiet_failures':quiet,'prior_comparison':contrast,'events':rows,'matrix_patch_max_difference_n':A['maximum_all_body_floor_minus_patch_vector_norm_n_in_selected_rows'],'input_rechecked':True}
if __name__=='__main__':
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--analysis',type=Path,required=True);a=p.parse_args();print(json.dumps(interpret(a.analysis),indent=2,allow_nan=False))
