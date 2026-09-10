from pathlib import Path
import json,hashlib,numpy as np
P=Path(__file__).resolve().parent;R=P.parent/'reference_physics_results_008';D=R/'run/wave'
audit=json.loads((R/'remote_audit.json').read_text());bad=[f for f,h in audit['raw_payloads'].items() if hashlib.sha256((R/f).read_bytes()).hexdigest()!=h];assert not bad
d=np.load(D/'trace.npz',allow_pickle=False);rows=json.loads((D/'reference_states.json').read_text());results=[x.get('result',x.get('reset')) for x in rows];times=d['time_s'][:,0];legs=list(d['legs']);events=[]
bytime={}
for result in results:
 if result['valid'][0]:bytime[round(result['target_time_s'],6)]=result
for result in results:
 s=result['state'];sw=s['swing']
 if sw is None:continue
 key=(s['current_leg'],sw['start_s'])
 if events and key==(events[-1]['leg'],events[-1]['start_s']):continue
 i=legs.index(s['current_leg']);start=sw['start_s'];idx=int(np.argmin(abs(times-start)));assert abs(times[idx]-start)<1e-8
 events.append({'leg':s['current_leg'],'start_s':start,'start_index':idx,'swing':sw,'leg_index':i,'preload_world_m':s['reference_minus_measured_preload_world_m'][i]})
for n,e in enumerate(events):
 stop=events[n+1]['start_index'] if n+1<len(events) else len(times)-1;i=e['leg_index'];idx=e['start_index'];c=d['distal_contact'][idx:stop+1,0,i];z=d['reference_point_world_m'][idx:stop+1,0,i,2];runs=[];run=None
 for k,contact in enumerate(c):
  if not contact and run is None:run=idx+k
  if contact and run is not None:runs.append([run,idx+k-1]);run=None
 if run is not None:runs.append([run,stop])
 e.update(end_index=stop,stride_m=float(np.linalg.norm(np.asarray(e['swing']['endpoint_world_m'])-np.asarray(e['swing']['coefficients'])[0])),no_force_contact_runs=runs,max_lift_from_swing_start_m=float(z.max()-z[0]),max_force_n=float(d['normal_force_world_n'][idx:stop+1,0,i,2].max()),min_force_n=float(d['normal_force_world_n'][idx:stop+1,0,i,2].min()))
 e['unloading_episodes']=[]
 for a,b in runs:
  base=max(idx,a-1);e['unloading_episodes'].append({'start_index':a,'end_index':b,'time_after_swing_start_s':float(times[a]-e['start_s']),'consecutive_off_samples':b-a+1,'max_lift_from_last_loaded_sample_m':float(d['reference_point_world_m'][a:b+1,0,i,2].max()-d['reference_point_world_m'][base,0,i,2]),'pre_off_planar_motion_from_swing_start_m':float(np.linalg.norm(d['reference_point_world_m'][base,0,i,:2]-d['reference_point_world_m'][idx,0,i,:2]))})
last=results[-1];s=last['state'];e=events[-1];i=e['leg_index'];idx=e['start_index'];time=times[-1];sw=e['swing'];u=(time-sw['start_s'])/sw['duration_s'];uh=min(u/sw['horizontal_duration_fraction'],1);k=np.arange(6);virtual=(u**k)@np.asarray(sw['coefficients']);virtual[2]+=sw['lift_m']*64*u**3*(1-u)**3;virtual[:2]=((uh**k)@np.asarray(sw['horizontal_coefficients']))[:2]
window=[]
for n in range(idx,len(times)):
 point=d['reference_point_world_m'][n,0,i];vel=d['reference_point_velocity_world_mps'][n,0,i];prev=d['reference_point_world_m'][n-1,0,i]
 window.append({'index':n,'time_s':float(times[n]),'contact':bool(d['distal_contact'][n,0,i]),'point_valid':bool(d['contact_point_valid'][n,0,i]),'normal_force_z_n':float(d['normal_force_world_n'][n,0,i,2]),'toe_world_m':point.tolist(),'reported_toe_velocity_mps':vel.tolist(),'toe_backward_difference_mps':((point-prev)/.02).tolist(),'planar_displacement_from_swing_start_m':float(np.linalg.norm(point[:2]-d['reference_point_world_m'][idx,0,i,:2]))})
report={'raw_payloads_verified':len(audit['raw_payloads']),'trace_sha256':hashlib.sha256((D/'trace.npz').read_bytes()).hexdigest(),'source_manifest_sha256':audit['source_manifest_sha256'],'scope':'Read-only diagnosis of actual008; no successor physics proof','confirmed_touchdowns':s['confirmed_touchdowns'],'scheduled_liftoffs':s['liftoffs'],'failure_reason':last['failure_reason'],'failure_time_s':float(time),'failure_swing_phase':float(u),'final_predicates':{'flight_seen':s['flight_seen'],'consecutive_no_force_contact_samples':s['flight_count'],'measured_lift_m':s['measured_flight_lift_m'],'minimum_measured_lift_m':last['diagnostics']['configuration']['minimum_measured_lift_m'],'passed_apex':bool(u>=.5),'actual_descent_seen':s['actual_descent_seen']},'failed_swing_virtual_foot_at_return_m':virtual.tolist(),'failed_swing_virtual_lift_from_loaded_anchor_m':float(sw['lift_m']*64*u**3*(1-u)**3),'failed_swing_virtual_planar_travel_m':float(np.linalg.norm(virtual[:2]-np.asarray(sw['coefficients'])[0,:2])),'failed_swing_actual_planar_travel_m':window[-1]['planar_displacement_from_swing_start_m'],'failed_swing_body_displacement_m':(d['position_world_m'][-1,0]-d['position_world_m'][idx,0]).tolist(),'actual_to_virtual_error_at_return_m':(d['reference_point_world_m'][-1,0,i]-virtual).tolist(),'events':events,'failed_swing_window':window,'post_settle_peak_requested_torque_nm':float(np.abs(d['computed_torque_nm'][199:]).max()),'post_settle_min_distal_contacts':int(d['distal_contact'][199:].sum(-1).min())}
(P/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['events','failed_swing_window']},indent=2))
print('events',json.dumps([{k:e[k] for k in ['leg','start_s','stride_m','unloading_episodes','preload_world_m']} for e in events],indent=2))
