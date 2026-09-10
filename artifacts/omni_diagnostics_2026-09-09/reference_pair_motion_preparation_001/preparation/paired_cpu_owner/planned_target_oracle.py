"""One planned opposing-pair schedule: executable target feasibility, no contact simulator."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'oracle'))
from wave_reference import WaveContactReference,AdvancedHorizontalSwing,LEGS,tensor
from reference_residual import ReferenceResidualTarget,ResidualConfig

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def load_inputs():
 plan=json.loads((HERE/'PLAN.json').read_text())
 for key in ('source_files','inputs'):
  for f,h in plan[key].items():
   if sha(HERE/f)!=h:raise ValueError('Frozen input/source mismatch: '+f)
 with np.load(HERE/'inputs/actual_pair001_startup_trace.npz',allow_pickle=False) as z:
  names=tuple(z['joint_names'].tolist())
  snapshot={k:z[k][-1].copy() for k in z.files if k!='joint_names'}
 return plan,names,snapshot


def evaluate(plan,names,snapshot):
 """Targets only. Timed planned landings are explicitly not measured confirmations."""
 g=WaveContactReference(names);reset=g.reset(snapshot);q0=g.q.copy();dt=plan['control_dt_s']
 runtime_q0=g._runtime(q0)[None].copy();lower=g._runtime(g.lower);upper=g._runtime(g.upper)
 cfg=ResidualConfig('formal_004',.02,.25,2.,8.)
 controller=ReferenceResidualTarget(names,dict(zip(names,lower)),dict(zip(names,upper)),1,cfg)
 controller.reset(runtime_q0,runtime_q0)
 pairs=[tuple(LEGS.index(n) for n in group) for group in plan['pair_order']]
 measured=g._read(snapshot);actual_com=g._com(g._leg(measured['joint_position_rad']),g.position,g.R0)
 actual_margins={'+'.join(LEGS[i] for i in pair):float(g.helper.support_margin(measured['contact_point_world_m'],[i for i in range(6) if i not in pair],actual_com)) for pair in pairs}
 start=g.time;start_p=g.position.copy();current=None;hold_until=g.time;order=0;events=[];rows=[];failure=None;quiet_time=None
 n=round((plan['initial_hold_s']+plan['motion_s']+plan['stop_s'])/dt)
 rows.append(dict(time_s=0.,q=runtime_q0[0].copy(),v=np.zeros(18),a=np.zeros(18),
                  desired_position_world_m=g.position.copy(),command=np.zeros(3),requested=np.zeros(3),
                  points_world_m=g.anchors.copy(),planned_support_mask=np.ones(6,bool),projected_margin_m=float(g.helper.support_margin(measured['contact_point_world_m'],range(6),actual_com)),active_pair_index=-1))
 maxlag=0.;reject_target=None
 for k in range(n):
  elapsed=k*dt
  requested=np.array(plan['requested_forward_left_yaw']) if plan['initial_hold_s']<=elapsed<plan['initial_hold_s']+plan['motion_s'] else np.zeros(3)
  # This is a planned endpoint completion only. No contact flags or landings are fabricated.
  if current is not None and g.time>=current['end']-1e-9:
   for i,curve in current['curves'].items():g.anchors[i]=curve.end.copy()
   events.append({'kind':'planned_endpoint_complete_not_measured_landing','time_s':elapsed,'pair':[LEGS[i] for i in current['pair']]})
   current=None;hold_until=g.time+plan['handoff_hold_s']
  if current is None and g.time>=hold_until-1e-9 and np.linalg.norm(requested)>0:
   pair=pairs[order%len(pairs)];pair_index=order%len(pairs);order+=1
   predicted_p,predicted_R=g._predict(requested,plan['placement_horizon_s'])
   curves={}
   for i in pair:
    end=predicted_R@g.neutral[i]+predicted_p;end[2]=g.anchors[i,2]
    curves[i]=AdvancedHorizontalSwing(g.time,plan['swing_s'],g.anchors[i],end,plan['lift_m'],plan['horizontal_fraction'])
   current={'pair':pair,'curves':curves,'end':g.time+plan['swing_s'],'pair_index':pair_index}
   events.append({'kind':'planned_pair_swing_not_measured_liftoff','time_s':elapsed,'pair':[LEGS[i] for i in pair],
                  'planned_stride_m':[float(np.linalg.norm(curves[i].end-g.anchors[i])) for i in pair]})
  finite_stop=(np.linalg.norm(requested)==0 and current is None and np.max(abs(g.command))<=g.cfg.stop_command_tolerance and np.max(abs(g.command_rate))<=g.cfg.stop_command_rate_tolerance)
  if finite_stop:g.command[:]=0.;g.command_rate[:]=0.
  g.position,g.yaw,g.command,g.command_rate=g._advance(requested,dt);g.time+=dt
  points=g.anchors.copy()
  if current:
   for i,curve in current['curves'].items():points[i]=curve.sample(g.time)[0]
  local=(g._body().rotation_wb.T@(points-g.position).T).T
  ik=g.g.ik(tensor(local[None]));q=ik['q_checked'][0].numpy()
  if current is None and np.max(abs(g.command))<1e-14 and np.max(abs(g.command_rate))<1e-12:q=g.q.copy()
  v=(q-g.q)/dt;a=(v-g.v)/dt
  support=[i for i in range(6) if current is None or i not in current['pair']]
  # Hypothetical planted contact centres retain the initial recorded preload offset.
  planned_contacts=g.anchors-g.preload_world
  com=g._com(q,g.position,g._body().rotation_wb)
  margin=float(g.helper.support_margin(planned_contacts,support,com))
  try:
   if not bool(ik['valid'].all()):raise ValueError('Unreachable target; no IK clipping')
   g._bounds(q,v,a)
   if margin<plan['proposed_support_margin_m']:raise ValueError('Planned four-support projected COM margin below0.05m')
   emitted=controller.step(g._runtime(q)[None],np.zeros((1,18)),reference_valid=np.ones(1,bool),dt=dt)
   executable=emitted['target_position_rad'][0].numpy();lag=float(np.max(abs(executable-g._runtime(q))));maxlag=max(maxlag,lag)
   if lag!=0:raise ValueError('Nonzero zero-residual target lag')
  except (ValueError,RuntimeError) as exc:
   failure={'time_s':(k+1)*dt,'error':str(exc),'planned_pair':None if current is None else [LEGS[i] for i in current['pair']],
            'attempted_max_velocity_rad_s':float(abs(v).max()),'attempted_max_acceleration_rad_s2':float(abs(a).max()),
            'attempted_joint_margin_rad':float(np.minimum(q-g.lower,g.upper-q).min()),'planned_support_margin_m':margin}
   reject_target=np.stack((g._runtime(q),g._runtime(v),g._runtime(a)));break
  mask=np.zeros(6,bool);mask[support]=True
  rows.append(dict(time_s=(k+1)*dt,q=executable.copy(),v=emitted['target_velocity_rad_s'][0].numpy().copy(),a=emitted['target_acceleration_rad_s2'][0].numpy().copy(),
                   desired_position_world_m=g.position.copy(),command=g.command.copy(),requested=requested.copy(),points_world_m=points.copy(),
                   planned_support_mask=mask,projected_margin_m=margin,active_pair_index=-1 if current is None else current['pair_index']))
  g.q=q;g.v=v
  if elapsed>=plan['initial_hold_s']+plan['motion_s'] and finite_stop and quiet_time is None:quiet_time=(k+1)*dt
 data={k:np.stack([r[k] for r in rows]) for k in rows[0]}
 maxidx_v=np.unravel_index(abs(data['v']).argmax(),data['v'].shape);maxidx_a=np.unravel_index(abs(data['a']).argmax(),data['a'].shape)
 margins={'+'.join(LEGS[i] for i in pair):float(data['projected_margin_m'][data['active_pair_index']==j].min()) if np.any(data['active_pair_index']==j) else None for j,pair in enumerate(pairs)}
 displacement=data['desired_position_world_m'][-1]-start_p
 report={'scope':plan['scope'],'candidate_id':plan['candidate_id'],'completed_target_sequence':failure is None and len(rows)==n+1,'failure':failure,
         'planned_controls':n,'accepted_controls':len(rows)-1,'actual_physics_controls':0,'actual_new_contact_confirmations':0,
         'target_velocity_peak_rad_s':float(abs(data['v']).max()),'target_acceleration_peak_rad_s2':float(abs(data['a']).max()),
         'velocity_peak':{'time_s':float(data['time_s'][maxidx_v[0]]),'runtime_joint':names[maxidx_v[1]],'value_rad_s':float(data['v'][maxidx_v])},
         'acceleration_peak':{'time_s':float(data['time_s'][maxidx_a[0]]),'runtime_joint':names[maxidx_a[1]],'value_rad_s2':float(data['a'][maxidx_a])},
         'max_zero_residual_executable_lag_rad':maxlag,'minimum_executed_joint_margin_rad':float(np.minimum(data['q']-lower,upper-data['q']).min()),
         'proposed_reference_margin_rad':.02,'actual_initial_snapshot_projected_margins_by_excluded_pair_m':actual_margins,
         'planned_minimum_projected_support_margins_by_swing_pair_m':margins,'minimum_planned_projected_margin_m':float(data['projected_margin_m'].min()),
         'planned_pair_swings':sum(e['kind'].startswith('planned_pair_swing') for e in events),'planned_completed_pair_endpoints':sum(e['kind'].startswith('planned_endpoint_complete') for e in events),
         'planned_liftoffs_after_stop_request':sum(e['kind'].startswith('planned_pair_swing') and e['time_s']>=plan['initial_hold_s']+plan['motion_s'] for e in events),
         'reference_quiet_time_s':quiet_time,'reference_stop_latency_s':None if quiet_time is None else quiet_time-(plan['initial_hold_s']+plan['motion_s']),
         'final_reference_quiet_duration_s':None if quiet_time is None else float(data['time_s'][-1]-quiet_time),
         'final_virtual_desired_displacement_world_m':displacement.tolist(),'requested_peak_forward_mps':float(data['requested'][:,0].max()),
         'admitted_target_equals_request_no_hidden_derating':True,'reported_rates_used_for_target_feasibility':False,
         'residual_core_contract':cfg.contract(),'events':events,'GPU_launches':0,'physics_admitted':False,'five_support_wave_gates_changed':False,'PPO_ready':False}
 if reject_target is not None:data['first_rejected_qva']=reject_target
 return report,data


def main():
 if (HERE/'report.json').exists():raise FileExistsError('Preserve existing evidence; run in a new copy')
 plan,names,snapshot=load_inputs();started=time.perf_counter();report,data=evaluate(plan,names,snapshot)
 report.update(plan_sha256=sha(HERE/'PLAN.json'),study_source_sha256=sha(Path(__file__)),runtime_joint_names=list(names),CPU_seconds=time.perf_counter()-started)
 np.savez_compressed(HERE/'planned_targets.npz',**data,joint_names=np.asarray(names))
 (HERE/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('events','runtime_joint_names','residual_core_contract')},indent=2,allow_nan=False))
if __name__=='__main__':main()
