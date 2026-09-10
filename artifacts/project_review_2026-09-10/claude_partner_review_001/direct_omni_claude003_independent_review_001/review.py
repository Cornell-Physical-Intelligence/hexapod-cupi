"""Read-only NumPy replay of actual pilot LR, command/reset coverage and quiet actions."""
from pathlib import Path
import argparse,ast,hashlib,json,math
import numpy as np
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--campaign-root',type=Path,default=HERE.parent/'direct_omni_matched_pilots_publication_001');parser.add_argument('--output',type=Path,default=HERE);args=parser.parse_args();args.output.mkdir(exist_ok=True,parents=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for v in iter(lambda:f.read(1<<20),b''):h.update(v)
 return h.hexdigest()
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def stats(v):
 x=np.asarray(v,dtype=float);return {'minimum':float(x.min()),'median':float(np.median(x)),'mean':float(x.mean()),'maximum':float(x.max())}
def method(p,cls,name):
 c=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef) and n.name==cls)
 return next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
source=HERE/'inputs';source_map=json.loads((HERE/'INPUTS_SHA256.json').read_text())
for k,v in source_map.items():require(sha(source/k)==v,'Source changed: '+k)
reset=method(source/'hexapod_env.py','HexapodEnv','_reset_idx')
# Top-level expression proves the sampler call is not beneath an evaluation guard.
sampler=[n for n in reset.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='_sample_commands']
require(len(sampler)==1,'Parent sampler no longer unconditional')
body=ast.unparse(reset)
require('omni_evaluation' not in body,'Unexpected parent evaluation path')
read_inputs={};report={'schema':'independent_claude003_actual50_review_v1','source_sha256':source_map,'parent_reset_sampler_unconditional_line':sampler[0].lineno,'source_manifest_sha256':'64144b2c2466f17c809554ef53f82b736b4199136cc821451d7ac2788752bd2e','branches':{},'Stage2_complete':False}
for branch in ['curriculum','caps']:
 d=args.campaign_root/branch/'raw/run';train=d/'train';receipt=json.loads((train/'training_receipt.json').read_text())
 require(receipt['complete'] and receipt['updates_completed']==50 and receipt['reload']['passed'],'Not completed 50-update receipt')
 for f in [train/'training_receipt.json',train/'training_trace.npz',train/'training_joint_trace.npz',train/'training_events.json',d/'final_stop/stop_trace.npz',d/'final_stop/stop_diagnostics.json']:
  read_inputs[str(f)]=sha(f)
 updates=receipt['optimizer_updates'];lr=np.array([u['learning_rate'] for u in updates]);require([u['completed_update'] for u in updates]==list(range(1,51)),'Missing update')
 lrout={'update_end_values':lr.tolist(),**stats(lr),'floor_end_update_numbers':(np.flatnonzero(np.isclose(lr,1e-5,atol=1e-12,rtol=0))+1).tolist(),'scope':'One post-update LR sample; no minibatch KL or full intra-update LR trace recorded.'}
 if branch=='caps':
  losses={k:np.array([u['losses'][k] for u in updates]) for k in ['caps_temporal','caps_spatial','caps_weighted','caps_valid_pair_fraction']}
  caps={k:{'values':v.tolist(),**stats(v),'first10_mean':float(v[:10].mean()),'last10_mean':float(v[-10:].mean())} for k,v in losses.items()}
  caps['temporal_first10_to_last10_relative_change']=float(losses['caps_temporal'][-10:].mean()/losses['caps_temporal'][:10].mean()-1)
 else:caps={'not_evaluated':True,'reason':'Zero-weight branch returns without actor calls; zero recorded CAPS losses are not evidence of zero policy variation.'}
 with np.load(train/'training_trace.npz') as z:
  cmd=z['command'];age=z['age_s'];terminated=z['terminated'];truncated=z['truncated'];times=z['time_s'];shapes={k:list(z[k].shape) for k in ['command','age_s','terminated','truncated','target_delta_rms_rad']}
 require(cmd.shape==(1200,1024,3) and age.shape==(1200,1024),'Unexpected actual coverage')
 require(np.isfinite(cmd).all() and np.isfinite(age).all(),'Nonfinite trace')
 require(np.array_equal(times,(np.arange(1200)+1)*.02),'Actual control cadence differs')
 quiet=np.arange(1024)%4==0;require(np.array_equal(cmd[:,quiet],np.zeros_like(cmd[:,quiet])),'A dedicated quiet row moved its command')
 done=terminated|truncated;events=json.loads((train/'training_events.json').read_text())
 ledger_pairs={(int(e['control'])-1,int(i)) for e in events for i in e['ids']};require(ledger_pairs==set(zip(*np.nonzero(done))),'Event ledger disagrees with trace')
 reset_t,reset_e=np.nonzero(done[:-1]);require(np.max(np.abs(age[reset_t+1,reset_e]-.02))<1e-7,'Reset elapsed age not restarted')
 require(np.array_equal(cmd[reset_t+1,reset_e],np.zeros((len(reset_t),3),dtype=cmd.dtype)),'First post-reset command not zero')
 noevent=~done[:-1];max_age_error=float(np.abs(np.diff(age,axis=0)[noevent]-.02).max());require(max_age_error<3e-6,'Non-reset elapsed cadence differs')
 targets=np.zeros_like(cmd);replay_mask=np.zeros(cmd.shape[:2],bool);episodes=[];reset_comparisons=[];all_motion=[];max_nonzero_run=0
 for env in range(1024):
  starts=[0]+(np.flatnonzero(done[:,env])+1).tolist();starts=[s for s in starts if s<1200];ends=starts[1:]+[1200];prev=None
  for episode,(start,end) in enumerate(zip(starts,ends)):
   s=cmd[start:end,env];nz=np.any(np.abs(s)>1e-9,axis=1)
   edges=np.flatnonzero(np.diff(np.r_[False,nz,False]));runs=edges[1::2]-edges[::2]
   if len(runs):max_nonzero_run=max(max_nonzero_run,int(runs.max()))
   stable=(np.max(np.abs(np.diff(s,axis=0)),axis=1)<1e-7)&nz[:-1] if len(s)>1 else np.array([],bool)
   ids=np.flatnonzero(stable);motion=None
   if quiet[env]:motion=np.zeros(3,np.float32)
   elif len(ids):motion=s[ids[np.argmax(np.linalg.norm(s[ids],axis=1))]].copy()
   if motion is not None:
    advancing=np.arange(end-start)+1
    moving=((advancing+(env%4)*200)%800<400)&~quiet[env]
    targets[start:end,env]=motion*moving[:,None]
    replay_mask[start:max(start,end-1),env]=True
    if not quiet[env]:all_motion.append({'env':env,'episode':episode,'start_control':start+1,'end_control':end,'motion':motion.tolist()})
   row={'env':env,'episode':episode,'start_control':start+1,'end_control':end,'dedicated_quiet':bool(quiet[env]),'observed_plateau':None if motion is None else motion.tolist()}
   episodes.append(row)
   if episode and not quiet[env]:
    eligible=prev is not None and motion is not None
    reset_comparisons.append({'env':env,'reset_control':start,'pre':None if prev is None else prev.tolist(),'post':None if motion is None else motion.tolist(),'eligible_observed_plateaus_both_sides':eligible,'changed':bool(not np.array_equal(prev,motion)) if eligible else None})
   prev=motion
 # Independent vectorized recurrence on episode spans with actually observed targets.
 delta=targets[:-1,:,:2]-cmd[:-1,:,:2];length=np.linalg.norm(delta,axis=-1,keepdims=True)
 factor=np.minimum(np.float32(.005)/np.maximum(length,np.float32(1e-9)),1)
 predicted=np.concatenate((cmd[:-1,:,:2]+delta*factor,cmd[:-1,:,2:3]+np.clip(targets[:-1,:,2:3]-cmd[:-1,:,2:3],np.float32(-.016),np.float32(.016))),axis=-1)
 mask=replay_mask[:-1]&~done[:-1]
 err=np.max(np.abs(predicted-cmd[1:]),axis=-1);max_replay_error=float(err[mask].max());require(max_replay_error<2e-7,'Observed command schedule differs from independent recurrence')
 eligible=[x for x in reset_comparisons if x['eligible_observed_plateaus_both_sides']];unchanged=[x for x in eligible if not x['changed']]
 require(not unchanged,'Observed episode motion did not redraw')
 moves=np.asarray([x['motion'] for x in all_motion]);trans=np.linalg.norm(moves[:,:2],axis=1)>1e-6;turn=np.abs(moves[:,2])>1e-6
 bearings=(np.arctan2(moves[trans,1],moves[trans,0])%(2*np.pi));bins=np.bincount(np.minimum((bearings/(2*np.pi)*16).astype(int),15),minlength=16)
 schedule={'all_command_rows':shapes,'dedicated_quiet_rows':int(quiet.sum()),'quiet_commands_exact_zero':True,'terminal_env_events':int(done.sum()),'terminations':int(terminated.sum()),'timeouts':int(truncated.sum()),'post_reset_events_with_next_sample':len(reset_t),'post_reset_commands_exact_zero':True,'post_reset_age_seconds':.02,'max_nonreset_age_step_error_s':max_age_error,'episodes_observed':len(episodes),'nonquiet_episodes_with_observed_plateau':len(all_motion),'nonquiet_reset_events':len(reset_comparisons),'eligible_adjacent_plateau_comparisons':len(eligible),'changed_adjacent_plateaus':len(eligible)-len(unchanged),'not_comparable_resets':len(reset_comparisons)-len(eligible),'command_recurrence_checked_pairs':int(mask.sum()),'command_recurrence_max_abs_error':max_replay_error,'max_contiguous_nonzero_command_controls':max_nonzero_run,'category_episode_counts':{'translation_only':int((trans&~turn).sum()),'yaw_only':int((~trans&turn).sum()),'combined':int((trans&turn).sum())},'yaw_positive_episodes':int((moves[:,2]>1e-6).sum()),'yaw_negative_episodes':int((moves[:,2]<-1e-6).sum()),'translation_bearing_bins_16':bins.tolist(),'limitations':'Actual command/age/event summary covers all 1024 rows. Target random draws and schedule.age were not directly logged; replay is restricted to episodes with observed settled motion plateaus. No distribution uniformity or independent random draw proof from finite samples.'}
 with np.load(train/'training_joint_trace.npz') as z:
  detail={'env_ids':z['env_ids'].tolist(),'shape':list(z['raw_policy_action'].shape),'scope':'8 sampled replicas only; raw training actions include exploration, not logged actor means.'}
 with np.load(d/'final_stop/stop_trace.npz') as z:
  time=z['time_s'];window=time>=22.;c=z['command'];q=z['joint_target_rad'];raw=z['raw_policy_action'];lim=z['target_slew_limited_fraction'];term=z['terminated']|z['truncated']
  require(np.max(np.abs(c[window]))==0,'Final quiet window command is not zero')
  pair=window[1:]&window[:-1];valid=~term[:-1]&~term[1:];dq=np.diff(q,axis=0);da=np.diff(raw,axis=0)
  quiet_rows=[]
  for env in range(48):
   x=dq[pair&valid[:,env],env];a=da[pair&valid[:,env],env];limited=lim[window,env]
   quiet_rows.append({'env':env,'paired_steps':len(x),'executed_joint_delta_max_p95_rad':float(np.max(np.percentile(np.abs(x),95,axis=0))),'raw_mean_delta_rms':float(np.sqrt(np.mean(a*a))),'raw_mean_delta_abs_p95':float(np.percentile(np.abs(a),95)),'executed_delta_rms_rad':float(np.sqrt(np.mean(x*x))),'fraction_joint_deltas_at_0_04_bound':float(np.mean(np.abs(x)>=.039999)),'mean_slew_limited_fraction':float(limited.mean())})
  evalscope={'trace_shape':list(q.shape),'window_start_s':22.,'window_end_s':32.,'policy_actions_deterministic':True,'no_terminal_pairs_included':True,'per_replica':quiet_rows,'raw_actor_step_rms_summary':stats([x['raw_mean_delta_rms'] for x in quiet_rows]),'joint_step_bound_fraction_summary':stats([x['fraction_joint_deltas_at_0_04_bound'] for x in quiet_rows]),'scope':'Actual deterministic cold stop rollout at zero command; distinct from training minibatch policy-mean regularizer. Executed target is clipped/slewed and is not action_scale times raw adjacent actor means.'}
 report['branches'][branch]={'learning_rate':lrout,'CAPS':caps,'schedule':schedule,'training_detail_coverage':detail,'actual_final_quiet':evalscope}
 (args.output/(branch+'_reset_plateaus.json')).write_text(json.dumps({'episodes':episodes,'adjacent_reset_comparisons':reset_comparisons},indent=2)+'\n')
report['claims']={'learning_rate_stuck_at_floor':'Refuted for the actual pilots; post-update rates rebound. Intra-update effective rate and KL are not fully logged.','aggregate_CAPS_proves_50_update_impossibility':'Unsupported; training aggregate RMS, quiet per-joint p95, different states/normalization and nonlinear actuator mapping cannot be equated.','parent_evaluation_flag_prevents_resampling':'Refuted by pinned unconditional parent call and observed post-reset motion redraws.','quiet_failure':'Confirmed from unchanged actual evaluation. A small aggregate CAPS decline did not deliver qualified quiet standing.'}
report['inputs_sha256']=read_inputs
(args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({b:{'lr':report['branches'][b]['learning_rate']['median'],'lr_floor_count':len(report['branches'][b]['learning_rate']['floor_end_update_numbers']),'schedule':report['branches'][b]['schedule'],'quiet_raw_step_rms':report['branches'][b]['actual_final_quiet']['raw_actor_step_rms_summary'],'quiet_at_bound':report['branches'][b]['actual_final_quiet']['joint_step_bound_fraction_summary']} for b in ['curriculum','caps']},indent=2))
