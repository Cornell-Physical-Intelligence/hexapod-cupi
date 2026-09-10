from pathlib import Path
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output;OUT.mkdir(parents=True,exist_ok=False)
RAW=ROOT/'artifacts/omni_diagnostics_2026-09-09/direct_omni_extended_caps_002/curated/raw/run'
TRACE=RAW/'final_stop/stop_trace.npz'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(4<<20),b''):h.update(x)
 return h.hexdigest()
def stats(x):
 x=np.asarray(x);return {'mean':float(np.mean(x)),'median':float(np.median(x)),'p05':float(np.quantile(x,.05)),'p95':float(np.quantile(x,.95)),'min':float(np.min(x)),'max':float(np.max(x))}
def corr(a,b):
 a=a-a.mean(axis=0);b=b-b.mean(axis=0);den=np.sqrt((a*a).sum(axis=0)*(b*b).sum(axis=0));return np.divide((a*b).sum(axis=0),den,out=np.zeros_like(den),where=den>1e-18)
def frequency(x,dt=.02):
 x=np.asarray(x,dtype=np.float64);x=x-x.mean(axis=0);axis=np.linspace(-1,1,len(x));slope=(x*axis[:,None,None]).sum(axis=0)/(axis*axis).sum();x=x-axis[:,None,None]*slope
 spec=np.abs(np.fft.rfft(x*np.hanning(len(x))[:,None,None],axis=0))**2;spec[0]=0;f=np.fft.rfftfreq(len(x),dt);aggregate=spec.sum(axis=(1,2));total=aggregate.sum()
 bands={str(lo)+'-'+str(hi)+'Hz':float(aggregate[(f>=lo)&((f<hi) if hi<25 else (f<=hi))].sum()/total) for lo,hi in [(0,2),(2,5),(5,10),(10,15),(15,20),(20,25)]}
 dominant=f[np.argmax(spec,axis=0)]
 return {'aggregate_peak_hz':float(f[np.argmax(aggregate)]),'dominant_peak_hz_per_joint_row':stats(dominant),'power_bands':bands,'lag_correlations':{str(lag):stats(corr(x[:-lag],x[lag:])) for lag in [1,2,3,4,5,10]},'aggregate_frequencies_hz':f.tolist(),'aggregate_power_fraction':(aggregate/total).tolist()}
with np.load(TRACE) as z:data={k:z[k] for k in z.files}
q=data['joint_position_rad'];t=data['joint_target_rad'];u=data['unfiltered_target_rad'];a=data['action'];raw=data['raw_policy_action'];done=data['terminated']|data['truncated'];dt=.02
assert q.shape==(1600,48,18) and np.allclose(np.diff(data['time_s']),dt) and np.isfinite(q).all()
resets=[{'control':int(i+1),'env_id':int(j),'time_s':float(data['time_s'][i]),'terminated':bool(data['terminated'][i,j]),'truncated':bool(data['truncated'][i,j])} for i,j in zip(*np.nonzero(done))]
result={'trace_sha256':sha(TRACE),'checkpoint_sha256':json.loads((RAW/'final_stop/stop_diagnostics.json').read_text())['checkpoint_sha256'],'shape':list(q.shape),'sample_rate_hz':50,'joint_names':data['joint_names'].tolist(),'resets_preserved':resets,'windows':{},'scope':'Descriptive50Hz spectra/correlation; no causal feedback attribution, no SDK-rate substitution, excluded full windows retained in reset ledger.'}
for label,start,end in [('initial_stand',100,200),('motion',300,800),('settled_stop',850,1100),('quiet',1100,1600)]:
 good=(~done[start:end].any(0)) & (data['age_s'][start:end].min(0)>=2) & (np.abs(np.diff(data['age_s'][start:end],axis=0)-dt).max(0)<1e-4)
 ids=np.flatnonzero(good);v={k:x[start:end,ids].astype(np.float64) for k,x in [('q',q),('target',t),('request',u),('clipped_action',a),('raw_action',raw),('sdk_rate',data['joint_velocity_rad_s'])]}
 if label!='motion':assert np.max(np.abs(data['command'][start:end,ids]))<1e-6
 delta=np.diff(v['target'],axis=0);dq=np.diff(v['q'],axis=0)/dt
 flips=(delta[1:]*delta[:-1]<0);nonzero=(np.abs(delta[1:])>1e-5)&(np.abs(delta[:-1])>1e-5)
 row={'controls':[start+1,end],'time_s':[float(data['time_s'][start]),float(data['time_s'][end-1])],'eligible_env_ids':ids.tolist(),'excluded_env_ids':np.flatnonzero(~good).tolist(),'command_abs_max':np.abs(data['command'][start:end,ids]).max(axis=(0,1)).tolist(),'target_step_abs':stats(np.abs(delta)),'target_at_slew_fraction':float((np.abs(delta)>=.04-2e-6).mean()),'target_step_flip_fraction_given_nonzero':float(flips[nonzero].mean()),'action_raw_abs':stats(np.abs(v['raw_action'])),'action_clipped_fraction':float((np.abs(v['raw_action'])>1).mean()),'actual_interval_angle_rms_per_joint':stats(np.sqrt((dq*dq).mean(0))),'raw_sdk_rate_rms_per_joint':stats(np.sqrt((v['sdk_rate']**2).mean(0))),'target_tracking_error_abs_rad':stats(np.abs(v['target']-v['q'])),'requested_saturation_fraction':float((np.abs(data['computed_torque_nm'][start:end,ids])>1.6).mean()),'spectra':{k:frequency(x) for k,x in v.items()},'cross_correlations':{}}
 # q[t] and torque are post-action/end-of-control. A positivelag here means the second signal is later.
 for n,x,y in [('target_to_q',v['target'],v['q']),('raw_action_to_q',v['raw_action'],v['q']),('raw_action_to_raw_sdk_rate',v['raw_action'],v['sdk_rate']),('clipped_action_to_next_raw_action',v['clipped_action'],v['raw_action'])]:
  row['cross_correlations'][n]={str(lag):stats(corr(x[:len(x)-lag] if lag>0 else x[-lag:] if lag<0 else x,y[lag:] if lag>0 else y[:len(y)+lag] if lag<0 else y)) for lag in range(-5,6)}
 result['windows'][label]=row
# Exact observable pipeline identities without invented sensor noise reconstruction.
step=t[1:]-t[:-1];valid=~done[:-1]&~done[1:];pred=t[:-1]+np.clip(u[1:]-t[:-1],-.04,.04)
result['pipeline_numeric']={'clipped_action_equals_clip_raw_max':float(np.abs(a-np.clip(raw,-1,1)).max()),'filtered_equals_unfiltered_max':float(np.abs(data['filtered_target_rad']-u).max()),'executed_target_slew_recurrence_max_valid_pairs':float(np.abs(pred-t[1:])[valid].max()),'target_step_abs_max_valid_pairs':float(np.abs(step)[valid].max()),'reset_crossing_pairs_excluded':int((~valid).sum())}
(OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
compact={k:{x:v[x] for x in ['eligible_env_ids','excluded_env_ids','command_abs_max','target_at_slew_fraction','target_step_flip_fraction_given_nonzero','action_clipped_fraction','actual_interval_angle_rms_per_joint','requested_saturation_fraction']}|{'peaks':{n:{'peak':s['aggregate_peak_hz'],'bands':s['power_bands'],'lag1_median':s['lag_correlations']['1']['median']} for n,s in v['spectra'].items()}} for k,v in result['windows'].items()}
print(json.dumps({'resets':resets,'pipeline_numeric':result['pipeline_numeric'],'windows':compact},indent=2))
