"""Counterfactual replay of immutable measurements, never physical continuation."""
from pathlib import Path
from dataclasses import replace
import hashlib,importlib.util,json,sys
import numpy as np
from wave_reference import WaveContactReference,WaveConfig
HERE=Path(__file__).resolve().parent
PARENT=HERE/'inputs/oracle_wave004.py'
spec=importlib.util.spec_from_file_location('wave004_replay_oracle',PARENT);old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)

def replay(number):
 path=HERE/'inputs'/f'actual_wave{number}_trace.npz'
 with np.load(path,allow_pickle=False) as raw:d={key:raw[key] for key in raw.files}
 n=len(d['time_s']);names=tuple(d['joint_names'])
 config_args={'lift_m':.005 if number=='002' else .007,'horizontal_duration_fraction':1. if number in ['002','003'] else .8}
 #004 historical physics used full-duration XY, but the immediate parent already
 #introduced .8. Compare exact parent/default semantics and preserve its12mm rejection.
 parent=old.WaveContactReference(names,replace(old.WaveConfig(),**config_args));new=WaveContactReference(names,replace(WaveConfig(),**config_args))
 def snapshot(k):return {key:d[key][k].copy() for key in d if d[key].shape[:1]==(n,)}
 parent.reset(snapshot(199));new.reset(snapshot(199));outcomes={};maximum={'q_ref':0.,'v_ref':0.,'a_ref':0.};last={}
 for k in range(199,n):
  pair={}
  for key,controller in [('parent',parent),('successor',new)]:
   if key in outcomes:continue
   output=controller.step(snapshot(k),[.005,0,0]);last[key]=output;pair[key]=output
   if not output['valid'][0]:outcomes[key]={'rejected':True,'sample':k,'time_s':float(d['time_s'][k,0]),'reason':output['failure_reason']}
  if set(pair)=={'parent','successor'} and all(v['valid'][0] for v in pair.values()):
   for key in maximum:maximum[key]=max(maximum[key],float(np.abs(pair['parent'][key]-pair['successor'][key]).max()))
 for key in ['parent','successor']:
  output=last[key];state=output['state'];outcomes.setdefault(key,{'rejected':False,'recorded_prefix_ended_sample':n-1,'next_unexecuted_target_time_s':output['target_time_s']})
  outcomes[key].update(confirmed_touchdowns=state['confirmed_touchdowns'],scheduled_liftoffs=state['liftoffs'],last_mode=state['mode'],last_flight_seen=state['flight_seen'],last_flight_count=state['flight_count'],last_measured_lift_m=state['measured_flight_lift_m'])
  if key=='successor':outcomes[key].update({name:state[name] for name in ['raw_force_free_samples','raw_force_free_runs','unqualified_contact_returns','last_unqualified_return_time_s','last_unqualified_run_samples','last_unqualified_lift_m']})
 return {'recording':number,'trace_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'replay_config':config_args,'prefix_target_max_parent_difference':maximum,'outcomes':outcomes,'physical_continuation_executed':False}

if __name__=='__main__':
 rows=[replay(n) for n in ['002','003','004','008']]
 report={'scope':'Counterfactual existing-measurement replay; actual physical verdicts preserved separately','parent_controller_sha256':hashlib.sha256(PARENT.read_bytes()).hexdigest(),'successor_controller_sha256':hashlib.sha256((HERE/'wave_reference.py').read_bytes()).hexdigest(),'cases':rows,'historical002_note':'Actual002 failed the older wave001 early-touch rule; both immediate parent wave004 and this successor retain the previously reviewed provisional landing fix. This is not a retroactive physical pass.'}
 (HERE/'historical_replay_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
