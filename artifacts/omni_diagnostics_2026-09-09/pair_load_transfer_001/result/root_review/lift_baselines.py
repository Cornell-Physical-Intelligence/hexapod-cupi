"""Compare the same raw toe points against two explicitly different baselines."""
from pathlib import Path
import json,numpy as np
HERE=Path(__file__).resolve().parent;PAIR=HERE.parent/'run/pair'
d=np.load(PAIR/'trace.npz');reset=json.loads((PAIR/'pair_reset.json').read_text());initial=np.asarray(reset['state']['initial_measured_toes_world_m'])
rows=[]
for leg,name in [(1,'lm'),(4,'rm')]:
 contact=d['distal_contact'][:,0,leg]&d['contact_point_valid'][:,0,leg];z=d['reference_point_world_m'][:,0,leg,2]
 begin=int(np.flatnonzero(~contact)[0]);end=begin
 while end<len(contact) and not contact[end]:end+=1
 peak=begin+int(z[begin:end].argmax())
 rows.append({'leg':name,'flight_begin_diagnostic_index':begin,'flight_end_diagnostic_index':end,'peak_diagnostic_index':peak,
  'peak_physical_time_s':float(d['time_s'][peak,0]),'reset_baseline_z_m':float(initial[leg,2]),'last_planted_baseline_z_m':float(z[begin-1]),
  'peak_raw_point_z_m':float(z[peak]),'lift_above_reset_baseline_m':float(z[peak]-initial[leg,2]),
  'lift_above_last_planted_baseline_m':float(z[peak]-z[begin-1]),'last_planted_minus_reset_baseline_m':float(z[begin-1]-initial[leg,2])})
p=HERE/'lift_baselines.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps({'same_raw_reference_point_channel':True,'cause':'Different declared time baselines: initial settled reset versus last measured planted sample before flight. No FK/SDK coordinate conversion is involved.','rows':rows},indent=2)+'\n')
