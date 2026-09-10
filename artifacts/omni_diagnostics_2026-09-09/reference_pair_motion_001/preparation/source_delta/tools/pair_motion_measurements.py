"""Read-only per-control evidence/gates; full failing rows are retained by caller."""
import numpy as np
LEGS=('lf','lm','lr','rf','rm','rr')

def check_sensor_row(row):
 for key in ('sensor_all_valid','sensor_contact_valid'):
  if not np.asarray(row[key],dtype=bool).all():raise ValueError('Sensor clock evidence invalid; no contact admission')
 if np.any(row['sensor_age_s']!=0.) or np.any(row['sensor_outdated']):raise ValueError('Stale contact data')
 if not np.array_equal(row['sensor_timestamp_s'],row['sensor_last_update_s']) or not np.array_equal(row['sensor_timestamp_s'],row['sensor_expected_timestamp_s']):raise ValueError('Sensor clock coverage mismatch')
 contact=np.asarray(row['distal_contact'],dtype=bool)
 if np.any(contact & ~np.asarray(row['contact_point_valid'],dtype=bool)):raise ValueError('Distal support point unknown')
 if np.any(contact & (np.linalg.norm(row['normal_force_world_n'],axis=-1)<=1.)):raise ValueError('Claimed distal support has <=1N measured normal force')

def torque_window(rows):
 if len(rows)!=8:raise ValueError('Exactly8 physics substeps required percontrol')
 requested=np.stack([r['computed_torque_nm'] for r in rows]);applied=np.stack([r['applied_torque_nm'] for r in rows])
 if not np.isfinite(requested).all() or not np.isfinite(applied).all():raise ValueError('Nonfinite complete substep torque')
 result={'max_requested_nm':float(abs(requested).max()),'max_applied_nm':float(abs(applied).max()),'substeps':8}
 if result['max_requested_nm']>1.6 or result['max_applied_nm']>1.60001:raise ValueError('Paired400Hz requested/applied torque limit exceeded')
 return result

def measured_support(g,row):
 """Do not advance reference phase/counters while inspecting the last real row."""
 check_sensor_row(row);m=g._read(row)
 if m['terminal'] or m['base_contact'] or any(m[k].any() for k in ('shaft_contact','coxa_contact','femur_contact')):raise ValueError('Measured terminal or nonfoot contact')
 margin=g._support(m,g.active_pair)
 required=[i for i in range(6) if g.active_pair is None or i not in g.active_pair]
 if g.active_pair is not None:
  for i in g.active_pair:
   if g.foot_cycles[i].current_leg is None:
    required.append(i)
    if not m['contact'][i]:raise ValueError('Confirmed paired foot lost support beforehandoff')
 drift=np.linalg.norm(m['reference_point_world_m']-g.measured_anchors,axis=-1)
 if drift[required].max()>g.cfg.maximum_stance_drift_m:raise ValueError('Measured retained toe drift exceeds20mm')
 error=float(np.linalg.norm(m['position_world_m'][:2]-g.position[:2]))
 if error>g.cfg.maximum_body_tracking_error_m:raise ValueError('Measured body tracking exceeds35mm')
 return {'support_margin_m':float(margin),'required_leg_indices':required,'required_contact_mask':[bool(m['contact'][i]) for i in required],
         'retained_toe_drift_m':[float(drift[i]) for i in required],'body_tracking_error_m':error,
         'physical_time_s':float(m['time_s']),'controller_time_s':g.time,'no_controller_state_advance':True}
