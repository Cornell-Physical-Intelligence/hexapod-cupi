#!/usr/bin/env python3
"""Combine the measured yaw envelope and user-confirmed unchanged-zero pitch travel."""
import argparse,copy,hashlib,json,math
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--yaw-review',type=Path,required=True);p.add_argument('--yaw-geometry',type=Path,required=True);p.add_argument('--source-model',type=Path,required=True)
 p.add_argument('--revision',required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--femur-limits-deg',type=float,nargs=2,default=[-120.,80.]);p.add_argument('--tibia-limits-deg',type=float,nargs=2,default=[-5.,180.])
 a=p.parse_args();yaw=json.loads(a.yaw_review.read_text());geometry=json.loads(a.yaw_geometry.read_text());model=json.loads(a.source_model.read_text())
 names={j['name'] for j in model['joints'] if j['name'].endswith('_coxa_yaw')}
 if len(names)!=6 or set(yaw['joints'])!=names:raise ValueError('All six final yaw intervals required')
 if geometry['source_model_sha256']!=sha(a.source_model):raise ValueError('Yaw evidence applies to a different source model')
 if geometry['minimum_requested_buffer_m']!=.0005 or geometry['femur_tibia_groups_included']:raise ValueError('Unexpected yaw review scope/buffer')
 legs={x['leg']:x for x in geometry['legs']}
 for name,entry in yaw['joints'].items():
  prior=geometry['joints'][name]
  if entry['lower_rad']<prior['lower_rad']-1e-12 or entry['upper_rad']>prior['upper_rad']+1e-12:raise ValueError('Published yaw bounds exceed buffered geometry interval')
  if abs(entry['zero_shift_rad']-(entry['lower_rad']+entry['upper_rad'])/2)>1e-12:raise ValueError('Yaw neutral is not interval midpoint')
 d={'schema':1,'revision':a.revision,'provenance':{'pitch_source':'user_confirmed_travel_about_current_viewer_zero','femur_limits_deg':a.femur_limits_deg,'tibia_limits_deg':a.tibia_limits_deg,
  'pitch_zero_changes_authorized':False,'limits_are_not_collision_free_envelope':True,'yaw_review_filename':a.yaw_review.name,'yaw_review_sha256':sha(a.yaw_review),'yaw_geometry_sha256':sha(a.yaw_geometry),'yaw_geometry_filename':a.yaw_geometry.name,'source_model_sha256':sha(a.source_model),
  'yaw_scope':'intrinsic coxa-group versus standoff-plate interval only;0.5mm endpoint separation;femur/tibia excluded'},'joints':{}}
 for j in model['joints']:
  name=j['name']
  if name.endswith('_coxa_yaw'):
   entry=copy.deepcopy(yaw['joints'][name]);entry['absolute_previous_zero_azimuth_deg']=legs[name[:2]]['old_radial_zero_azimuth_deg']
  else:
   limits=a.femur_limits_deg if name.endswith('_femur_pitch') else a.tibia_limits_deg
   entry={'lower_rad':math.radians(limits[0]),'upper_rad':math.radians(limits[1]),'bounds_frame':'previous_viewer_zero','zero_shift_rad':0.,
    'provenance':{'source':'user_current_viewer_zero','limits_deg':limits,'explicit_user_confirmation':True,'pitch_zero_unchanged':True,'collision_free_envelope':False}}
  d['joints'][name]=entry
 a.out.write_text(json.dumps(d,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
 print(json.dumps({'revision':a.revision,'joints':len(d['joints']),'out':str(a.out),'sha256':sha(a.out)},indent=2))

if __name__=='__main__':main()
