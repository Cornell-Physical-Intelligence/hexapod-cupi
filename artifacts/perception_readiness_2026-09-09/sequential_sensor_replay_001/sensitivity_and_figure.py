"""Conservative optical/uncertainty sensitivity; no extra physics acquisition."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from replay import ROOT,pose_matrix,footprint_mask,SHAPE
from sensor_mount_math import optical_visibility

r=json.loads((ROOT/'report.json').read_text());mount=json.loads((ROOT/'inputs/mounts.json').read_text())
d=dict(np.load(ROOT/'inputs/motion_source004.npz'));vis=dict(np.load(ROOT/'visibility.npz'));maps=dict(np.load(ROOT/'map_masks.npz'))
profile=mount['profile'];cameras=mount['selected_rig']['cameras'];points=vis['points'];rows=[];accepted_depth=[]
for minimum in (.07,.10,.20,.26,.30):
 for roi in (1.,.90,.80):
  whole=[];needed=[]
  for frame,index in enumerate(vis['indices']):
   body=pose_matrix(d,index);ground=np.einsum('ni,ij->nj',points-body[:3,3],body[:3,:3]);union=np.zeros(len(points),bool)
   for camera_index,camera in enumerate(cameras):
    optical=np.einsum('ni,ij->nj',ground-np.asarray(camera['body_position_m']),np.asarray(camera['optical_rotation_body']))
    optical_ok=optical_visibility(optical,{**profile,'min_z_m':minimum},roi_fraction=roi)
    union|=optical_ok&vis['clear'][frame,camera_index]
    if minimum==.07 and roi==1.:accepted_depth.extend(optical[vis['clear'][frame,camera_index],2].tolist())
   required,_,_=footprint_mask(points,d['planned_footprint_centres_world_m'][index+2])
   whole.append(union.mean());needed.append(union.reshape(SHAPE)[required].mean())
  rows.append(dict(effective_min_optical_depth_m=minimum,effective_image_roi_fraction=roi,
    mean_instantaneous_ground_grid_fraction=float(np.mean(whole)),mean_required_footprint_visible_fraction=float(np.mean(needed)),
    basis='Conservative synthetic usable-depth/crop sensitivity; nominal profile retains its original vendor-derived bounds'))
noise=[]
# Actual per-point registration adds1mm position variance and a nonnegative
#range-dependent rotation term. These are assumptions, not sensor measurements.
for sigma in (.003,.010,.015,.020):
 total_sigma=np.sqrt(sigma*sigma+.001**2+(.001*.30)**2)
 noise.append(dict(assumed_isotropic_point_sigma_m=sigma,assumed_300mm_lever_total_sigma_m=float(total_sigma),
    usable_under_15mm_contract=bool(total_sigma<=.015),basis='Unqualified synthetic noise/pose sensitivity'))
out=dict(scope='Optical and assumed uncertainty sensitivity on the same recorded motion, meshes and sample grid',
    nominal_optical_depth_range_of_visible_points_m=[float(min(accepted_depth)),float(max(accepted_depth))],
    optical_cases=rows,uncertainty_cases=noise,hardware_qualification=False)
(ROOT/'sensitivity.json').write_text(json.dumps(out,indent=2)+'\n')
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for n,color in [(2,'#9a6bb5'),(4,'#d88b35'),(6,'#168475')]:
 case=next(x for x in r['cases'] if x['id']==f'{n}cam_continuous');t=np.array([x['receive_time_s'] for x in case['frames']])
 axs[0,0].plot(t,[x['required_footprint_usable_fraction'] for x in case['frames']],color=color,label=f'{n} cameras')
axs[0,0].set(title='Required six-foot regions: retained usable cells',ylabel='Fraction of required cells',ylim=(0,1));axs[0,0].legend()
for name,label,style in [('6cam_continuous','Continuous acquisition','-'),('6cam_dropout','600 ms synthetic dropout','--')]:
 case=next(x for x in r['cases'] if x['id']==name);t=np.array([x['receive_time_s'] for x in case['frames']])
 axs[0,1].plot(t,[x['required_footprint_usable_fraction'] for x in case['frames']],style,label=label)
axs[0,1].axvspan(8.04,8.64,color='gray',alpha=.15);axs[0,1].set(title='A stored cell is not always usable',ylim=(0,1),ylabel='Required usable fraction');axs[0,1].legend(fontsize=8)
case=next(x for x in r['cases'] if x['id']=='6cam_continuous');frames=case['frames'];t=[x['receive_time_s'] for x in frames]
for key,label in [('instantaneous_mesh_clear_fraction','Instantaneous visibility'),('retained_usable_fraction','Retained usable ≤250 ms'),('ever_observed_fraction','Ever observed (includes stale)')]:
 axs[1,0].plot(t,[x[key] for x in frames],label=label)
axs[1,0].set(title='Separate visibility, usable map and accumulated observations',ylabel='Fraction of fixed 1.2 m square grid',xlabel='Recorded simulation time (s)');axs[1,0].legend(fontsize=8)
k=len(vis['indices'])-1;instant=maps['6cam_continuous_instantaneous'][k];usable=maps['6cam_continuous_usable'][k];ever=maps['6cam_continuous_ever'][k];required=maps['6cam_continuous_required'][k]
state=np.zeros(len(points));state[ever]=1;state[usable]=2;state[instant]=3
from matplotlib.colors import ListedColormap
axs[1,1].scatter(points[:,0],points[:,1],c=state,cmap=ListedColormap(['#d8d8d8','#ce963c','#7bbbae','#168475']),s=8,vmin=0,vmax=3,marker='s')
axs[1,1].scatter(points[required,0],points[required,1],s=24,facecolors='none',edgecolors='#8b1f40',linewidths=.9,label='Required cells')
axs[1,1].set(title='Final frame: holes remain unknown / stale',xlabel='World X (m)',ylabel='World Y (m)',aspect='equal')
from matplotlib.patches import Patch
handles=[Patch(color='#d8d8d8',label='Never observed'),Patch(color='#ce963c',label='Observed, stale'),Patch(color='#7bbbae',label='Retained usable'),Patch(color='#168475',label='Visible now')]
handles+=axs[1,1].get_legend_handles_labels()[0]
axs[1,1].legend(handles=handles,fontsize=7,loc='lower left',ncol=2)
for ax in axs.ravel():ax.grid(alpha=.15)
fig.suptitle('Actual004 motion + synthetic flat acquisition + exact C mesh occlusion',fontsize=14)
fig.savefig(ROOT/'sequential_coverage.png',dpi=160)
print(json.dumps(out,indent=2))
