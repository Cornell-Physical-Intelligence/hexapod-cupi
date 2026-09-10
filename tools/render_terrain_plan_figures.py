#!/usr/bin/env python3
"""Render analytical/concept figures; these are not CAD or Isaac screenshots."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from terrain_readiness import terrain_mesh

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/project_review_2026-09-04'
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'font.family':'DejaVu Sans','savefig.facecolor':'white'})
blue='#1674a5';green='#257d62';orange='#b86b1d';dark='#253442'
fig,(ax,bx)=plt.subplots(1,2,figsize=(14,6),layout='constrained',gridspec_kw={'width_ratios':[1,1.2]})
xml=ET.parse(ROOT/'robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf').getroot()
ax.add_patch(Ellipse((0,0),.25,.45,facecolor='#e7ebee',edgecolor='#758592'))
for leg in ('lf','lm','lr','rf','rm','rr'):
    j=next(x for x in xml.findall('joint') if x.get('name')==f'{leg}_coxa_yaw')
    p=np.fromstring(j.find('origin').get('xyz'),sep=' ')[:2];p[1]*=-1
    r=p/np.linalg.norm(p);foot=p+.17*r
    ax.plot([p[0],foot[0]],[p[1],foot[1]],color='#99a4ab',linewidth=4,zorder=1)
    ax.add_patch(Circle(foot,.055,fill=False,linestyle='--',edgecolor=green))
    ax.scatter(*p,s=55,c=blue,marker='s',zorder=3)
    ax.annotate('',xy=p+.10*r,xytext=p,arrowprops={'arrowstyle':'->','color':blue,'lw':1.8})
    ax.text(*(foot+.072*r),leg.upper(),ha='center',va='center',color=dark)
ax.scatter(0,0,s=180,c=orange,marker='o',label='Mid-360 / central rig')
ax.scatter(0,.25,s=115,c=dark,marker='s',label='D455: proposed front bracket')
ax.plot([],[],color=blue,marker='s',linestyle='',label='Close-depth camera candidates')
ax.plot([],[],color=green,linestyle='--',label='Illustrative foot regions')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.03),frameon=False,fontsize=9)
ax.set_xlim(-.46,.46);ax.set_ylim(-.57,.54);ax.set_aspect('equal');ax.axis('off')
ax.set_title('Mount concept: one close view per foot sector',loc='left',pad=20,color=dark)
ax.annotate('',xy=(0,.48),xytext=(0,.41),arrowprops={'arrowstyle':'->','color':dark})
ax.text(0,.51,'Forward (body −Y)',ha='center',color=dark)
h=.205;near=h/np.tan(np.deg2rad(7))
bx.axhline(0,color=dark,lw=1.5)
bx.fill_between([0,near],0,.028,color=orange,alpha=.22)
bx.plot([0,near],[h,0],color=orange,lw=2)
bx.scatter(0,h,color=orange,s=80,zorder=3)
bx.add_patch(Rectangle((-.03,.105),.22,.025,facecolor='#c6cfd5',edgecolor='#758592'))
bx.plot([0,0],[.13,h],color='#758592',lw=3)
bx.annotate('Illustrative optical height: 0.205 m',xy=(0,h),xytext=(.18,.265),
            arrowprops={'arrowstyle':'-','color':dark},color=dark)
bx.text(.55,.157,'Lowest ray: 7° below horizontal',rotation=-12,color=orange)
bx.annotate(f'First ideal ground hit\n{near:.2f} m from sensor',xy=(near,0),xytext=(1.43,.09),ha='center',
            arrowprops={'arrowstyle':'->','color':dark},color=dark)
bx.text(.73,.037,'Near ground outside this LiDAR view',ha='center',color=orange,fontsize=10)
bx.set(xlim=(-.07,2.03),ylim=(-.03,.31),xlabel='Horizontal distance from sensor (m)',ylabel='Height above ground (m)')
bx.set_title('Panoramic FoV does not establish foothold visibility',loc='left',pad=20,color=dark)
bx.grid(alpha=.15)
fig.suptitle('Proposed rig — CAD hip anchors, unvalidated camera brackets',fontsize=16,color=dark)
fig.savefig(OUT/'sensor_mount_concept.png',dpi=160);plt.close(fig)

fig,ax=plt.subplots(figsize=(14,6),layout='constrained');ax.set_xlim(0,14);ax.set_ylim(0,6);ax.axis('off')
def box(x,y,w,h,text,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=.08',facecolor=color,edgecolor='none'))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',color='white',fontsize=11)
def arrow(start,end):ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#6b7780','lw':1.6})
box(.2,4.2,2.7,1,'Flat PPO\ntracking + motor gates',blue)
box(3.6,4.2,2.7,1,'Mild terrain pilot\nverified contact geometry',blue)
box(7,4.2,2.7,1,'Terrain teacher\nprivileged simulation data',blue)
box(10.4,4.2,3.1,1,'Recurrent student\nactual sensor observations',green)
for a,b in [(2.9,3.6),(6.3,7),(9.7,10.4)]:arrow((a,4.7),(b,4.7))
box(.2,2.5,2.7,.9,'C packaging + leg stand\npayload / actuator model',dark)
box(3.6,2.5,2.7,.9,'Mount sweep + sensor bags\ncoverage / clocks / calibration',orange)
box(7,2.5,2.7,.9,'Estimator + local map\nuncertainty / age / support',orange)
box(10.4,.65,3.1,1,'Jetson + whole robot\nqualified survey mission',green)
arrow((1.55,3.4),(4.95,4.2));arrow((6.3,2.95),(7,2.95));arrow((9.7,2.95),(11.2,4.2));arrow((11.95,4.2),(11.95,1.65))
ax.text(.2,1.55,'Available now: CPU geometry, interfaces, evaluation analysis and mount candidates.',color=dark,fontsize=12)
ax.text(.2,1.,'Gate each transition. Preserve flat performance and test unseen terrain / sensor failures.',color=dark,fontsize=12)
ax.set_title('Parallel preparation, followed by evidence-based training steps',loc='left',fontsize=17,color=dark)
fig.savefig(OUT/'terrain_training_sequence.png',dpi=160);plt.close(fig)

fig=plt.figure(figsize=(16,4.1),layout='constrained')
for i,family in enumerate(('smooth_rough','ramp','step','ridge','pit')):
    ax=fig.add_subplot(1,5,i+1,projection='3d')
    v,f,m=terrain_mesh(family,7103)
    # Plot triangles from the actual prepared geometry, not an interpolated image.
    triangles=v[f]
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    triangles=triangles[normals[:,2]>1e-12]
    surf=Poly3DCollection(triangles,linewidth=0,rasterized=True)
    values=triangles[:,:,2].mean(1)
    surf.set_facecolor(plt.cm.viridis((values-values.min())/(np.ptp(values)+1e-9)))
    ax.add_collection3d(surf)
    ax.set(xlim=(-1.5,1.5),ylim=(-1.5,1.5),zlim=(-.15,.20))
    ax.set_box_aspect((1,1,.35));ax.view_init(elev=28,azim=-125)
    ax.set_title(family.replace('_',' ').title()+('\nAvoidance only' if family=='pit' else ''),fontsize=11)
    ax.set_xticks([-1,1]);ax.set_yticks([-1,1]);ax.set_zticks([-.1,0,.1]);ax.tick_params(labelsize=8)
fig.suptitle('Prepared collision fixtures (metres) — Isaac import/contact validation pending',fontsize=15)
fig.savefig(OUT/'terrain_fixture_gallery.png',dpi=150);plt.close(fig)
print(json.dumps({'figures':[str(OUT/x) for x in ('sensor_mount_concept.png','terrain_training_sequence.png','terrain_fixture_gallery.png')]}))
