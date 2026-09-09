#!/usr/bin/env python3
"""Read-only CAD/collider fit audit; sampled surface errors are not mesh proofs."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np
import scipy
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'tools'))
from audit_mkii_stance import _origin
import mkii_fourbar_kinematics as kin


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_mesh(path):
    data = path.read_bytes()
    count = struct.unpack_from('<I', data, 80)[0]
    if len(data) != 84+50*count:
        raise ValueError('Expected the pinned binary STL: '+str(path))
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3,3)), ('attribute','<u2')])
    triangles = np.frombuffer(data, dtype=dtype, offset=84)['vertices'].astype(float)
    points = np.unique(triangles.reshape(-1,3), axis=0)
    hull = points[ConvexHull(points).vertices]
    cross = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
    twice_area = np.linalg.norm(cross, axis=1)
    normal = cross/np.maximum(twice_area[:,None], 1e-30)
    return {'triangles': triangles, 'points': points, 'hull': hull,
            'centroids': triangles.mean(1), 'area': twice_area/2, 'normals':normal}


def shapes(link):
    rows=[]
    for index, collision in enumerate(link.findall('collision')):
        s=collision.find('geometry')[0]
        rows.append({'index':index, 'kind':s.tag, 'transform':_origin(collision),
                     'radius':float(s.get('radius',0)), 'length':float(s.get('length',0)),
                     'half_size':np.array([float(x)/2 for x in s.get('size','0 0 0').split()])})
    return rows


def signed_distance(points, collider):
    t=collider['transform'];q=np.einsum('ij,jk->ik',points-t[:3,3],t[:3,:3])
    if collider['kind']=='sphere':
        return np.linalg.norm(q,axis=1)-collider['radius']
    if collider['kind']=='box':
        v=np.abs(q)-collider['half_size']
    else:
        v=np.stack((np.linalg.norm(q[:,:2],axis=1)-collider['radius'], np.abs(q[:,2])-collider['length']/2),1)
    return np.linalg.norm(np.maximum(v,0),axis=1)+np.minimum(v.max(1),0)


def union_distance(points, colliders):
    result=np.min(np.stack([signed_distance(points,c) for c in colliders],1),1)
    if not np.isfinite(result).all():raise ValueError('Nonfinite surface distance')
    return result


def support(colliders, directions):
    result=[]
    for c in colliders:
        t=c['transform']; d=np.einsum('ij,jk->ik',directions,t[:3,:3])
        if c['kind']=='sphere': extent=np.full(len(d),c['radius'])
        elif c['kind']=='box': extent=np.einsum('ij,j->i',np.abs(d),c['half_size'])
        else: extent=np.linalg.norm(d[:,:2],axis=1)*c['radius']+np.abs(d[:,2])*c['length']/2
        result.append(np.einsum('ij,j->i',directions,t[:3,3])+extent)
    return np.max(result,axis=0)


def transform_points(points,t):
    result=np.einsum('ij,kj->ik',points,t[:3,:3])+t[:3,3]
    if not np.isfinite(result).all():raise ValueError('Nonfinite transformed geometry')
    return result


def summary(values, weights=None):
    result={'max_outside_mm':float(max(0,values.max())*1000),
            'sample_count':int(len(values))}
    for tolerance in [0.1,1.,3.,5.]:
        mask=values>tolerance/1000
        result[f'outside_{tolerance:g}mm_percent']=float(100*(weights[mask].sum()/weights.sum() if weights is not None and weights.sum()>0 else mask.mean()))
    return result


def directions():
    i=np.arange(194);z=1-2*(i+.5)/len(i);theta=i*math.pi*(3-math.sqrt(5))
    radial=np.sqrt(1-z*z)
    return np.vstack((np.stack([radial*np.cos(theta),radial*np.sin(theta),z],1), np.eye(3),-np.eye(3)))


def main():
    previous_errors=np.seterr(divide='raise',over='raise',invalid='raise')
    root=ET.parse(kin.URDF).getroot()
    contract=json.loads(kin.CONTRACT.read_text())
    poses=kin.forward_kinematics(contract['joint_frames'],contract['default_joint_positions_rad'])
    meshes={}; links=[];parts=[];feet=[];special={};input_meshes={};foot_geometry={}
    for link in root.findall('link'):
        name=link.get('name'); colliders=shapes(link); all_hulls=[];all_dist=[];all_area=[];all_vertex=[]
        direction=np.vstack((directions(),poses[name][2,:3],-poses[name][2,:3]))
        for index,visual in enumerate(link.findall('visual')):
            filename=visual.find('geometry/mesh').get('filename').split('/')[-1]
            meshpath=kin.URDF.parent.parent/'meshes'/filename
            if filename not in meshes:
                meshes[filename]=read_mesh(meshpath);input_meshes[filename]=sha(meshpath)
            mesh=meshes[filename];t=_origin(visual)
            # All current visual scales are identity; do not silently ignore new scales.
            if visual.find('geometry/mesh').get('scale','1 1 1')!='1 1 1':raise ValueError('New mesh scale requires review')
            vertices=transform_points(mesh['points'],t); centers=transform_points(mesh['centroids'],t)
            hull=transform_points(mesh['hull'],t)
            vd=union_distance(vertices,colliders);cd=union_distance(centers,colliders)
            all_hulls.append(hull);all_dist.append(cd);all_area.append(mesh['area']);all_vertex.append(vd)
            part={'link':name,'visual_index':index,'mesh':filename,
                  'surface_centroid_area_weighted':summary(cd,mesh['area']),
                  'vertex_sample':summary(vd),
                  'worst_vertex_link_m':vertices[int(np.argmax(vd))].tolist()}
            parts.append(part)
            if filename=='silicone_foot.stl':
                pads=[c for c in colliders if c['kind']=='sphere']
                pad_vd=union_distance(vertices,pads);pad_cd=union_distance(centers,pads)
                world=poses[name];height=contract['reset_root_height_m']
                vertex_z=np.einsum('ij,j->i',vertices,world[2,:3])+world[2,3]+height
                centroid_z=np.einsum('ij,j->i',centers,world[2,:3])+world[2,3]+height
                normals_world_z=np.einsum('ij,kj,k->i',mesh['normals'],t[:3,:3],world[2,:3])
                band=(centroid_z<=vertex_z.min()+.010)&(normals_world_z<-.25)&(mesh['area']>0)
                hmesh=np.einsum('ij,kj->ik',hull,direction).max(0);hpad=support(pads,direction);hfull=support(colliders,direction)
                foot={'leg':name.split('_')[0],'spheres_only_vertex_sample':summary(pad_vd),
                      'spheres_only_surface_centroid_area_weighted':summary(pad_cd,mesh['area']),
                      'complete_tibia_vertex_sample':summary(vd),
                      'complete_tibia_surface_centroid_area_weighted':summary(cd,mesh['area']),
                      'lower_10mm_downward_surface_samples':int(band.sum()),
                      'lower_10mm_downward_spheres_only_area_weighted':summary(pad_cd[band],mesh['area'][band]),
                      'lower_10mm_downward_complete_tibia_area_weighted':summary(cd[band],mesh['area'][band]),
                      'cad_pad_min_z_at_nominal_reset_m':float(vertex_z.min()),
                      'collider_pad_min_z_at_nominal_reset_m':float(-support(pads,np.array([-world[2,:3]]))[0]+world[2,3]+height),
                      'sampled_support_sphere_overhang_max_mm':float(max(0,(hpad-hmesh).max())*1000),
                      'sampled_support_sphere_underreach_max_mm':float(max(0,(hmesh-hpad).max())*1000),
                      'sphere_center_distance_mm':float(np.linalg.norm(pads[0]['transform'][:3,3]-pads[1]['transform'][:3,3])*1000),
                      'sphere_radius_mm':float(pads[0]['radius']*1000)}
                feet.append(foot)
                foot_geometry[name]={'hull':hull,'pads':pads}
                if name=='lf_tibia':special={'triangles':transform_points(mesh['triangles'].reshape(-1,3),t).reshape(-1,3,3),'colliders':colliders,'pads':pads,'world':world}
        hulls=np.concatenate(all_hulls)
        mesh_support=np.einsum('ij,kj->ik',hulls,direction).max(0);collider_support=support(colliders,direction)
        excess=collider_support-mesh_support;under=-excess
        cdist=np.concatenate(all_dist);vdist=np.concatenate(all_vertex);area=np.concatenate(all_area)
        vertices_bottom=-float(np.einsum('ij,j->i',hulls,-poses[name][2,:3]).max())+poses[name][2,3]+contract['reset_root_height_m']
        collider_bottom=-float(support(colliders,np.array([-poses[name][2,:3]]))[0])+poses[name][2,3]+contract['reset_root_height_m']
        links.append({'link':name,'colliders':len(colliders),'visual_instances':len(link.findall('visual')),
                      'vertex_sample':summary(vdist),'surface_centroid_area_weighted':summary(cdist,area),
                      'sampled_support_overhang_max_mm':float(max(0,excess.max())*1000),
                      'sampled_support_underreach_max_mm':float(max(0,under.max())*1000),
                      'overhang_direction_link':direction[int(np.argmax(excess))].tolist(),
                      'overhang_collision_index':int(np.argmax([support([c],direction[[int(np.argmax(excess))]])[0]for c in colliders])),
                      'underreach_direction_link':direction[int(np.argmax(under))].tolist(),
                      'cad_min_z_at_nominal_reset_m':vertices_bottom,'collider_min_z_at_nominal_reset_m':collider_bottom})
    for foot in feet:
        name=foot['leg']+'_tibia';data=foot_geometry[name];cases=[]
        for offsets in itertools.product([-.30,0.,.30],repeat=3):
            active={n:contract['default_joint_positions_rad'][n]for n in contract['active_joint_names']}
            for suffix,offset in zip(kin.ACTIVE_SUFFIXES,offsets):active[foot['leg']+'_'+suffix]+=offset
            pose=kin.forward_kinematics(contract['joint_frames'],kin.expand_active(active))[name]
            downward=-pose[2,:3]
            mesh_min=-np.einsum('ij,j->i',data['hull'],downward).max()
            pad_min=-support(data['pads'],np.array([downward]))[0]
            cases.append({'motor_offsets_rad':list(offsets),'collider_minus_cad_lowest_z_mm':float((pad_min-mesh_min)*1000)})
        foot['planned_target_envelope_27_poses']={'scope':'Static kinematic combinations of nominal+[-.30,0,.30] on three active motors; no continuous or physical sweep guarantee',
            'minimum_bias_case':min(cases,key=lambda x:x['collider_minus_cad_lowest_z_mm']),
            'maximum_bias_case':max(cases,key=lambda x:x['collider_minus_cad_lowest_z_mm'])}
    output={'schema':'hexapod.collider_fit.v1','source':{'linkage_urdf_sha256':sha(kin.URDF),
            'kinematics_sha256':sha(kin.CONTRACT),'usd_sha256':sha(ROOT/contract['usd_path_relative']),
            'audit_script_sha256':sha(__file__),'meshes_sha256':input_meshes},
            'versions':{'numpy':np.__version__,'scipy':scipy.__version__},
            'method':{'surface':'All unique STL vertices and all triangle centroids transformed by each visual origin; exact point-to-primitive signed-distance formulas; min across union. Positive values are actual undercoverage at those sample points.',
            'area':'Triangle-centroid quadrature weighted by triangle area; not an exact exterior-surface fraction, and internal/overlapping part surfaces remain included.',
            'support':'202 specified directions per link (194 Fibonacci sphere +6 local axes +2 nominal verticals). Mesh support uses its convex-hull vertices; shape support analytic. Positive collider-minus-CAD support proves protrusion outside CAD convex envelope for that direction. Max over directions is sampled, not worst over all directions.',
            'foot_band':'Triangle centroid within10mm of lowest nominal pad vertex and outward winding normal with worldZ<-.25; a nominal contact-relevant sample, not every rough-terrain contact.',
            'scope':'Surface fit only. No mesh watertightness/solid-union proof, contact deformation/friction/compliance, self-collision sweep, rigid-body simulation or hardware qualification.'},
            'links':links,'footpads':feet,'parts':parts}
    (OUT/'report.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    with (OUT/'links.csv').open('w',newline='') as f:
        fields=['link','colliders','visual_instances','max_vertex_outside_mm','area_centroid_outside_1mm_percent','support_overhang_mm','support_underreach_mm','cad_reset_bottom_mm','collider_reset_bottom_mm']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in links:writer.writerow(dict(zip(fields,[r['link'],r['colliders'],r['visual_instances'],r['vertex_sample']['max_outside_mm'],r['surface_centroid_area_weighted']['outside_1mm_percent'],r['sampled_support_overhang_max_mm'],r['sampled_support_underreach_max_mm'],r['cad_min_z_at_nominal_reset_m']*1000,r['collider_min_z_at_nominal_reset_m']*1000])))
    np.seterr(**previous_errors)
    plot_foot(special)
    print(json.dumps({'links':len(links),'primitives':sum(r['colliders']for r in links),'visual_instances':len(parts),'unique_meshes':len(meshes),'foot':feet[0]},indent=2))


def plot_foot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    pads=data['pads'];a,b=[p['transform'][:3,3]for p in pads];center=(a+b)/2
    u=(b-a)/np.linalg.norm(b-a)
    down=-data['world'][2,:3];v=down-u*(u@down);v/=np.linalg.norm(v)
    normal=np.cross(u,v)
    segments=[]
    for tri in data['triangles']:
        signed=(tri-center)@normal
        hit=[]
        for i,j in [(0,1),(1,2),(2,0)]:
            if signed[i]*signed[j]<0:
                hit.append(tri[i]+signed[i]/(signed[i]-signed[j])*(tri[j]-tri[i]))
        if len(hit)==2:
            p=np.array(hit)-center
            segments.append(np.stack((p@u,p@v),1)*1000)
    x,y=np.meshgrid(np.linspace(-.036,.036,721),np.linspace(-.024,.024,481))
    points=center+x.reshape(-1,1)*u+y.reshape(-1,1)*v
    sdf=union_distance(points,data['colliders']).reshape(x.shape)*1000
    pad_sdf=union_distance(points,pads).reshape(x.shape)*1000
    fig,ax=plt.subplots(figsize=(10,6.4))
    fig.subplots_adjust(left=.09,right=.99,top=.90,bottom=.19)
    ax.contourf(x*1000,y*1000,sdf,levels=[-1000,0],colors=['#dcf1e6'])
    ax.contour(x*1000,y*1000,sdf,levels=[0],colors=['#20845a'],linewidths=2)
    ax.contour(x*1000,y*1000,pad_sdf,levels=[0],colors=['#3279bd'],linewidths=1.3,linestyles='--')
    ax.add_collection(LineCollection(segments,colors='#222b36',linewidths=1.8))
    for color,label,style in [('#222b36','Actual silicone CAD section','-'),('#20845a','Complete tibia collider union','-'),('#3279bd','Two foot spheres only','--')]:
        ax.plot([],[],color=color,label=label,linestyle=style)
    ax.set(xlim=(-36,36),ylim=(-24,24),xlabel='Along pad (mm)',ylabel='Toward nominal ground (mm)',title='Left-front pad: exact CAD section versus primitive contact geometry')
    ax.invert_yaxis();ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=9)
    fig.text(.5,.045,'Plane contains both sphere centers and nominal downward direction. Collider contour grid 0.1 mm; no deformation simulated.',ha='center',fontsize=8)
    fig.savefig(OUT/'foot_section.png',dpi=180)
    fig.savefig(OUT/'foot_section.svg')
    plt.close(fig)


if __name__=='__main__':main()
