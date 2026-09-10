#!/usr/bin/env python3
"""Audit exported CAD contact geometry without substituting visual/inertial data.
Run with the repository venv: python audit_collisions.py SOURCE_EXPORT OUTPUT_DIR.
Requires installed numpy/scipy/trimesh/rtree only. No geometry libraries installed.
All meshes stay in original Onshape part coordinates, in metres.
"""
from __future__ import annotations
import argparse, hashlib, json, math, importlib.metadata, importlib.util
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import ConvexHull


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def clipped_surface_points(mesh, bounds):
    # Exact triangle clipping; includes intersection vertices on original faces.
    triangles=mesh.triangles
    use=np.ones(len(triangles),dtype=bool)
    for axis,(lo,hi) in enumerate(bounds):
        if lo is not None:use &= triangles[:,:,axis].max(axis=1)>=lo-1e-10
        if hi is not None:use &= triangles[:,:,axis].min(axis=1)<=hi+1e-10
    out=[]
    for tri in triangles[use]:
        poly=list(tri)
        for axis,(lo,hi) in enumerate(bounds):
            for value,sign in [(lo,1),(hi,-1)]:
                if value is None or not poly:continue
                new=[]
                for a,b in zip(poly,[*poly[1:],poly[0]]):
                    da=sign*(a[axis]-value); db=sign*(b[axis]-value)
                    if da>=-1e-12:new.append(a)
                    if (da < -1e-12) != (db < -1e-12):
                        new.append(a+(b-a)*(value-a[axis])/(b[axis]-a[axis]))
                poly=new
        out.extend(poly)
    return np.unique(np.round(np.array(out),12),axis=0)


def reduced_hull(points, tol=.00005, max_vertices=124):
    points=points[ConvexHull(points).vertices]
    # Add worst violating source vertex until tolerance or vertex budget met.
    seed_directions=np.array([[x,y,z] for x in [-1,0,1] for y in [-1,0,1] for z in [-1,0,1] if x or y or z])
    ids=sorted(set(np.argmax(points@seed_directions.T,axis=0).tolist()))
    for _ in range(max_vertices):
        hull=ConvexHull(points[ids]);equ=hull.equations
        distances=points@equ[:,:3].T+equ[:,3]
        worst=distances.max(axis=1);idx=int(worst.argmax())
        if worst[idx]<=tol or len(ids)>=max_vertices:break
        ids.append(idx)
    mesh=trimesh.convex.convex_hull(points[ids])
    # Exact Euclidean distance of omitted convex-source vertices to result.
    pq=trimesh.proximity.ProximityQuery(mesh)
    _,dist,_=pq.on_surface(points)
    return mesh,{'source_hull_vertices':len(points),'output_vertices':len(mesh.vertices),
        'output_triangles':len(mesh.faces),'max_source_vertex_to_hull_m':float(dist.max()),
        'target_facet_deviation_m':tol,'max_vertices':max_vertices}


def fibonacci(n):
    z=1-2*(np.arange(n)+.5)/n;phi=np.arange(n)*math.pi*(3-math.sqrt(5));r=np.sqrt(1-z*z)
    return np.stack([z,r*np.cos(phi),r*np.sin(phi)],axis=1)


def support(vertices,directions):
    return np.concatenate([np.max(d@vertices.T,axis=1) for d in np.array_split(directions,max(1,len(directions)//512))])


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    assets=args.source/'assets';args.output.mkdir(parents=True,exist_ok=True);(args.output/'meshes').mkdir(exist_ok=True)
    names=['tibia','femur_first_stage','first_joint_top','first_joint_bottom_plate','tibia_attatchment_plate','motor_bearing_holder','top_enclosure','bottom_enclosure']
    report={'schema_version':1,'units':'metres, kilograms, radians','status':'offline geometry candidates, not Isaac-qualified','software_versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','trimesh','rtree']},'installed_decomposition_packages':{p:bool(importlib.util.find_spec(p)) for p in ['coacd','vhacdx','pyvhacd']},'mesh_audit':{},'foot':{},'candidates':[],
    'limitations':['CAD mass and inertia must remain the independent authoritative values; no collision-volume mass recomputation.',
    'CAD mate graph, physical hard stops, compliance and friction are not present in these STL surfaces.',
    'Contact audit is geometric and rigid only; no dynamic Isaac validation was performed.',
    'Distal cap hulls fill their inaccessible enclosed interiors; hollow-shaft cavities cannot be represented as one hull when exposed.',
    'Only candidate mesh files recorded here are generated; do not replace the entire tibia collision with the cap.']}
    for name in names:
        path=assets/(name+'.stl');m=trimesh.load(path,force='mesh');h=m.convex_hull
        report['mesh_audit'][name]={'sha256':sha(path),'vertices':len(m.vertices),'triangles':len(m.faces),'watertight':bool(m.is_watertight),
            'bounds_m':m.bounds.tolist(),'solid_mesh_volume_m3':float(m.volume),'single_hull_volume_m3':float(h.volume),'single_hull_volume_ratio':float(h.volume/m.volume)}
    tibia=trimesh.load(assets/'tibia.stl',force='mesh');v=tibia.vertices
    # Tip is a rounded CAD cap ending in a 2.5mm square flat. Four quadrants
    # prevent PhysX vertex budget from coarsening the important contact surface.
    cap_meshes=[]
    for yi,sy in enumerate([-1,1]):
        for zi,sz in enumerate([-1,1]):
            bounds=[(.115,None),(None,0) if sy<0 else (0,None),(None,0) if sz<0 else (0,None)]
            points=clipped_surface_points(tibia,bounds)
            points=np.vstack([points,[.115,0,0],[float(v[:,0].max()),0,0]])
            mesh,metrics=reduced_hull(points,tol=.00003,max_vertices=124)
            fn=f'tibia_foot_cap_y{sy:+d}_z{sz:+d}.stl';path=args.output/'meshes'/fn;mesh.export(path)
            metrics.update({'file':'meshes/'+fn,'sha256':sha(path),'source':'tibia.stl','clip_bounds_m':bounds,'convex':bool(mesh.is_convex),'watertight':bool(mesh.is_watertight),'kind':'foot_cap_quadrant','within_255_vertex_and_triangle_budget':len(mesh.vertices)<=255 and len(mesh.faces)<=255})
            report['candidates'].append(metrics);cap_meshes.append(mesh)
    cap_vertices=np.concatenate([m.vertices for m in cap_meshes])
    dirs=fibonacci(24000);cone=dirs[dirs[:,0]>=.5]
    original=support(v,cone);candidate=support(cap_vertices,cone)
    # Compare plausible sphere preserving x=130mm extreme, and best radial fit.
    fitpoints=v[v[:,0]>.115];fit=np.linalg.lstsq(np.c_[2*fitpoints,np.ones(len(fitpoints))],(fitpoints*fitpoints).sum(1),rcond=None)[0]
    center=fit[:3];radius=math.sqrt(fit[3]+sum(center*center));resid=np.linalg.norm(fitpoints-center,axis=1)-radius
    spheres=[{'label':'sphere_with_16p25mm_radius_and_matching_extreme','center_m':[.11375,0,0],'radius_m':.01625},
             {'label':'least_squares_sphere_on_distal_vertices','center_m':center.tolist(),'radius_m':radius,'radial_fit_rms_m':float(np.sqrt(np.mean(resid*resid))),'radial_fit_max_abs_m':float(abs(resid).max())}]
    for sp in spheres:
        error=cone@np.array(sp['center_m'])+sp['radius_m']-original
        sp['forward_60deg_contact_support_error_m']={'max_overfill':float(max(0,error.max())),'max_undercoverage':float(max(0,-error.min())),'rms':float(np.sqrt(np.mean(error*error)))}
    # Surface sampling excludes clipping faces. Vertices and centroids examine
    # original exterior facets, not internal cap-quadrant partition surfaces.
    samples=[]
    for m in cap_meshes:
        keep=(m.face_normals[:,0]>1e-5)&(m.triangles_center[:,0]>.115+1e-7)
        tr=m.triangles[keep]
        samples.append(np.concatenate([tr.mean(axis=1),(tr[:,0]+tr[:,1])/2,(tr[:,1]+tr[:,2])/2,(tr[:,2]+tr[:,0])/2]))
    samples=np.concatenate(samples);d=np.concatenate([trimesh.proximity.signed_distance(tibia,x) for x in np.array_split(samples,max(1,len(samples)//300))])
    caperr=candidate-original
    report['foot']={'local_axis_toward_foot':[1,0,0],'local_knee_axis':[0,1,0],'hinge_center_reference_m':[0,0,0],
        'extreme_contact_plane_x_m':float(v[:,0].max()),'distal_flat_yz_bounds_m':[[-.00125,-.00125],[.00125,.00125]],
        'cap_start_x_m':.115,'cap_interpretation':'CAD rounded cap on rounded-square shaft, not a sphere; 2.5mm square distal flat.',
        'convenience_tip_frame_m':[.13,0,0],'tip_frame_is_not_sphere_center':True,
        'fitted_sphere_candidates_rejected':spheres,
        'compound_cap_forward_60deg_support':{'directions':len(cone),'max_overfill_m':float(max(0,caperr.max())),'max_undercoverage_m':float(max(0,-caperr.min())),'rms_error_m':float(np.sqrt(np.mean(caperr*caperr)))},
        'compound_cap_external_surface_samples':{'samples':len(samples),'max_outside_original_m':float(max(0,-d.min())),'max_inside_original_m':float(max(0,d.max())),'max_absolute_m':float(abs(d).max())},
        'shaft_cross_sections':[]}
    for x in [.06,.08,.10,.115,.12,.125,.129]:
        pts=tibia.section(plane_origin=[x,0,0],plane_normal=[1,0,0]).vertices
        report['foot']['shaft_cross_sections'].append({'x_m':x,'yz_bounds_m':[pts[:,1:].min(0).tolist(),pts[:,1:].max(0).tolist()]})
    # Large open U cavities must remain open. Part-level split planes below are
    # geometry measurements, not an automatically approved collider set.
    report['plate_recommendations']=[
        {'mesh':'femur_first_stage.stl','type':'U bracket','side_plate_y_ranges_m':[[-.030282,-.02825],[.02825,.030282]],'base_z_range_m':[-.0016,.000432],'open_gap_m':.0565,'recommendation':'split into base and two side plates; small bolt holes may be suppressed only after quantified clearance review'},
        {'mesh':'first_joint_bottom_plate.stl','type':'U bracket with bearing through-holes','side_bearing_hole_center_xz_m':[-.049,.0261],'base_bearing_hole_center_xy_m':[0,0],'bearing_hole_radius_m':.016,'side_plate_y_ranges_m':[[-.0261,-.0245],[.0245,.0261]],'base_z_range_m':[0,.0016],'open_gap_m':.049,'recommendation':'base plus independently segmented side plates; preserve large bearing holes with ring sectors instead of solid plate hulls'},
        {'mesh':'tibia_attatchment_plate.stl','type':'U bracket with bearing through-holes','side_bearing_hole_center_xz_m':[0,-.028502],'bearing_hole_radius_m':.016,'side_plate_y_ranges_m':[[-.026532,-.0245],[.0245,.026532]],'base_z_range_m':[-.003632,-.0016],'open_gap_m':.049,'recommendation':'base plus independently segmented side plates; preserve large bearing holes with ring sectors instead of solid plate hulls'},
        {'mesh':'tibia.stl','type':'fork plus hollow shaft and closed cap','fork_inner_y_m':[-.02825,.02825],'fork_outer_y_m':[-.03325,.03325],'recommendation':'keep two fork cheeks and curved transition separate; a single whole-part hull fills the 56.5mm fork opening; supplied four cap hulls are a contact-only component'}]
    report['physx_authoring_notes']={'explicit_hull_vertex_limit':124,'explicit_hull_max_triangles':244,'default_hull_vertex_limit_64_would_change_mesh':True,'single_convex_max_vertices_and_faces':255,'required_next_steps':['Author convexHull approximation with PhysxConvexHullCollisionAPI hullVertexLimit=124 for these 4 meshes.','Preserve CAD inertias explicitly; never infer mass from candidate hull volume.','Cook in pinned Isaac Sim and inspect output geometry, warnings and contact fidelity.','Add fork, shaft and structural collisions; cap-only is not a complete tibia collider.'],'sources':['https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Geometry.html','https://docs.omniverse.nvidia.com/kit/docs/usdrt.scenegraph/7.6.1/api/classusdrt_1_1_physx_schema_physx_convex_hull_collision_a_p_i.html','https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html','https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/dev_guide/guides/collision_guide.html']}
    (args.output/'collision_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'foot':report['foot'],'candidate_count':len(report['candidates']),'bytes':sum(x.stat().st_size for x in args.output.rglob('*') if x.is_file())},indent=2))

if __name__=='__main__':main()
