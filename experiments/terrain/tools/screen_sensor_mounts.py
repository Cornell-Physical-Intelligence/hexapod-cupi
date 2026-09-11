#!/usr/bin/env python3
"""Reproducible CPU mount pre-screen. No CAD edits, GPU or robot connections.

Outputs geometry-only visibility and a conservative clearance screen using every
URDF visual mesh's enclosing box. A box hit is POSSIBLE self-occlusion, not an
exact triangle hit. A clear segment is clear of the included CAD meshes only.
"""

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

from experiments.terrain.tools.sensor_mount_math import apply_transform, axis_rotation, box_corners, look_outward, nominal_profile, optical_visibility, replay_visibility_contract, segments_hit_boxes, transform

ROOT = Path(__file__).resolve().parents[3]
LEGS = ('lf', 'lm', 'lr', 'rf', 'rm', 'rr')
SOURCE_URL = 'https://www.realsenseai.com/download/21345/?tmstv=1780360410'


def vector(value, default='0 0 0'):
    return np.fromstring(value or default, sep=' ')


def origin(element):
    e = element.find('origin')
    return np.eye(4) if e is None else transform(vector(e.get('xyz')), vector(e.get('rpy')))


def stl_bounds(path):
    raw = path.read_bytes()
    n = struct.unpack_from('<I', raw, 80)[0]
    if len(raw) != 84+50*n:
        raise ValueError(f'Expected binary STL with exact triangle count: {path}')
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attr', '<u2')])
    vertices = np.frombuffer(raw, dtype=dtype, offset=84)['vertices'].reshape(-1, 3)
    if not np.isfinite(vertices).all():
        raise ValueError(f'Nonfinite mesh: {path}')
    return np.stack((vertices.min(axis=0), vertices.max(axis=0))).astype(float), n


class RobotBounds:
    def __init__(self, urdf):
        self.urdf = urdf
        root = ET.parse(urdf).getroot()
        self.joints = list(root.findall('joint'))
        self.boxes, self.meshes, self.visual_count = {}, {}, 0
        self.links, self.parts = {}, {}
        for link in root.findall('link'):
            name = link.get('name')
            self.links[name] = link
            bounds, parts = [], []
            for visual in link.findall('visual'):
                mesh = visual.find('geometry/mesh')
                if mesh is None:
                    raise ValueError('This screen requires every visual to be a mesh')
                filename = mesh.get('filename')
                path = ROOT/'robot'/filename.removeprefix('package://')
                if path not in self.meshes:
                    box, triangles = stl_bounds(path)
                    self.meshes[path] = dict(bounds=box, triangles=triangles,
                                            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                corners = box_corners(self.meshes[path]['bounds']) * vector(mesh.get('scale'), '1 1 1')
                points = apply_transform(corners, origin(visual))
                bounds.append(np.stack((points.min(axis=0), points.max(axis=0))))
                scale=np.eye(4);scale[:3,:3]=np.diag(vector(mesh.get('scale'), '1 1 1'))
                parts.append((path,origin(visual)@scale))
            self.boxes[name] = np.array(bounds)
            self.parts[name] = parts
            self.visual_count += len(bounds)
        if len(self.links) != 19 or len(self.joints) != 18:
            raise ValueError('Expected production serial asset:19 links/18 joints')

    def forward(self, q):
        out = {'body': np.eye(4)}
        remaining = list(self.joints)
        while remaining:
            changed = False
            for j in remaining[:]:
                parent = j.find('parent').get('link')
                if parent not in out:
                    continue
                out[j.find('child').get('link')] = (out[parent] @ origin(j) @
                    axis_rotation(vector(j.find('axis').get('xyz')), q[j.get('name')]))
                remaining.remove(j)
                changed = True
            if not changed:
                raise ValueError('Disconnected or cyclic URDF')
        return out

    def clearance(self, camera_body, targets_body, transforms):
        blocked = np.zeros(len(targets_body), dtype=bool)
        body_blocked = blocked.copy()
        for link, boxes in self.boxes.items():
            matrix = transforms[link]
            o = (camera_body-matrix[:3, 3]) @ matrix[:3, :3]
            p = (targets_body-matrix[:3, 3]) @ matrix[:3, :3]
            broad = np.stack((boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)))[None]
            indices = np.flatnonzero(~blocked & segments_hit_boxes(o, p, broad))
            if len(indices):
                hit = segments_hit_boxes(o, p[indices], boxes)
                blocked[indices[hit]] = True
                if link == 'body':
                    body_blocked[indices[hit]] = True
        return ~blocked, body_blocked

    def foot_centres(self, fk):
        points = []
        for leg in LEGS:
            spheres = self.links[leg+'_tibia'].findall('collision')
            spheres = [s for s in spheres if s.find('geometry/sphere') is not None]
            distal = max(spheres, key=lambda s: np.linalg.norm(origin(s)[:3, 3]))
            points.append(apply_transform(origin(distal)[:3, 3], fk[leg+'_tibia']))
        return np.array(points)


class ExactMeshScreen:
    """Optional CPU triangle refinement; requires trimesh+rtree in an isolated env."""
    def __init__(self, robot):
        import trimesh
        self.trimesh=trimesh
        self.robot=robot
        self.meshes={}

    def clearance(self,camera_body,targets_body,transforms):
        blocked=np.zeros(len(targets_body),dtype=bool)
        origin_ambiguous=np.zeros(len(targets_body),dtype=bool)
        for link,boxes in self.robot.boxes.items():
            matrix=transforms[link]
            camera=(camera_body-matrix[:3,3])@matrix[:3,:3]
            targets=(targets_body-matrix[:3,3])@matrix[:3,:3]
            broad=np.stack((boxes[:,0].min(axis=0),boxes[:,1].max(axis=0)))[None]
            potential=np.flatnonzero(~blocked & segments_hit_boxes(camera,targets,broad))
            if not len(potential):
                continue
            for box,(path,part_transform) in zip(boxes,self.robot.parts[link]):
                indices=potential[~blocked[potential]]
                indices=indices[segments_hit_boxes(camera,targets[indices],box[None])]
                if not len(indices):
                    continue
                # No claim of a valid camera installation when the camera origin
                # lies inside a part bound; thin/open vendor meshes complicate
                # winding-based inside tests. Retain explicit uncertainty.
                if ((camera>=box[0]) & (camera<=box[1])).all():
                    origin_ambiguous[indices]=True
                if path not in self.meshes:
                    self.meshes[path]=self.trimesh.load_mesh(path,process=False)
                mesh=self.meshes[path]
                inv=np.linalg.inv(part_transform)
                o=apply_transform(camera,inv)
                p=apply_transform(targets[indices],inv)
                d=p-o
                lengths=np.linalg.norm(d,axis=1)
                good=lengths>1e-9
                locations,rays,_=mesh.ray.intersects_location(np.repeat(o[None],good.sum(),axis=0),
                    d[good]/lengths[good,None],multiple_hits=False)
                if len(rays):
                    # Parameter along the same transformed segment, avoiding
                    # assumptions about uniform mesh scale.
                    params=np.linalg.norm(locations-o,axis=1)/lengths[good][rays]
                    original=indices[good][rays]
                    world_length=np.linalg.norm(targets_body[original]-camera_body,axis=1)
                    hit=(params>=0)&(params < 1-.001/np.maximum(world_length,1e-9))
                    blocked[original[hit]]=True
        return ~blocked & ~origin_ambiguous,blocked,origin_ambiguous


def exact_finalists(args):
    start=time.time()
    report=json.loads((args.out/'report.json').read_text())
    robot=RobotBounds(ROOT/report['urdf_path'])
    exact=ExactMeshScreen(robot)
    points=np.array(report['targets']['points_world_m']);sectors=np.array(report['targets']['sector']);bands=np.array(report['targets']['band'])
    # Two finalists have the best nominal close-depth limits of each family.
    selected=[r for r in report['rig_comparisons'] if r['id'] in ('D405_848x480_6cam','D435_640x360_6cam')]
    cases=[s for s in report['states'] if s['pose'] in ('stance','tripod_a_lift','tripod_b_lift') and s['height_m']==.124]
    results=[]
    for rig in selected:
        masks=[];optics=[];ambiguous=[]
        for state in cases:
            body=transform((0,0,state['height_m']),np.radians((state['roll_deg'],state['pitch_deg'],0)))
            ground=(points-body[:3,3])@body[:3,:3]
            fk=robot.forward(report['pose_joint_values'][state['pose']])
            visible=np.zeros(len(points),dtype=bool);optical_union=visible.copy();ambiguity_union=visible.copy()
            for index in rig['member_indices']:
                c=report['candidates'][index];p=np.array(c['body_position_m']);rotation=np.array(c['optical_rotation_body']);profile=report['profiles'][c['profile_id']]
                inview=optical_visibility((ground-p)@rotation,profile)
                optical_union|=inview
                ids=np.flatnonzero(inview & ~visible)
                left_ok,_,left_ambiguous=exact.clearance(p,ground[ids],fk)
                right_ok,_,right_ambiguous=exact.clearance(p+profile['baseline_m']*rotation[:,0],ground[ids],fk)
                visible[ids]|=left_ok&right_ok
                ambiguity_union[ids]|=left_ambiguous|right_ambiguous
            masks.append(visible);optics.append(optical_union);ambiguous.append(ambiguity_union&~visible)
            print(f'Exact {rig["id"]} {state}: {visible.mean():.1%} clear',flush=True)
        mask=np.array(masks)
        results.append(dict(rig_id=rig['id'],members=rig['members'],cases=cases,
            triangle_clearance=statistics(mask,sectors,bands),
            optics_only_upper_bound=statistics(np.array(optics),sectors,bands),
            origin_inside_bound_ambiguity_fraction=float(np.mean(ambiguous)),
            qualification=False))
        np.savez_compressed(args.out/(rig['id']+'_exact.npz'),clear=mask,optics_only=np.array(optics),origin_ambiguous=np.array(ambiguous))
    output=dict(version=1,status='sampled_exact_triangle_ray_refinement',results=results,
        base_report_sha256=hashlib.sha256((args.out/'report.json').read_bytes()).hexdigest(),
        source_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        trimesh_version=exact.trimesh.__version__,elapsed_seconds=time.time()-start,
        method='Nearest forward triangle surface intersections on both stereo-to-ground segments; all URDF visual parts are eligible occluders. Camera origins inside a part bound remain explicitly ambiguous.',
        limits='Only27nominal-height stance/lift attitude cases per finalist. No final C geometry, physical depth/noise, housing/harness/payload occlusion or continuous gait/stop sweep. Nominal inferred rectified optical intrinsics. Not an actual camera rendering.')
    (args.out/'exact_finalists.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


def exact_camera_masks(robot,exact,cameras,profile,points,fk,body):
    ground=(points-body[:3,3])@body[:3,:3]
    masks=[];optics=[]
    for camera in cameras:
        p=np.array(camera['body_position_m']);rotation=np.array(camera['optical_rotation_body'])
        inview=optical_visibility((ground-p)@rotation,profile)
        ids=np.flatnonzero(inview)
        left_ok,_,_=exact.clearance(p,ground[ids],fk)
        right_ok,_,_=exact.clearance(p+profile['baseline_m']*rotation[:,0],ground[ids],fk)
        mask=np.zeros(len(points),dtype=bool);mask[ids]=left_ok&right_ok
        masks.append(mask);optics.append(inview)
    return np.array(masks),np.array(optics)


def expand_exact_mounts(args):
    """CPU geometric hypothesis search; positions are not CAD-approved brackets."""
    start=time.time();report=json.loads((args.out/'report.json').read_text())
    robot=RobotBounds(ROOT/report['urdf_path']);exact=ExactMeshScreen(robot)
    points=np.array(report['targets']['points_world_m']);sectors=np.array(report['targets']['sector']);bands=np.array(report['targets']['band'])
    profile=report['profiles']['D405_848x480']
    fk=robot.forward(report['pose_joint_values']['stance']);body=transform((0,0,.124))
    original={c['leg_sector']:c for c in report['candidates'] if c['id'].endswith('D405_z100_p45_848x480')}
    results=[]
    for offset,height,pitch in itertools.product((0.,.04,.08,.12),(.07,.10,.14),(45,60,75,90)):
        cameras=[]
        for leg in LEGS:
            p=np.array(original[leg]['body_position_m']);horizontal=np.array([p[0],p[1],0.]);horizontal/=np.linalg.norm(horizontal)
            p+=offset*horizontal;p[2]=height
            cameras.append(dict(id=f'{leg}_D405_out{offset:.2f}_z{height:.2f}_p{pitch}',leg_sector=leg,
                body_position_m=p.tolist(),optical_rotation_body=look_outward(p,pitch).tolist()))
        mask,optical=exact_camera_masks(robot,exact,cameras,profile,points,fk,body)
        stats=statistics(mask.any(axis=0)[None],sectors,bands)
        results.append(dict(id=f'D405_out{offset:.2f}_z{height:.2f}_p{pitch}',outboard_offset_m=offset,
            height_above_plate_m=height,downward_pitch_deg=pitch,cameras=cameras,
            nominal_triangle_clearance=stats,nominal_optics=statistics(optical.any(axis=0)[None],sectors,bands)))
    results.sort(key=lambda r:(r['nominal_triangle_clearance']['worst_sector_mean_fraction'],r['nominal_triangle_clearance']['mean_fraction']),reverse=True)
    cases=[s for s in report['states'] if s['pose'] in ('stance','tripod_a_lift','tripod_b_lift') and s['height_m']==.124]
    finalists=[]
    for candidate in results[:3]:
        masks=[];optics=[]
        for state in cases:
            body=transform((0,0,state['height_m']),np.radians((state['roll_deg'],state['pitch_deg'],0)))
            fk=robot.forward(report['pose_joint_values'][state['pose']])
            mask,optical=exact_camera_masks(robot,exact,candidate['cameras'],profile,points,fk,body)
            masks.append(mask);optics.append(optical)
        masks=np.array(masks);optics=np.array(optics)
        comparisons=[]
        for count in (2,4,6):
            alternatives=[]
            for ids in itertools.combinations(range(6),count):
                stats=statistics(masks[:,ids].any(axis=1),sectors,bands)
                alternatives.append((stats['worst_case_sector_fraction'],stats['worst_sector_mean_fraction'],stats['mean_fraction'],ids,stats))
            _,_,_,ids,stats=max(alternatives,key=lambda a:a[:3])
            comparisons.append(dict(camera_count=count,sectors=[LEGS[i] for i in ids],triangle_clearance=stats,
                optics_only=statistics(optics[:,ids].any(axis=1),sectors,bands)))
        finalists.append(dict(candidate,rig_comparisons=comparisons))
        np.savez_compressed(args.out/(candidate['id']+'_expanded_exact.npz'),clear=masks,optics_only=optics)
        print(f'Expanded {candidate["id"]}: {comparisons[-1]["triangle_clearance"]}',flush=True)
    best=max(finalists,key=lambda c:(c['rig_comparisons'][-1]['triangle_clearance']['worst_sector_mean_fraction'],c['rig_comparisons'][-1]['triangle_clearance']['mean_fraction']))
    sample=np.load(args.out/(best['id']+'_expanded_exact.npz'))['clear']
    nominal_index=next(i for i,s in enumerate(cases) if s['roll_deg']==0 and s['pitch_deg']==0 and s['pose']=='stance')
    map_replay=dict(source_rig=best['id'],source_case=cases[nominal_index],
        fixture_scope='sector-by-target-index array, not a spatial map or actual sensor replay; synthetic flat heights and5mm sigma; actual geometric visibility mask',
        cases=replay_visibility_contract(sample[nominal_index].any(axis=0)))
    (args.out/'map_contract_replay.json').write_text(json.dumps(map_replay,indent=2)+'\n')
    output=dict(version=1,status='proposed_outboard_mount_geometric_search',profile=profile,
        candidates_nominal=results,finalists=finalists,finalist_cases=cases,
        search_dimensions=dict(outboard_m=[0,.04,.08,.12],height_above_plate_m=[.07,.10,.14],pitch_deg=[45,60,75,90]),
        source_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_report_sha256=hashlib.sha256((args.out/'report.json').read_bytes()).hexdigest(),elapsed_seconds=time.time()-start,
        physical_qualification=False,
        limits='48 uniform six-camera rigs screened at nominal stance, top3 then27attitude/lift cases; exhaustive2/4/6sector subsets only within those3rigs. Outboard positions may collide with moving hardware and require actual housing, bracket, stiffness and harness design. No final C model or real sensing. Optical-frame convention and exclusions from base report apply.')
    (args.out/'expanded_exact_mounts.json').write_text(json.dumps(output,indent=2)+'\n')
    write_exact_figure(args.out,report,output,points,sectors)
    print(json.dumps(dict(path=str(args.out/'expanded_exact_mounts.json'),elapsed=time.time()-start),indent=2))


def write_exact_figure(out,report,expanded,points,sectors):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    baseline_file=out/'D405_848x480_6cam_exact.npz'
    if not baseline_file.exists():
        return
    base=np.load(baseline_file)
    best=max(expanded['finalists'],key=lambda c:(c['rig_comparisons'][-1]['triangle_clearance']['worst_sector_mean_fraction'],c['rig_comparisons'][-1]['triangle_clearance']['mean_fraction']))
    new=np.load(out/(best['id']+'_expanded_exact.npz'))
    index=next(i for i,s in enumerate(expanded['finalist_cases']) if s['roll_deg']==0 and s['pitch_deg']==0 and s['pose']=='stance')
    fig,axes=plt.subplots(1,3,figsize=(15,6),gridspec_kw={'width_ratios':[1,1,1.1]})
    baseline_rig=next(r for r in report['rig_comparisons'] if r['id']=='D405_848x480_6cam')
    baseline_cameras=[report['candidates'][i] for i in baseline_rig['member_indices']]
    for ax,seen,inview,cameras,title in [
        (axes[0],base['clear'][index],base['optics_only'][index],baseline_cameras,'Initial six-view concept'),
        (axes[1],new['clear'][index].any(axis=0),new['optics_only'][index].any(axis=0),best['cameras'],'Expanded six-view proposal')]:
        for mask,color,label in [(~inview,'#c5cbd4','Outside stereo envelope'),(inview & ~seen,'#dba047','Blocked by CAD / unresolved'),(seen,'#119d83','Both rays clear of CAD')]:
            ax.scatter(points[mask,0],-points[mask,1],c=color,s=17,label=label)
        for camera in cameras:
            p=camera['body_position_m'];ax.scatter(p[0],-p[1],marker='s',s=48,c='#2868c6',edgecolor='white',linewidth=.7,zorder=4)
        for leg in LEGS:
            xy=points[sectors==leg,:2].mean(axis=0)
            ax.text(xy[0],-xy[1],leg.upper(),ha='center',va='center',weight='bold',fontsize=8,bbox=dict(fc='white',ec='none',alpha=.8,pad=1))
        ax.add_patch(Rectangle((-.1,-.2),.2,.4,fc='#34425a',alpha=.12))
        ax.set_title(f'{title}\n{seen.mean()*100:.0f}% of nominal target samples clear',fontsize=11)
        ax.set_aspect('equal');ax.set_xlim(-.45,.45);ax.set_ylim(-.50,.50);ax.grid(alpha=.15)
        ax.set_xlabel('Left (+X), m');ax.set_ylabel('Forward (−Y), m')
    axes[0].legend(loc='upper left',fontsize=7,framealpha=.94)
    ax=axes[2];rows=best['rig_comparisons'];x=np.arange(3)
    overall=[100*r['triangle_clearance']['mean_fraction'] for r in rows]
    worst=[100*r['triangle_clearance']['worst_sector_mean_fraction'] for r in rows]
    worstcase=[100*r['triangle_clearance']['worst_case_sector_fraction'] for r in rows]
    ax.bar(x-.24,overall,.24,label='All-sector mean',color='#119d83')
    ax.bar(x,worst,.24,label='Worst-sector mean',color='#2871b9')
    ax.bar(x+.24,worstcase,.24,label='Worst case + sector',color='#dfa344')
    for shift,values in [(-.24,overall),(0,worst),(.24,worstcase)]:
        for xx,value in zip(x,values):
            ax.text(xx+shift,value+2,f'{value:.0f}',ha='center',fontsize=8)
    ax.set_xticks(x,['2 cameras','4 cameras','6 cameras']);ax.set_ylim(0,100);ax.set_ylabel('Target samples with both CAD rays clear, %');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.set_title('Expanded layout: 27 attitude/lift cases',fontsize=11);ax.legend(fontsize=8,loc='upper left')
    fig.suptitle('Actual CAD ray checks reject the first mount layout; broader positions improve coverage',fontsize=14,weight='bold',y=.99)
    fig.text(.5,.035,f'Expanded proposal: {best["outboard_offset_m"]*1000:.0f} mm outboard of each hip, {best["height_above_plate_m"]*1000:.0f} mm above plate, {best["downward_pitch_deg"]}° down. Optical origins only; brackets and final C CAD unresolved.\nNominal intrinsics; no terrain/material/depth-quality validation. Targets are proposed ground samples, not proven reachable support. Unseen support stays unknown.',ha='center',fontsize=9,color='#354259')
    fig.tight_layout(rect=(0,.12,1,.94));fig.savefig(out/'exact_mount_comparison.png',dpi=160);plt.close(fig)


def poses(robot, stance):
    q0 = {j.get('name'): stance['coxa_yaw_rad' if 'coxa' in j.get('name') else
                               'femur_pitch_rad' if 'femur' in j.get('name') else 'tibia_pitch_rad']
          for j in robot.joints}
    result = [('stance', q0)]
    for label, tripod in [('tripod_a_lift', ('lf', 'lr', 'rm')), ('tripod_b_lift', ('rf', 'rr', 'lm'))]:
        q = dict(q0)
        for leg in tripod:
            q[leg+'_femur_pitch'] += .3
            q[leg+'_tibia_pitch'] += .2
        result.append((label, q))
    # Whole-group stop poses screen extreme obstruction; these are not physically
    # admissible gaits and do not claim complete swept-volume coverage.
    for kind in ('coxa', 'femur', 'tibia'):
        for end in ('lower', 'upper'):
            q = dict(q0)
            for j in robot.joints:
                if kind in j.get('name'):
                    q[j.get('name')] = float(j.find('limit').get(end))
            result.append((kind+'_'+end, q))
    return result


def target_points(feet):
    points, sectors, bands = [], [], []
    for leg, foot in zip(LEGS, feet):
        for band, extent, count, min_r, max_r in [('next_foot', .06, 5, .025, .085),
                                                ('lookahead', .14, 7, .08, .16)]:
            for dx, dy in itertools.product(np.linspace(-extent, extent, count), repeat=2):
                if min_r <= math.hypot(dx, dy) <= max_r:
                    points.append([foot[0]+dx, foot[1]+dy, 0.])
                    sectors.append(leg)
                    bands.append(band)
    return np.array(points), np.array(sectors), np.array(bands)


def candidates_and_profiles(manifest, overrides):
    profiles = [nominal_profile('D405', 848, 480, .07)]
    for model, near in [('D435', (.28, .195, .15)), ('D455', (.52, .35, .26))]:
        profiles += [nominal_profile(model, w, h, z) for (w, h), z in zip(((1280,720),(848,480),(640,360)), near)]
    for profile in profiles:
        if profile['id'] in overrides:
            profile.update(overrides[profile['id']])
            if not profile.get('calibrated'):
                raise ValueError('Supplied intrinsics must declare calibrated:true and their source')
    output = []
    for candidate in manifest['candidates']:
        for profile in profiles:
            if candidate['model'] == profile['model']:
                output.append(dict(candidate, id=candidate['id']+'_'+profile['id'].split('_', 1)[1],
                                   profile_id=profile['id']))
    # Available D455: a proposed front-edge optical centre, not the obsolete
    # prototype transform. It is deliberately not counted as a close rig camera.
    for z, pitch in itertools.product((.04,.07,.10), (15,22.5,30)):
        p = [0., -.23, z]
        for profile in profiles:
            if profile['model'] == 'D455':
                output.append(dict(id=f'nose_D455_z{int(z*1000)}_p{pitch}_{profile["width"]}x{profile["height"]}',
                    leg_sector='nose', model='D455', body_position_m=p,
                    optical_rotation_body=look_outward(p,pitch).tolist(), downward_pitch_deg=pitch,
                    profile_id=profile['id'], cad_fit_checked=False))
    return output, {p['id']: p for p in profiles}


def statistics(mask, sectors, bands):
    result = dict(mean_fraction=float(mask.mean()))
    result['per_sector_fraction'] = {leg: float(mask[..., sectors == leg].mean()) for leg in LEGS}
    result['worst_sector_mean_fraction'] = min(result['per_sector_fraction'].values())
    result['worst_case_sector_fraction'] = min(float(mask[:, sectors == leg].mean(axis=1).min()) for leg in LEGS)
    result['per_band_fraction'] = {band: float(mask[..., bands == band].mean()) for band in ('next_foot','lookahead')}
    return result


def select_rigs(candidates, profiles, optic, clear, sectors, bands, operating=None):
    if operating is None:
        operating=np.ones(len(clear),dtype=bool)
    rigs = []
    for profile in profiles:
        if profile.startswith('D455'):
            continue
        # Same mode/height/pitch on each populated sector makes this a bounded
        # reproducible family, not an assertion of a global combinatorial optimum.
        for count in (2,4,6):
            possibilities = []
            for z,pitch in itertools.product((.04,.07,.10),(35,45,60)):
                ids = {c['leg_sector']: i for i,c in enumerate(candidates)
                       if c['profile_id'] == profile and np.isclose(c['body_position_m'][2], z)
                       and c['downward_pitch_deg'] == pitch}
                for legs in itertools.combinations(LEGS,count):
                    indices = [ids[leg] for leg in legs]
                    visible = clear[:,indices].any(axis=1)
                    optical = optic[:,indices].any(axis=1)
                    stats = statistics(visible[operating],sectors,bands)
                    score = (stats['worst_case_sector_fraction'],stats['worst_sector_mean_fraction'],stats['mean_fraction'])
                    possibilities.append((score,indices,stats,statistics(optical[operating],sectors,bands),z,pitch))
            score,indices,stats,optical_stats,z,pitch = max(possibilities,key=lambda x:x[0])
            rigs.append(dict(id=f'{profile}_{count}cam', profile_id=profile, camera_count=count,
                members=[candidates[i]['id'] for i in indices], member_indices=indices,
                height_above_plate_m=z, downward_pitch_deg=pitch,
                sensor_body_mass_kg=profiles[profile]['mass_kg']*count,
                configurations_compared=len(possibilities), conservative_clearance=stats,
                optics_only_upper_bound=optical_stats,
                all_joint_stop_stress_clearance=statistics(clear[:,indices].any(axis=1),sectors,bands),
                physical_qualification=False))
    return rigs


def ground_footprint(candidate,profile,plate_height=.124):
    """Bounded 5 mm flat-plane raster, independent of body/leg occlusion."""
    x,y=np.meshgrid(np.arange(-.8,.8001,.005),np.arange(-1.2,1.2001,.005))
    points=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,-plate_height)))
    p=np.array(candidate['body_position_m']);r=np.array(candidate['optical_rotation_body'])
    optical=np.einsum('ni,ij->nj',points-p,r)
    if not np.isfinite(optical).all():
        raise ValueError('Nonfinite ground-footprint projection')
    visible=optical_visibility(optical,profile)
    seen=points[visible];z=optical[visible,2]
    if not len(seen):
        return dict(visible_grid_points=0)
    distances=np.linalg.norm(seen[:,:2]-p[:2],axis=1)
    return dict(plane='level ground,plate height0.124m',grid_resolution_m=.005,
        bounded_domain_body_xy_m=[[-.8,-1.2],[.8,1.2]],
        visible_grid_points=len(seen),approx_area_in_domain_m2=float(len(seen)*.005**2),
        body_xy_bounds_m=np.stack((seen[:,:2].min(axis=0),seen[:,:2].max(axis=0))).tolist(),
        horizontal_distance_from_optical_origin_min_m=float(distances.min()),
        horizontal_distance_from_optical_origin_max_m=float(distances.max()),
        optical_depth_min_m=float(z.min()),optical_depth_max_m=float(z.max()),
        domain_clips_footprint=bool(np.any(np.abs(seen[:,0])>.794)|np.any(np.abs(seen[:,1])>1.194)),
        actual_depth_validity_or_occlusion_verified=False)


def write_figure(out, report, candidates, points, sectors, bands, clear, optic, states):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    preferred = [r for r in report['rig_comparisons'] if r['profile_id']=='D405_848x480']
    nominal = next(i for i,s in enumerate(states) if s['height_m']==.124 and s['roll_deg']==0 and s['pitch_deg']==0 and s['pose']=='stance')
    fig, axes = plt.subplots(2,3,figsize=(15,9),gridspec_kw={'height_ratios':[1, .64]})
    for ax,rig in zip(axes[0],preferred):
        indices=rig['member_indices']
        geometric=optic[nominal,indices].any(axis=0)
        seen=clear[nominal,indices].any(axis=0)
        for mask,color,label in [(~geometric,'#c3c9d1','Outside optical envelope'),(geometric & ~seen,'#e4a23f','Possible CAD occlusion'),(seen,'#159c84','Clear of CAD bounds')]:
            ax.scatter(points[mask,0],-points[mask,1],c=color,s=14,label=label)
        for i in indices:
            c=candidates[i];p=np.asarray(c['body_position_m']);v=np.asarray(c['optical_rotation_body'])[:,2]
            ax.arrow(p[0],-p[1],v[0]*.07,-v[1]*.07,width=.003,color='#2c64c7',length_includes_head=True)
        for leg in LEGS:
            xy=points[sectors==leg,:2].mean(axis=0)
            ax.text(xy[0],-xy[1],leg.upper(),fontsize=9,ha='center',va='center',weight='bold',bbox=dict(fc='white',ec='none',alpha=.8,pad=1))
        ax.add_patch(Rectangle((-.10,-.20),.20,.40,fc='#425068',alpha=.14,ec='#425068'))
        ax.set_title(f'{rig["camera_count"]} D405 views | {rig["height_above_plate_m"]*1000:.0f} mm above plate | {rig["downward_pitch_deg"]}° down',fontsize=11)
        ax.set_aspect('equal');ax.set_xlabel('Body left (+X), m');ax.set_ylabel('Body forward (−Y), m')
        ax.grid(alpha=.15);ax.set_xlim(-.53,.53);ax.set_ylim(-.57,.57)
    axes[0,0].legend(loc='upper left',fontsize=7,framealpha=.9)
    matrix=np.array([[r['conservative_clearance']['per_sector_fraction'][leg] for leg in LEGS] for r in preferred])
    ax=axes[1,0];im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(6),[s.upper() for s in LEGS]);ax.set_yticks(range(3),['2 cameras','4 cameras','6 cameras'])
    for i,j in itertools.product(range(3),range(6)):
        ax.text(j,i,f'{matrix[i,j]*100:.0f}%',ha='center',va='center',fontsize=10,color='white' if matrix[i,j]>.6 else '#172338')
    ax.set_title('Per-sector clear fraction, stance/lift samples',fontsize=11)
    ax=axes[1,1]
    for pidx,profile in enumerate(('D405_848x480','D435_1280x720','D435_848x480','D435_640x360')):
        rows=[r for r in report['rig_comparisons'] if r['profile_id']==profile]
        ax.plot([r['camera_count'] for r in rows],[100*r['conservative_clearance']['worst_sector_mean_fraction'] for r in rows],'o-',label=profile.replace('_',' '))
    ax.set_ylim(0,100);ax.set_xticks([2,4,6]);ax.set_xlabel('Number of close cameras');ax.set_ylabel('Worst sector mean clear, %');ax.grid(alpha=.2);ax.legend(fontsize=7);ax.set_title('Resolution and minimum depth matter',fontsize=11)
    axes[1,2].axis('off')
    axes[1,2].text(0,1,'GEOMETRIC PRE-SCREEN\n\nProduction CAD, not finalized C hardware.\n27 attitudes/heights × 9 sampled leg poses.\nRigs ranked on stance + 2 lift poses only;\njoint-stop stress results in JSON report.\nNominal inferred optical intrinsics.\nAll 1,927 CAD visual bounds included.\n\nClear = both rays miss included bounds.\nOrange = box hit; exact occlusion unresolved.\nCables, payload and camera bodies absent.\nNo depth accuracy / outdoor qualification.\n\nTargets are proposed, not proven reachable.',va='top',fontsize=10,color='#26354d',linespacing=1.3)
    fig.suptitle('Close-ground sensor coverage: count, pose and near-depth constraints',fontsize=16,weight='bold')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'mount_screen.png',dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidates',type=Path,default=ROOT/'artifacts/terrain_readiness_2026-09-09/mount_candidates.json')
    ap.add_argument('--out',type=Path,default=ROOT/'artifacts/sensor_mount_study_2026-09-09')
    ap.add_argument('--intrinsics',type=Path,help='JSON object keyed by profile ID; selected calibrated rectified intrinsics')
    ap.add_argument('--exact-finalists',action='store_true',help='Refine two six-camera finalists from an existing report with optional trimesh+rtree')
    ap.add_argument('--expand-exact-mounts',action='store_true',help='Search48outboard/height/angle proposals, then refine top3 using actual triangles')
    args=ap.parse_args();start=time.time();args.out.mkdir(parents=True,exist_ok=True)
    if args.exact_finalists:
        exact_finalists(args)
        return
    if args.expand_exact_mounts:
        expand_exact_mounts(args)
        return
    manifest=json.loads(args.candidates.read_text());urdf=ROOT/manifest['source_urdf']
    if hashlib.sha256(urdf.read_bytes()).hexdigest()!=manifest['source_urdf_sha256']:
        raise ValueError('Candidate geometry provenance no longer matches source URDF')
    robot=RobotBounds(urdf)
    stance=json.loads((urdf.parent.parent/'stance.json').read_text())
    pose_list=poses(robot,stance);fk=[robot.forward(q) for _,q in pose_list]
    feet=robot.foot_centres(fk[0]);points,sectors,bands=target_points(feet)
    candidates,profiles=candidates_and_profiles(manifest,json.loads(args.intrinsics.read_text()) if args.intrinsics else {})
    groups={}
    for i,c in enumerate(candidates):
        key=(tuple(c['body_position_m'])+tuple(np.array(c['optical_rotation_body']).ravel())+
             (profiles[c['profile_id']]['baseline_m'],))
        groups.setdefault(key,[]).append(i)
    states=[];optics=[];clearance=[];roi80=[];body_block=[];optical_heights={c['id']:[] for c in candidates}
    for height,roll,pitch in itertools.product((.114,.124,.134),(-10,0,10),(-10,0,10)):
        body=transform((0,0,height),np.radians((roll,pitch,0)))
        targets_body=(points-body[:3,3])@body[:3,:3]
        base_opt=np.zeros((len(candidates),len(points)),dtype=bool);base_roi=base_opt.copy()
        for indices in groups.values():
            c=candidates[indices[0]];p=np.array(c['body_position_m']);rotation=np.array(c['optical_rotation_body'])
            optical=(targets_body-p)@rotation
            for i in indices:
                base_opt[i]=optical_visibility(optical,profiles[candidates[i]['profile_id']])
                base_roi[i]=optical_visibility(optical,profiles[candidates[i]['profile_id']],.8)
                optical_heights[candidates[i]['id']].append(float(apply_transform(p,body)[2]))
        for (name,q),link_transforms in zip(pose_list,fk):
            state_clear=np.zeros_like(base_opt);state_body=np.zeros_like(base_opt)
            for indices in groups.values():
                candidate=candidates[indices[0]]
                p=np.array(candidate['body_position_m'])
                p_right=p+profiles[candidate['profile_id']]['baseline_m']*np.array(candidate['optical_rotation_body'])[:,0]
                selected=np.flatnonzero(base_opt[indices].any(axis=0))
                ok,blocked_body=robot.clearance(p,targets_body[selected],link_transforms)
                right_ok,right_blocked_body=robot.clearance(p_right,targets_body[selected],link_transforms)
                ok &= right_ok
                blocked_body |= right_blocked_body
                for i in indices:
                    state_clear[i,selected]=base_opt[i,selected]&ok
                    state_body[i,selected]=base_opt[i,selected]&blocked_body
            states.append(dict(height_m=height,roll_deg=roll,pitch_deg=pitch,pose=name))
            optics.append(base_opt);clearance.append(state_clear);roi80.append(state_clear&base_roi);body_block.append(state_body)
        print(f'Completed height={height:.3f}, roll={roll:+d}, pitch={pitch:+d}; {len(states)} cases',flush=True)
    optic,clear,roi,body_hits=map(np.array,(optics,clearance,roi80,body_block))
    operating=np.array([s['pose'] in ('stance','tripod_a_lift','tripod_b_lift') for s in states])
    rigs=select_rigs(candidates,profiles,optic,clear,sectors,bands,operating)
    reports=[]
    for i,c in enumerate(candidates):
        heights=optical_heights[c['id']]
        reports.append(dict(c,optical_height_ground_min_m=min(heights),optical_height_ground_max_m=max(heights),
            optics_only_upper_bound=statistics(optic[:,i],sectors,bands),
            conservative_clearance=statistics(clear[:,i],sectors,bands),
            central_80pct_image_clearance=statistics(roi[:,i],sectors,bands),
            operating_pose_clearance=statistics(clear[operating,i],sectors,bands),
            nominal_ground_footprint=ground_footprint(c,profiles[c['profile_id']]),
            possible_body_occlusion_fraction=float(body_hits[:,i].mean())))
    report=dict(version=1,geometry_scope='Production serial CAD; not finalized C72.5/126mm model',
        status='cpu_geometry_prescreen_only',urdf_path=str(urdf.relative_to(ROOT)),
        urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),candidate_input_sha256=hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
        mesh_count=len(robot.meshes),visual_bounds_count=robot.visual_count,case_count=len(states),
        candidate_count=len(candidates),target_count=len(points),sectors=list(LEGS),
        optical_convention='+X right,+Y down,+Z depth; origins denote left-imager depth origin, not housing centre',
        body_convention='forward=-Y,left=+X,up=+Z; plate origin',
        source_specs=SOURCE_URL,profiles=profiles,states=states,pose_joint_values={name:q for name,q in pose_list},
        source_tool_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('sensor_mount_math.py'))},
        targets=dict(points_world_m=points.tolist(),sector=sectors.tolist(),band=bands.tolist(),
            explanation='Flat ground patches anchored to production neutral foot XY; 30–85mm nearby offsets and80–160mm lookahead samples; illustrative, not verified reach/stopping support'),
        production_stance_foot_centres_body_m=feet.tolist(),candidates=reports,rig_comparisons=rigs,
        occlusion_method='Each actual visual STL vertex bound is transformed into a conservative per-part AABB in its link frame; both left/right imager-to-target segments include all19 moving links. Box hits may be false positives; no exact triangle occlusion claimed.',
        selection_method='Enumerate every2/4/6sector subset within each shared height/pitch/mode; use81stance/lift cases to maximize worst-case sector, then worst-sector mean, then overall mean. Joint-stop poses reported separately because they are not dynamically admissible gaits. Mixed heights/angles/models not exhaustively searched.',
        excluded=['full continuous joint sweep','recorded learned gait poses','finalized C CAD','camera housings/brackets','payload/cables','terrain occlusion','real intrinsics/distortion unless overridden','stereo matching failures','optical dirt/sun/texture','timing/bandwidth','mass/inertia integration','visibility map memory'],
        minimum_measurements=['final C assembly and full joint/gait pose traces','left/right calibrated intrinsics at exact mode and depth setting','measured optical-to-housing and housing-to-body transforms','housing/bracket/harness swept clearances','real valid-depth/height-error masks outdoors','sensor-to-map delay and jitter under full USB load','usable support corridor and actual deceleration'],
        elapsed_seconds=time.time()-start)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.out/'profiles.json').write_text(json.dumps(profiles,indent=2)+'\n')
    np.savez_compressed(args.out/'visibility_masks.npz',optics_only=optic,cad_bounds_clear=clear,central80_clear=roi,
                        body_bounds_hits=body_hits,points=points,sectors=sectors,bands=bands)
    (args.out/'mesh_provenance.json').write_text(json.dumps({str(p.relative_to(ROOT)):{k:v for k,v in d.items() if k!='bounds'} for p,d in robot.meshes.items()},indent=2)+'\n')
    write_figure(args.out,report,candidates,points,sectors,bands,clear,optic,states)
    print(json.dumps(dict(output=str(args.out),seconds=time.time()-start,rigs=[dict(id=r['id'],clear=r['conservative_clearance']['mean_fraction'],worst_sector=r['conservative_clearance']['worst_sector_mean_fraction'],worst_case_sector=r['conservative_clearance']['worst_case_sector_fraction']) for r in rigs]),indent=2))


if __name__=='__main__':
    main()
