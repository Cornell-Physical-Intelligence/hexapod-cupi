"""Read-only URDF structural, inertial, mesh and kinematic audit."""
from pathlib import Path
import ast
import hashlib
import json
import struct
import xml.etree.ElementTree as ET
from collections import Counter
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / 'robot/hexapod_mkii_assy'
OUT = Path(__file__).parent
LEGS = 'lf lm lr rf rm rr'.split()
errors = []

def check(value, message):
    if not value:
        errors.append(message)

def vec(s):
    return np.array([float(x) for x in s.split()])

def origin(element):
    e = element.find('origin')
    t = np.eye(4)
    if e is not None:
        t[:3, 3] = vec(e.get('xyz', '0 0 0'))
        t[:3, :3] = Rotation.from_euler('xyz', vec(e.get('rpy', '0 0 0'))).as_matrix()
    return t

def inertia(link):
    e = link.find('inertial')
    a = {k: float(v) for k, v in e.find('inertia').attrib.items()}
    I = np.array([[a['ixx'], a['ixy'], a['ixz']], [a['ixy'], a['iyy'], a['iyz']], [a['ixz'], a['iyz'], a['izz']]])
    return float(e.find('mass').get('value')), origin(e), I

def stl(path):
    raw = path.read_bytes()
    n = struct.unpack_from('<I', raw, 80)[0] if len(raw) >= 84 else 0
    if len(raw) == 84 + n * 50:
        a = np.frombuffer(raw, dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')]), offset=84)
        check(np.isfinite(a['normal']).all(), f'{path.name}: non-finite normals')
        v = a['vertices'].astype(float)
    else:
        v = np.array([vec(line.strip()[7:]) for line in raw.decode().splitlines() if line.strip().startswith('vertex ')]).reshape(-1,3,3)
    check(len(v)>0 and np.isfinite(v).all(), f'{path.name}: empty/non-finite STL')
    return v

limits = json.loads((PKG/'joint_limits.json').read_text())
stance = json.loads((PKG/'stance.json').read_text())
report = json.loads((PKG/'assembly_report.json').read_text())
u = ET.parse(PKG/'urdf/hexapod_mkii_serial.urdf').getroot()
links = {e.get('name'):e for e in u.findall('link')}
joints = {e.get('name'):e for e in u.findall('joint')}
check(len(links)==len(u.findall('link'))==19,'duplicate/wrong link count')
check(len(joints)==len(u.findall('joint'))==18,'duplicate/wrong joint count')
check(set(links)=={'body'}|{f'{l}_{s}' for l in LEGS for s in ['coxa','femur','tibia']},'unexpected links')
children = []
for name,j in joints.items():
    parent,child = j.find('parent').get('link'), j.find('child').get('link')
    children.append(child)
    check(parent in links and child in links and parent!=child, f'{name}: invalid parent/child')
    leg,kind=name.split('_',1)
    expected={'coxa_yaw':('body',f'{leg}_coxa'),'femur_pitch':(f'{leg}_coxa',f'{leg}_femur'),'tibia_pitch':(f'{leg}_femur',f'{leg}_tibia')}
    check((parent,child)==expected[kind],f'{name}: wrong chain')
    check(j.get('type')=='revolute' and j.find('mimic') is None, f'{name}: not independent revolute')
    check(abs(np.linalg.norm(vec(j.find('axis').get('xyz')))-1)<1e-8,f'{name}: non-unit axis')
    a={k:float(v) for k,v in j.find('limit').attrib.items()}
    check(a['lower']<a['upper'] and a['effort']>0 and a['velocity']>0,f'{name}: invalid limits')
    check(all(abs(a[k]-limits[kind][k])<1e-8 for k in ['lower','upper']),f'{name}: JSON limits mismatch')
    q=stance[kind+'_rad']
    check(a['lower']<=q<=a['upper'],f'{name}: reset outside limits')
    for v in j.find('dynamics').attrib.values(): check(float(v)>=0,f'{name}: negative dynamics')
check(len(set(children))==18 and set(links)-set(children)=={'body'},'invalid rooted tree')

for e in u.iter():
    for k in ['xyz','rpy','rgba','size','radius','length','scale','ixx','ixy','ixz','iyy','iyz','izz','lower','upper','effort','velocity','damping','friction','value']:
        if k in e.attrib: check(np.isfinite(vec(e.attrib[k])).all(), f'{e.tag}: non-finite {k}')
materials={m.get('name') for m in u.findall('material')}
for m in u.findall('.//visual/material'): check(m.get('name') in materials, 'unresolved material')

mesh_cache={}
mass_data={}
shape_counts=Counter()
for name,l in links.items():
    check(len(l.findall('inertial'))==1 and len(l.findall('collision'))>0 and len(l.findall('visual'))>0,f'{name}: missing inertial/collision/visual')
    m,T,I=inertia(l)
    eig=np.linalg.eigvalsh(I)
    check(m>0 and eig[0]>0 and eig[-1]<=sum(eig[:2])+1e-10, f'{name}: nonphysical mass/inertia')
    expected=report['link_masses_kg'][name]
    if name.endswith('_femur'): expected+=sum(report['link_masses_kg'][name.split('_')[0]+'_'+s] for s in ['tibia_push_lever','tibia_pushrod'])
    check(abs(m-expected)<1e-7,f'{name}: mass mismatch with report')
    mass_data[name]={'mass_kg':m,'principal_moments_kg_m2':eig.tolist()}
    for visual in l.findall('visual'):
        mesh=visual.find('geometry/mesh')
        uri=mesh.get('filename')
        check(uri.startswith('package://hexapod_mkii_assy/'),f'{name}: unexpected mesh URI {uri}')
        path=PKG/uri.removeprefix('package://hexapod_mkii_assy/')
        check(path.exists() and path.name.isascii(),f'{name}: missing/non-ASCII mesh {path}')
        if path not in mesh_cache: mesh_cache[path]=stl(path)
        check(np.all(vec(mesh.get('scale','1 1 1'))>0),f'{name}: invalid mesh scale')
    for c in l.findall('collision'):
        geom=list(c.find('geometry'))
        check(len(geom)==1,f'{name}: malformed collision')
        g=geom[0]
        shape_counts[g.tag]+=1
        check(g.tag in ['box','cylinder','sphere'],f'{name}: unexpected collider {g.tag}')
        for value in g.attrib.values(): check(np.all(vec(value)>0),f'{name}: invalid collider dimensions')
    if name.endswith('_tibia'): check(len(l.findall('collision/geometry/sphere'))==2,f'{name}: missing foot spheres')

def fk(q):
    ts={'body':np.eye(4)}
    axes={}
    remaining=dict(joints)
    while remaining:
        old=len(remaining)
        for name,j in list(remaining.items()):
            parent=j.find('parent').get('link')
            if parent not in ts: continue
            T=ts[parent]@origin(j)
            axis=vec(j.find('axis').get('xyz'))
            axes[name]=(T[:3,3].copy(),T[:3,:3]@axis)
            R=np.eye(4); R[:3,:3]=Rotation.from_rotvec(axis*q.get(name,0)).as_matrix()
            ts[j.find('child').get('link')]=T@R
            del remaining[name]
        if len(remaining)==old: raise ValueError('cycle/unreachable links')
    return ts,axes

def lowest(T,g):
    if g.tag=='sphere': extent=float(g.get('radius'))
    elif g.tag=='box': extent=np.dot(np.abs(T[2,:3]),vec(g.get('size'))/2)
    else:
        az=T[2,2]
        extent=abs(az)*float(g.get('length'))/2+float(g.get('radius'))*np.sqrt(max(0,1-az*az))
    return float(T[2,3]-extent)

def pose(femur,tibia,height):
    q={f'{leg}_{s}':v for leg in LEGS for s,v in [('coxa_yaw',0),('femur_pitch',femur),('tibia_pitch',tibia)]}
    ts,axes=fk(q)
    foot={}; other=[]; coms={}; visual_bounds=[]
    for name,l in links.items():
        m,T,I=inertia(l); coms[name]=(m,(ts[name]@T)[:3,3])
        for c in l.findall('collision'):
            g=list(c.find('geometry'))[0]; T=ts[name]@origin(c)
            z=lowest(T,g)
            if name.endswith('_tibia') and g.tag=='sphere':
                leg=name.split('_')[0]
                if leg not in foot or z<foot[leg][0]: foot[leg]=(z,T[:3,3])
            else: other.append((z,name,g.tag))
        for v in l.findall('visual'):
            mesh=v.find('geometry/mesh'); path=PKG/mesh.get('filename').removeprefix('package://hexapod_mkii_assy/')
            # Exact bounds of every visual triangle vertex after the URDF transforms.
            vertices=mesh_cache[path].reshape(-1,3)*vec(mesh.get('scale','1 1 1'))
            T=ts[name]@origin(v); p=np.einsum('ij,kj->ki',T[:3,:3],vertices)+T[:3,3]
            check(np.isfinite(p).all(),f'{name}: non-finite transformed mesh')
            visual_bounds.append((p.min(0),p.max(0)))
    mass=sum(m for m,c in coms.values()); com=sum(m*c for m,c in coms.values())/mass
    supports=np.array([foot[l][1][:2] for l in LEGS]); hull=ConvexHull(supports)
    support_margin=float(np.min(-(hull.equations[:,:2]@com[:2]+hull.equations[:,2])))
    torque={}
    for leg in LEGS:
        torque[leg]={}
        for kind,parts in [('coxa_yaw',['coxa','femur','tibia']),('femur_pitch',['femur','tibia']),('tibia_pitch',['tibia'])]:
            pos,axis=axes[f'{leg}_{kind}']
            moment=np.cross(foot[leg][1]-pos,[0,0,mass*9.81/6])
            for part in parts:
                m,c=coms[f'{leg}_{part}']; moment+=np.cross(c-pos,[0,0,-m*9.81])
            torque[leg][kind]=float(-axis@moment)
    return {'femur_rad':femur,'tibia_rad':tibia,'root_height_m':height,'foot_bottom_z_at_root_zero_m':{l:float(foot[l][0]) for l in LEGS},'foot_center_body_m':{l:foot[l][1].tolist() for l in LEGS},'foot_spread_z_mm':float(np.ptp([x[0] for x in foot.values()])*1000),'ground_clearance_nonfoot_at_reset_m':min(x[0] for x in other)+height,'lowest_nonfoot':sorted(other)[:6],'visual_bounds_body_m':[np.min([b[0] for b in visual_bounds],axis=0).tolist(),np.max([b[1] for b in visual_bounds],axis=0).tolist()],'com_body_m':com.tolist(),'support_margin_m':support_margin,'static_equal_vertical_load_torques_nm':torque}

zero,axes=fk({})
mounts={}
for leg in LEGS:
    point,axis=axes[f'{leg}_coxa_yaw']
    hip=zero[f'{leg}_femur'][:3,3]
    heading=np.arctan2(*(hip-point)[:2][::-1]); radial=np.arctan2(point[1],point[0])
    error=np.rad2deg(np.arctan2(np.sin(heading-radial),np.cos(heading-radial)))
    mounts[leg]={'origin_m':point.tolist(),'positive_axis_body':axis.tolist(),'radial_heading_error_deg':float(error)}
    check(axis[2]>0.999 and abs(error)<0.1,f'{leg}: wrong yaw direction/radial zero')

cfg_ast=ast.parse((ROOT/'isaaclab/hexapod_rl/asset_cfg.py').read_text())
cfg={}
for node in cfg_ast.body:
    if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
        try: cfg[node.targets[0].id]=ast.literal_eval(node.value)
        except (ValueError,TypeError): pass
for key,jkey in [('COXA_LIMITS','coxa_yaw'),('FEMUR_LIMITS','femur_pitch'),('TIBIA_LIMITS','tibia_pitch')]:
    check(tuple(cfg[key])==tuple(limits[jkey][x] for x in ['lower','upper']),f'asset_cfg {key} mismatch')
for key,skey in [('STANCE_ROOT_HEIGHT_M','root_height_m'),('STANCE_RESET_ROOT_HEIGHT_M','reset_root_height_m'),('STANCE_FEMUR_RAD','femur_pitch_rad'),('STANCE_TIBIA_RAD','tibia_pitch_rad')]:
    check(cfg[key]==stance[skey],f'asset_cfg {key} mismatch')

result={'urdf_sha256':hashlib.sha256((PKG/'urdf/hexapod_mkii_serial.urdf').read_bytes()).hexdigest(),'errors':errors,'link_count':len(links),'joint_count':len(joints),'unique_meshes':len(mesh_cache),'mesh_unique_triangles':sum(len(v) for v in mesh_cache.values()),'visual_instances':len(u.findall('.//visual')),'collision_primitives':dict(shape_counts),'mass_total_kg':sum(d['mass_kg'] for d in mass_data.values()),'inertias':mass_data,'mounts':mounts,'local_reset':pose(stance['femur_pitch_rad'],stance['tibia_pitch_rad'],stance['reset_root_height_m']),'spark_reset':pose(0.5,-0.8,0.084)}
(OUT/'static_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['inertias','mounts']},indent=2))
raise SystemExit(bool(errors))
