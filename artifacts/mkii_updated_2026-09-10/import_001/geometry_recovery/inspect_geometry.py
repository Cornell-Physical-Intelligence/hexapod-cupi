import sys,json,collections
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
sys.path.insert(0,'/Users/andreboufama/Documents/CUPI/HEXAPOD/robot/tools')
from import_onshape_hexapod import MeshCache,make_T
src=Path('/tmp/hexapod-urdf-intake-20260910')
root=ET.parse(src/'robot.urdf').getroot(); instances=[];cache=MeshCache([src/'assets'])
for idx,vis in enumerate(root.findall('link/visual')):
 f=vis.find('geometry/mesh').get('filename').split('/')[-1];o=vis.find('origin');p=np.fromstring(o.get('xyz'),sep=' ');r=np.fromstring(o.get('rpy'),sep=' ');T=make_T(p,r); m=cache.get(f)
 instances.append(dict(index=idx,file=f,T=T,center=T[:3,:3]@m['cen']+p))
for f,n in collections.Counter(i['file'] for i in instances).items():
 m=cache.get(f);v=m['verts'];print(f,n,'bbox',np.round(np.stack([v.min(0),v.max(0)])*1000,3).tolist(),'cen',np.round(m['cen']*1000,3).tolist(),'volume_cc',round(m['vol']*1e6,4))
for i in instances:
 if i['file'] in ['femur_first_stage.stl','tibia_attatchment_plate.stl','tibia.stl','motor_flange.stl','motor_bearing_holder.stl','bearing_insert.stl'] or i['file'].startswith(('revolve','mirror','cirpattern')):
  print('inst',i['index'],i['file'],'center',np.round(i['center']*1000,3).tolist(),'origin',np.round(i['T'][:3,3]*1000,3).tolist(),'axes',np.round(i['T'][:3,:3],5).tolist())
