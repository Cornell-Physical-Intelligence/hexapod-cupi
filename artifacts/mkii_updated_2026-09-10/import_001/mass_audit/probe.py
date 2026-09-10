import pickle,numpy as np,json
from pathlib import Path
from collections import defaultdict
class Stub:
 def __setstate__(self,s): self.__dict__.update(s)
allowed={('onshape_to_robot.robot',n):type(n,(Stub,),{}) for n in ['Robot','Link','Part']}
allowed[('onshape_to_robot.geometry','Mesh')]=type('Mesh',(Stub,),{})
allowed.update({('numpy._core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype})
class Reader(pickle.Unpickler):
 def find_class(self,m,n):
  if (m,n) not in allowed: raise ValueError((m,n))
  return allowed[m,n]
src=Path('/tmp/hexapod-urdf-intake-20260910')
with (src/'robot.pkl').open('rb') as f:r=Reader(f).load()
g=defaultdict(list)
for l in r.links:
 for p in l.parts:g[p.meshes[0].filename].append(p)
for f,ps in g.items():
 meta=json.loads((src/Path(f).with_suffix('.part')).read_text())
 print(len(ps),sum(p.mass for p in ps),meta['documentId'],f)
print('frames',[(l.name,vars(l).get('frames')) for l in r.links])
print('joint closures',r.joints,r.closures)
