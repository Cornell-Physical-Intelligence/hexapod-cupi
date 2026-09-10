from pathlib import Path
import json,numpy as np
from pxr import Usd,UsdGeom
from scipy.spatial import ConvexHull
root=Path(__file__).resolve().parents[2]
meta=json.loads((root/'tmp/updated_native_standing_001/geometry/geometry.json').read_text())
stage=Usd.Stage.Open(str(root/'artifacts/mkii_updated_2026-09-10/usd_002/rs05_mass_corrected/robot.usda'))
s=meta['shapes'][0];p=stage.GetPrimAtPath(s['path']);m=UsdGeom.Mesh(p)
v=np.array(m.GetPointsAttr().Get(),float);faces=np.array(m.GetFaceVertexIndicesAttr().Get()).reshape(-1,3);assert set(m.GetFaceVertexCountsAttr().Get())=={3}
cut=s['cap_lower_x_m'];edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
a,b=v[edges[:,0]],v[edges[:,1]];cross=(a[:,0]<cut)&(b[:,0]>cut)|(a[:,0]>cut)&(b[:,0]<cut);a=a[cross];b=b[cross]
points=a+(b-a)*((cut-a[:,0])/(b[:,0]-a[:,0]))[:,None]
old=v[v[:,0]<cut];new=np.concatenate([old,points])
rng=np.random.default_rng(20260910);directions=np.concatenate([np.eye(3),-np.eye(3),rng.normal(size=(5000,3))]);directions/=np.linalg.norm(directions,axis=1,keepdims=True)
old=old[ConvexHull(old).vertices];new=new[ConvexHull(new).vertices]
delta=np.min(old@directions.T,axis=0)-np.min(new@directions.T,axis=0);i=int(delta.argmax())
print(json.dumps({'scope':'CPU source-shape geometric counterexample, no native physics or claim of realized pose','source_mesh':s['path'],'crossing_edges':len(points),'max_overstatement_sampled_m':float(delta[i]),'direction_source':directions[i].tolist(),'old_min_m':float((old@directions[i]).min()),'clipped_min_m':float((new@directions[i]).min()),'directions':len(directions)},indent=2))
