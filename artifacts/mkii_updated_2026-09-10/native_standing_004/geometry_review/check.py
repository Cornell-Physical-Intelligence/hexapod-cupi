from pathlib import Path
import sys,json,math,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];source=ROOT/'tmp/updated_native_standing_002';raw=ROOT/'tmp/canonical_native_standing_terminal_001/run/standing';sys.path.insert(0,str(source))
from standing_math import Geometry
from inspect_core import rotation
s=json.loads((raw/'session.json').read_text());names=s['body_names'];p=json.loads((raw/'failed_partial_step.json').read_text());initial=json.loads((raw/'initial_reset.json').read_text())['post_reset']['link_pose_xyzw']
with np.load(source/'geometry/geometry_extrema.npz')as z:clouds={k:z[k]for k in z.files}
geo=Geometry(json.loads((source/'geometry/geometry.json').read_text()),clouds,names)
with np.load(raw/'substeps_000.npz')as z:poses=list(z['link_pose_xyzw'])
poses=[np.asarray(initial)]+poses+[np.asarray(p['link_pose_xyzw'])]
max_element=0.;max_scalar=0.;samples=0;finite=True
for pose in poses:
 actual=geo.clearance(pose);scalar=[math.inf,math.inf]
 for j,name in enumerate(names):
  q=pose[0,j];z=rotation(q[3:])[2]
  for kind,key in enumerate(['all__','non_toe__']):
   cloud=clouds[key+name];mat=cloud@z+q[2];element=(cloud*z).sum(-1)+q[2];py=np.array([math.fsum([float(row[0])*float(z[0]),float(row[1])*float(z[1]),float(row[2])*float(z[2])])+float(q[2])for row in cloud])
   finite=finite and np.isfinite(mat).all()and np.isfinite(element).all()and np.isfinite(py).all();max_element=max(max_element,float(abs(mat-element).max()));max_scalar=max(max_scalar,float(abs(mat-py).max()));samples+=len(cloud);scalar[kind]=min(scalar[kind],float(py.min()))
 for kind in [0,1]:assert abs(actual[kind][0]-scalar[kind])<1e-14
assert finite and max_element<1e-14 and max_scalar<1e-14
print(json.dumps({'scope':'Local actual15-pose all19-body clearance arithmetic check only; no native or future standing result','poses':len(poses),'plane_dot_products':samples,'all_finite':bool(finite),'max_matrix_vs_elementwise_m':max_element,'max_matrix_vs_python_fsum_m':max_scalar,'all_full_geometry_minima_match_atol_m':1e-14,'warnings_suppressed':False,'source_freeze_sha256':hashlib.sha256((source/'FREEZE_SHA256.json').read_bytes()).hexdigest()},indent=2))
