from pxr import Usd, UsdPhysics, UsdGeom, Gf, Sdf
import json
from pathlib import Path
import hashlib

p=Path('/workspace/hexapod/robot/hexapod_mkii_assy/usd/hexapod_mkii_serial/hexapod_mkii_serial.usda')
s=Usd.Stage.Open(str(p))
cache=UsdGeom.XformCache()
def val(v):
    if isinstance(v,(str,int,float,bool)) or v is None: return v
    if isinstance(v,Sdf.AssetPath): return {'asset':v.path,'resolved':v.resolvedPath}
    if isinstance(v,(Gf.Quatf,Gf.Quatd,Gf.Quath)): return [v.GetReal(),*v.GetImaginary()]
    try: return [val(x) for x in v]
    except TypeError: return str(v)
result={'meters_per_unit':UsdGeom.GetStageMetersPerUnit(s),'up_axis':UsdGeom.GetStageUpAxis(s),'default_prim':str(s.GetDefaultPrim().GetPath()),'bodies':[],'joints':[],'articulation_roots':[],'collisions':[],'unresolved_assets':[],'layers':[]}
for prim in s.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        result['bodies'].append({'name':prim.GetName(),'path':str(prim.GetPath()),'schemas':prim.GetAppliedSchemas(),'world_transform':val(cache.GetLocalToWorldTransform(prim)),'attributes':{a.GetName():val(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith(('physics:','physx'))}})
    if prim.IsA(UsdPhysics.Joint):
        result['joints'].append({'name':prim.GetName(),'type':prim.GetTypeName(),'attributes':{a.GetName():val(a.Get()) for a in prim.GetAttributes()},'relationships':{r.GetName():val(r.GetTargets()) for r in prim.GetRelationships()}})
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI): result['articulation_roots'].append(str(prim.GetPath()))
for prim in Usd.PrimRange(s.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        result['collisions'].append({'path':str(prim.GetPath()),'type':prim.GetTypeName(),'world_transform':val(cache.GetLocalToWorldTransform(prim)),'attributes':{a.GetName():val(a.Get()) for a in prim.GetAttributes() if a.GetName() in ['radius','height','size','axis','physics:collisionEnabled','extent']}})
    for a in prim.GetAttributes():
        if a.GetTypeName()==Sdf.ValueTypeNames.Asset:
            v=a.Get()
            if v and v.path and not v.resolvedPath: result['unresolved_assets'].append([str(prim.GetPath()),a.GetName(),v.path])
for layer in s.GetUsedLayers():
    path=Path(layer.realPath)
    result['layers'].append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None})
print(json.dumps(result,indent=2))
