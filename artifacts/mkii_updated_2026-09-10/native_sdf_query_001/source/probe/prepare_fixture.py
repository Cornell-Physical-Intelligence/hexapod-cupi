"""Prepare a bounded source-geometry oracle; never initializes Isaac or a GPU."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from mesh_oracle import oracle, topology, ray_distance
from score_probe import CELL_M


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def spread(centers, indices, count):
    indices=list(map(int,indices))
    if not indices:return []
    selected=[indices[0]]
    while len(selected)<min(count,len(indices)):
        distances=np.min(np.sum((centers[indices,None]-centers[selected][None])**2,axis=-1),axis=1)
        selected.append(indices[int(np.argmax(distances))])
    return selected


def prepare(asset, output):
    from pxr import Usd,UsdGeom,UsdPhysics
    asset=Path(asset).resolve();output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    # The nine-file map is copied from exact canonical inspector003, not guessed.
    bindings=json.loads((Path(__file__).parent/'ASSET_SHA256.json').read_text())
    for n,d in bindings.items():
        if sha(asset/n)!=d:raise ValueError('Changed canonical asset '+n)
    stage=Usd.Stage.Open(str(asset/'geometry.usdc'))
    g=UsdGeom.Mesh(stage.GetPrimAtPath('/Geometry/mesh_056'))
    v=np.asarray(g.GetPointsAttr().Get(),dtype=np.float32)
    if not np.all(np.asarray(g.GetFaceVertexCountsAttr().Get())==3):raise ValueError('Not triangle mesh')
    f=np.asarray(g.GetFaceVertexIndicesAttr().Get(),dtype=np.int32).reshape(-1,3)
    manifest=json.loads((asset/'geometry_manifest.json').read_text())
    item=next(x for x in manifest if x['mesh']=='mesh_056_tibia.stl')
    if hashlib.sha256(v.tobytes()).hexdigest()!=item['vertex_array_sha256']:raise ValueError('Vertex identity')
    if hashlib.sha256(f.tobytes()).hexdigest()!=item['triangle_index_sha256']:raise ValueError('Triangle identity')
    info=topology(v,f)
    if not info['closed'] or not info['consistently_oriented'] or info['signed_volume_m3']<=0:
        raise ValueError('Signed oracle requires a closed consistently oriented positive-volume mesh')
    tri=v.astype(float)[f];centers=tri.mean(1);norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm/=np.linalg.norm(norm,axis=1)[:,None]
    masks={'distal_face':(centers[:,0]>.1299)&(norm[:,0]>.999),
           'distal_edge':(centers[:,0]>.115)&(norm[:,0]>.15)&(norm[:,0]<.85),
           'shaft_surface':(centers[:,0]>.035)&(centers[:,0]<.075),
           'fork_surface':centers[:,0]<.02}
    points=[];regions=[];source_faces=[];outward=[]
    for label,mask in masks.items():
        for face in spread(centers,np.flatnonzero(mask),6):
            for offset in (-.0015,-.0006,.0006,.0015):
                points.append(centers[face]+offset*norm[face]);regions.append(label);source_faces.append(face);outward.append(norm[face])
    base_count=len(points)
    candidates=np.array([[x,y,z]for x in (-.015,0.,.015,.03,.06,.09,.12)for y in (-.02,-.01,0.,.01,.02)for z in (-.01,0.,.01)])
    candidate_oracle=oracle(candidates,v,f);void=[]
    for i,(p,r) in enumerate(zip(candidates,candidate_oracle)):
        if r['distance_m']<.0007 or abs(r['winding'])>.01:continue
        chords=[]
        for axis in np.eye(3):
            plus=ray_distance(p,axis,v,f);minus=ray_distance(p,-axis,v,f)
            if plus is not None and minus is not None and min(plus,minus)>.0005 and max(plus,minus)<.05:
                chords.append({'axis':axis.tolist(),'positive_hit_m':plus,'negative_hit_m':minus})
        if chords:void.append((i,chords))
    void_by_index=dict(void)
    chosen=spread(candidates,list(void_by_index),8)
    void_records=[]
    for i in chosen:
        void_records.append({'query_index':len(points),'chords':void_by_index[i]})
        points.append(candidates[i]);regions.append('void_between_surfaces');source_faces.append(-1);outward.append(np.zeros(3))
    results=oracle(np.asarray(points),v,f)
    # Calibration needs signed variation and varied gradients, away from the surface.
    anchor_candidates=[i for i,r in enumerate(results[:base_count]) if abs(r['distance_m'])>3*CELL_M
                       and np.dot(r['gradient'],outward[i])>.98]
    anchors=[]
    for label in masks:
        subset=[i for i in anchor_candidates if regions[i]==label]
        anchors.extend(spread(np.asarray(points),subset,4))
    if len(anchors)<12 or not any(results[i]['distance_m']<0 for i in anchors):raise ValueError('Insufficient signed semantic anchors')
    stencils=[];h=CELL_M
    # Bounded independent derivative checks on six well-behaved anchors.
    for i in anchors[:6]:
        query=np.array([points[i]+sign*h*axis for axis in np.eye(3)for sign in (1,-1)])
        rr=oracle(query,v,f);fd=np.array([(rr[j]['distance_m']-rr[j+1]['distance_m'])/(2*h)for j in (0,2,4)])
        if np.max(np.abs(fd-results[i]['gradient']))>.05:continue
        pairs=[]
        for j in (0,2,4):
            pairs.append([len(points),len(points)+1])
            for k in (j,j+1):points.append(query[k]);regions.append('derivative_stencil');source_faces.append(-1);results.append(rr[k])
        stencils.append({'center':i,'pairs':pairs,'h_m':h,'source_fd_max_error':float(np.max(np.abs(fd-results[i]['gradient'])))})
    if len(stencils)<2:raise ValueError('Insufficient smooth derivative fixtures')
    robot=Usd.Stage.Open(str(asset/'robot.usda'));shapes=[]
    for prim in robot.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetAttribute('hexapod:sourceMesh').Get()=='mesh_056_tibia.stl':
            path=str(prim.GetPath());link=path.split('/')[2];lp=robot.GetPrimAtPath('/Robot/'+link)
            W=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
            L=np.asarray(UsdGeom.Xformable(lp).ComputeLocalToWorldTransform(0)).T
            local=np.linalg.inv(L)@W
            if not np.allclose(local[:3,:3].T@local[:3,:3],np.eye(3),atol=2e-6):raise ValueError('Scaled/sheared shape needs separate contract')
            shapes.append({'path':path,'link':link,'shape_to_link':local.tolist()})
    if len(shapes)!=6:raise ValueError('Six exact canonical tibia shapes required')
    np.savez_compressed(output/'mesh.npz',vertices=v,faces=f)
    fixture={'schema':'canonical_tibia_sdf_fixture_v1','asset_files':bindings,'mesh_manifest':item,'topology':info,
             'mesh_npz_sha256':sha(output/'mesh.npz'),'nominal_cell_m':CELL_M,'shapes':shapes,
             'points':np.asarray(points).tolist(),'regions':regions,'source_triangles':source_faces,
             'distance_m':[r['distance_m']for r in results],'gradient':[r['gradient']for r in results],
             'winding':[r['winding']for r in results],'semantic_anchor_indices':anchors,
             'void_chords':void_records,'derivative_stencils':stencils,
             'scope':'Exact triangulated-source reference; not native SDF/contact admission'}
    (output/'fixture.json').write_text(json.dumps(fixture,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'points':len(points),'anchors':len(anchors),'stencils':len(stencils),'voids':len(void_records),'shapes':len(shapes),'topology':info},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--asset-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();prepare(a.asset_root,a.output)
