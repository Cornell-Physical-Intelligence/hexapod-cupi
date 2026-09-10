"""Small NumPy triangle oracle; no simulator or proprietary implementation."""
import numpy as np


def topology(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError('Expected triangle geometry')
    if not np.isfinite(v).all() or f.min() < 0 or f.max() >= len(v):
        raise ValueError('Invalid geometry')
    tri = v[f]
    normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    if (np.linalg.norm(normal, axis=1) < 1e-15).any():
        raise ValueError('Degenerate triangle')
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    ordered = np.sort(edges, axis=1)
    _, inverse, counts = np.unique(ordered, axis=0, return_inverse=True, return_counts=True)
    signs = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    winding = np.bincount(inverse, weights=signs)
    volume = np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()/6
    return {'closed': bool(np.all(counts == 2)), 'consistently_oriented': bool(np.all(winding == 0)),
            'signed_volume_m3': float(volume), 'vertices': len(v), 'triangles': len(f)}


def oracle(points, vertices, faces):
    """Exact triangle proximity plus oriented solid-angle winding, outside positive.

    Gradients use nearest-surface displacement, not an assumed CAD face normal.
    On the surface the distance is zero and the gradient is explicitly invalid.
    """
    p = np.asarray(points, dtype=np.float64)
    v = np.asarray(vertices, dtype=np.float64); f = np.asarray(faces, dtype=np.int64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('Invalid query points')
    a, b, c = np.moveaxis(v[f], 1, 0)
    ab, ac = b-a, c-a
    aa = np.einsum('ij,ij->i', ab, ab); bb = np.einsum('ij,ij->i', ab, ac)
    cc = np.einsum('ij,ij->i', ac, ac); den = aa*cc-bb*bb
    if (den <= 1e-30).any(): raise ValueError('Degenerate triangle')
    rows = []
    for x in p:
        ap = x-a
        d = np.einsum('ij,ij->i', ap, ab); e = np.einsum('ij,ij->i', ap, ac)
        u = (cc*d-bb*e)/den; w = (aa*e-bb*d)/den
        projected = a+u[:, None]*ab+w[:, None]*ac
        inside = (u >= 0) & (w >= 0) & (u+w <= 1)
        candidates = [projected]
        distances = [np.where(inside, ((projected-x)**2).sum(1), np.inf)]
        for start, end in ((a,b),(b,c),(c,a)):
            edge=end-start
            t=np.clip(np.einsum('ij,ij->i', x-start, edge)/(edge*edge).sum(1),0,1)
            closest=start+t[:,None]*edge
            candidates.append(closest);distances.append(((closest-x)**2).sum(1))
        distances=np.asarray(distances);which,face=np.unravel_index(np.argmin(distances),distances.shape)
        closest=candidates[which][face];unsigned=float(np.sqrt(distances[which,face]))
        va,vb,vc=a-x,b-x,c-x
        na,nb,nc=[np.linalg.norm(z,axis=1)for z in (va,vb,vc)]
        numerator=np.einsum('ij,ij->i',va,np.cross(vb,vc))
        denominator=na*nb*nc+np.einsum('ij,ij->i',va,vb)*nc+np.einsum('ij,ij->i',vb,vc)*na+np.einsum('ij,ij->i',vc,va)*nb
        winding=float(np.arctan2(numerator,denominator).sum()/(2*np.pi))
        signed=-unsigned if abs(winding)>.5 else unsigned
        gradient=(x-closest)/signed if unsigned>1e-10 else np.zeros(3)
        rows.append({'distance_m':signed,'gradient':gradient.tolist(),'gradient_valid':unsigned>1e-10,
                     'winding':winding,'triangle':int(face),'closest_point':closest.tolist()})
    return rows


def ray_distance(point, direction, vertices, faces):
    """Nearest positive triangle intersection, used only to label void chords."""
    tri=np.asarray(vertices)[np.asarray(faces)];a,b,c=np.moveaxis(tri,1,0);ab=b-a;ac=c-a
    d=np.asarray(direction);h=np.cross(np.broadcast_to(d,ac.shape),ac);det=np.einsum('ij,ij->i',ab,h)
    inv=np.divide(1.,det,out=np.zeros_like(det),where=np.abs(det)>1e-14)
    s=np.asarray(point)-a;u=inv*np.einsum('ij,ij->i',s,h);q=np.cross(s,ab)
    w=inv*(q@d);t=inv*np.einsum('ij,ij->i',ac,q)
    valid=(np.abs(det)>1e-14)&(u>=-1e-10)&(w>=-1e-10)&(u+w<=1+1e-10)&(t>1e-7)
    return float(t[valid].min()) if valid.any() else None


def frame_prediction(points, vertices, faces, shape_to_frame):
    """Counterfactual: both query coordinates and gradient are in another frame."""
    t=np.asarray(shape_to_frame,dtype=np.float64)
    if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1],atol=1e-10):
        raise ValueError('Invalid frame transform')
    if not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=2e-6,rtol=0) or np.linalg.det(t[:3,:3])<.999:
        raise ValueError('Frame transform must be rigid, with no scale or reflection')
    local=(np.linalg.inv(t)@np.c_[points,np.ones(len(points))].T).T[:,:3]
    rows=oracle(local,vertices,faces)
    gradient=np.asarray([r['gradient']for r in rows])@t[:3,:3].T
    return np.c_[[r['distance_m']for r in rows],gradient]
