"""CPU geometry and observation checks for the proposed terrain stage.

No Isaac imports, robot mutations, GPU launches, or hardware commands.
"""
from __future__ import annotations

import math
import numpy as np


def body_to_navigation(vectors):
    """Export body (-Y forward,+X left) to forward/left/up coordinates."""
    a = np.asarray(vectors, dtype=float)
    if a.shape[-1:] != (3,) or not np.isfinite(a).all():
        raise ValueError('Expected finite vectors with final dimension 3')
    return np.stack((-a[..., 1], a[..., 0], a[..., 2]), axis=-1)


def optical_rotation_body(radial_xy, downward_pitch_deg):
    """Columns are optical +X right,+Y down,+Z forward, expressed in body."""
    r = np.asarray(radial_xy, dtype=float)
    if r.shape != (2,) or not np.isfinite(r).all() or np.linalg.norm(r) == 0:
        raise ValueError('Nonzero finite radial direction required')
    if not math.isfinite(downward_pitch_deg) or not 0 < downward_pitch_deg < 90:
        raise ValueError('Pitch must be between 0 and 90 degrees')
    r = r / np.linalg.norm(r)
    pitch = math.radians(downward_pitch_deg)
    forward = np.array([r[0] * math.cos(pitch), r[1] * math.cos(pitch), -math.sin(pitch)])
    right = np.array([r[1], -r[0], 0.])
    down = np.cross(forward, right)
    return np.column_stack((right, down, forward))


def terrain_channels(height_m, variance_m2, observed, capture_time_s, *, now_s,
                     max_age_s=.25, max_std_m=.015, height_scale_m=.20):
    """Encode HxW patch as height, usable-mask, normalized age, normalized sigma.

    Unknown/stale/uncertain/future data is never usable. Zero height is only
    numerical padding accompanied by a zero mask, not evidence of flat support.
    Geometric visibility does not establish material strength or traversability.
    """
    arrays = [np.asarray(a) for a in (height_m, variance_m2, observed, capture_time_s)]
    h, var, seen, captured = arrays
    if h.ndim != 2 or any(a.shape != h.shape for a in arrays):
        raise ValueError('All patch arrays must share HxW shape')
    if seen.dtype != np.bool_:
        raise ValueError('Observed mask must be boolean')
    if not all(math.isfinite(x) for x in (now_s, max_age_s, max_std_m, height_scale_m)):
        raise ValueError('Time/threshold values must be finite')
    if min(max_age_s, max_std_m, height_scale_m) <= 0:
        raise ValueError('Thresholds must be positive')
    h, var, captured = (np.asarray(a, dtype=float) for a in (h, var, captured))
    age = now_s - captured
    usable = (seen & np.isfinite(h) & np.isfinite(var) & (var >= 0)
              & (var <= max_std_m ** 2) & np.isfinite(age)
              & (age >= 0) & (age <= max_age_s))
    out = np.empty((*h.shape, 4), dtype=np.float32)
    out[..., 0] = np.where(usable, np.clip(h / height_scale_m, -1, 1), 0)
    out[..., 1] = usable
    out[..., 2] = np.where(usable, np.clip(age / max_age_s, 0, 1), 1)
    safe_var = np.where(np.isfinite(var) & (var >= 0), var, max_std_m ** 2)
    out[..., 3] = np.where(usable, np.clip(np.sqrt(safe_var) / max_std_m, 0, 1), 1)
    return out


def support_region_observed(channels, required_cells):
    """Fail closed if any required support cell is unavailable, including NaNs."""
    a, required = np.asarray(channels), np.asarray(required_cells)
    if a.ndim != 3 or a.shape[-1] != 4 or required.shape != a.shape[:2] or required.dtype != np.bool_:
        raise ValueError('Expected HxWx4 channels and HxW boolean region')
    if not required.any():
        return False
    return bool(np.isfinite(a[required]).all() and (a[..., 1][required] == 1).all())


def box_mesh(x0, x1, y0, y1, z0, z1):
    if not np.isfinite([x0, x1, y0, y1, z0, z1]).all() or not (x0 < x1 and y0 < y1 and z0 < z1):
        raise ValueError('Box intervals must have finite, positive extent')
    v = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                  [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]], dtype=float)
    f = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                  [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return v, f


def combine_meshes(meshes):
    vs, fs, offset = [], [], 0
    for v, f in meshes:
        vs.append(v); fs.append(f + offset); offset += len(v)
    return np.concatenate(vs), np.concatenate(fs)


def terrain_mesh(family, seed, *, difficulty=1.):
    """3x3 m fixtures in world X/Y, Z up. Start zone is X < -0.65 m.

    Sharp steps and pit walls are explicit boxes, not sloping height-field
    interpolation. Pits are avoidance fixtures, not initial walking curriculum.
    """
    if not 0 < difficulty <= 1:
        raise ValueError('difficulty must be in (0,1]')
    rng = np.random.default_rng(seed)
    meta = {'family': family, 'seed': int(seed), 'difficulty': difficulty,
            'size_m': [3., 3.], 'start_xy_m': [-1., 0.],
            'coordinate_frame': 'world XYZ, Z up; mesh X is course forward',
            'collision_approximation': 'none; preserve triangle surface',
            'material_parameters': 'not calibrated; assign explicitly in Isaac',
            'physical_validation': False}
    if family in ('smooth_rough', 'ramp'):
        xy = np.linspace(-1.5, 1.5, 76)
        x, y = np.meshgrid(xy, xy, indexing='ij')
        gate = np.clip((x + .65) / .4, 0, 1)
        if family == 'smooth_rough':
            phase = rng.uniform(-np.pi, np.pi, 3)
            raw = (np.sin(x * 9 + phase[0]) * np.cos(y * 7 + phase[1])
                   + .4 * np.sin(x * 15 + y * 10 + phase[2])) / 1.4
            z = .02 * difficulty * gate * raw
            meta.update(max_abs_height_m=.02*difficulty, spacing_m=.04)
        else:
            slope = math.tan(math.radians(float(rng.uniform(2, 5)*difficulty)))
            cross = float(rng.uniform(-.4,.4)) * slope
            z = slope * np.maximum(x + .65, 0) + cross * y * gate
            meta.update(slope_x=slope, slope_y=cross, spacing_m=.04)
        v = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        faces=[]
        n=len(xy)
        for i in range(n-1):
            for j in range(n-1):
                a=i*n+j; faces.extend(((a,a+n,a+1),(a+n,a+n+1,a+1)))
        return v, np.asarray(faces), meta
    if family == 'step':
        h=float(rng.uniform(.01,.02)*difficulty)
        edge=float(rng.uniform(-.15,.15))
        meta.update(step_height_m=h, edge_x_m=edge)
        v,f=combine_meshes([box_mesh(-1.5,edge,-1.5,1.5,-.25,0),
                            box_mesh(edge,1.5,-1.5,1.5,-.25,h)])
        return v,f,meta
    if family == 'ridge':
        h=float(rng.uniform(.01,.02)*difficulty); width=float(rng.uniform(.12,.22))
        meta.update(ridge_height_m=h, width_m=width)
        v,f=combine_meshes([box_mesh(-1.5,1.5,-1.5,1.5,-.25,0),
                            box_mesh(-width/2,width/2,-1.5,1.5,0,h)])
        return v,f,meta
    if family == 'pit':
        half=float(rng.uniform(.12,.22)); depth=.12
        meta.update(pit_half_width_m=half, pit_depth_m=depth,
                    curriculum='avoidance_only', required_behavior='stop or replan; no blind crossing')
        v,f=combine_meshes([box_mesh(-1.5,-half,-1.5,1.5,-.3,0),
                            box_mesh(half,1.5,-1.5,1.5,-.3,0),
                            box_mesh(-half,half,-1.5,-half,-.3,0),
                            box_mesh(-half,half,half,1.5,-.3,0),
                            box_mesh(-half,half,-half,half,-.3,-depth)])
        return v,f,meta
    raise ValueError(f'Unknown terrain family: {family}')


def validate_mesh(v, f):
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError('Expected triangle mesh')
    if not np.isfinite(v).all() or f.dtype.kind not in 'iu' or f.min() < 0 or f.max() >= len(v):
        raise ValueError('Non-finite vertices or invalid indices')
    tri = v[f]
    area2 = np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    if (area2 <= 1e-12).any():
        raise ValueError('Degenerate triangles')
    return {'vertices':len(v),'triangles':len(f),'bounds_m':[v.min(axis=0).tolist(),v.max(axis=0).tolist()]}


def mesh_usda(v, f):
    """Static collision-enabled mesh; runtime import remains an Isaac gate."""
    validate_mesh(v,f)
    points=',\n'.join('        ('+', '.join(f'{x:.7f}' for x in row)+')' for row in v)
    indices=', '.join(str(int(x)) for x in f.ravel())
    counts=', '.join('3' for _ in f)
    return ('#usda 1.0\n(\n    defaultPrim = "Terrain"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n'
            'def Mesh "Terrain" (\n    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]\n)\n{\n'
            '    bool physics:collisionEnabled = true\n    uniform token physics:approximation = "none"\n'
            '    uniform token subdivisionScheme = "none"\n    bool doubleSided = true\n'
            '    point3f[] points = [\n'+points+'\n    ]\n'
            '    int[] faceVertexCounts = ['+counts+']\n'
            '    int[] faceVertexIndices = ['+indices+']\n}\n')
