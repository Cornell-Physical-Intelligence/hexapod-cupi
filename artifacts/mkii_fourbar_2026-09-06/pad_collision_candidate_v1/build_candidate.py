"""Generate an unqualified convex pad envelope, without editing runtime assets.

Dependencies are isolated from the repository environment; see requirements.txt.
All mesh coordinates remain in the exact original STL frame and meter units.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np
import scipy
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MESH = ROOT / 'robot/hexapod_mkii_assy/meshes/silicone_foot.stl'
URDF = ROOT / 'robot/hexapod_mkii_assy/urdf/hexapod_mkii_linkage.urdf'
KIN = ROOT / 'configs/mkii_fourbar_v3_kinematics.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_stl(path):
    data = path.read_bytes()
    count = struct.unpack_from('<I', data, 80)[0]
    if len(data) != 84 + 50 * count:
        raise ValueError('Expected exact binary STL length')
    dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attr', '<u2')])
    return np.frombuffer(data, dtype=dtype, count=count, offset=84)['vertices'].astype(np.float64)


def write_stl(path, hull):
    triangles = []
    for indices, plane in zip(hull.simplices, hull.equations):
        tri = hull.points[indices].copy()
        if np.dot(np.cross(tri[1]-tri[0], tri[2]-tri[0]), plane[:3]) < 0:
            tri[[1, 2]] = tri[[2, 1]]
        triangles.append(tri)
    with path.open('wb') as f:
        f.write(b'Hexapod pad convex candidate v1; original STL frame; meters'.ljust(80, b' '))
        f.write(struct.pack('<I', len(triangles)))
        for tri in triangles:
            normal = np.cross(tri[1]-tri[0], tri[2]-tri[0])
            normal /= np.linalg.norm(normal)
            f.write(struct.pack('<12fH', *normal, *tri.reshape(-1), 0))


def closest_on_triangles(points, triangles):
    """Euclidean distance to an arbitrary triangle surface, by every face."""
    a, b, c = np.moveaxis(triangles, 1, 0)
    ab, ac = b-a, c-a
    normals = np.cross(ab, ac)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    pa = points[:, None, :] - a[None, :, :]
    height = np.einsum('nfi,fi->nf', pa, normals)
    projection = points[:, None, :] - height[:, :, None] * normals[None, :, :]
    ap = projection - a[None, :, :]
    d00 = np.einsum('fi,fi->f', ab, ab)
    d01 = np.einsum('fi,fi->f', ab, ac)
    d11 = np.einsum('fi,fi->f', ac, ac)
    d20 = np.einsum('nfi,fi->nf', ap, ab)
    d21 = np.einsum('nfi,fi->nf', ap, ac)
    denom = d00*d11-d01*d01
    if np.any(denom <= 0):
        raise ValueError('Degenerate hull triangle')
    v = (d11*d20-d01*d21)/denom
    w = (d00*d21-d01*d20)/denom
    in_face = (v >= -1e-10) & (w >= -1e-10) & (v+w <= 1+1e-10)
    nearest = projection.copy()
    distance_sq = np.where(in_face, height**2, np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end-start
        length_sq = np.einsum('fi,fi->f', edge, edge)
        alpha = np.clip(np.einsum('nfi,fi->nf', points[:, None, :]-start, edge)/length_sq, 0, 1)
        q = start + alpha[:, :, None]*edge
        ds = np.sum((points[:, None, :]-q)**2, axis=2)
        better = ds < distance_sq
        nearest[better] = q[better]
        distance_sq = np.minimum(distance_sq, ds)
    chosen = np.argmin(distance_sq, axis=1)
    q = nearest[np.arange(len(points)), chosen]
    distance = np.sqrt(distance_sq[np.arange(len(points)), chosen])
    return distance, q


def closest_on_convex(points, hull):
    """Euclidean distance to a closed convex polyhedron, by all triangle faces."""
    distance, q = closest_on_triangles(points, hull.points[hull.simplices])
    inside = np.max(points @ hull.equations[:, :3].T + hull.equations[:, 3], axis=1) <= 1e-12
    distance[inside] = 0
    q[inside] = points[inside]
    return distance, q


def sampled_concavity_fill(candidate_triangles, source_triangles, divisions=8):
    """Sampled candidate-surface deviation, with a Lipschitz coverage bound.

    Distance to any closed surface is 1-Lipschitz. The barycentric grid covers
    each triangle within max_edge/divisions, providing a conservative upper
    bound on unsigned surface deviation. Winding classifies only the witness.
    """
    samples = []
    for i in range(divisions+1):
        for j in range(divisions+1-i):
            samples.append(candidate_triangles[:, 0] + i/divisions*(candidate_triangles[:, 1]-candidate_triangles[:, 0])
                           + j/divisions*(candidate_triangles[:, 2]-candidate_triangles[:, 0]))
    points = np.unique(np.concatenate(samples), axis=0)
    best = (-1., None, None)
    for start in range(0, len(points), 16):
        p = points[start:start+16]
        distance, nearest = closest_on_triangles(p, source_triangles)
        index = int(np.argmax(distance))
        if distance[index] > best[0]:
            best = (float(distance[index]), p[index], nearest[index])
    distance, witness, nearest = best
    a, b, c = np.moveaxis(source_triangles-witness, 1, 0)
    na, nb, nc = [np.linalg.norm(p, axis=1) for p in (a, b, c)]
    determinant = np.einsum('ij,ij->i', a, np.cross(b, c))
    denominator = na*nb*nc + np.einsum('ij,ij->i', a, b)*nc + np.einsum('ij,ij->i', b, c)*na + np.einsum('ij,ij->i', c, a)*nb
    winding = float(np.sum(2*np.arctan2(determinant, denominator))/(4*math.pi))
    max_edge = max(float(np.linalg.norm(candidate_triangles[:, i]-candidate_triangles[:, j], axis=1).max()) for i, j in ((0, 1), (1, 2), (2, 0)))
    return {'sampled_maximum_candidate_surface_to_cad_surface_distance_m': distance,
            'uniform_upper_bound_on_unsigned_surface_distance_m': distance+max_edge/divisions,
            'sampling_cover_radius_bound_m': max_edge/divisions,
            'sample_points': len(points), 'barycentric_divisions_per_edge': divisions,
            'witness_candidate_mesh_point_m': witness.tolist(), 'nearest_cad_surface_point_m': nearest.tolist(),
            'witness_generalized_winding_number_in_cad': winding,
            'witness_is_outside_cad_solid': bool(abs(winding) < 1e-6),
            'scope': 'Unsigned triangle-surface deviation; witness outside status uses oriented CAD winding. Upper bound is conservative (1-Lipschitz + grid cover radius), not a tight full-volume or signed-distance bound.'}


def origin(element):
    node = element.find('origin')
    xyz = np.array([float(v) for v in node.get('xyz', '0 0 0').split()])
    r, p, y = [float(v) for v in node.get('rpy', '0 0 0').split()]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    t = np.eye(4)
    t[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                 [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
    t[:3, 3] = xyz
    return t


def source_placements(root):
    rows = []
    for leg in ('lf', 'lm', 'lr', 'rf', 'rm', 'rr'):
        link = root.find(f"link[@name='{leg}_tibia']")
        visual = next(v for v in link.findall('visual') if v.find('geometry/mesh').get('filename').endswith('silicone_foot.stl'))
        scale = [float(v) for v in visual.find('geometry/mesh').get('scale', '1 1 1').split()]
        if scale != [1, 1, 1]:
            raise ValueError('Candidate expects the observed meter-scale mesh')
        t = origin(visual)
        spheres = [c for c in link.findall('collision') if c.find('geometry/sphere') is not None]
        centers = [(np.linalg.inv(t) @ np.r_[origin(c)[:3, 3], 1])[:3].tolist() for c in spheres]
        radii = [float(c.find('geometry/sphere').get('radius')) for c in spheres]
        if len(centers) != 2 or radii[0] != radii[1]:
            raise ValueError('Exact bisector proof requires two equal spheres')
        rows.append({'link': leg+'_tibia', 'source_visual_origin': visual.find('origin').attrib,
                     'mesh_scale': scale, 'link_from_mesh': t.tolist(),
                     'existing_sphere_centers_in_mesh_m': centers, 'existing_sphere_radii_m': radii,
                     'proposed_mesh': 'silicone_pad_convex64.stl', 'proposed_approximation': 'convexHull'})
    return rows


def sphere_surface_gap(triangles, centers, radius):
    """Maximum on every source triangle, partitioned by the spheres' bisector.

    On each Voronoi side, distance to the nearer sphere is a convex function.
    Its maximum over a clipped triangle is at an original or bisector vertex.
    """
    normal = centers[1]-centers[0]
    midpoint = centers.mean(axis=0)
    candidates = [triangles.reshape(-1, 3)]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = triangles[:, i], triangles[:, j]
        da, db = (a-midpoint) @ normal, (b-midpoint) @ normal
        cross = ((da < 0) & (db > 0)) | ((da > 0) & (db < 0))
        t = da[cross]/(da[cross]-db[cross])
        candidates.append(a[cross] + t[:, None]*(b[cross]-a[cross]))
    points = np.unique(np.concatenate(candidates), axis=0)
    distances = np.min(np.linalg.norm(points[:, None, :]-centers, axis=2)-radius, axis=1)
    index = int(np.argmax(distances))
    return {'maximum_outside_distance_m': float(max(0, distances[index])),
            'witness_point_mesh_m': points[index].tolist(), 'partition_vertices_evaluated': len(points),
            'scope': 'Entire triangulated CAD surface, equal-radius sphere Voronoi partition, floating-point geometry'}


def fibonacci(count=4096):
    z = 1-2*(np.arange(count)+.5)/count
    phi = np.arange(count)*math.pi*(3-math.sqrt(5))
    r = np.sqrt(1-z*z)
    return np.column_stack([r*np.cos(phi), r*np.sin(phi), z])


def support(points, directions):
    return np.max(points @ directions.T, axis=0)


def kinematic_poses(contract):
    poses = {'body': np.eye(4)}
    pending = {k: v for k, v in contract['joint_frames'].items() if not v['exclude_from_articulation']}
    while pending:
        before = len(pending)
        for name, frame in list(pending.items()):
            if frame['body0'] not in poses:
                continue
            q = contract['default_joint_positions_rad'][name]
            r = np.eye(4); r[:2, :2] = [[math.cos(q), -math.sin(q)], [math.sin(q), math.cos(q)]]
            poses[frame['body1']] = poses[frame['body0']] @ np.array(frame['body0_from_hinge_matrix']) @ r @ np.linalg.inv(frame['body1_from_hinge_matrix'])
            del pending[name]
        if len(pending) == before:
            raise ValueError('Disconnected physical tree')
    return poses


def main():
    source = read_stl(MESH)
    vertices = np.unique(source.reshape(-1, 3), axis=0)
    source_ids = {tuple(p): i for i, p in enumerate(vertices)}
    source_edges = {}
    for tri in source:
        ids = [source_ids[tuple(p)] for p in tri]
        for a, b in zip(ids, ids[1:]+ids[:1]):
            source_edges.setdefault(tuple(sorted((a, b))), []).append((a, b))
    source_manifold = all(len(v) == 2 and v[0] == tuple(reversed(v[1])) for v in source_edges.values())
    if not source_manifold:
        raise ValueError('Source mesh must be a closed consistently wound manifold for winding interpretation')
    full_hull = ConvexHull(vertices)
    extrema = sorted(set(int(np.argmax(vertices @ d)) for d in np.r_[np.eye(3), -np.eye(3)]))
    selected = extrema.copy()
    tradeoff = []
    for count in range(len(selected), 65):
        candidate = ConvexHull(vertices[selected])
        distances, nearest = closest_on_convex(vertices[full_hull.vertices], candidate)
        if count in (16, 24, 32, 48, 64):
            tradeoff.append({'vertices': len(candidate.vertices), 'triangles': len(candidate.simplices),
                             'max_cad_hull_to_candidate_distance_m': float(distances.max()),
                             'volume_m3': float(candidate.volume)})
        if count == 64:
            break
        selected.append(int(full_hull.vertices[int(np.argmax(distances))]))
        if len(set(selected)) != len(selected):
            raise ValueError('Greedy refinement repeated a vertex')
    mesh_out = OUT/'silicone_pad_convex64.stl'
    write_stl(mesh_out, candidate)
    reread = read_stl(mesh_out)
    candidate_vertices = np.unique(reread.reshape(-1, 3), axis=0)
    if len(candidate_vertices) > 64 or not all(any(np.array_equal(p, v) for v in vertices) for p in candidate_vertices):
        raise ValueError('Serialized candidate is not <=64 exact source vertices')
    candidate = ConvexHull(candidate_vertices)
    distances, nearest = closest_on_convex(vertices, candidate)
    witness = int(np.argmax(distances))
    witness_direction = (vertices[witness]-nearest[witness])/distances[witness]
    directions = np.r_[fibonacci(), np.eye(3), -np.eye(3), full_hull.equations[:, :3], candidate.equations[:, :3], witness_direction[None, :]]
    cad_support = support(vertices, directions)
    candidate_support = support(candidate_vertices, directions)
    placements = source_placements(ET.parse(URDF).getroot())
    centers = np.array(placements[0]['existing_sphere_centers_in_mesh_m'])
    radius = placements[0]['existing_sphere_radii_m'][0]
    sphere_support = np.max(centers @ directions.T, axis=0)+radius
    max_halfspace = float(np.max(candidate_vertices @ full_hull.equations[:, :3].T+full_hull.equations[:, 3]))
    if max_halfspace > 1e-10:
        raise ValueError('Candidate extends outside the exact CAD convex hull')
    # Independent topological/winding checks on serialized triangles.
    indices = {tuple(p): i for i, p in enumerate(candidate_vertices)}
    edges = {}
    for tri in reread:
        ids = [indices[tuple(p)] for p in tri]
        for a, b in zip(ids, ids[1:]+ids[:1]):
            edges.setdefault(tuple(sorted((a, b))), []).append((a, b))
    manifold = all(len(v) == 2 and v[0] == tuple(reversed(v[1])) for v in edges.values())
    volume = float(np.sum(np.einsum('ij,ij->i', reread[:, 0], np.cross(reread[:, 1], reread[:, 2])))/6)
    source_signed_volume = float(np.sum(np.einsum('ij,ij->i', source[:, 0], np.cross(source[:, 1], source[:, 2])))/6)
    if not manifold or volume <= 0 or not np.isclose(volume, candidate.volume, atol=1e-14, rtol=1e-10):
        raise ValueError('Candidate manifold, winding or volume failed')
    poses = kinematic_poses(json.loads(KIN.read_text()))
    nominal = []
    for row in placements:
        world = poses[row['link']] @ np.array(row['link_from_mesh'])
        down = world[:3, :3].T @ np.array([0., 0., -1.])
        h = float(support(vertices, down[None, :])[0])
        hc = float(support(candidate_vertices, down[None, :])[0])
        cs = np.array(row['existing_sphere_centers_in_mesh_m'])
        hs = float(np.max(cs @ down)+row['existing_sphere_radii_m'][0])
        nominal.append({'link': row['link'], 'down_direction_mesh': down.tolist(),
                        'candidate_bottom_minus_cad_bottom_m': h-hc,
                        'sphere_bottom_minus_cad_bottom_m': h-hs})
    # Plane-normal sweep around the LF nominal stance direction; this is not a joint workspace sweep.
    down = np.array(nominal[0]['down_direction_mesh']); helper = np.eye(3)[np.argmin(np.abs(down))]
    u = np.cross(down, helper); u /= np.linalg.norm(u); v = np.cross(down, u)
    sweep = []
    for tilt_deg in range(0, 91, 15):
        tilt = math.radians(tilt_deg)
        az = np.deg2rad(np.arange(0, 360, 5))
        normals = math.cos(tilt)*down + math.sin(tilt)*(np.cos(az)[:, None]*u+np.sin(az)[:, None]*v)
        hs = support(vertices, normals)
        hc = support(candidate_vertices, normals)
        hb = np.max(centers @ normals.T, axis=0)+radius
        sweep.append({'tilt_from_nominal_degrees': tilt_deg, 'azimuth_samples': len(az),
                      'candidate_bottom_error_min_m': float(np.min(hs-hc)), 'candidate_bottom_error_max_m': float(np.max(hs-hc)),
                      'sphere_bottom_error_min_m': float(np.min(hs-hb)), 'sphere_bottom_error_max_m': float(np.max(hs-hb))})
    report = {
        'schema': 'hexapod.pad_collision_candidate.v1', 'status': 'unqualified_rigid_envelope_proposal',
        'runtime_assets_modified': False, 'units': 'meters', 'coordinate_frame': 'Original silicone_foot.stl frame, no recenter/rotation/rescale',
        'sources': {str(p.relative_to(ROOT)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in (MESH, URDF, KIN)},
        'dependencies': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__},
        'algorithm': 'Greedy farthest Euclidean CAD-hull vertex refinement, starting with signed-axis support extrema, exact source vertices only, 64-vertex budget',
        'original_mesh': {'triangles': len(source), 'unique_vertices': len(vertices), 'convex_hull_vertices': len(full_hull.vertices),
                          'closed_consistently_wound_manifold': source_manifold,
                          'signed_mesh_volume_m3': source_signed_volume, 'convex_hull_volume_m3': float(full_hull.volume)},
        'tradeoff': tradeoff,
        'candidate': {'path': mesh_out.name, 'sha256': sha(mesh_out), 'bytes': mesh_out.stat().st_size,
                      'vertices': len(candidate_vertices), 'triangles': len(reread), 'volume_m3': volume,
                      'closed_consistently_wound_manifold': manifold, 'all_vertices_exact_source_vertices': True,
                      'maximum_cad_surface_to_candidate_distance_m': float(distances[witness]),
                      'witness_cad_mesh_point_m': vertices[witness].tolist(), 'witness_candidate_point_m': nearest[witness].tolist(),
                      'witness_direction': witness_direction.tolist(),
                      'bound_scope': 'Distance to a closed convex set is convex: vertex maximum bounds every original CAD triangle and its convex hull. Candidate is inside CAD convex hull. Thus this directed Hausdorff distance bounds all directional support deficits, within floating-point tolerance; it does not bound filling of CAD concavities.'},
        'existing_two_spheres': sphere_surface_gap(source, centers, radius),
        'concavity_fill': sampled_concavity_fill(reread, source),
        'sampled_support_comparison': {'directions': len(directions), 'sampling': '4096 Fibonacci + six axes + full CAD/candidate hull normals + exact Hausdorff witness direction',
                                       'candidate_max_deficit_m': float(np.max(cad_support-candidate_support)),
                                       'candidate_max_excess_m': float(np.max(candidate_support-cad_support)),
                                       'two_spheres_max_deficit_m': float(np.max(cad_support-sphere_support)),
                                       'two_spheres_max_excess_m': float(np.max(sphere_support-cad_support)),
                                       'warning': 'Support functions cannot detect the two-sphere waist or CAD concavities; use the separate full-surface gap check.'},
        'nominal_physical_stance': nominal, 'lf_plane_normal_sweep': sweep, 'placements': placements,
        'limitations': ['Not installed or cooked in PhysX; importer/cooking must preserve the measured envelope.',
                        'A convex hull fills source concavities and internal sockets; no self-collision compatibility is implied.',
                        'No new mass, COM, or inertia is derived from the collision volume.',
                        'Rigid geometry does not identify silicone stiffness, damping, friction, or hysteresis.',
                        'These are exact tessellated-CAD comparisons, not hardware dimensional measurements or a contact/solver qualification.']}
    (OUT/'geometry_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(OUT/'support_samples.npz', directions=directions, cad_support_m=cad_support,
                        candidate_support_m=candidate_support, existing_two_spheres_support_m=sphere_support)
    print(json.dumps({'candidate': report['candidate'], 'tradeoff': tradeoff, 'spheres': report['existing_two_spheres'], 'nominal': nominal}, indent=2))


if __name__ == '__main__':
    main()
