"""Classify each native ground-contact patch in a common world frame."""
from __future__ import annotations

import numpy as np

from .capture import CONTACT_CATEGORIES


def classify_patches(data, sensor_map, link_poses, origins, geometry, legs):
    """Retain patch loads before aggregation, including opposing nonfoot loads."""
    force, points, normals, separation, counts, starts = map(np.asarray, data)
    count = len(link_poses)
    capacity = len(force)
    if (force.shape != (capacity, 1) or points.shape != (capacity, 3)
            or normals.shape != points.shape or separation.shape != force.shape
            or counts.shape != (len(sensor_map), 1) or starts.shape != counts.shape
            or counts.dtype.kind not in 'iu' or starts.dtype.kind not in 'iu'
            or np.any(counts < 0) or np.any(starts < 0)):
        raise ValueError('Invalid native contact buffer layout')
    world = np.asarray(link_poses, dtype=float).copy()
    world[:, :, :3] += np.asarray(origins)[:, None]
    feet = np.zeros((count, len(legs), 3))
    other = np.zeros((count, len(CONTACT_CATEGORIES), 3))
    nonfoot = np.zeros(count, dtype=bool)
    patches, used = [], set()
    for index, (replica, body) in enumerate(sensor_map):
        start, size = int(starts[index, 0]), int(counts[index, 0])
        if start + size > capacity or used.intersection(range(start, start+size)):
            raise ValueError('Contact buffer overflow or overlapping ranges')
        used.update(range(start, start+size))
        ids = np.arange(start, start+size)
        if not size:
            continue
        f, p, n, d = force[ids, 0], points[ids], normals[ids], separation[ids, 0]
        if not all(np.isfinite(v).all() for v in (f, p, n, d)):
            raise ValueError('Nonfinite used contact patch')
        inactive = (f == 0) & np.all(n == 0, axis=1) & (d == 0)
        if np.any((abs(np.linalg.norm(n, axis=1)-1) > 1e-3) & ~inactive):
            raise ValueError('Invalid patch normal')
        local = None
        toe = np.zeros(size, dtype=bool)
        if body.endswith('_tibia'):
            pose = world[replica, geometry.body_names.index(body)]
            toe, local = geometry.cap_contains_batch(body, p, np.repeat(pose[None], size, axis=0))
            feet[replica, legs.index(body[:2])] += (f[toe, None]*n[toe]).sum(axis=0)
            category = 'shaft'
        else:
            category = 'coxa' if body.endswith('_coxa') else 'femur' if body.endswith('_femur') else 'body'
        body_nonfoot = (f[~toe, None]*n[~toe]).sum(axis=0)
        other[replica, CONTACT_CATEGORIES.index(category)] += body_nonfoot
        nonfoot[replica] |= bool(np.any(abs(f[~toe]) > 1.) or np.linalg.norm(body_nonfoot) > 1.)
        for offset, buffer_index in enumerate(ids):
            patches.append({'buffer_index': int(buffer_index), 'env': replica, 'body': body,
                'category': 'toe' if toe[offset] else category, 'normal_force_n': float(f[offset]),
                'point_world_m': p[offset].tolist(), 'normal_world': n[offset].tolist(),
                'separation_m': float(d[offset]), 'inactive_zero_normal': bool(inactive[offset]),
                'shape_point_m': None if local is None else local[offset].tolist()})
    if len(used) >= capacity:
        raise ValueError('Contact capacity exhausted; completeness is unknown')
    return {'distal_contact': np.linalg.norm(feet, axis=-1) > 1., 'distal_force_world_n': feet,
            'nonfoot_contact': nonfoot, 'nonfoot_force_world_n': other, 'patches': patches}
