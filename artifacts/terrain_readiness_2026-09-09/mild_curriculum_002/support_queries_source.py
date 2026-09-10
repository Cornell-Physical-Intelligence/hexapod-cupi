"""Batched exact-triangle terrain queries for a future privileged teacher.

This is geometry truth, not a sensor, a traversability guarantee or a Gym task.
Unknown/missing surfaces remain NaN with explicit masks; pit floors are visible
geometry but are never marked as allowed walking support.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass
class SurfaceQuery:
    height_m: torch.Tensor
    normal: torch.Tensor
    geometry_hit: torch.Tensor
    inside_course: torch.Tensor
    avoidance_hazard: torch.Tensor
    support_geometry: torch.Tensor


@dataclass
class RelativeHeight:
    height_m: torch.Tensor
    reference_height_m: torch.Tensor
    valid: torch.Tensor
    terrain_height_span_m: torch.Tensor


def _expand_prefix(value, shape, *, device, dtype):
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.ndim < len(shape):
        value = value.reshape(*value.shape, *([1] * (len(shape) - value.ndim)))
    return torch.broadcast_to(value, shape)


class TerrainSupportQueries:
    """Uniform XY index over unmodified triangles, with batched Torch queries.

    Build once on CPU and move immutable tensors to the future simulation device.
    Query memory is O(chunk_size * cell_candidates), not O(points * all_triangles).
    Input records use load_catalog's (metadata, USD path, vertices, faces) format.
    Course poses support translation and yaw only, retaining Z-up geometry.
    """

    def __init__(self, records, *, cell_size_m=.05, chunk_size=8192, device="cpu", dtype=torch.float32):
        if not records or not math.isfinite(cell_size_m) or cell_size_m <= 0 or chunk_size < 1:
            raise ValueError("Nonempty fixtures, positive cell size and chunk size required")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Terrain queries require a floating-point geometry type")
        from terrain_fixture_checks import validate_mesh
        self.device, self.dtype = torch.device(device), dtype
        self.fixture_ids = tuple(record[0]["id"] for record in records)
        if len(set(self.fixture_ids)) != len(self.fixture_ids):
            raise ValueError("Fixture IDs must be unique")
        self.cell_size_m, self.chunk_size = float(cell_size_m), int(chunk_size)
        triangles, bounds, dimensions, pits, lists = [], [], [], [], []
        offset = 0
        for metadata, _, vertices, faces in records:
            validate_mesh(vertices, faces)
            tri = vertices[faces]
            normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            # A vertical triangle has zero projected area and cannot be a
            # downward surface hit. Sharp walls remain in the collision USD.
            tri = tri[np.abs(normal[:, 2]) > 1e-12]
            lo, hi = vertices[:, :2].min(0), vertices[:, :2].max(0)
            size = np.maximum(1, np.ceil((hi - lo) / cell_size_m).astype(int))
            cells = {}
            for local, triangle in enumerate(tri):
                # Include both cells at a boundary with a small metre-space
                # padding, without multiplying every cell's candidate count by
                # whole neighbouring bins. Exact triangle tests decide hits.
                padding = 5e-6
                start = np.floor((triangle[:, :2].min(0) - lo - padding) / cell_size_m).astype(int)
                end = np.floor((triangle[:, :2].max(0) - lo + padding) / cell_size_m).astype(int)
                start = np.maximum(0, start)
                end = np.minimum(size - 1, end)
                for x in range(start[0], end[0] + 1):
                    for y in range(start[1], end[1] + 1):
                        cells.setdefault((x, y), []).append(offset + local)
            triangles.append(tri)
            offset += len(tri)
            bounds.append(np.r_[lo, hi])
            dimensions.append(size)
            pits.append(metadata.get("pit_half_width_m", 0.) if metadata["family"] == "pit" else 0.)
            lists.append(cells)
        if not offset:
            raise ValueError("No downward-queryable triangles")
        dimensions = np.asarray(dimensions)
        max_x, max_y = dimensions.max(0)
        max_candidates = max(len(value) for cells in lists for value in cells.values())
        if offset >= np.iinfo(np.int32).max:
            raise ValueError("Triangle count exceeds the prepared int32 spatial index")
        indices = np.full((len(records), max_x, max_y, max_candidates), offset, dtype=np.int32)
        for fixture, cells in enumerate(lists):
            for (x, y), candidates in cells.items():
                indices[fixture, x, y, :len(candidates)] = candidates
        tri = np.concatenate(triangles)
        # A sentinel triangle yields no hit; no negative indexing ever aliases
        # missing candidates to the last valid surface.
        tri = np.concatenate((tri, np.zeros((1, 3, 3))), axis=0)
        self.triangles = torch.as_tensor(tri, device=device, dtype=dtype)
        self.candidates = torch.as_tensor(indices, device=device)
        self.bounds = torch.as_tensor(np.asarray(bounds), device=device, dtype=dtype)
        self.dimensions = torch.as_tensor(dimensions, device=device)
        self.pit_half_width = torch.as_tensor(pits, device=device, dtype=dtype)
        self.sentinel = offset
        self.statistics = dict(fixtures=len(records), triangles=offset, max_cell_candidates=max_candidates,
            spatial_index_bytes=indices.nbytes, cell_size_m=cell_size_m,
            implementation="exact vertical triangle intersection; indexed candidates; teacher geometry truth")

    def query_local(self, xy, fixture_indices, *, ray_origin_z_m=2., boundary_margin_m=0.):
        """Course-local XY → highest triangle below a fixed finite ray origin.

        Shape (...,2), with fixture_indices broadcast over leading environment
        dimensions. Boundary margins invalidate support; actual geometry hits
        are still reported. Inputs outside the course never clamp onto its edge.
        """
        xy = torch.as_tensor(xy, device=self.device, dtype=self.dtype)
        if xy.ndim < 2 or xy.shape[-1] != 2:
            raise ValueError("Expected (...,points,2) or (points,2) coordinates")
        if not math.isfinite(ray_origin_z_m) or not math.isfinite(boundary_margin_m) or boundary_margin_m < 0:
            raise ValueError("Finite ray origin and nonnegative boundary margin required")
        shape = xy.shape[:-1]
        fixture = _expand_prefix(fixture_indices, shape, device=self.device, dtype=torch.long).reshape(-1)
        if bool(((fixture < 0) | (fixture >= len(self.fixture_ids))).any()):
            raise ValueError("Fixture index is outside the selected atlas")
        xy = xy.reshape(-1, 2)
        chunks = []
        for start in range(0, len(xy), self.chunk_size):
            point, index = xy[start:start + self.chunk_size], fixture[start:start + self.chunk_size]
            bounds = self.bounds[index]
            finite = torch.isfinite(point).all(-1)
            inside_geometry = finite & (point >= bounds[:, :2]).all(-1) & (point <= bounds[:, 2:]).all(-1)
            inside_course = finite & (point >= bounds[:, :2] + boundary_margin_m).all(-1) & (
                point <= bounds[:, 2:] - boundary_margin_m).all(-1)
            safe_point = torch.where(finite[:, None], point, bounds[:, :2])
            cell = torch.floor((safe_point - bounds[:, :2]) / self.cell_size_m).long()
            cell = torch.minimum(torch.maximum(cell, torch.zeros_like(cell)), self.dimensions[index] - 1)
            ids = self.candidates[index, cell[:, 0], cell[:, 1]]
            triangles = self.triangles[ids]
            a, u, v = triangles[:, :, 0], triangles[:, :, 1] - triangles[:, :, 0], triangles[:, :, 2] - triangles[:, :, 0]
            denominator = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
            divisor = torch.where(denominator.abs() > 1e-12, denominator, torch.ones_like(denominator))
            delta = safe_point[:, None] - a[..., :2]
            beta = (delta[..., 0] * v[..., 1] - delta[..., 1] * v[..., 0]) / divisor
            gamma = (u[..., 0] * delta[..., 1] - u[..., 1] * delta[..., 0]) / divisor
            z = a[..., 2] + beta * u[..., 2] + gamma * v[..., 2]
            accepted = ((ids != self.sentinel) & (denominator.abs() > 1e-12) & (beta >= -1e-6)
                & (gamma >= -1e-6) & (beta + gamma <= 1 + 1e-6) & (z <= ray_origin_z_m)
                & inside_geometry[:, None])
            height, selected = torch.where(accepted, z, -torch.inf).max(-1)
            hit = torch.isfinite(height)
            normal = torch.linalg.cross(u, v, dim=-1)
            normal = normal[torch.arange(len(point), device=self.device), selected]
            normal = normal * torch.where(normal[:, 2:3] < 0, -1., 1.)
            normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1e-12)
            normal = torch.where(hit[:, None], normal, torch.nan)
            height = torch.where(hit, height, torch.nan)
            half_width = self.pit_half_width[index]
            hazard = finite & (half_width > 0) & (point.abs() <= half_width[:, None] + boundary_margin_m).all(-1)
            support = hit & inside_course & ~hazard
            chunks.append((height, normal, hit, inside_course, hazard, support))
        if not chunks:
            raise ValueError("An empty query batch is not a support observation")
        arrays = [torch.cat([chunk[i] for chunk in chunks]).reshape(*shape, *(() if i != 1 else (3,))) for i in range(6)]
        return SurfaceQuery(*arrays)

    def query_world(self, points_world, fixture_indices, *, course_origins_world, course_yaw_rad=0., boundary_margin_m=0.):
        """World XYZ → world-Z surface height, preserving course yaw/translation."""
        points = torch.as_tensor(points_world, device=self.device, dtype=self.dtype)
        if points.ndim < 2 or points.shape[-1] != 3:
            raise ValueError("Expected world points ending in XYZ")
        shape = points.shape[:-1]
        origins = torch.as_tensor(course_origins_world, device=self.device, dtype=self.dtype)
        if origins.shape[-1:] != (3,):
            raise ValueError("Course origins must end in XYZ")
        origins = origins.reshape(*origins.shape[:-1], *([1] * (len(shape) - origins.ndim + 1)), 3)
        origins = torch.broadcast_to(origins, points.shape)
        yaw = _expand_prefix(course_yaw_rad, shape, device=self.device, dtype=self.dtype)
        if not torch.isfinite(origins).all() or not torch.isfinite(yaw).all():
            raise ValueError("Course poses must be finite")
        delta = points - origins
        cosine, sine = yaw.cos(), yaw.sin()
        local = torch.stack((cosine * delta[..., 0] + sine * delta[..., 1],
                             -sine * delta[..., 0] + cosine * delta[..., 1]), -1)
        query = self.query_local(local, fixture_indices, boundary_margin_m=boundary_margin_m)
        finite = torch.isfinite(points).all(-1)
        normal = query.normal
        normal = torch.stack((cosine * normal[..., 0] - sine * normal[..., 1],
                              sine * normal[..., 0] + cosine * normal[..., 1], normal[..., 2]), -1)
        return SurfaceQuery(torch.where(finite, query.height_m + origins[..., 2], torch.nan),
                            torch.where(finite[..., None], normal, torch.nan), query.geometry_hit & finite,
                            query.inside_course & finite, query.avoidance_hazard & finite,
                            query.support_geometry & finite)

    def point_clearances(self, points_world, fixture_indices, **course_pose):
        """Signed vertical clearance; negative values expose penetration.

        Returns (clearance, geometry query). A pit's floor can give a numeric
        clearance while support_geometry remains false. Callers must use masks.
        """
        points = torch.as_tensor(points_world, device=self.device, dtype=self.dtype)
        query = self.query_world(points, fixture_indices, **course_pose)
        return points[..., 2] - query.height_m, query

    def relative_base_height(self, root_positions_world, support_samples_world, fixture_indices,
                             *, required_samples, observation_usable=None, **course_pose):
        """Root Z minus the mean terrain height at required support samples.

        All required samples must have eligible geometry; missing/pit/boundary
        samples invalidate the whole reference. Optional observation_usable must
        already include sensor age/error checks. No selection by current contact
        is implicit. This mean-height primitive does not claim a feasible support
        plane, stable force balance, or a completed terrain reward design.
        """
        points = torch.as_tensor(support_samples_world, device=self.device, dtype=self.dtype)
        root = torch.as_tensor(root_positions_world, device=self.device, dtype=self.dtype)
        required = torch.as_tensor(required_samples, device=self.device)
        if points.ndim != 3 or root.shape != (points.shape[0], 3) or required.shape != points.shape[:2] or required.dtype != torch.bool:
            raise ValueError("Expected BxPx3 samples, Bx3 roots and BxP boolean required mask")
        query = self.query_world(points, fixture_indices, **course_pose)
        usable = query.support_geometry
        if observation_usable is not None:
            observed = torch.as_tensor(observation_usable, device=self.device)
            if observed.dtype != torch.bool or observed.shape != required.shape:
                raise ValueError("Observation usability must match the boolean required mask")
            usable = usable & observed
        count = required.sum(-1)
        valid = (count > 0) & (~required | usable).all(-1) & torch.isfinite(root).all(-1)
        height = torch.where(required & usable, query.height_m, 0.).sum(-1) / count.clamp_min(1)
        low = torch.where(required & usable, query.height_m, torch.inf).min(-1).values
        high = torch.where(required & usable, query.height_m, -torch.inf).max(-1).values
        return RelativeHeight(torch.where(valid, root[:, 2] - height, torch.nan),
                              torch.where(valid, height, torch.nan), valid,
                              torch.where(valid, high - low, torch.nan))
