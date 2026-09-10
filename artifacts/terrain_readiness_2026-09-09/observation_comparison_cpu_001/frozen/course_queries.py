"""CPU course-placement adapter over the existing exact-triangle query oracle.

No Isaac scene mutation, controller, actor or physics defaults are supplied.
"""
from dataclasses import dataclass, asdict
import re
import torch
from hexapod_terrain.support_queries import TerrainSupportQueries


@dataclass(frozen=True)
class UnresolvedRuntime:
    actor_sha256: str | None
    source_sha256: str | None
    physics_sha256: str | None

    def __post_init__(self):
        for value in asdict(self).values():
            if value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("Runtime identities must be explicit None or SHA256")

    def receipt(self):
        return {**asdict(self), "runtime_admitted": False, "CPU_adapter_only": True}


class CourseQueries:
    def __init__(self, oracle: TerrainSupportQueries, *, fixture_indices,
                 origins_world, yaw_rad, boundary_margin_m, runtime: UnresolvedRuntime):
        if oracle.device.type != "cpu":
            raise ValueError("This bounded adapter is CPU-only")
        if not isinstance(runtime, UnresolvedRuntime):
            raise TypeError("An explicit unresolved runtime identity is required")
        self.oracle, self.runtime = oracle, runtime
        self.boundary_margin_m = float(boundary_margin_m)
        if not torch.isfinite(torch.tensor(self.boundary_margin_m)) or self.boundary_margin_m < 0:
            raise ValueError("Finite nonnegative course-edge margin required")
        n = len(fixture_indices)
        if n < 1:
            raise ValueError("At least one course is required")
        self.num_envs = n
        f, o, y = self._placement(fixture_indices, origins_world, yaw_rad, n)
        self._fixtures, self._origins, self._yaw = f, o, y
        self._epochs = torch.zeros(n, dtype=torch.long)

    def _placement(self, fixture_indices, origins, yaw, n):
        f = torch.as_tensor(fixture_indices)
        if f.dtype not in (torch.int32, torch.int64) or f.shape != (n,):
            raise ValueError("One integer fixture index per course is required")
        f = f.to(dtype=torch.long, device="cpu").clone()
        o = torch.as_tensor(origins, dtype=self.oracle.dtype, device="cpu").clone()
        y = torch.as_tensor(yaw, dtype=self.oracle.dtype, device="cpu").clone()
        if o.shape != (n, 3) or y.shape != (n,) or not torch.isfinite(o).all() or not torch.isfinite(y).all():
            raise ValueError("Finite course XYZ and yaw per environment required")
        if ((f < 0) | (f >= len(self.oracle.fixture_ids))).any():
            raise ValueError("Fixture index outside this immutable oracle")
        return f, o, y

    def rows(self, env_ids):
        rows = torch.as_tensor(env_ids)
        if rows.ndim != 1 or rows.dtype not in (torch.int32, torch.int64) or rows.numel() == 0:
            raise ValueError("Nonempty integer environment rows required")
        rows = rows.to(dtype=torch.long, device="cpu")
        if rows.unique().numel() != rows.numel() or ((rows < 0) | (rows >= self.num_envs)).any():
            raise ValueError("Duplicate or out-of-bounds environment row")
        return rows

    def snapshot(self):
        return {"fixture_indices": self._fixtures.clone(), "origins_world": self._origins.clone(),
                "yaw_rad": self._yaw.clone(), "epochs": self._epochs.clone()}

    def reset_rows(self, env_ids, *, fixture_indices, origins_world, yaw_rad):
        """Replace selected query placements only; this never writes physical state."""
        rows = self.rows(env_ids)
        f, o, y = self._placement(fixture_indices, origins_world, yaw_rad, len(rows))
        self._fixtures[rows], self._origins[rows], self._yaw[rows] = f, o, y
        self._epochs[rows] += 1

    def _points(self, rows, points):
        points = torch.as_tensor(points, device="cpu", dtype=self.oracle.dtype)
        if points.ndim != 3 or points.shape[0] != len(rows) or points.shape[-1] != 3 or points.shape[1] < 1:
            raise ValueError("Points must be selected-environment x point x XYZ")
        return points

    def course_to_world(self, env_ids, points_course):
        rows = self.rows(env_ids); p = self._points(rows, points_course)
        c, s = self._yaw[rows, None].cos(), self._yaw[rows, None].sin()
        rotated = torch.stack((c*p[..., 0]-s*p[..., 1], s*p[..., 0]+c*p[..., 1], p[..., 2]), -1)
        return rotated + self._origins[rows, None]

    def query_world(self, env_ids, points_world):
        rows = self.rows(env_ids); p = self._points(rows, points_world)
        return self.oracle.query_world(p, self._fixtures[rows],
            course_origins_world=self._origins[rows], course_yaw_rad=self._yaw[rows],
            boundary_margin_m=self.boundary_margin_m)

    def relative_base_height(self, env_ids, root_world, support_samples_world, *, required_samples,
                             observation_usable=None):
        rows = self.rows(env_ids); p = self._points(rows, support_samples_world)
        return self.oracle.relative_base_height(root_world, p, self._fixtures[rows],
            required_samples=required_samples, observation_usable=observation_usable,
            course_origins_world=self._origins[rows], course_yaw_rad=self._yaw[rows],
            boundary_margin_m=self.boundary_margin_m)

    def point_clearances(self, env_ids, points_world):
        rows = self.rows(env_ids); p = self._points(rows, points_world)
        query = self.query_world(rows, p)
        return p[..., 2] - query.height_m, query
