"""CPU observation seam: existing triangle truth or causally delivered map data.

WorldPoints must already contain transformed, masked, uncertainty-labelled data.
This adapter does not simulate sensors, estimate pose, add noise, or infer support
from a contact. The caller explicitly labels synthetic/simulated provenance.
"""
from dataclasses import dataclass
import copy
import math
import numpy as np
from perception_replay import LocalHeightMap, WorldPoints, validate_transform
from terrain_readiness import terrain_channels
from course_queries import CourseQueries


@dataclass
class MapPacket:
    mode: str
    channels: np.ndarray
    sample_world_xy: np.ndarray
    assembled_time_s: float
    reset_epoch: int
    geometry_truth: object | None
    provenance: tuple[str, ...]
    registration: str = "caller_supplied_pose; oracle_or_estimated_must_be_declared"


class TerrainObservationModes:
    MODES = {"blind", "ideal_teacher", "corrupted_map"}
    PROVENANCE = {"synthetic_corrupted_map", "simulated_lidar", "simulated_depth", "fused_simulated_map"}

    def __init__(self, courses: CourseQueries, *, map_origins_xy, map_sizes_xy,
                 world_frame, clock_id, reset_time_s, registration):
        if not world_frame or not clock_id or registration not in {"oracle_localization", "estimated_localization"}:
            raise ValueError("Explicit world frame, clock and registration type required")
        if not math.isfinite(reset_time_s):
            raise ValueError("Finite reset time required")
        self.courses, self.world_frame, self.clock_id = courses, world_frame, clock_id
        self.registration = registration
        n = courses.num_envs
        self._origins, self._sizes = self._specs(map_origins_xy, map_sizes_xy, n)
        self._maps = [self._new_map(i) for i in range(n)]
        self._pending = [[] for _ in range(n)]
        self._sources = [set() for _ in range(n)]
        self._epoch = np.zeros(n, dtype=np.int64)
        self._course_epoch = courses.snapshot()["epochs"].numpy().copy()
        self._reset_time = np.full(n, reset_time_s, dtype=float)
        self._last_read = np.full(n, reset_time_s, dtype=float)

    def _specs(self, origins, sizes, n):
        o, s = np.asarray(origins, float).copy(), np.asarray(sizes, float).copy()
        if o.shape != (n, 2) or s.shape != (n, 2) or not np.isfinite(o).all() or not np.isfinite(s).all() or (s <= 0).any():
            raise ValueError("Finite map origins and positive sizes required for every selected row")
        return o, s

    def _new_map(self, i):
        return LocalHeightMap(origin_xy=self._origins[i], size_xy=self._sizes[i],
                              resolution_m=.02, world_frame=self.world_frame)

    def _row(self, row):
        rows = self.courses.rows([row]); i = int(rows[0])
        if self.courses.snapshot()["epochs"][i].item() != self._course_epoch[i]:
            raise ValueError("Course reset requires an explicit matching observation reset")
        return i

    def epoch(self, row):
        return int(self._epoch[self._row(row)])

    def reset_rows(self, env_ids, *, now_s, map_origins_xy, map_sizes_xy):
        """Drop only selected maps/packets; old epoch packets cannot re-enter."""
        rows = self.courses.rows(env_ids).numpy()
        o, s = self._specs(map_origins_xy, map_sizes_xy, len(rows))
        if not math.isfinite(now_s) or (now_s < self._last_read[rows]).any():
            raise ValueError("Reset cannot rewind the common clock")
        for k, i in enumerate(rows):
            self._origins[i], self._sizes[i] = o[k], s[k]
            self._maps[i] = self._new_map(i); self._pending[i].clear(); self._sources[i].clear()
            self._epoch[i] += 1; self._reset_time[i] = self._last_read[i] = now_s
            self._course_epoch[i] = self.courses.snapshot()["epochs"][i].item()

    def enqueue(self, row, cloud: WorldPoints, *, receive_time_s, clock_id, reset_epoch, provenance):
        i = self._row(row)
        if clock_id != self.clock_id or reset_epoch != self._epoch[i] or provenance not in self.PROVENANCE:
            raise ValueError("Wrong clock, reset epoch or unqualified stream provenance")
        if not isinstance(cloud, WorldPoints) or cloud.world_frame != self.world_frame or not cloud.source_sensor or not cloud.calibration_id:
            raise ValueError("Expected identified WorldPoints in the declared world frame")
        p, v, t = np.asarray(cloud.points_m), np.asarray(cloud.variance_z_m2), np.asarray(cloud.capture_time_s)
        if p.ndim != 2 or p.shape[1] != 3 or v.shape != (len(p),) or t.shape != (len(p),):
            raise ValueError("Malformed world point packet")
        if (not math.isfinite(receive_time_s) or receive_time_s < self._last_read[i]
            or not np.isfinite(p).all() or not np.isfinite(v).all() or not np.isfinite(t).all()
            or (v < 0).any() or (t > receive_time_s).any() or (t < self._reset_time[i]).any()):
            raise ValueError("Invalid uncertainty, future capture, late receipt or pre-reset evidence")
        # Ownership transfers by value: caller mutation cannot change queued data.
        self._pending[i].append((float(receive_time_s), copy.deepcopy(cloud), provenance))
        self._pending[i].sort(key=lambda item: item[0])

    def packet(self, row, mode, world_from_body, *, now_s):
        i = self._row(row)
        if mode not in self.MODES or not math.isfinite(now_s) or now_s < self._last_read[i]:
            raise ValueError("Unknown observation mode or nonmonotonic assembly time")
        pose = validate_transform(world_from_body)
        self._last_read[i] = now_s
        while self._pending[i] and self._pending[i][0][0] <= now_s:
            _, cloud, source = self._pending[i].pop(0)
            self._maps[i].integrate(cloud); self._sources[i].add(source)
        patch = self._maps[i].local_patch(pose, now_s=now_s, extent_m=2., resolution_m=.02)
        if patch['channels'].shape != (100, 100, 4):
            raise ValueError("Map cell-centre convention changed")
        channels, truth = patch['channels'].copy(), None
        sources = tuple(sorted(self._sources[i]))
        if mode == "blind":
            channels[..., 0:2] = 0.; channels[..., 2:4] = 1.
            sources = ()
        elif mode == "ideal_teacher":
            xy = patch['world_xy']; points = np.concatenate((xy, np.zeros((100, 100, 1))), axis=-1)
            truth = self.courses.query_world([i], points.reshape(1, -1, 3))
            height = truth.height_m.numpy().reshape(100, 100) - pose[2, 3]
            hit = truth.geometry_hit.numpy().reshape(100, 100)
            channels = terrain_channels(height, np.zeros_like(height), hit,
                                         np.full_like(height, now_s), now_s=now_s)
            sources = ("collision_geometry_truth",)
        return MapPacket(mode, channels, patch['world_xy'].copy(), float(now_s), int(self._epoch[i]),
                         truth, sources, self.registration)
