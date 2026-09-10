# Runnable CPU terrain observation seams

This implements the first two seams of `terrain_observation_comparison_plan_001`.
It creates no actor, Gym task, Isaac scene, physics defaults or GPU workload.
`UnresolvedRuntime(None, None, None)` explicitly keeps the selected actor, source
and physics unresolved; providing hashes still does not imply admission.

`CourseQueries` wraps the existing `TerrainSupportQueries` exact-triangle oracle.
It maps declared course-local points through yaw/XYZ placements, queries surface
height/normals, exposes signed clearances and delegates required-support relative
base height. Missing geometry stays NaN; visible pit floors remain ineligible;
an inset course edge is distinct from a geometry miss. Selected placement resets
validate before mutation and increment their own epochs. This is query state,
not a substitute for the future native scene-origin synchronization/admission.

`TerrainObservationModes` uses the existing `LocalHeightMap`, `WorldPoints` and
`terrain_channels` types. It returns exactly 100 × 100 × 4 cell-centred channels,
rows forward/columns left, gravity aligned, height relative to the supplied plate
position. The three modes are `blind`, `ideal_teacher` and `corrupted_map`.
Only ideal mode returns separate collision-geometry truth. A visible pit can be
observed while its separate geometry-support flag is false: neither visibility
nor map usability certifies traversability. Blind output contains no terrain
values/truth. Confirmed contacts do not populate the map.

The noisy-mode seam **accepts** already transformed/masked `WorldPoints` with
explicit synthetic/simulated provenance, calibration identifier, capture times,
variance, receipt time, clock and reset epoch. It does not itself generate sensor
noise, mask the robot, estimate pose or claim hardware calibration. The caller
must use the existing acquisition/registration functions to establish that
lineage. A queued packet is deep-copied; data is integrated only after its stated
receipt. Future capture, invalid variance/frame/clock and old-reset packets
reject. Existing map logic prevents older acquired data from replacing newer
cells. The unchanged 250 ms capture-age and 15 mm uncertainty contract decides
usability; missing/stale values are padded with zero height and zero usable mask.

All three modes share an explicit `oracle_localization` or
`estimated_localization` label. Calling a mode with an ideal pose is not a
deployment result. Blind/masked packets still include sampling coordinates as
debug metadata; a future actor packer must consume only the declared channels,
never debug world coordinates or `geometry_truth`.

`reset_rows` discards only those rows' map/queued data and increments their map
epochs. A changed course epoch blocks observation assembly until an explicit
map reset; other rows retain their maps, packets and independent read clocks.
Call course placement reset first, then observation reset, with the same selected
rows. Both methods mutate CPU adapter state only. World time must not rewind.

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run python -B -m unittest discover \
  -s tmp/terrain_observation_comparison_001 -p 'test_*.py'

PYTHONPATH=tools:isaaclab:tmp/terrain_observation_comparison_001 \
  PYTHONDONTWRITEBYTECODE=1 uv run python -B \
  tmp/terrain_observation_comparison_001/smoke.py \
  --catalog artifacts/terrain_readiness_2026-09-09/mild_curriculum_002/terrain_catalog.json \
  --fixture-id mild0_train_ramp_1103 --output /tmp/terrain_observation_smoke_new.json
```

The tests compare a translated/rotated admitted ramp to independent triangle
intersections, exercise an original pit and a deliberately missing synthetic
triangle, verify required support/negative clearance, exact map centres/axes,
oracle byte parity, receipt/lease timing and selective reset/input ownership.
The included smoke result is one-point synthetic acquisition, not a sensor or
robot rollout. `ORACLE_SHA256.json` binds the reused code/data; no frozen inputs
were edited. Device throughput, native course import, controller reset, actor
packing/reload and full robot terrain admissions remain separate work.
