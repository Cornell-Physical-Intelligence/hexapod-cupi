# Synthetic CPU support-envelope supervisor001

10 September 2026. A bounded implementation of the first gap in `../stage3_perception_review_001/REVIEW.md`. **No actor integration, main changes, GPU run, physically admitted stopping model or real-sensor qualification.** Eleven CPU tests pass.

## Implemented distinction

A cell can be **observed**, **usable now**, **usable at its required future use**, and **geometrically eligible**; these are separate masks. A correctly observed120mm pit floor remains useful height evidence but is not eligible support. Unknown, stale, uncertain, future-dated, outside-course and outside-map regions cannot become zero-height support.

`MapSnapshot` copies the existing `LocalHeightMap` arrays without filling holes, with explicit odom frame, common acquisition clock, calibration ID, source mode, dimensions and resolution. `terrain_channels` supplies the unchanged current usability rules. The raw map remains intact; normalized height padding does not enter geometry decisions.

`map_eligibility` is a deliberately bounded synthetic height-geometry assessment. It requires a supplied reference support surface and mission boundary, then checks drop/step limits and fully observed3×3 height/slope neighborhoods. It does not invent a reference zero plane; the demos explicitly supply a known synthetic plane. Sharp edges/unknown neighborhoods are conservatively ineligible. It does not estimate material strength, support forces, friction or3D overhang clearance.

`teacher_eligibility` reuses `TerrainSupportQueries` and preserves its explicit geometry-hit/course/pit/support masks. That adapter is allowed only for `ideal_teacher` snapshots. Teacher geometry cannot silently complete sensor-mode eligibility. The existing query's geometric support mask is not a new slope/reach/stability certificate; the teacher curriculum still needs its admitted terrain envelope.

## Required swept envelope

`swept_foot_envelope` rasterizes explicit body-fixed pad locations over a supplied piecewise-linear XY/unwrapped-yaw path. It includes pad radius, declared position uncertainty, cell footprint and conservative inter-sample motion bounds. Checking only the starting and ending footprint misses the demo's intermediate pit; the full rotational sweep rejects it. Any swept physical disk crossing map bounds rejects instead of clipping onto the last known cell.

This is a **synthetic support bound**, not a gait planner or measured braking prediction. All supplied pad positions are treated as possible support throughout the path. Actual integration must supply the controller's validated stance/swing/stop trajectories, including ongoing swings, and add full pad/shaft/body collision assessment. An XY/yaw curve passing this function is not permission to execute it on the robot.

`assess` combines the geometry and observation masks for every required cell, rejects wrong frame/clock/time/provenance, and emits requested/admitted twist, explicit reason counts, validity deadline and controller ID. It never sends joint targets, disables feedback or freezes the robot. On rejection its zero admitted twist is a **request** for the controller to perform a supported stop/replan. `stop_feasibility_proven` remains false in every result.

No future camera reobservation is assumed: evidence must remain usable at each cell's latest required time. With the current250ms lease, the demo's short100ms path can pass but its500ms stopping path rejects even though all cells are usable now. This exposes the unresolved persistent-map versus measured-stop-duration contract; it does not solve it by increasing the age limit. Query-time pose-drift growth, scene changes and covariance calibration remain outstanding.

## Tests and artifacts

Eleven tests cover known-flat approval, observed-but-ineligible pit, unknown cells, stale/uncertain/future/negative-variance rejection, evidence expiring before stopping use, wrong frame/clock, course/raster boundaries, rotational sweeps, empty masks, the existing map adapter and strict teacher/sensor provenance separation. They use synthetic geometry; no reported success establishes real terrain traversal.

`report.json` records the expected outcomes: start and end poses individually pass, the pit-intersecting rotational sweep fails, a short flat support path passes and a longer path fails on evidence expiry. `swept_support.png` illustrates the endpoint-versus-sweep distinction.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tmp/stage3_support_supervisor_001 -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 python3 tmp/stage3_support_supervisor_001/demo.py
```

The next discriminator is sequential acquisition-time replay over measured full-C motion, scoring instantaneous visibility, retained usable map and required-envelope coverage separately. Current mount percentages and this synthetic raster are not accumulated-route coverage. The next physics integration must preserve the exact source/checkpoint/command identity and independently measure actual stopping; no current actor accepts this prototype.

`SOURCE_SHA256.json` records the reused modules and preparation evidence. `FREEZE_SHA256.json` freezes this prototype; no gait source or physical gate was edited.
