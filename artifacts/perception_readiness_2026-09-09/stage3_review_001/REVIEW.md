# Stage 3 perception and supervisor review

10 September 2026. Read-only review; no main changes, sensor purchases, GPU work or architecture admission. The qualification draft correctly separates ideal teachers, corrupted/simulated observations and real sensor qualification. **It does not explicitly conflate sampled visibility with accumulated coverage.** Its main implementation gap is the missing executable support/stopping supervisor and route-time coverage test.

## What already executes

`tools/perception_replay.py` unprojects identified rectified depth, applies acquisition-time rigid transforms, masks robot returns and integrates latest valid evidence into a fixed odom height grid. It preserves holes and pits and refuses to refresh cells with older observations. `tools/terrain_readiness.py` produces four channels and checks whether all cells in a supplied support mask are usable. All14 existing perception tests pass on this review's host.

This is a synthetic CPU interface, not a moving-C sensor renderer, ROS2 node, localization estimator, terrain controller, supervisor, or measured onboard pipeline. The demonstration integrates six views at the same synthetic acquisition time and pose. The 1.2m map/0.8m returned patch defaults differ from the draft's proposed2m patch; a2m×2m/20mm schema must be explicitly configured and versioned. No current495/498 actor can simply consume the additional40,000 channel values without a new observation/encoder contract.

## Visibility and accumulation must have different denominators

The mount study's16.1% initial and78.1% expanded results are **instantaneous visibility averaged over discrete geometric target/pose cases**, on a hashed serial CAD snapshot. They are not spatial map completeness, all-time union coverage, current C reachable-foot coverage, route coverage, or hardware valid-depth performance. Cases cannot be unioned as though they occurred in one registered, time-consistent walk. The synthetic replay's1,031/3,600 fresh cells also describes one fixture, not a navigation result.

An accumulated map may retain an occluded cell seen earlier, but only while its acquisition age, pose-registration uncertainty and environment assumptions remain acceptable. Report three separate quantities along an actual time-indexed motion trace: instantaneous valid-depth coverage; retained usable-map coverage; and the fraction of the **required predicted support/stopping envelope** with usable, geometrically eligible evidence. The last is the supervisor's denominator. Raw map percentages must not authorize motion.

The proposed250ms age bound is particularly consequential: at0.005m/s the body travels only1.25mm within it; at proposed0.05–0.12m/s,12.5–30mm. Terrain observed150mm ahead would age1.25–3s before nominal arrival at those latter speeds if not reobserved, excluding gait phase and braking. At the present slow wave it ages30s. Therefore neither “we saw it ahead” nor “the current camera cannot see it” alone decides support usability. Test reacquisition and occlusion duration. Do not increase retention simply to improve coverage; longer-lived evidence requires a separately justified covariance/scene-change model. `LocalHeightMap.local_patch` currently does not add query-time pose drift to already stored cell variance.

## Minimum implementable interface

| Boundary | Required payload/decision |
| --- | --- |
| Map snapshot | Version, source mode, clock/frame/calibration IDs, acquisition and query times, odom origin/resolution/dimensions, gravity-aligned body patch transform and exact channel ordering/scales. Keep raw height/variance/acquisition/observed arrays for the supervisor; normalized actor channels alone lose information. Unknown/outside/stale/future stays unusable. |
| Controller prediction | Actual estimated pose/twist with uncertainty, requested twist, current phase/contact/reference state, and bounded candidate foot placements plus complete stop trajectories under the admitted controller. Include current swings and body/shaft/pad swept volume. A current-body footprint or `speed×delay` circle is insufficient. |
| Geometry assessment | Distinguish **observed**, **usable**, **support eligible** and **collision free**. A precisely observed120mm pit floor is usable height data but is ineligible support. The existing `support_region_observed` checks only data availability. Eligibility needs the declared step/slope/reach/drop bounds and obstacle/boundary layers; a single height cannot certify overhang clearance. |
| Supervisor result | Requested/admitted twist, valid-until time, selected predicted stopping/support envelope, reason codes and rejected cells, stop/replan/observe decision, and exact controller state/schema ID. Missing data commands a physically executed supported stop while feedback remains active; never freeze pose or substitute terrain truth. Record derating so it cannot count as original-command tracking success. |

The map schema can be shared between an `ideal_teacher` ray adapter and a `synthetic_interface` replay adapter, but provenance and results remain separate. Teacher privileged inputs must stay explicitly outside the deployable student. Header/calibration plausibility checks reject malformed transforms; they cannot detect a numerically valid but physically wrong calibration without independent geometry or consistency evidence.

## Priorities before the large qualification matrix

1. **Freeze a small interface and execute it on CPU.** Use the existing four-channel/raw-map functions with explicit2m configuration and controller-state input. Implement the geometry-eligibility and envelope generator; avoid defining them through simulator truth in the deployable path. Test known pit, boundary, unknown hole, stale/future frame, shifted transform and stream loss. Verify the result preserves the finite-stop controller's action-state continuity.
2. **Replay a measured full-C motion sequence in time.** Generate acquisition-time moving-leg masks for the selected geometry, integrate sequential views, and score instantaneous/map/required-envelope coverage separately for forward, strafe, yaw and arcs. Exercise body translation, leg occlusion, reacquisition, rolling-window recentering/outside cells, pose drift and dropout. Current fixed-odom grid suffices for a bounded initial route; a production rolling map is still absent. Failures identify coverage or latency gaps without a camera purchase decision.
3. **Exercise the same supervisor in the physics loop before qualification.** Start with an explicitly labeled ideal teacher and injected map faults; grade real supported stopping and complete pre-reset traces. Sensor-mode student training and matched comparison come after this interface works. Real selected-mode calibration, clock alignment, actual sensor masks/noise, localization and Jetson timing remain a separate measured qualification campaign.

No current artifact closes Stage3. The existing draft's observation-mode separation is sound; add these interface/coverage milestones as prerequisites, without equating CPU replay or sensor ownership with real perception readiness.

## Evidence used

- `tmp/stage3_qualification_plan_001/QUALIFICATION_PLAN.md` and `qualification_draft.json`.
- `tools/perception_replay.py`, `tools/terrain_readiness.py`, `robot/tests/test_perception_replay.py`.
- `artifacts/perception_readiness_2026-09-09/README.md` and `report.json`.
- `artifacts/sensor_mount_study_2026-09-09/README.md`, `report.json`, `expanded_exact_mounts.json`.

Exact input hashes are in `SOURCE_SHA256.json`. This note preserves the earlier sensor study's source and hardware limitations; no new hardware recommendation research was performed.
