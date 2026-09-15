# Three draft milestone issues

These are parent issues to open this week, not promises of completion this week. The checklists below are proposed acceptance contracts. M1’s numerical limits are already agreed; other missing limits need a proposal from the responsible lead before the affected work becomes Ready.

For each issue, name one accountable owner and a different reviewer. The owner posts a short approach, risks and proposed child tasks. Subleads review child PRs and reproduce their results; James accepts the milestone demonstration. A merged PR or completed investigation does not by itself complete the parent issue.

## Issue 1 — Demonstrate smooth omnidirectional walking and reliable stops

**Outcome:** The approved detailed hexapod follows commands in every horizontal direction, turns, and stops steadily on level hard ground, with gait quality comparable to the accepted forward-walking reference.

**Suggested staffing:** Two contributors; platform lead reviews. Owner unassigned.

**Starting point:** [Approved model](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/e9d5e775b3de76e9836c0139b557e2fcc4af047f/robot/active_model.json), [walking requirements](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/e9d5e775b3de76e9836c0139b557e2fcc4af047f/ARCHITECTURE.md#2-accepted-physical-and-control-baseline), and [current standing failures](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/e9d5e775b3de76e9836c0139b557e2fcc4af047f/artifacts/mkii_updated_2026-09-10/native_standing32_rejected_005/README.md). Walking videos use earlier models. Only 10 of 32 detailed-model standing cases pass; training on that model has not started.

**Deliver:** A saved controller, repeatable evaluation command, measured results and a short video showing the same evaluated controller.

**Success criteria**

- [ ] The approved model passes its required standing checks before training begins.
- [ ] Evaluation covers forward/backward, left/right, diagonals, both turning directions, simultaneous translation and turning, direction changes, and stops after movement.
- [ ] Each case meets the agreed tracking, stability, motor and foot-contact limits. Report failures individually, including falls and resets.
- [ ] Stop tests demonstrate settling and sustained stillness, rather than merely reaching the end of a route.
- [ ] James accepts the gait’s smoothness against the existing forward reference. Measurements accompany the comparison; the earlier model’s torque failure remains disclosed.
- [ ] The reviewer reproduces the result from the recorded source, model, configuration and controller files.

**Before execution:** The platform lead proposes the command speeds, durations, repeats, tracking tolerances and stop limits for the detailed model. Agree these before evaluating candidates. Preserve existing applicable gates and the recorded comparison limiter; historical gates retain their model-specific scope.

**Implementation decisions belong to the owner:** Standing diagnosis, training configuration, reward/curriculum changes and evaluation implementation, within the approved model and control architecture. Use child issues for bounded investigations. Rough terrain, navigation and physical deployment are outside this issue.

**Tracking:** Parent of controller work under Pages `stage2`. Standing diagnosis is a child task. Keep this milestone open until the full demonstration passes.

## Issue 2 — Produce a local 3D map from multiple simulated scans

**Outcome:** Combine stationary Mid-360 scans taken from several known positions into one saved, inspectable 3D map of a test scene.

**Suggested staffing:** Two contributors; survey lead reviews. Owner unassigned.

**Starting point:** [M1 specification](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/e9d5e775b3de76e9836c0139b557e2fcc4af047f/ARCHITECTURE.md#m1-stationary-simulated-scan-export-and-reload), [existing sensor configuration](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/e9d5e775b3de76e9836c0139b557e2fcc4af047f/isaaclab/hexapod_phase3/sensor_cfg.py) and [scan model](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/e9d5e775b3de76e9836c0139b557e2fcc4af047f/isaaclab/hexapod_phase3/mid360_pattern.py). Use simulator-provided sensor positions and orientations. Walking and sensor-based position estimation are separate dependencies for later integration.

**Deliver:** Recorded scans with their sensor poses, a combined map, a way to view it, and a repeatable geometry report against the known scene.

**Success criteria**

- [ ] M1 passes first: exact point/validity/pose preservation; every expected noise-free hit within 1 mm; at least 95% of expected noisy hits within 7 cm, counting misses against the total and reporting false returns separately.
- [ ] Scans from different positions and orientations align into the same local coordinate system. Overlapping surfaces agree with each other and the known scene within the agreed map-error limits.
- [ ] The output clearly distinguishes observed surfaces from areas with no measurements.
- [ ] Each input scan retains its acquisition time, sensor pose and sensor configuration identity.
- [ ] A separate process reloads the saved map and source scans without the original simulator session.
- [ ] The reviewer reproduces the map and error report. The report states that sensor poses came from the simulator.

**Before the multi-scan task becomes Ready:** The survey lead proposes the scene, scan locations, reference checks and numerical map-error limits. Agree them before evaluating the combined map. M1’s single-scan thresholds do not automatically define final map accuracy.

**Implementation decisions belong to the owner:** Storage format, map representation, scan-combination method and viewer. Preserve M1’s existing fixture and sensor settings. Physical mounting, localization, moving scans and terrain-aware walking are outside this issue.

**Tracking:** Parent mapping issue under Pages `mission`; M1 is its first child. Link the sensor component on the system graph. It can proceed independently of the walking milestone.

## Issue 3 — Rehearse a drawn-area survey with simulated motion

**Outcome:** An operator draws an area, approves a proposed survey route, runs it, and receives a clear record of completed and missed coverage. Use ideal position information and a simple motion simulator.

**Suggested staffing:** One contributor; survey lead reviews, with platform-lead review of motion commands. Owner unassigned.

**Starting point:** [Mission requirements and interfaces](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/e9d5e775b3de76e9836c0139b557e2fcc4af047f/ARCHITECTURE.md#3-system-boundaries-and-ownership) and [existing waypoint follower](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/e9d5e775b3de76e9836c0139b557e2fcc4af047f/packages/hexapod_nav/hexapod_nav/waypoint.py). The follower is an example; the drawing interface, coverage planner and mission controls remain to be built.

**Deliver:** A runnable operator demonstration and automated mission-state/coverage tests, including interrupted runs.

**Success criteria**

- [ ] Draw a simple survey polygon, inspect the route and explicitly approve it before motion. Reject invalid boundaries; changing an approved request requires new approval.
- [ ] During the agreed test routes, the simulated robot’s full footprint stays within the approved area or entry corridor.
- [ ] The robot stops at measurement locations and records simulated measurement completion.
- [ ] Pause stops movement; resume continues remaining work; abort stops and preserves a readable partial result. Abort requires a new approved mission to restart.
- [ ] Coverage comes from valid simulated measurement footprints. Missing or invalid measurements leave gaps; duplicate records do not increase coverage.
- [ ] A complete run meets the agreed coverage target. An interrupted run shows remaining areas and its termination reason.
- [ ] The motion simulator and simulated measurement responses connect through explicit interfaces that walking and mapping can later implement. The demonstration identifies its ideal-position and simulated-measurement assumptions.

**Before implementation becomes Ready:** The survey lead proposes a small set of boundary cases, robot and sensor footprints, coverage target and stop-response tolerance. Agree these with the platform lead and James. Use level, hard, obstacle-free conditions for this first rehearsal.

**Implementation decisions belong to the owner:** Route algorithm, operator interface, mission-state implementation and test-double design. Physical robot dynamics, real localization, scientific sensor accuracy and obstacle avoidance are outside this issue.

**Tracking:** Parent mission-software issue under Pages `mission`, linked to boundary → coverage → commands. It can proceed independently of gait and mapping.

## Tracking all three

Use one board with Needs definition, Ready, In progress, Review and Accepted. Show blockers explicitly. Each child issue links one parent, its deliverable and its reviewer. Keep one active child task per contributor. Record owner-proposed design choices in the issue or PR rather than a new planning document.

Pages shows the parent milestone’s status and evidence. GitHub holds assignments, child tasks and review discussion. The shared integration decisions are coordinates/units, motion-command meaning, scan/pose records and measurement-completion behavior; agree only the fields needed by the next child tasks.

Research execution remains paused until James resumes it. Drafting and assigning these issues does not start simulation or training.
