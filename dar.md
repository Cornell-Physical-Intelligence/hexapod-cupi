# Mission and requirements

Reviewed with the project lead on 4 September 2026. [PLAN.md](docs/PLAN.md) is the living architecture, timeline and decision record; [STATUS.md](STATUS.md) describes current evidence and execution state. The [previous requirements page](docs/archive/DAR-before-autonomy-review-2026-09-04.md) is retained as historical context.

## Confirmed mission

An operator draws a bounded zone on a map, places the hexapod near it, and starts a mission. The robot registers the zone, approaches through an approved corridor, follows zigzag passes across traversable ground with little deviation, and provides a steady platform for survey acquisition. It reports covered, missed and unreachable areas and can pause, stop and resume predictably.

The final research direction is robust movement on irregular outdoor terrain in any commanded direction. Initial releases qualify explicit terrain and operating envelopes. “Any terrain” and “foolproof” are ambitions, not claims an untested policy can satisfy.

## Confirmed constraints and responsibilities

- Six legs with three actuators each; CAD assembly is the mechanical baseline. Only CAD is complete. A single-leg test comes next, followed by remaining-parts ordering and assembly.
- Use learned omnidirectional locomotion beneath a separately tested coverage/navigation system. Do not prescribe a tripod phase or hard-coded footfall schedule.
- Favor vision, navigation LiDAR and IMU. Navigation sensors are independent of survey sensing; ordinary GNSS can provide coarse anchoring. No operational RTK base station is required or planned.
- Accurate local line following and absolute placement of a satellite-map boundary are separate requirements. Boundary registration must have sufficient measured confidence for the moving robot's footprint and stopping margin.
- The deployment computer is the already-owned Jetson Orin Nano. Actuators are RobStride RS05 with XT30 2+2 connections; CAN topology, adapter/controller and final harness are to be validated. Sensors will likely use Ethernet and USB.
- Terrestrial LiDAR is the initial survey payload. Cornell Geo Data's sensor team owns acquisition and processing; platform autonomy supplies motion, calibrated pose/time and a generic payload interface. Future payloads must fit a qualified mass/COM, clearance, power and stability envelope.
- Mechanical work proceeds in parallel on weight, strength, fit, stops/linkage and endurance. Electrical and embedded teams jointly qualify loaded CAN/power timing, fault isolation and stopping behavior. CS owns most autonomy implementation and integration.
- Available Spark compute and AI implementation support enable parallel software development. Physical measurements, procurement, calibration and held-out qualification remain necessary.

## Requirements still to close

The initial working area is approximately 10 m × 10 m on packed ground, based on the prior mission record. Proposed tracking targets are p95 cross-track error ≤0.10 m, maximum ≤0.25 m, p95 heading error ≤5°, and ≥95% coverage of the agreed traversable target using the actual survey footprint. These are **proposals**, not achieved or lead-ratified values. [The plan](docs/PLAN.md#5-acceptance-gates) specifies how to measure them and qualify terrain.

Finalize useful survey footprint and overlap, payload mass/COM and stability limits, absolute geolocation accuracy, mission duration and useful speed, operator-link needs, field site, staffing and hard deadline. Prior records mention a $6,000 budget, tentative 1–2 kg payload and long-range telemetry; reconcile those with the current teams rather than treating them as newly confirmed constraints.

## Development sequence

1. Correct and validate the simulator asset, reset and coordinate/action contracts.
2. Train flat omnidirectional motion, then a qualified terrain curriculum and deployable perceptive policy.
3. In parallel, develop camera/LiDAR/IMU localization, near-ground terrain mapping, polygon coverage, operator controls and runtime/motor emulation.
4. Integrate simulated missions early, with explicit truth-versus-estimated observations and fault tests.
5. Use the vertical-carriage leg stand to identify motor/linkage/contact dynamics, update the model and screen policies before and after each material change.
6. Transfer to the assembled robot through restrained tests, free supervised motion, indoor coverage and bounded field trials with independent ground truth.

Target an integrated simulation prototype in four software weeks and a candidate to begin hardware transfer in six. Physical field demonstration is conditionally targeted about 2–3 weeks after assembled-robot readiness H, if software and hardware gates pass. H and the final deadline are not yet known. See the [timeline](artifacts/project_review_2026-09-04/TIMELINE.svg).

The latest instruction is to review and push this plan, **without launching new runs**. [NEXT_RUNS.md](docs/NEXT_RUNS.md) is preparatory only. Further context updates this mission and the living plan; it does not erase prior evidence or automatically certify a changed system.
