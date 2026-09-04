# DAR: the requirements and the path to them

Source: `DAR.png` at the repo root (the team lead's "dead ass requirements"
slide, 2026-09-03). This page transcribes it, turns it into testable
statements, and lays out the steps from the current repo to the demo.
`docs/ROADMAP.md` carries the detailed milestones and gates; when the two
disagree, this page states the mission and the roadmap is the one to update.

## 1. What the slide says

```text
Goal:     Make a hexapod robot that can survey a bounded area drawn on the fly.
Inputs:   Team R&D, ELL space, funding $6000.
Sponsors: Gold CU Geodata, Picogrid. Silver (compass mark), Onshape.
          Bronze Tantalus, UPS.

Concept of operations:
  1. User draws a bounded zone on a map.
  2. User places the robot near the region.
  3. Robot steadily traverses the area while providing a steady platform
     for data collection.
```

In plain words: the mission is **cover a user-drawn area steadily**. It is
narrower than the earlier "explore unknown terrain, possibly with an air
unit" framing. Covering a drawn polygon is what robot lawnmowers and farm
robots do; most of the pieces exist off the shelf, and the one genuinely
hard part, a steady walking platform, is the part this repo already works on.

## 2. The numbers the slide does not give

"Steadily" and "steady platform" are not testable until these are written
down. Fill in every blank before the gates below are set. Each line names
who owns the answer.

| Requirement | Blank to fill | Owner |
|---|---|---|
| Area size | Largest polygon to cover in one mission (m x m) | Team lead |
| Terrain | Mown grass / rough grass / gravel / slope up to __ deg / obstacles yes or no | Team lead |
| Mission time | Battery must last __ minutes of walking | MechE + EE |
| Deck steadiness | Roll and pitch within +-__ deg; vertical bounce below __ m/s RMS; yaw wander below __ rad/s | Data-collection sponsor |
| Sweep spacing | Distance between passes = data sensor footprint = __ m | Data-collection sponsor |
| Position accuracy | Robot must know where it is to within __ m | Follows from sweep spacing |
| Payload | Data sensor mass __ kg, size, power, mount | Sponsor + MechE |
| Speed | Minimum useful walking speed __ m/s (area / time) | Follows from the rows above |
| Operator | Draws on a phone / laptop; range to robot __ m | Team lead |

The repo's locomotion gates (`docs/TRAINING.md` section 6) already measure
deck steadiness as a composite score, yaw wander, and speed. Once the blanks
are filled, those thresholds are re-derived from this table, as a written gate
change with a decision record, never silently.

## 3. Where the repo is today

- **Legs.** Walking policies exist for the old mock robot model (Phase-0
  lineage). The real CAD assembly (asset v1, 8.26 kg) is in pull request #9
  under its own task ID and has not been trained on yet.
- **Eyes.** A Livox Mid-360 lidar is on hand. Its mount position is decided
  (upright, deck centre, at least 75 mm above the deck). It is not yet on the
  robot model as a link, and no data has been recorded from it.
- **Brain.** The three-layer architecture (perception, locomotion,
  navigation) and the small contracts between them are written in
  `docs/ARCHITECTURE.md`. The navigation seam exists in code as a
  velocity-command producer. No coverage planner exists yet.
- **Hardware.** No motor driver, runtime, or safety layer has run on the
  physical robot.

## 4. The steps

Each step names the roadmap workstream it belongs to (A asset, B locomotion,
C perception, D navigation, E hardware) and what proves it done.

1. **Walk on the real robot model.** (A, B) Import asset v1 on the Spark, run
   the standing validator, confirm the joint order it prints, retrain
   flat-ground walking for the 8.26 kg body. Done when `validate.py` passes
   and a policy meets the steadiness gate in simulation.
2. **Make steadiness the grade, not speed.** (B) Keep the deck composite as
   the primary objective and set its threshold from section 2. A slow robot
   that does not bounce beats a fast one that does, and slow walking needs
   less motor torque, which is the binding constraint on this robot.
3. **Hardware bring-up.** (E) CAN driver for the 18 RS05 motors, the runtime
   that turns policy output into motor commands, a watchdog, an emergency
   stop, and a restrained-robot test before the first free step. Runs on the
   bench in parallel with step 1.
4. **Know where the robot is.** (C) "Draw on a map, place the robot near it"
   means the robot must match its own position to that map. Standard answer:
   a GPS module for coarse absolute position plus lidar odometry from the
   Mid-360 for smooth position in between. Fallback if GPS is deferred: the
   user draws the zone relative to the robot's starting point on a blank
   canvas. Weaker, but it removes a sensor from the critical path.
5. **The drawing.** (D) A web page on a phone or laptop with a map; the user
   draws a polygon and presses go. The page sends one message to the robot:
   the area of interest (contract C4 in the architecture). Write the message
   down before building the page.
6. **The route.** (D) Turn the polygon into back-and-forth sweep lines and
   walk them. This is coverage path planning; the standard navigation stack
   (ROS 2 Nav2) has a coverage plugin for it. The elevation map from the
   Mid-360 marks where not to step. Output is a stream of small velocity
   commands to the walking policy (contract C1), which already has a seam in
   `packages/hexapod_nav`.
7. **Safety from the start.** (D, E) The drawn polygon is also a fence the
   robot never leaves. Stop when position is uncertain, when the map is
   stale, when tilt exceeds a limit, and on the hardware emergency stop.
8. **Data collection.** (C, D) Decide the deck sensor and what "collecting"
   means. Record only while the deck is inside the steadiness limits, and
   stamp every record with position. This is what the sponsors see.

## 5. The demo ladder

Each rung is a demo for the team lead. Each rung finds problems the previous
one hid. Do not skip rungs.

| Rung | What is shown | Where | Depends on |
|---|---|---|---|
| 1 | Walking policy on asset v1 passing the steadiness gate | Simulation | Step 1, 2 |
| 2 | User draws a polygon; simulated robot sweeps it on flat ground | Simulation | Step 5, 6 |
| 3 | Real robot walks under joystick, restrained then free | Bench, hallway | Step 3 |
| 4 | Draw a polygon; real robot sweeps a flat indoor area | Indoor | Rungs 2, 3, step 4 fallback |
| 5 | Same outdoors on grass with GPS | Field | Step 4 |
| 6 | Data sensor on the deck, recording only when steady | Field | Step 8 |

Rungs 1 and 2 need only the Spark. Rung 2 can start now against the existing
Phase-0 walking policy while step 1 retrains, because the coverage planner
only talks to the policy through velocity commands.

## 6. Budget notes ($6000)

Rough, to be priced by the team:

- Onboard computer (NVIDIA Jetson class): the largest single item.
- GPS module (RTK-capable if sweep spacing demands it).
- Battery, power distribution, emergency-stop hardware.
- Data sensor: sponsor-provided or budgeted here.
- The Mid-360 is already owned. Training compute is the shared Spark.

## 7. What changes in the repo because of this page

- `docs/ROADMAP.md` milestones M2 to M4 were written for exploration and an
  air unit. They are now rewritten around the concept of operations above,
  with section 2 as the source of every numeric gate; the decision record is
  `docs/decisions/0004-mission-is-bounded-area-coverage.md` (proposed, awaiting
  team acceptance).
- Move `DAR.png` under `docs/` next to this page once the team is happy with
  the wording, and keep this page in step with the slide if the slide
  changes.
- The Mid-360 link and payload mass go on asset v1 (already listed as an M0
  follow-up in pull request #9).
