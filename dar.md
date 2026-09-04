# DAR: the requirements and the path to them

Source: `DAR.png` at the repo root, the team lead's "dead ass requirements"
slide (2026-09-03). This page transcribes the slide, turns it into testable
statements, and lists the steps from the current repo to the demo.
`docs/PLAN.md` carries the detailed milestones and gates. If the two disagree,
this page states the mission and you update the plan.

## 1. The slide text

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

The mission is **cover a user-drawn area with a steady deck**. It is narrower
than the earlier "explore unknown terrain, possibly with an air unit" framing.
Robot lawnmowers and farm robots cover drawn polygons today, and most of the
pieces exist off the shelf. The hard part is a steady walking platform, and
this repo already works on that part.

## 2. The numbers the slide does not give

"Steadily" and "steady platform" become testable after the team writes these
numbers down. The team lead answered on 2026-09-04. Three rows stay open.

| Requirement | Blank to fill | Answer (2026-09-04) | Status |
|---|---|---|---|
| Area size | Largest polygon to cover in one mission | A few dozen feet on a side, about 10 m x 10 m | Set |
| Terrain | Surface class, slope, obstacles | Start on a path. Aim for forest later; forest is too hard as a first target | Set for M1/M2; forest deferred |
| Mission time | Minutes of walking per battery | Not a constraint now; optimize later | Deferred |
| Deck steadiness | Roll and pitch, vertical bounce, yaw wander limits | Minimize. The lead expects software compensation to absorb much of it. Orion is thinking about limits | Open (Orion) |
| Sweep spacing | Distance between passes, from the data sensor footprint | To find by experiment in sim and in the field. Orion will report back | Open (Orion) |
| Position accuracy | Localization error the sweep tolerates | Derived from sweep spacing once that row closes. RTK GNSS is out of budget, so a standard GNSS fix (metres) anchors the polygon and lidar-inertial odometry carries position inside it | Derived; RTK excluded |
| Payload | Data sensor mass, size, power, mount | Under 1 to 2 kg. Mass is not a constraint now because the robot needs a weight reduction anyway | Set (bound only) |
| Speed | Minimum useful walking speed | Not a constraint now; make it faster later | Deferred |
| Operator | Medium and range | Medium is open. Range is long, so LoRa (or cellular) for telemetry | Set (long range) |

Consequences for the gates:

- The deck-stability composite stays a relative grade. The Stage2C thresholds
  in `docs/TRAINING.md` §6 stand until Orion sets absolute limits. A written
  gate change then re-derives them.
- Speed and mission time carry no gate. The current `0.240 m/s` floor is a
  Stage2C artifact and does not come from this table.
- Terrain for M1 is flat ground and for M2 is a packed path with a stated
  slope. Forest enters the plan only after a field sweep on a path.
- Position comes from lidar-inertial odometry inside a 10 m square. A
  standard GNSS fix is good to a few metres and cannot hold sweep lines on
  its own, so it anchors the drawn polygon to the world and no more. The
  start-anchored local frame in `docs/PLAN.md` §6 P1 is the fallback.
- The operator link is a long-range radio. This closes the medium part of
  hardware decision 4 in `docs/PLAN.md` §2 toward LoRa or cellular.
- The mechanical team plans a weight reduction. Asset v1 masses (8.26 kg)
  will change, so the M0 mass gate compares against whichever robot exists at
  measurement time and says which.

Reference mission the lead pointed to: Rodriguez-Sanchez, Johnsen, and Li,
"A Ground Mobile Robot for Autonomous Terrestrial Laser Scanning-Based Field
Phenotyping" (arXiv:2404.04404). It uses RTK-GNSS plus sensor fusion for
localization and route optimization over crop trials. This robot replaces
RTK with lidar-inertial odometry.

## 3. The repo today

- **Legs.** Walking policies exist for the old mock robot model (Phase-0
  lineage). The real CAD assembly (asset v1, 8.26 kg) is merged under its
  own task ID. Nobody has trained on it yet.
- **Eyes.** The team owns a Livox Mid-360 lidar and has chosen its mount
  position (upright, deck centre, at least 75 mm above the deck). The robot
  model does not carry it as a link yet, and nobody has recorded data from it.
- **Brain.** `docs/PLAN.md` §1 defines the three-layer architecture
  (perception, locomotion, navigation) and the small contracts between the
  layers. The navigation seam exists in code as a velocity-command producer.
  No coverage planner exists yet.
- **Hardware.** No motor driver, runtime, or safety layer has run on the
  physical robot.

## 4. The steps

Each step names its workstream (A asset, B locomotion, C perception, D
navigation, E hardware) and the evidence that closes it.

1. **Walk on the real robot model.** (A, B) Import asset v1 on the Spark, run
   the standing validator, confirm the joint order it prints, and retrain
   flat-ground walking for the 8.26 kg body. Done when `validate.py` passes
   and a policy meets the steadiness gate in simulation.
2. **Grade steadiness first.** (B) Keep the deck composite as the primary
   objective and set its threshold from section 2. A slow robot with a steady
   deck serves the mission. Slow walking also needs less motor torque, and
   torque is the binding constraint on this robot.
3. **Hardware bring-up.** (E) CAN driver for the 18 RS05 motors, the runtime
   that turns policy output into motor commands, a watchdog, an emergency
   stop, and a restrained-robot test before the first free step. This runs
   on the bench in parallel with step 1.
4. **Know where the robot is.** (C) "Draw on a map, place the robot near it"
   requires the robot to match its own position to that map. The standard
   answer is a GPS module for coarse absolute position plus lidar odometry
   from the Mid-360 for smooth position in between. If the team defers GPS,
   the user draws the zone relative to the robot's starting point on a blank
   canvas. That option is weaker and removes a sensor from the critical path.
5. **The drawing.** (D) A web page on a phone or laptop with a map. The user
   draws a polygon and presses go. The page sends one message to the robot:
   the area of interest (contract C4 in the plan). Write the message down
   before you build the page.
6. **The route.** (D) Turn the polygon into back-and-forth sweep lines and
   walk them. This is coverage path planning, and the standard navigation
   stack (ROS 2 Nav2) has a coverage plugin for it. The elevation map from the
   Mid-360 marks where the robot must not step. The output is a stream of
   small velocity commands to the walking policy (contract C1), which already
   has a seam in `packages/hexapod_nav`.
7. **Safety from the start.** (D, E) The drawn polygon is also a fence the
   robot stays inside. Stop when position is uncertain, when the map is
   stale, when tilt exceeds a limit, and on the hardware emergency stop.
8. **Data collection.** (C, D) Decide the deck sensor and define a valid
   record. Record only while the deck is inside the steadiness limits, and
   stamp each record with position. This is the sponsors' deliverable.

## 5. The demo ladder

Each rung is a demo for the team lead. Each rung exposes problems the earlier
rung could not show. Do not skip rungs.

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
talks to the policy only through velocity commands.

## 6. Budget notes ($6000)

Rough list for the team to price:

- Onboard computer (NVIDIA Jetson class): the largest single item.
- GPS module, standard GNSS. RTK is out of budget.
- Battery, power distribution, emergency-stop hardware.
- Data sensor: sponsor-provided or budgeted here.
- The team already owns the Mid-360. Training compute is the shared Spark.

## 7. Repo changes that follow from this page

- `docs/PLAN.md` milestones M2 to M4 follow the concept of operations above,
  with section 2 as the source of each numeric gate. The decision record is
  ADR-0004 in `docs/PLAN.md` §7 (proposed, awaiting team acceptance).
- Move `DAR.png` under `docs/` next to this page after the team approves the
  wording, and keep this page in step with the slide if the slide changes.
- The Mid-360 link and payload mass go on asset v1 (an M0 item in
  `docs/PLAN.md` §6).
