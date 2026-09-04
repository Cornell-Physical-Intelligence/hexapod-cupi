# ADR-0004: The mission is steady coverage of an operator-drawn bounded area

Status: proposed (2026-09-03)

## Context

The program documents written on 2026-08-27 (`docs/ROADMAP.md`,
`docs/ARCHITECTURE.md`, ADRs 0001-0003) framed the mission as autonomous
exploration of an area of interest in unknown terrain, possibly designated by
an air unit. On 2026-09-03 the team lead issued the requirements slide kept at
the repo root as `DAR.png` and transcribed in `dar.md`:

```text
Goal: make a hexapod robot that can survey a bounded area drawn on the fly.
Concept of operations: the user draws a bounded zone on a map; the user places
the robot near the region; the robot steadily traverses the area while
providing a steady platform for data collection.
Funding: $6000. Sponsors: CU Geodata, Picogrid, Onshape, Tantalus, UPS.
```

That is a coverage mission, not an exploration mission: the area is known and
bounded before the robot moves, the operator is a person with a map, and the
measured output is a steady data-collection platform, not discovery.

## Decision

The mission of record is the one in `dar.md`. Concretely:

1. **Coverage, not exploration.** Navigation produces a sweep of the drawn
   polygon (coverage path planning) bounded by that polygon as a geofence.
   Frontier exploration and air-unit hand-off leave the roadmap; they may
   return by a later ADR.
2. **Contract C4 is the operator's drawn polygon.** `AreaOfInterest` is
   produced by an operator map page on a phone or laptop, in map coordinates
   (WGS84 with GPS, or a local frame relative to the robot's start when GPS is
   deferred). The message is written down before the page is built.
3. **Steadiness is the primary locomotion grade.** The deck-stability
   composite, yaw wander, and tilt limits are set from the numeric
   requirements in `dar.md` §2 once the team lead fills them in, through a
   written gate change. Speed is a secondary requirement derived from area and
   mission time.
4. **Localization joins perception.** A GPS receiver (RTK if sweep spacing
   demands it) fused with Mid-360 lidar-inertial odometry gives the robot its
   position in the operator's map frame. Contract C3 carries that pose.
5. **Data collection is a deliverable.** Records are taken only while the deck
   is inside the steadiness limits and are stamped with position.

ADRs 0001 (rebuild the asset), 0002 (height scan, not lidar points) and 0003
(ROS 2 for perception and navigation) stand unchanged; they are how this
mission is built.

## Consequences

- `docs/ROADMAP.md` §1 and milestones M2-M4 are rewritten around the concept of
  operations and the demo ladder in `dar.md` §5. Gates that were phrased in
  terms of exploration coverage become sweep-coverage, steadiness-during-
  traversal, and zero-fence-violation gates.
- `docs/ARCHITECTURE.md` §1, §2, §4 (C4), §5 and §6 replace the air unit with
  the operator map page and add GPS to the perception layer.
- The numeric requirement table in `dar.md` §2 is the source for every gate
  threshold added after this date. Until it is filled in, gates are written
  with named blanks, never with guessed numbers.
- The earlier exploration framing stays in the accepted-history of ADRs
  0001-0003 and in `docs/archive/`; nothing is deleted.
