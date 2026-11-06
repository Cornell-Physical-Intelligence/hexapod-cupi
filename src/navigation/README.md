# Navigation

`WaypointFollower` turns `Pose2D` goals into commands from the omniwalk
training bank only: eight translation bearings at 45 degree spacing at 0.025
or 0.05 m/s, pure yaw at ±0.2 rad/s, or stand. It holds the body heading at a
multiple of 45 degrees in the map frame and drives holonomically, so the
eight-connected grid routes of the planner map onto bank bearings; a test
asserts the navigation-side bank equals `command_bank(TaskConfig())`. Its
default envelope is the bank's per-axis extremes and a consumer may supply a
narrower one. A translation command is held for the training minimum of 2 s
before it is chosen again, so the policy sees the hold lengths it was trained
on; reaching a goal or tripping the heading gate ends a hold. The follower has no obstacle avoidance. Position will come
from onboard GPS only ([navigation contract](../../ARCHITECTURE.md#navigation-and-commands)); the pose estimate remains pending.
Planning remains pending. Navigation imports shared contracts, not simulation.

```sh
uv run python -m unittest discover -s src/navigation/tests
```
