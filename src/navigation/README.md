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
on; reaching a goal or tripping the heading gate ends a hold. The heading gate has hysteresis (turn from 0.05 rad, release at 0.01 rad), and a goal is retired at 0.06 m, inside the 0.12 m stop tolerance by the body's stopping distance. The follower has no obstacle avoidance. Position will come
from onboard GPS only ([navigation contract](../../ARCHITECTURE.md#navigation-and-commands)); the pose estimate remains pending.
Navigation imports shared contracts, not simulation.

## Route planning

`region.py`, `grid.py`, `graph.py` and `planner.py` hold the route-planning seam for
[#20](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/issues/20) task 3.
Geometry is [Shapely](https://shapely.readthedocs.io/)'s. A `SurveyRegion` holds
`shapely.Polygon`s for the boundary, exclusions and optional corridor, and rejects
invalid (self-crossing) drawings on construction. `SurveyRegion.feasible(margin)`
shrinks the free space by the `PlanMargin` with a negative `buffer`, and
`OccupancyGrid.from_region` marks a cell passable when its center lies inside that
shape. Every passable cell is equally good: there is no risk weighting, so routes are
shortest in meters. `CostGraph` in `graph.py` is the interface `a_star` searches; `GridGraph` is its eight-connected
implementation over an `OccupancyGrid`, with octile distance as the heuristic and no
diagonal cutting between blocked cells.

The margin holds at cell centers only approximately. Shapely's `buffer` approximates rounded
corners with straight segments, so a feasible point can fall short of the margin by
up to about 1% of it. Straight steps between passable centers are not checked, so a
step can pass closer still: up to about `cell_size² / (4 × margin)`, which is about
3 cm at 0.25 m cells and 5 mm at 0.1 m. When considering whether these tolerances are /
aren't acceptable, note that these may be much smaller than the GPS and stop-tolerance
terms in `PlanMargin` (with the GPS tolerance in particular being on a scale of multiple
meters). If a route must keep the margin exactly, add the shortfall as a term in `PlanMargin` rather than changing
the geometry.

`a_star` in `planner.py` searches one start cell to one goal cell and returns a
`Route`, or `None` when there is no route, including when either cell is blocked or
off the grid. The caller decides what a blocked start or goal means. Chaining routes through the stop
points remains pending.

One property the tests enforce that is easy to miss: the tie-break must be a total
order on the node coordinates rather than on discovery order, because an operator
approves a route before motion and the same request has to produce the same route.

`PlanMargin` carries the route clearance as three separate terms, one of which is the
GPS position error bound. It is a **per-plan input, never a constant**: a rehearsal on
ideal poses passes `0.0`, and the plain-GPS bound of several meters leaves no feasible
region inside #20's 6 x 4 m area at all. `OccupancyGrid.passable_cell_count` returning zero
is that result, reported rather than raised.

```sh
uv run python -m unittest discover -s src/navigation/tests
```
