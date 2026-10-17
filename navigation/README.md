# Navigation

You can exercise the existing `WaypointFollower` with `Pose2D` and an explicit
command envelope. Its default example limits preserve the historical demo;
they qualify no policy. The follower has no obstacle avoidance. Position will come
from onboard GPS only (ARCHITECTURE R-03); the pose estimate remains pending.
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
shortest in meters. `CostGraph` in `graph.py` is the interface a planner searches and
the contract other mission work builds against; `GridGraph` is its eight-connected
implementation over an `OccupancyGrid`, with octile distance as the heuristic and no
diagonal cutting between blocked cells.

The margin holds at cell centers only approximately. `buffer` approximates rounded
corners with straight segments, so a feasible point can fall short of the margin by
up to about 1% of it. Straight steps between passable centers are not checked, so a
step can pass closer still: up to about `cell_size² / (4 × margin)`, which is about
3 cm at 0.25 m cells and 5 mm at 0.1 m. These are accepted because they are far
smaller than the GPS and stop-tolerance terms in `PlanMargin`. If a route must keep
the margin exactly, add the shortfall as a term in `PlanMargin` rather than changing
the geometry.

**`a_star` in `planner.py` is not implemented yet.** It raises `NotImplementedError`,
and its docstring states the required behavior; the tests in `tests/test_planner.py`
that call it fail until it is written.

One property the tests enforce that is easy to miss: the tie-break must be a total
order on the node coordinates rather than on discovery order, because an operator
approves a route before motion (R-01) and the same request has to produce the same
route.

`PlanMargin` carries R-05's clearance as three separate terms, one of which is R-03's
GPS position error bound. It is a **per-plan input, never a constant**: a rehearsal on
ideal poses passes `0.0`, and the plain-GPS bound of several meters leaves no feasible
region inside #20's 6 x 4 m area at all. `OccupancyGrid.passable_cell_count` returning zero
is that result, reported rather than raised.

```sh
uv run python -m unittest discover -s navigation/tests
```
