# Navigation

You can exercise the existing `WaypointFollower` with `Pose2D` and an explicit
command envelope. Its default example limits preserve the historical demo;
they qualify no policy. The follower has no obstacle avoidance. Position will come
from onboard GPS only (ARCHITECTURE R-03); the pose estimate remains pending.
Navigation imports shared contracts, not simulation.

## Route planning

`region.py`, `grid.py`, `graph.py` and `planner.py` hold the route-planning seam for
[#20](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/issues/20) task 3.
`CostGraph` in `graph.py` is the interface a planner searches and the contract other
mission work builds against; `GridGraph` is its eight-connected implementation over a
`CostGrid`, which rasterises a `SurveyRegion` inflated by a `PlanMargin`.

**The algorithms are not implemented yet.** Seven method bodies raise
`NotImplementedError` behind a `CHALLENGE` comment stating the required result, and 42
of the tests in `tests/` fail until they are written. Planning therefore still remains
pending; what exists is the scaffold, the validation and the specification.

Two properties the tests enforce that are easy to miss. The heuristic must be scaled
by `CostGrid.min_cell_cost` or it stops being admissible as soon as cells carry
different costs, which yields plausible but suboptimal routes. And the tie-break must
be a total order on the node coordinates rather than on discovery order, because an
operator approves a route before motion (R-01) and the same request has to produce the
same route.

`PlanMargin` carries R-05's clearance as three separate terms, one of which is R-03's
GPS position error bound. It is a **per-plan input, never a constant**: a rehearsal on
ideal poses passes `0.0`, and the plain-GPS bound of several metres leaves no feasible
region inside #20's 6 x 4 m area at all. `CostGrid.passable_cell_count` returning zero
is that result, reported rather than raised.

```sh
uv run python -m unittest discover -s navigation/tests
```
