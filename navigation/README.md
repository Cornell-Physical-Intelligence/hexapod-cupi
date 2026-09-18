# Navigation

You can exercise the existing `WaypointFollower` with `Pose2D` and an explicit
command envelope. Its default example limits preserve the historical demo;
they qualify no policy. The follower has no obstacle avoidance or localization.
Planning remains pending. Navigation imports shared contracts, not simulation.

```sh
uv run python -m unittest discover -s navigation/tests
```
