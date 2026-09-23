# Mission

You can reuse the Mid-360 ray pattern and sensor transport/noise prototype under
`sensors/`. CPU tests preserve the transport regressions. The ray-pattern module
requires the pinned Isaac Lab runtime. Navigation uses GPS only, so these
prototypes have no navigation role.

Survey request handling and mission controls remain pending. GeoData owns survey
recording and export. The approved M1 fixture and limits remain in
[ARCHITECTURE](../ARCHITECTURE.md#6-roadmap) as a record; this project schedules
no M1 work.
Historical robot-mounted sensor scenes remain in the archive; their old model
and mount assumptions do not define the approved robot.

```sh
uv run python -m unittest discover -s mission/tests
```
