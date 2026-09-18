# Mission

You can reuse the Mid-360 ray pattern and sensor transport/noise prototype under
`sensors/`. CPU tests preserve the transport regressions. The ray-pattern module
requires the pinned Isaac Lab runtime.

Survey request handling, recording and export remain pending. The approved M1
fixture and limits remain in [ARCHITECTURE](../ARCHITECTURE.md#6-roadmap).
Historical robot-mounted sensor scenes remain in the archive; their old model
and mount assumptions do not define the approved robot.

```sh
uv run python -m unittest discover -s mission/tests
```
