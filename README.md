# Hexapod

You develop a robot that surveys an operator-drawn region and stops to collect
measurements for a 3D terrain map. You can inspect the approved robot in the
[viewer](robot/hexapod_mkii_updated_v1/README.md#inspection) and review recordings
on the [progress site](https://cornell-physical-intelligence.github.io/hexapod-cupi/).

The consolidated simulation foundation supports stock PPO and an optional
trajectory optimizer. The recorded forward trajectory passes its screen;
the tested PPO policies fail walking acceptance. See [STATUS](STATUS.md) for
measured results. Paper-method implementation remains ahead of you in
[TRAINING](docs/TRAINING.md#proposed-reproduction-sequence).

| Directory | Maintained scope |
| --- | --- |
| `robot/` | Approved direct-drive model and hash-pinned simulation inputs. |
| `locomotion/` | Simulation, reward, stock PPO, evaluation and guarded launch; optional optimizer in `priors/`. |
| `contracts/` | Velocity commands, planar poses and checkpoint identity fields. |
| `navigation/` | Waypoint follower example; planner and localization remain pending. |
| `mission/` | Sensor-pattern and transport prototypes; survey recording/export remain pending. |
| `viewer/` | Joint and part inspection of the approved robot. |
| `tools/`, `configs/` | Asset preparation, source verification and archive/publication commands. |
| `docs/`, `site/` | Reference procedures and the published progress record. |

Read [ARCHITECTURE](ARCHITECTURE.md) for requirements and boundaries,
[CONTRIBUTING](CONTRIBUTING.md) for setup and checks, and [CLAUDE](CLAUDE.md)
for agent constraints. Use the [reference index](docs/README.md) for a specific
procedure. Start walking work with the [kernel code map](locomotion/README.md).

Use the [open issues](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/issues)
for assignments. Keep new run payloads outside this checkout and selected public
media in `site/assets/`. The [archive guide](docs/PIPELINE_LINEAGES.md) explains
restoration from Git. A shallow clone avoids downloading retired payload history.
