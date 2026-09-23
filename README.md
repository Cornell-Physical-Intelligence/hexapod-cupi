# Hexapod

You develop a robot that walks an approved route through an operator-drawn
region and stops at designated locations. GeoData owns survey data collection
outside this repository. You can inspect the approved robot in the
[viewer](robot/hexapod_mkii_updated_v1/README.md#inspection) and review recordings
on the [progress site](https://cornell-physical-intelligence.github.io/hexapod-cupi/).

The consolidated simulation foundation supports stock PPO and an optional
trajectory optimizer. [STATUS](https://cornell-physical-intelligence.github.io/hexapod-cupi/#findings) lists measured results; run
`python3 tools/project_site.py status` for a local, untracked copy. Paper-method
implementation remains ahead of you in
[TRAINING](docs/TRAINING.md#proposed-reproduction-sequence).

| Directory | Maintained scope |
| --- | --- |
| `robot/` | Approved direct-drive model and hash-pinned simulation inputs. |
| `locomotion/` | Simulation, reward, stock PPO, evaluation and guarded launch; optional optimizer in `priors/`. |
| `contracts/` | Velocity commands, planar poses and checkpoint identity fields. |
| `navigation/` | Waypoint follower example; planner and localization remain pending. |
| `mission/` | Sensor-pattern and transport prototypes; mission controls remain pending. GeoData owns survey recording and export. |
| `viewer/` | Joint and part inspection of the approved robot. |
| `tools/`, `configs/` | Asset preparation, source verification and archive/publication commands. |
| `docs/`, `site/` | Reference procedures and the published progress record. |

Read [ARCHITECTURE](ARCHITECTURE.md) for requirements and boundaries,
[CONTRIBUTING](CONTRIBUTING.md) for setup and checks, and [AGENTS](AGENTS.md)
for agent constraints. Start walking work with the [kernel code map](locomotion/README.md).
Read a reference below when its subject is part of your task.

| Reference | Maintained purpose |
| --- | --- |
| [Training](docs/TRAINING.md) | Kernel contract and the paper-reproduction roadmap. |
| [Operations](docs/OPERATIONS.md) | Host access, ownership, launch procedure and recovery controls. |
| [Compute coordination](docs/SPARK_COMPUTE_COORDINATION.md) | Current reservation policy; release requires James's direction. |
| [Progress publication](docs/PROJECT_SITE.md) | Registry updates and Pages publication. |
| [Source lineages](docs/PIPELINE_LINEAGES.md) | Source manifests and archive restoration. |
| [Updated CAD import](docs/UPDATED_CAD_IMPORT.md) | Approved direct-drive geometry, joint conventions and import review. |
| [RS05 review](docs/RS05_SPEC_REVIEW.md) | Motor specification provenance and unmeasured hardware assumptions. |
| [Leg stand](docs/LEG_STAND_HARDWARE.md) | Component selection, wiring and open calibration requirements. |

Git history retains superseded plans and handoffs. Load them to answer a specific
historical question; their instructions do not authorize current work.

Use the [open issues](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/issues)
for assignments. Keep new run payloads outside this checkout and selected public
media in `site/assets/`. The [archive guide](docs/PIPELINE_LINEAGES.md) explains
restoration from Git. A shallow clone avoids downloading retired payload history.
