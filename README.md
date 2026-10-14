# Hexapod

We are building a six-legged robot that walks a planned route across an area
drawn on a map, finds its way with GPS alone and stops at each survey point.
GeoData collects the survey data; this repository covers the robot.

**Where we are (12 October 2026):** in simulation, one training run of our
learned walking controller passed all 13 movement tests: walking in eight
directions, turning both ways, standing still and stopping. We are repeating
that run in four controlled training runs to check that the result holds. The
[progress site](https://cornell-physical-intelligence.github.io/hexapod-cupi/)
shows the videos and the roadmap, and the [viewer](robot/hexapod_mkii_updated_v1/README.md#inspection)
shows each part and joint of the approved robot.

## What is here

| Directory | Contents |
| --- | --- |
| `robot/` | The approved robot model: 18 motors, 7.47 kg, with hash-pinned simulation inputs. |
| `locomotion/` | Simulation, rewards, training, the movement tests and the guarded Spark launcher. |
| `navigation/` | A waypoint follower example; the route planner and GPS position come later. |
| `contracts/` | Shared types: speed and direction commands, planar poses, checkpoint identity. |
| `mission/` | Sensor-pattern and transport prototypes. |
| `viewer/` | The browser viewer for the robot's parts and joints. |
| `site/`, `docs/`, `tools/`, `configs/` | The progress site, guides, checks and archive records. |

`navigation/` and `mission/` do not import simulation code.

## Guides

- [Contributing](CONTRIBUTING.md): setup and the checks we run before each push.
- [Architecture](ARCHITECTURE.md): requirements and system boundaries.
- [Locomotion code map](locomotion/README.md) and [training](docs/TRAINING.md).
- [Spark](SPARK.md) and [operations](docs/OPERATIONS.md): jobs on the shared GPU server.
- [Progress site](docs/PROJECT_SITE.md): how we update the page.
- Hardware: [CAD import](docs/UPDATED_CAD_IMPORT.md), [RS05 motors](docs/RS05_SPEC_REVIEW.md) and the [leg stand](docs/LEG_STAND_HARDWARE.md).
- [Source lineages](docs/PIPELINE_LINEAGES.md): retired files and how to restore them. A shallow clone skips their history.

We track assignments in [GitHub issues](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/issues).
