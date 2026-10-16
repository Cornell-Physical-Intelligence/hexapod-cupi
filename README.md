# Hexapod

We are building a six-legged robot that walks a planned route across a mapped
area and stops at each survey point. It finds its way with GPS alone.

**Where we are (12 October 2026):** in simulation, one training run of our
learned walking controller passed all 13 movement tests: walking in eight
directions, turning both ways, standing still and stopping. We are training that
setup twice more, and twice without its new reward terms for comparison, to
check that the result holds and that the new terms cause it. The
[progress site](https://cornell-physical-intelligence.github.io/hexapod-cupi/)
shows the videos and the roadmap, and the [viewer](robot/hexapod_mkii_updated_v1/README.md#inspection)
shows each part and joint of the approved robot.

## Contents

| Directory | Contents |
| --- | --- |
| `src/locomotion/` | Simulation, rewards, training, the movement tests and the guarded Spark launcher. |
| `src/navigation/` | A waypoint follower example; the route planner and GPS position come later. |
| `src/contracts/` | Shared message types, such as speed and direction commands and planar poses. |
| `src/mission/` | Sensor-pattern and transport prototypes. |
| `robot/` | The approved robot model (18 motors, 7.47 kg) and its hash-pinned simulation inputs. |
| `viewer/` | The browser viewer for the robot's parts and joints. |
| `site/`, `docs/`, `tools/`, `configs/`, `tests/` | The progress site, guides, checks, archive records and repository tests. |

`src/` holds the importable Python packages; `uv sync` installs them in editable
mode, so `uv run python -m locomotion...` works from any directory inside the checkout.
`navigation` and `mission` do not import `locomotion`. The Mid-360 ray
pattern in `src/mission/sensors/` needs the Isaac Lab runtime.

## Guides

- [Contributing](CONTRIBUTING.md): setup and the checks we run before each push.
- [Architecture](ARCHITECTURE.md): requirements and system boundaries.
- [Locomotion code map](src/locomotion/README.md) and [training](docs/TRAINING.md).
- [Spark](SPARK.md) and [operations](docs/OPERATIONS.md): jobs on the shared GPU server.
- [Progress site](docs/PROJECT_SITE.md): how we update the page.
- [CAD import](docs/UPDATED_CAD_IMPORT.md) and [RS05 motors](docs/RS05_SPEC_REVIEW.md): the approved model and its motors.
- [Leg stand](docs/LEG_STAND_HARDWARE.md): the single-leg test stand.
- [Source lineages](docs/PIPELINE_LINEAGES.md): retired files and how to restore them. A shallow clone skips their history.

We track assignments in [GitHub issues](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/issues).
