# Terrain import preparation

This package is opt-in preparation for the terrain stage. It does not register a
PPO environment, change the flat policy, alter the production robot, or launch a
Spark job. The selected full C study robot remains a user-authorized geometry
study: 72.5 mm femur, 126 mm tibia, fixed coxa, transferred CAD mass/inertia and
RS05 limits. Its production CAD fit is still separate work.

## Original fixture physics passed

`tools/validate_terrain_fixtures.py` runs two independent levels:

- `--cpu-only`: opens each actual USDA with OpenUSD, compares its vertices and
  triangle indices against the prepared NPZ, verifies metre/Z-up frames and
  static `physics:approximation = "none"` collision, and checks the start pad,
  negative pit floor and lack of outside-fixture support. All 30 fixtures passed
  this audit. Evidence is in
  `artifacts/terrain_readiness_2026-09-09/cpu_usd_audit_003/validation.json`.
- Isaac mode: imports fixtures into a new scene with no plane, binds explicit
  physics materials, enables PhysX scene queries, and compares an Isaac Lab
  ideal terrain RayCaster against independent triangle intersections and PhysX
  collision queries. Three 10 mm-radius, 50 g spheres per fixture fall onto the
  start pad and selected features. Filtered contact reports, final surface gap
  and vertical motion must agree. The pit-centre sphere must reach its negative
  floor. All 30 original fixtures **passed this runtime check** in Spark attempt
  `terrain_fixture_smoke_003` on 2026-09-09. The 90 probes and 2,430 rays passed;
  maximum ray-height error was 0.000000189 m and maximum absolute settled sphere
  bottom gap was 0.0000319 m over 2.5 seconds of simulated time.

The exact remote evidence is
`/home/orionh/HEXAPOD_runs/mock_length_study_20260909/terrain_fixture_smoke_003/fixtures/validation.json`.
Its frozen source is the sibling `terrain_smoke_source_003` directory; the
executed `tools/validate_terrain_fixtures.py` SHA-256 is
`60a1c9da5f01537492cf64d99d8140e2342d4b03c5a54ed3e96bd08a7f9bde9d`.
The result has `status = completed` and `all_fixture_smokes_passed = true`.
This qualifies fixture collision, ideal rays and probe contacts. Full C standing,
terrain walking, the new mild derivatives below and deployed perception retain
their separate runtime gates.

The default full run creates 30 fixtures, 90 contact probes and 30 terrain ray
sensors, then advances 500 steps at 5 ms: 2.5 seconds of simulated time. Isaac
startup/collision cooking dominate the unknown wall time. This is a short smoke,
not a long training campaign. Run a five-family subset first if startup diagnosis
is more useful than auditing all fixtures at once.

Inside the same Isaac Lab container, with the newly frozen source mounted at
`/workspace/hexapod` and a separate writable output mounted at `/outputs`:

```sh
PYTHONPATH=/workspace/hexapod/tools:/workspace/hexapod/isaaclab \
  /workspace/isaaclab/_isaac_sim/python.sh \
  /workspace/hexapod/tools/validate_terrain_fixtures.py \
  --catalog /workspace/hexapod/artifacts/terrain_readiness_2026-09-09/terrain_catalog.json \
  --fixtures train_smooth_rough_1103,train_ramp_1103,train_step_1103,train_ridge_1103,train_pit_1103 \
  --output /outputs/terrain_fixture_smoke_001 --headless --device cuda:0
```

Omit `--fixtures` for all 30. Use a fresh output directory for every attempt.
The script records failure details before exiting nonzero. A successful result
requires `status = completed` and `all_fixture_smokes_passed = true`; CPU success
or process exit alone must not be relabeled as runtime validation.

The root task owns scheduling. Freeze a new source, use the same existing two GPU
locks and unrelated-workload preflight as Stage2, and queue this after the higher
priority flat-policy work. Do not edit `omni_source_v*`, run beside PPO, or change
forecast restoration ownership. This script deliberately has no SSH, Docker,
systemd, forecasting or GPU-lock management.

## Full robot integration adapter

`fixture_adapter.MildTerrainSpec` stores the proposed 0.05–0.12 m/s and at most
0.20 rad/s mild-terrain command envelope, 1 mm contact offset and explicit
friction. These are pilot settings, not calibrated hardware limits.

`adapt_flat_cfg_for_fixture_smoke` copies an already admitted full-robot config.
The caller must pass the matching standing admission and identity fields
`variant`, `urdf_sha256`, `plan_sha256`, and `stance_index`. Asset path, actuators,
joint names, foot definitions and joint targets are copied; no fixed action index
order or new motor limits are introduced. The adapter accepts only
`purpose="standing_smoke"` and uses one environment. Pits are rejected from this
mild walking adapter; they remain in the import and future avoidance tests.

The existing environment's setup can create the imported terrain through the
adapter's `TerrainImporterCfg`. Its importer applies the physics material
explicitly: the upstream USD importer **does not apply** the configuration's
`physics_material`. Foot contact-point filters are changed from the former
`GroundPlane/CollisionPlane` to the actual `/World/ground/terrain` mesh. Leaving
the old filter would silently lose the contact-location evidence used to separate
feet from tibia shafts.

The start is the fixture's nominal flat pad at world X = -1 m. Body yaw is +90°:
the project's body -Y forward direction then points along course +X. This keeps
the production coordinate convention intact. Validate the complete current C
footprint against the start pad before admission; a point sample under the root
does not establish that all six pads fit on that pad. The adapter does not move
or rescale the original fixture, and resets retain the admitted root height.

Example integration in a new smoke runner after its asset-specific config is
constructed (tools/ and isaaclab/ must both be on PYTHONPATH):

```python
from hexapod_terrain.fixture_adapter import MildTerrainSpec, adapt_flat_cfg_for_fixture_smoke

terrain_cfg = adapt_flat_cfg_for_fixture_smoke(
    admitted_flat_cfg,
    MildTerrainSpec(catalog_path, "train_ramp_1103"),
    admission=standing_admission,
    asset_identity=current_asset_identity,
)
# Instantiate the existing full-robot environment with terrain_cfg, hold its
# admitted named-joint stance, and run the existing contact/torque checks.
```

The adapter suppresses inherited velocity commands, but a subclass such as
`OmniFlatEnv` overrides command sampling. Its caller must explicitly hold zero
commands and zero actions for this smoke. This package does not claim that merely
passing the config to an arbitrary subclass will override that subclass's logic.

### Runnable full C standing check

`tools/validate_terrain_robot.py` now supplies that runner. It requires the exact
full 32-environment/1000-step flat standing admission, its matching study package
and a successful fixture runtime admission. It verifies the original C geometry,
all 18 named stance targets, imported masses/COM/inertia and contact-report APIs
without editing the USD. The exact collision-mesh distal pad footprints were also
checked on CPU: all six fit the fixture's flat start with margin; a deliberately
misplaced starting pose is rejected by the test.

The runner uses `OmniFlatEnv(evaluation=True)` with zero command targets and zero
actions throughout, preserving the current observation/control architecture. It
advances 1000 control steps with the copied named stance and imports the fixture
through the explicit-material adapter. Acceptance requires 19 bodies, 18 joints,
six support contacts after settling, finite observations, zero terminations,
zero nonfoot/base contact, <=0.5% requested torque saturation and the unchanged
1.6 N·m applied torque cap (1.61 numerical tolerance). Output is `state.json`.
This verifies standing/contact integration only; it neither loads a walking
checkpoint nor performs terrain traversal or PPO.

After the fixture run passes, use a new frozen source and mount the existing
admitted study package at `/study`, its flat admission at `/flat-admission.json`,
the fixture `validation.json` at `/fixture-admission.json`, and writable results
at `/outputs`. The exact container invocation is:

```sh
PYTHONPATH=/workspace/hexapod/tools:/workspace/hexapod/isaaclab \
  /workspace/isaaclab/_isaac_sim/python.sh \
  /workspace/hexapod/tools/validate_terrain_robot.py \
  --package /study --admission /flat-admission.json \
  --fixture-admission /fixture-admission.json \
  --catalog /workspace/hexapod/artifacts/terrain_readiness_2026-09-09/terrain_catalog.json \
  --fixture-id train_ramp_1103 --variant f050_t060 --stance-index 0 \
  --steps 1000 --output /outputs/terrain_robot_standing_001 --headless --device cuda:0
```

Required additional source dependencies are the existing `omni_flat_env.py`,
`omni_flat_math.py` (and `omni_diagnostics.py` if the exact plan has overrides),
`tools/c_study_runtime.py`, the complete `experiments/c_length_study/runtime/`
directory and its pinned manifest, `robot/tools/generate_length_study.py`, and
`hexapod_terrain/robot_smoke.py`. Follow the
[C-study frozen-source publication contract](../../experiments/c_length_study/README.md#frozen-source-publication)
and include every vendored source in the campaign manifest. The runner verifies
and binds that runtime before AppLauncher; the production `hexapod_rl` shims
cannot substitute for it. Existing frozen snapshots retain their original bytes.
The `/study` mount must contain the original
`manifest.json`, **matching** `training_plan.json`, its URDF/mesh package and its
previously repaired `training_usd/f050_t060/` with payloads. No frozen inputs are
written. The helper raises if the USD needs repair; use the existing separate
asset preparation pipeline on a new artifact rather than patching it in place.

The installed Spark source was checked read-only: `SimulationCfg` directly owns
`enable_scene_query_support`, and its `physics` field accepts `PhysxCfg`. The
RayCaster configuration exposes `ray_alignment`. The original fixture runner
has passed; the full C standing runner remains unexecuted.

The first frozen fixture attempt (`terrain_smoke_source_001`, output
`terrain_fixture_smoke_001`) did not reach a written fixture audit after several
minutes and was stopped by the root scheduler to return priority to Stage2. The
process had no observed CUDA context. Its exact blocked Python frame is unknown;
do not label that attempt a geometry failure or a passed runtime check. The next
source deferred numpy/helper imports until after SimulationApp, wrote phase state
before startup, and emits a faulthandler traceback every 90 seconds through
startup/audit/runtime. AppLauncher suppresses ordinary stdout during app creation,
so phase messages go to stderr. This instrumentation is diagnostic; the import
order change is not proven to explain that first stall. Attempt 002 reached all
30 meshes and exposed the installed `SensorBase.update(dt, force_recompute=False)`
signature. Removing the unsupported `force_compute` argument and using the
sensor's lazy `data` access produced the successful attempt 003 above.

## Batched support and terrain-relative height queries

`support_queries.TerrainSupportQueries` builds a uniform XY index over the exact
triangle surfaces, then performs chunked Torch vertical intersections. Inputs can
be batched by environment and point; a course translation and yaw explicitly map
world coordinates into each fixture. Sharp steps retain their original triangle
edges. Vertical walls remain in collision geometry and are excluded only from
the downward intersection calculation.

Each result separates actual geometry hits, course containment, avoidance hazards
and eligible support geometry. Outside-course and missing surfaces remain NaN;
an index clamp used for safe lookup never creates a support hit. Pit floors remain
visible at their negative height while the pit footprint is marked ineligible
for walking. `boundary_margin_m` rejects support inside a requested course-edge
margin. Callers must query the entire required footprint or swept envelope;
checking only the root cannot enforce body or foot containment.

`point_clearances` returns signed vertical clearance, preserving negative
penetration. `relative_base_height` returns root Z minus the mean height of the
explicitly required support samples. Every required sample must have eligible
geometry and, when supplied, an affirmative observation-usability mask. One
unknown, stale, pit or out-of-bounds required sample invalidates the whole
reference; zero required samples also returns invalid/NaN. The output includes
the sampled terrain-height span. This is a geometry primitive: a mean reference
does not establish a feasible support plane, force balance or a finished reward.

The interface is teacher terrain truth. A deployed actor still needs the
perception contract's height, usability, age and uncertainty channels; exact
geometry is not silently substituted for a sensor map. Nothing here registers
an actor, task or training loop.

The recorded CPU smoke used 48 fixtures / 405,096 queryable triangles and queried
131,072 points (1,024 × 128). On local arm64 with Torch 2.8.0 and one CPU thread,
three batches took 0.103–0.108 seconds with a 12.44 MB spatial index. These are
CPU preparation measurements, not a CUDA throughput or 50 Hz control claim.
The benchmark records its exact captured source in
`artifacts/terrain_readiness_2026-09-09/mild_curriculum_002/query_cpu_smoke_002.json`
and `support_queries_source.py`. GPU memory, latency and simulator integration
remain to be measured. The 2 cm perception map contract does not imply querying
its full 101 × 101 grid for every robot on every control step.

## Reproducible proposed mild curriculum and resets

The original fixtures' nominal difficulty fields are insufficient to enforce a
local slope limit. Triangle measurements found up to 7.25° on blended ramps
and about 12.4° on the 20 mm rough fixture. `prepare_curriculum.py` therefore
writes separate derivatives with explicit vertex-Z scales, retaining the original
XY coordinates and every triangle. Continuous surfaces are capped at 4.99° before
USDA rounding, leaving margin under 5°. Steps and ridges keep their actual
10–20 mm discontinuities and vertical walls.

| Proposed level | Geometry families | Fixtures | Maximum continuous slope | Translation speed | Yaw rate limit |
| --- | --- | ---: | ---: | --- | ---: |
| 0 | Ramps, smooth roughness at quarter height | 12 | 3.152° | 0.05–0.08 m/s | 0.12 rad/s |
| 1 | Ramps, smooth roughness at half height, capped | 12 | 4.99° | 0.05–0.10 m/s | 0.15 rad/s |
| 2 | Capped ramps/roughness plus steps/ridges | 24 | 4.99° on continuous surfaces | 0.05–0.12 m/s | 0.20 rad/s |

The proposed command contract retains forward/left/yaw, continuous 360° bearings,
pure turns of both signs and mixed translation/turning. Each level proposes 25%
flat replay. There is no automatic promotion rule implemented and no terrain PPO
registration; held-out per-family and per-bearing traversal/stop results and
unchanged hardware limits must determine advancement.

The latest snapshot is
[`mild_curriculum_002/curriculum.json`](../../artifacts/terrain_readiness_2026-09-09/mild_curriculum_002/curriculum.json),
with a separately hashed catalog and 48 USDA/NPZ pairs. All 48 passed an OpenUSD
topology/vertex audit in `usd_audit/validation.json`. These derivatives still
require their own Isaac ray/contact smoke and full robot admission.

Its 1,536 reset poses contain 1,024 training and 512 held-out rows, generated with
independent PCG64 seed streams and explicit split membership. Pits stay in a
separate avoidance catalog and never enter walking reset selection. Every reset
uses the selected full C's exact named stance, current stance root height, zero
joint reset noise, ±20 mm XY jitter and a yaw/quaternion pair sampled over the
full circle. Conservative rectangles enclosing the actual distal collision
vertices of all six feet fit within the inset flat start; the smallest measured
margin among these resets is 58.85 mm. Identity, split, root height, orientation,
complete foot bounds and zero-height support are checked and invalid resets raise
an error. This CPU geometry check does not replace dynamic contact admission.

The course is finite at ±1.5 m in X/Y. Its proposed goal X is 0.85 m with a
60-second evaluation episode. A future reset/task adapter must keep the required
body/foot swept envelope within those boundaries and stop or replan before
unknown support. No backing plane can be introduced to extend this course. A
multi-robot collision atlas and environment-origin implementation remain a gate.

Reproduce into a fresh directory from the repository root:

```sh
PYTHONPATH=tools:isaaclab OMP_NUM_THREADS=1 \
  python3 -m hexapod_terrain.prepare_curriculum \
  --catalog artifacts/terrain_readiness_2026-09-09/terrain_catalog.json \
  --study-package robot/hexapod_mkii_length_study \
  --training-plan artifacts/omni_flat_2026-09-09/inputs/training_plan.json \
  --output /fresh/output --seed 20260909
```

## Gates remaining before terrain PPO

1. The original 30-fixture geometry/ray/contact smoke has passed. Repeat it for
   the separately derived mild assets below, using a fresh source and output.
2. Reuse the exact current full C asset's validated mass/inertia/actuator setup;
   pass the standing admission, then a terrain-start stance check with runtime
   18-joint/19-body/six-foot assertions and observed joint-name order. No test of
   spheres can substitute for this robot gate.
3. Add terrain-relative root/foot clearance and support-height rewards, reset
   heights and bounds. The current flat world-height penalty must not hold the
   body at its original world Z on a ramp. Preserve command-relative tracking and
   actual 1.6 N·m actuator constraints; do not relax flat qualification to promote
   terrain. The config adapter intentionally cannot enable training.
4. Add a tiled collision atlas and explicit environment origins before scaling
   above one robot. The current finite 3×3 m USD does not cover a grid of clones.
   Boundaries must terminate or replan, never be backed by an invisible plane.
5. Exercise reset, history, checkpoint loading and evaluation with the selected
   Stage2 architecture, initially on mild continuous ground and later step/ridge
   cases in both travel directions. Keep held-out fixtures out of training.
6. Keep teacher terrain truth and deployed sensor inputs separate. The present
   RayCaster is an ideal static terrain query and intentionally sees no robot
   links. It does not validate D455/Mid-360 models, motion distortion, occlusion,
   map uncertainty, camera mounting or sensor coverage. Actual sensor trials need
   those moving occluders and the perception agent's observation contract.

CPU regression tests include `robot/tests/test_terrain_fixtures.py`,
`robot/tests/test_terrain_support.py` and `robot/tests/test_terrain_curriculum.py`.
Standard tests
run without OpenUSD; the USD corruption test additionally runs in an isolated
`usd-core` environment. Runtime outputs always retain
`ready_for_terrain_training = false` because fixture import alone is insufficient.

API references checked during preparation:
[Isaac terrain importer source](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/terrains/terrain_importer.py),
[ray-caster configuration](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_cfg.py),
[scene-query setting](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/sim/simulation_cfg.html).
The installed Spark API is the runtime acceptance target; public API review
alone is preparation, while attempt 003 supplies measured fixture evidence.

## Full-C standing launcher and installed-API corrections

`isaaclab/deploy/hexapod-rl terrain-robot-smoke --source FROZEN_SOURCE --output FRESH_OUTPUT` runs only the exact 32×1,000 flat admission and a single-robot 1,000-control hold on the admitted original fixture's flat entry pad. It verifies the pinned C runtime, source, full-review plan, fixture evidence and unchanged per-run robot inputs. It owns both job-scoped locks, yields on coordination/resource changes and cleans up only its own immutable container identity. Forecast scheduling is managed by the separately recorded outer pause/restore procedure. No policy is loaded or trained.

The installed generic USD spawner authors a stronger Xform type over a root-Mesh reference. The adapter now creates the reference as Mesh, preserving exact geometry, collision checks, importer registration and material binding without editing fixture files. It deep-copies the admitted configuration because the installed configclass `copy()` drops controller fields attached after dataclass construction. Regression tests reproduce both problems, check all 30 original meshes, and retain configuration values without shared mutable state. The adapter also uses the installed SDK's XYZW quaternion convention; an independent rotation check verifies an upright robot with body −Y aimed along course +X.

Attempts [001](../../artifacts/terrain_readiness_2026-09-09/runtime/terrain_robot_smoke_001/README.md) and [002](../../artifacts/terrain_readiness_2026-09-09/runtime/terrain_robot_smoke_002/README.md) preserve those failures. Current execution and subsequent results belong in [STATUS](../../STATUS.md). A flat-entry hold does not establish ramp traversal, slope standing or terrain locomotion.


### Contact data must be complete

[Corrected full-C entry attempt 004](../../artifacts/terrain_readiness_2026-09-09/runtime/terrain_robot_smoke_004/README.md) ran without resets but failed its contact gate and logged 2,796 incomplete-contact warnings. Point/friction sensors now reserve a minimum of 128 records per prim, preserving larger configured values. `terrain_contact_evidence.audit_contact_log` hashes the completed log and rejects any reported incomplete contact/friction data before the host can admit either phase. This is an additional evidence check; raw physics gates still apply. The capacity revision needs a new full-C standing run and does not qualify walking or the derived curriculum.
