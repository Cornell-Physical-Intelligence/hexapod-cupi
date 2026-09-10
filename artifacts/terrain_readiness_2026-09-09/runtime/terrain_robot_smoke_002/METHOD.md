# Terrain standing retry 002

This attempt preserves failure001 and uses a new source, new output, new flat
admission and new forecasting pause. It remains a standing/contact smoke. No
policy, terrain traversal, derived-fixture admission or production physical
four-bar qualification is implied.

## Minimal import correction

The installed SDK source confirms the failure chain:

1. `TerrainImporter.import_usd()` constructs `UsdFileCfg` and calls its generic
   spawner.
2. `_spawn_from_usd_file()` calls `create_prim()` without a prim type.
3. `create_prim(prim_type="Xform")` authors the stronger local Xform type and
   then adds the USD reference. The fixture's default prim is itself a Mesh;
   that type is overridden while its CollisionAPI remains present.

The old scene therefore contains a non-Mesh collider, which the existing guard
correctly rejects. This is reproduced independently in OpenUSD for every one of
the 30 original fixture files.

`fixture_adapter.patch` adds a small `reference_fixture_mesh` helper that authors
a **Mesh** reference in the new scene and preserves the terrain importer's path
registration. Existing material binding, exact collision guard, contact filters
and every robot/admission threshold are unchanged. No referenced USD is edited.

The portable regression file `test_terrain_fixture_reference.py` belongs under
`robot/tests/` when root publishes the patch. Two OpenUSD tests passed against the
new source. They reproduce the old failure for all 30 originals, compare every
point and triangle after the corrected reference, verify source bytes are
unchanged, and retain duplicate-prim and convexification rejection. Environments
without OpenUSD explicitly skip these composition tests.

Installed files inspected read-only are under
`/home/orionh/IsaacLab/source/isaaclab/isaaclab/`:
`terrains/terrain_importer.py`, `sim/spawners/from_files/from_files.py`, and
`sim/utils/prims.py`. This uses the installed API as the acceptance target.

## Source and dispatch

The new remote source is
`/home/orionh/HEXAPOD_runs/mock_length_study_20260909/terrain_robot_source_002`.
Its complete 660-file manifest SHA-256 is
`e969141e7afac6e44963a50e0bd87c1d3f4095a4856644f92b75642f3724cfad`.
The local byte-matching source is `source/`. It copies immutable source001,
changes only the adapter and provenance, and adds the regression test. The pinned
16-file runtime, exact full-review plan, robot, original 30-fixture catalog and
attempt003 fixture admission are unchanged. All 660 manifested files match their
hashes. Host Python also created four `tools/__pycache__/*.pyc` cache files when
the host coordinator imported its helpers; these are preserved and are not
source-byte changes or evidence inputs. Future host launch commands will disable
bytecode writes as the simulator container already does.

The service is `hexapod-terrain-robot-smoke-002-20260909.service`, with output
`/home/orionh/HEXAPOD_runs/mock_length_study_20260909/terrain_robot_smoke_002`.
It repeats the complete 32-environment × 1,000-step flat admission, followed only
on success by the single full C robot's 1,000-step terrain-entry standing check.
The source is read-only; only a fresh per-run asset copy is writable during the
flat validator's existing inertia authoring. Ten-minute phase bounds, 22-minute
service bound, both job-scoped locks, source hashes, checkpoint-free commands,
resource/coordination checks and exact-container cleanup remain unchanged.

The initial retry dispatch correctly failed before a pause or robot output was
created because StormScope's restored timer had started another dispatch. The
identified service MainPID was 3158697 and its CUDA descendant 3159101. The user
and root task explicitly authorize pausing these identified forecasting services.
`launch_guarded_remote_v2.py` verifies CUDA ancestry before pausing; it never
signals an unrelated process. It records named-service state and graceful stop
semantics, arms recovery, stops the previously active timers and uses the named
service's normal SIGTERM/control-group stop. Existing outputs are preserved.

Pause014 has normal service-exit restoration and an independent 30-minute
fallback. Both locks and idle-workload checks must pass after the authorized stop
before any Isaac container starts. Only previously active timers are restored;
the interrupted forecast is eligible for later scheduler retry, not claimed to
have completed or resumed at an internal checkpoint. Previous pause013 and its
successful restoration remain preserved.

The dispatched outer command is:

```sh
ssh spark python3 /home/orionh/HEXAPOD_runs/mock_length_study_20260909/terrain_robot_smoke_002_preparation/launch_guarded_remote_v2.py
```

Do not rerun with the same labels. Outputs follow attempt001's layout:
`campaign.json`, `jobs/{flat,terrain}.json`, `logs/{flat,terrain}.log`,
`flat/{state,admission}.json`, `terrain/state.json` and pre/post flat asset hashes.
The result of the new runtime check must be recorded separately; CPU success
does not admit this robot on terrain.

## Measured outcome

The fresh flat32×1000 gate passed again with exactly the same metrics as attempt001
and no asset changes. Its admission SHA-256 is
`98b22e8bf7114cce9dcaf667f9a4b91f0ccf922c17ed2ac008f63bd7c452070b`.
The corrected reference passed terrain import and initialized the full robot's
19 bodies, 18 named joints and six feet. It then failed at reset, before stepping,
because `cfg.copy()` dropped `omni_observation_noise_scale`, a dynamically attached
controller field. The installed configclass `_copy_class` calls
`dataclasses.replace`, which only preserves declared fields. A separate minimal
copy correction and fresh runtime admission are still required.

All manifested source files and admitted robot inputs are preserved. The failed
service exited, no owned container or CUDA process remained, both locks were
free, and pause014 restored both StormScope timers to active/waiting at Unix
1789001391.2247028. Local evidence is in `results/run/` and
`results/forecast_pause_014/`. No full-C terrain standing pass is claimed.
