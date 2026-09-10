# Exact 48 mild-fixture Isaac preparation001

This standalone source is ready for a bounded **fixture-only** Isaac check of all 48 existing `mild_curriculum_002` derivatives. It loads no robot, actor, checkpoint or training task. A future pass qualifies these exact mesh/ray/probe imports; it does not qualify C standing, locomotion, perception or Stage 3 completion.

Use **`source_mild_fixtures_001_final`**. Its 105-payload map is `245c20e9318229d59e7132a6bc30b1fd1179460ea2700a2192c1647721a59a91`; host SHA is `15b05814152e219adfeba0cbbb35cdb84fc6e7c98dc174453f92c5f378688fb6`. The earlier unfrozen assembly directory `source_mild_fixtures_001` is preserved as a preparation preview and is not the dispatch source.

The final source contains seven runtime tools, lineage metadata, and 97 exact catalog/geometry files (catalog plus 48 USDA/NPZ pairs). It does not copy the full robot/PPO source or duplicate the original30 fixtures. Catalog SHA is `e494a82c767a368c4105ce3a2a88675b47bede0ea1a88bb91de32b992719fc02`; all ordered IDs and both USDA/NPZ hashes are bound in the contract and preflight. Original30 admission remains separate immutable evidence.

## Scope and unchanged checks

There are 18 smooth-rough, 18 ramp, six step and six ridge derivatives: 32 train and 16 held-out catalog entries. No pit is part of this derivative catalog. The maximum nonvertical triangle slope is 4.99° (vertical step/ridge walls are separately represented), and the highest absolute vertex is 0.25 m. These are source-geometry measurements, not traversability limits.

The harness comes from published main `22957c2062d532f47400f2adb30ef04e257b4b2b`. `run_isaac`, `audit` and geometry helper functions are unchanged. The only harness edits are a 45 s traceback before the 90 s startup deadline, and correcting CLI help from “all30” to “all catalog fixtures.” CPU/USD schema, exact triangle topology, no convexification/ground plane, RayCaster, PhysX ray and physical sphere-probe checks retain their original implementation and limits.

All48 meshes are imported in one scene with 4 m spacing, 48 ideal RayCaster anchors and 144 sphere probes. The harness makes 81 mesh rays and four PhysX scene queries per fixture, including an outside-mesh miss. Probe integration is 500×0.005 s = 2.5 s after the initial query step. The last40 probe steps retain finite contact-force evidence; each probe must pass the existing bottom-gap, contact fraction and vertical-speed checks. The runtime log must also contain no reported incomplete contact/friction data. The host binds and validates all48 ordered CPU/runtime rows, 3,888 mesh rays, 192 PhysX queries and 144 probe results; missing, duplicated, CPU-only, shortened or mismatched results cannot pass.

Material remains nominal friction1, restitution0. The terrain queries are ideal static geometry queries; no camera/LiDAR, self-occlusion, depth noise, sensing lease, robot foot loading or training generalization is measured.

## Ownership and launch

The host is `source_mild_fixtures_001_final/tools/launch_mild_fixtures_spark.py`. Its exact-container lookup and complete cleanup body are AST-identical to the reviewed source009 launcher. It acquires both `/opt/wx/gpu.lock` and `/tmp/hexapod-isaac-gpu.lock`, checks live workloads and the coordination-note hash, and recovers the exact container identity even if the Docker client has exited. Unknown inspect failures remain ownership uncertainty; they cannot be called absence. Source/catalog mount readonly. Source and all97 geometry hashes are checked before and after the phase, including failures; a terminal provenance/log failure cannot exit successfully.

There is exactly one allowed phase (`fixtures`), with a 600 s bound. `TERRAIN_PHASE loading_geometry_helpers` must appear within 90 s, after AppLauncher returns. Output is unbuffered and traceback starts at45 s. The launcher never pauses/restores another workload or starts a successor: root supplies the existing reviewed outer coordination/forecast guard, finite outer deadline and restoration fallback. If cleanup reports uncertainty, root must inspect ownership before restoring a competing workload.

Suggested remote names are `terrain_mild_fixture_source_001` and `terrain_mild_fixture_smoke_001`:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B /home/orionh/HEXAPOD_runs/mock_length_study_20260909/terrain_mild_fixture_source_001/tools/launch_mild_fixtures_spark.py --source /home/orionh/HEXAPOD_runs/mock_length_study_20260909/terrain_mild_fixture_source_001 --output /home/orionh/HEXAPOD_runs/mock_length_study_20260909/terrain_mild_fixture_smoke_001 --isaaclab /home/orionh/IsaacLab
```

Root's preflight loads source/tools on PYTHONPATH and calls `check_source(source)`. This source has 105 files and 97 geometry inputs, not the550 C-study robot assets or the16-file robot runtime. There is no robot runtime admission to check for a sphere/terrain-only job. Root must bind the exact final105-file map and host hash before dispatch. No GPU launch is claimed here.

## CPU verification and expected evidence

`cpu_audit/validation.json` is a fresh actual USD audit of all48 derivatives with the exact final harness hash. It reports `cpu_audit_completed_isaac_pending`, and the host rejects it as physical admission. `INTEGRATION_PREFLIGHT.json` verifies exact105-source/97-geometry identity and the final harness match.

`tests_002.log` records **13 passing CPU tests**, covering complete source/catalog identity, all48 strict result fields, missing/duplicate/incorrect geometry, numeric ray/contact failures hidden behind pass flags, readonly CLI/mode restriction, unchanged physics/cleanup AST, an exited Docker client with a still-running owned container, unknown inspect failure, 90 s AppReady timeout, output-inside-source rejection and terminal provenance/contact failures. The first test log records11 passing tests before adding the two final host-failure checks. Synthetic runtime receipts are labelled test fixtures and confer no physics admission.

```sh
PYTHONDONTWRITEBYTECODE=1 uv run python -B -m unittest discover -s tmp/terrain_mild_fixture_adapter_001 -p 'test_mild_fixtures.py' -v
```

Expected runtime artifacts are `campaign.json`, `jobs/fixtures.json`, contact-log audit, `logs/fixtures.log`, `inputs/fixture_before.sha256.json` and `fixtures/validation.json`, including exact CPU/runtime rows and failure traceback where available. Root's terminal wrapper adds all raw hashes, source/geometry verification, exact owned-container absence and forecast restoration. Preserve this preparation and the old30/raw failures; do not rewrite the catalog's pending status to imply locomotion qualification.
