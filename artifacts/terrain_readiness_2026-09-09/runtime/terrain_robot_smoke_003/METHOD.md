# Terrain standing attempt003

This source keeps failed attempts001/002 immutable and adds one small correction
to attempt002's typed-Mesh import adapter: `deepcopy(cfg)` replaces `cfg.copy()`.
The installed Isaac configclass implementation uses `dataclasses.replace` for
its copy method. That loses fields attached by `configure_omni` after instance
construction, including observation noise, target-filter state and reward
weights. The copy now retains those settings and nested independence. No reward,
actuator, geometry, observation or acceptance value changes.

`installed_copy_reproduction.json` records the actual installed SDK file hash
and executes its `_copy_class` function on CPU to reproduce the missing field.
The new portable `test_terrain_config_copy.py` extracts the real published
`configure_omni`, runs the terrain adapter and checks all configured fields,
actuator values, nested independence and contact filters. That test and the two
OpenUSD reference tests passed. `config_copy.patch` is the minimal difference
from attempt002; `fixture_adapter.py` is the final adapter containing both fixes.
Root owns main integration and Git publication; this preparation has not edited
main or production assets.

Source: `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/terrain_robot_source_003`.
Its 661-file source manifest SHA-256 is
`b6e2767475e514977d9ac26dbe50e408fe35f45a087c8908f91cb0770ce479bf`.
The byte-matching local copy is `source/`. It retains the published
`9d6107794c8d329dded8997b6c88f7ea0d8695dd` study code, pinned 16-file runtime,
full-review plan `6a234f2b1ffd4f30ba4470b5ebb806cb2964fa0cdc85f6e4433976749c98af2a`,
original 30-fixture catalog and its passed attempt003 fixture admission.
No policy checkpoint or new PPO code is loaded.

Service: `hexapod-terrain-robot-smoke-003-20260909.service`.
Output: `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/terrain_robot_smoke_003`.
Pause: `forecast_pause_015`, with exact-unit exit restoration and a separate
30-minute fallback. The host and container disable bytecode writes. Both shared
locks are job-scoped; each phase has a ten-minute bound and the service is capped
at 22 minutes. The owner yields on unrelated GPU work, coordination-note change,
low memory or an explicit stop, cleaning only its immutable container ID.

The sequence again requires **fresh 32-environment × 1,000-control flat standing**,
then **one full C robot × 1,000 controls on the original ramp's flat entry pad**.
The source mounts read-only. A separate per-run robot copy is writable only for
the established flat validator's inertia authoring, then mounted read-only for
terrain. Source and admitted asset hashes are checked across phases.

The output layout matches attempts001/002: `campaign.json`, two `jobs/*.json`,
two `logs/*.log`, `flat/state.json`, `flat/admission.json`, `terrain/state.json`
and before/after asset hashes. The fresh flat admission passed (32 environments,
1,000 controls), with post-settle maximum computed torque 0.536684 N*m and
zero terminations, truncations, nonfoot contacts or requested saturation.
Its admission hash is `b5e9b2a092bc1553a193749a6fcb9b60d7aa50442b0910a70e8a77511cdb7939`.

The terrain gate rejected: 1,000 terminations in 1,000 controls, two timeouts,
zero distal support and post-step root height constantly 0.136532515 m. The
reported 25.0507 N*m computed-torque maximum and 54.5% saturation are sampled
after automatic resets; they are not steady standing-load evidence. The next
bounded job captures the unchanged adapter before reset: installed Isaac Lab
InitialStateCfg specifies XYZW, while this frozen adapter supplied WXYZ.
No terrain standing, walking or derived-fixture admission follows from this run.

`results/` preserves the exact rejected terrain state, fresh flat admission,
logs, source identities and forecast pause015 restoration. The source661-file
manifest and all550 admitted asset hashes were rechecked after exit: no changes
and no extra source files. Both owned containers were cleaned, CUDA was empty,
both shared locks free, and both previously active forecast timers restored.
`results/post_run_verification.json` and `results/SHA256SUMS.json` record this.
Completion/restoration was at Unix1789003321.998.
 This is a standing integration test, not a terrain walking policy,
derived-fixture admission, sensor validation or physical four-bar qualification.
