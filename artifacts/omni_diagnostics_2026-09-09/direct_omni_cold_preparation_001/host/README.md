# Cold direct-PPO baseline host

This standard-library host allocates only fresh 32-replica/1,000-control standing, followed by the preserved 315/318 checkpoint over the exact 48-replica, twelve-case, twelve-second diagnostic. Baseline starts only after new-plan standing admission. Complete diagnostics remain measurements, not physical, training or Stage 2 admission. No training or video phase exists.

The cold source is the root-built 589-file manifest `4671eab012aaf79ce49769cfdc1667142237a4b4a45966fd0399a9caa900c51b`. The independent baseline contract freeze is `eb87f1dd456771950f7c1bf50ed0a51c1fc6ccb4631217e29d99fa8a84717095`; original checkpoint is `1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8`. The contract retains historical solver16/4, disabled external-force default, noise, actions, rewards and geometry, with an explicitly new .04 target-step profile. Source009 supplies host supervision only; its physical configuration is not imported into the cold environment.

## Supervision and read-only inputs

`launch_cold_spark.py` imports the exact source009 supervisor (926-file map `04942c62…e01e`, host file `9ebabaf2…d561`). Its `run_owned` and `owned_container` functions are unchanged, including locks, competing-work checks, 600-second phase deadlines, 90-second AppReady deadline, contact-overflow rejection, stop handling and exact container cleanup. Four explicit namespace adaptations are recorded:

1. The command callback invokes the cold entry adapter with the original cold CLI.
2. The pre-container source callback verifies the complete cold589 tree plus its legacy16 Python runtime. The full supervisor926 tree is independently verified before/after each phase.
3. `RUNTIME_TREE` is explicitly bound to the actual legacy16 canonical digest `abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280`. It equals the pinned vendored runtime when keys are package-relative. Full source-prefixed paths are retained in `LEGACY_RUNTIME_SHA256.json`; a different serialization is not claimed to be a different runtime.
4. A job-save wrapper corrects historical policy metadata. `policy_phase_requested` marks baseline intent. `no_policy_loaded` is true for standing, unknown/null during an unconfirmed baseline, and false only once the completed baseline state names the exact checkpoint. The inherited original value is preserved separately. No successful load is claimed merely because a job started.

Source/assets, checkpoint, and this adapter are mounted read-only. Baseline sees standing at both `/admission:ro` and `/output/standing:ro`, protecting the writable-parent alias. Exact complete standing hashes are checked before/after baseline and at terminal finalization. Source, contract, supervisor and host hashes are also rechecked; a late integrity failure makes the campaign failed while preserving raw outputs.

## Metadata-only legacy entry adapter

The legacy entry has no explicit readiness marker or runtime binding. `run_cold_entry.py` verifies the frozen589 source and actual16 runtime before execution, then compiles a copy of its AST with exactly two inserted statements:

- Emit `REFERENCE_SCREEN_APP_READY` immediately after the unchanged `app = AppLauncher(args).app` construction succeeds.
- Wrap the existing `save` callable to add verified runtime/import-origin metadata only to `state.json`.

All original function and class ASTs remain unchanged. AppLauncher is not monkeypatched. Every actually imported `hexapod_rl` module must resolve to the exact `/source/isaaclab/hexapod_rl` file and match the legacy map; the actual environment import is mandatory. No newer package import or assumed runtime identity can pass. State annotations are not physics qualification. The adapter does not change actor actions, reset behavior, target filtering, assets, or simulation settings.

## CLI

The outer guard can call `verify_inputs(args)` (also exported as `validate_inputs`) without creating output or launching a process. It returns the exact cold identity and checks this host freeze. All paths must resolve to the intended frozen inputs.

```sh
python3 launch_cold_spark.py \
  --source /path/to/direct_omni_cold_source_001 \
  --checkpoint /path/to/original.pt \
  --contract /path/to/baseline_contract \
  --supervisor-source /path/to/reference_physics_source_009 \
  --output /path/to/fresh_cold_run \
  --preflight-only
```

Removing `--preflight-only` is a GPU allocation and remains root-owned behind its separate pause/restore guard. Optional `--isaaclab` defaults to `/home/orionh/IsaacLab`. Outputs must be fresh and outside immutable inputs. A failed or changed standing result never triggers the second phase. There is no automatic retry, training continuation, or source writing.

Twelve CPU tests cover exact inherited supervision ASTs, actual pre-container cold-validation dispatch without starting a process, command/mount aliases, source runtime digest convention, unchanged old function ASTs, readiness-after-construction, wrong imported origins, truthful policy metadata, standing failure/mutation, and two-phase scope. The test harness uses repository-local frozen parent/contract files and can be invoked with `python3 -m unittest discover -s tmp/direct_omni_cold_host_001 -p 'test_*.py'`. The portable host itself needs only its frozen files plus explicitly supplied source/contract inputs. Actual container startup and baseline behavior remain unmeasured until root dispatches.
