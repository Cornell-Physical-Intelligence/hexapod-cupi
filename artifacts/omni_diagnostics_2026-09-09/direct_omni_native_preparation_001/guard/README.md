# Pause054: bounded direct-PPO training smoke guard

This external guard is derived from frozen `direct_omni_cold_guard_001`. It is
fully bound to the final native002 contract, 598-file source and eight-file host.
Root alone dispatches it. Preparation ran CPU tests and read-only previous-owner
checks; it did not pause forecasting or launch a GPU process.

The only allocation is `--allocation smoke --branch caps`: fresh standing,
32 environments × 24 controls × two PPO updates, final constant-command
diagnostics and final stop diagnostics. Each owned phase retains the host's
600-second bound. The outer service has 2,520 seconds, a 180-second stop timeout,
and a 45-minute independent restoration fallback. A completed smoke does not
allocate the separate 50-update pilot or establish Stage 2 acceptance.

- Source: `BASE/direct_omni_train_source_001`.
- Contract: `BASE/direct_omni_train_preparation_001` (native002).
- Host: `BASE/direct_omni_train_host_001/launch_train_spark.py`.
- Output: `BASE/direct_omni_train_smoke_001`.
- Pause: `BASE/forecast_pause_054`.
- Unit: `hexapod-direct-omni-train-smoke-001-20260910.service`.
- `BASE` is `/home/orionh/HEXAPOD_runs/mock_length_study_20260909`.

The guard calls the exact host's `selected_phases` and `verify_inputs` before
pause and immediately before dispatch, enforcing the original checkpoint,
315/318 widths, exact two-update positive-CAPS selection and all frozen bundles.
It does not carry forward the cold guard's `training_allowed=False` test.
The source009 supervisor remains pinned solely for established coordination,
resource and ownership tooling; its physics is not silently adopted.

Previous ownership is bound to cold001 invocation
`ae3b95c756db4aa4b639ee4230e8c958`, its completed campaign, both jobs, pause053's
original timer snapshot and restoration. All five receipt hashes must match;
the unit must be inactive/failed, and both exact container names and IDs must be
absent. An empty live invocation is permitted for a garbage-collected unit only
with those exact receipts; a conflicting nonempty invocation rejects. Live
read-only verification also passed and is recorded separately.

Existing protections remain: reject unrelated CUDA ancestry; gracefully stop
only the authorized forecasting services; acquire both GPU locks; arm fallback
before timer mutation; preserve exactly the previously active timers; inspect
and stop only exact owned containers; and keep unknown cleanup fail-closed.
The embedded restorer is byte-identical to the frozen cold guard. A changed input
after pause restores directly before any dispatch. An uncertain dispatch failure
first stops only the new owner, then invokes restoration.

Twenty-five CPU tests pass after final binding. These include execution of the
actual guard `main` with fake external effects, not merely source inspection:
fallback-before-pause ordering, two input checks, correct final selectors,
prelaunch-failure restoration and uncertain-dispatch cleanup. The eight inherited
embedded-restorer tests remain unchanged. No actual training success is claimed.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s tmp/direct_omni_train_guard_001 -p 'test_*.py'
```

`CPU_READINESS.json` binds the final source/host/contract and parent; the bundle
manifest preserves all tests, logs and exact previous-owner receipts.
