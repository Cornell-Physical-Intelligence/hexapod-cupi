# Supplemental deferral of the exact halo replay

Root may execute this helper under the user's existing authorization to defer
Forecasting-Pipeline work for HEXAPOD. It does not alter the frozen direct-PPO
guard, weather source, input files, shared coordination, GPU configuration or
unrelated services/timers. Preparation itself made read-only Spark calls only.

The only weather target is `stormscope-halo104-1010-20260910.service`, invocation
`441bd8a01ae1421d8b0cf60283fd200e`, MainPID3544965, with the exact recorded cgroup,
working directory, unit fragment and `halo_replay.py` SHA. The read-only identity
receipt preserves these values and the original command. A replacement invocation
is never stopped. If this invocation finishes before the stop, the helper skips
deferral and does not restart it.

`defer` arms a 90-minute fallback **before** stopping the exact unit with its
existing SIGTERM/control-group semantics. The weather script has no checkpoint
resume or signal handler. After process/cgroup exit, the helper copies the entire
`halo104-1010` output tree (about46MB at inspection), the complete sibling log,
script and unit fragment to a fresh `archive_NNN` directory beneath
`BASE/forecast_halo_defer_001`. Original output/log/source bytes are left in place.
Temporary files are preserved as temporary evidence, never counted as completed
frames. Source and copied trees are hashed and compared after copy; an incomplete
archive blocks all restart. A retry uses a new archive directory, preserving any
partial earlier copy. The manifest itself and exact archive inventory are bound.

`restore` can be run early by root, or attempted by the90-minute fallback. It
requires a complete immutable archive, stopped original/reviewed weather identity,
no active HEXAPOD service, no CUDA producer, no running HEXAPOD container, and
absence of the exact known direct-PPO job container names/IDs. Unknown inspection
errors fail closed. Both normal GPU locks are checked briefly and released
immediately before start; there is **no long-lived GPU reservation**. The replay's
own nonblocking weather lock remains the compute protection.

The helper starts only that original unit. If systemd garbage-collected its
transient definition, it recreates the same unit name with the captured original
argv/working directory and2-hour runtime/default stop settings. Installed systemd
was checked read-only: querying a collected unit returns `LoadState=not-found`,
inactive/PID0 and exit0. No persistent weather unit file is installed.

A bounded90-second readback then requires a new stable invocation, exact
ExecStart/cgroup/working directory and the process's own kernel FLOCK on the
weather lock device/inode. If it exits or loses the slot, restoration reports
failure without retry. `restored.json` certifies this startup/lock proof, not a
completed forecast. Re-running a restored helper is idempotent. A blocked fallback
leaves append-only failure evidence for root; it does not interrupt HEXAPOD or
keep retrying autonomously.

Restarting recomputes the original replay and may overwrite its working output;
the complete pre-restart archive remains immutable. **This is deferral and
recomputation, not mid-frame resume.** No new weather dataset, model or publishing
allocation is introduced.

After root transfers the whole frozen helper to
`BASE/forecast_halo_defer_helper_001`, the only operator commands are:

```sh
/usr/bin/python3 -B BASE/forecast_halo_defer_helper_001/defer_halo.py defer
/usr/bin/python3 -B BASE/forecast_halo_defer_helper_001/defer_halo.py restore
```

Here `BASE=/home/orionh/HEXAPOD_runs/mock_length_study_20260909`; expand the path
in the actual invocation. Root owns every mutation and dispatch.

CPU tests exercise real temporary archive files with fake external processes:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s tmp/forecast_halo_defer_helper_001 -p 'test_*.py'
```

The first owner test attempt exposed a fixture method named `run` shadowing
`unittest.TestCase.run`; its failed log remains. Renaming it `fake_run` changed
only the fixture. Final results and independent review are recorded separately.
