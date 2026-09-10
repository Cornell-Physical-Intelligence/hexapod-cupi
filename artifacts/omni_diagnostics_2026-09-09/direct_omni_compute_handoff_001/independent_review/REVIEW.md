# Exact weather replay deferral review

The reviewed helper is clear for the bounded, user-authorized deferral of only
`stormscope-halo104-1010-20260910.service`. It pins original invocation
`441bd8a01ae1421d8b0cf60283fd200e`, PID3544965, cgroup, script, unit fragment,
working directory and exact command. A replacement owner is rejected.

The90-minute fallback is armed before the exact stop request. All existing
output files—including partial temporary files—plus log, source and unit fragment
are copied and hash-verified after that service stops and before any recompute.
The original tree is retained. Existing archives are checked against their
original manifest hash and full inventory. A failed archive prevents restart.
The approximately46MB actual dataset stays remote; this receipt contains no
copies of those forecast outputs and does not claim a runtime stop occurred.

Restoration affects only that previously active unit. It checks protected
HEXAPOD services, CUDA processes, active containers, exact direct-owned container
names/IDs and both job locks. Unknown Docker inspection errors remain unknown.
The temporary verification locks are released before starting the weather
script, whose own nonblocking GPU lock protects its compute. A new invocation
must subsequently acquire that exact lock, proven using kernel PID/device/inode
fields; early exit or a90-second readiness miss fails without retry. If systemd
collected the stopped transient unit, only its same command and properties are
recreated. This is replay recomputation, not a mid-frame checkpoint resume.

Two review findings were corrected before clearance: archive manifest identity
was not originally rechecked, and the helper originally held the same lock
across weather startup. Six independent tests pass against the corrected code:
real temporary archive preservation, replaced manifest rejection, collected-unit
recreation with lock-release ordering, exact kernel ownership proof and negative
variants, replacement invocation/PID rejection, unknown Docker rejection and
previously inactive no-restart behavior. No GPU or service mutation was performed
by the reviewer. The first independent test log used an outdated fixture without
the newly added unit-fragment hash; that fixture was corrected, not the helper.

Actual read-only Spark checks confirmed that `systemctl --user show` for a
missing unit returns exit0 with `LoadState=not-found`, inactive state, PID0 and
empty invocation, so the recreation path is reachable. The current
active/activating/deactivating `hexapod-*.service` inventory was empty; no stale
active/exited unit blocked this check at review time.

The90-minute fallback is an attempt only if the GPU remains available. A busy
protected owner defers restoration; there is no retry framework. Root owns the
explicit early restore after the bounded pilot batch and the actual run audit.
No forecast completion guarantee is inferred from reacquiring the lock.

Root's separately reviewed guard002 changes only the expected coordination
hash to `22c615d67c6c4dc14ab370b291f8eddd72f8475ad34118b91a4237a34ee05fab`.
Its25 tests were not repeated in this helper review. Frozen source/host/native
behavior and pause054 remain separate.

Run the focused tests without contacting Spark:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tmp/forecast_halo_defer_independent_review_001 -p 'test_*.py' -v
```

`verification.json` binds the exact reviewed helper and identity input. This
independent receipt can be referenced by its owner freeze without copying or
editing any frozen helper payload.
