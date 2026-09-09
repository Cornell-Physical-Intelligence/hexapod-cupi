# Authorized weather CPU takeover: preserved execution evidence

The original takeover record documents the user's instruction to remove
competing processes and continue the hexapod work under full Spark priority.
The operation ran from **2026-09-06 03:13:57.588920 to 03:14:00.858089 UTC**.
This artifact preserves that completed operation; its verifier never sends
signals or changes services.

The prior read-only audit captured **21 weather process identities** at
03:09:29 UTC: twenty processes outside the materializer service plus the
materializer process. Immediately before acting, the controller rechecked
PID, start ticks, process group, argv and UID against that audit.

The recorded actions comprise:

- Twelve SIGTERM operations covering **20 non-service weather processes**:
  watcher groups, continuation parents and their separate child groups, and
  four batch queue/worker groups.
- A successful stop of the exact user service
  `corrdiff-radar-year2021-cPZhN6.service`, whose materializer PID was
  **1530836**. Its recorded policy was `Restart=no`, `KillMode=control-group`.

The final record has **no remaining targeted processes**. The service
reports `MainPID=0`, `ActiveState=inactive`, `SubState=dead`. No SIGKILL
action is recorded. The signal target set excludes the protected hexapod
PIDs **1716416, 1716170, 1712641 and 1711343**. No viewer/container stop is
recorded. The nominal hexapod campaign had already completed its failing
result before this CPU operation.

`outputs_deleted` is **false** in the original record. No output directories
were removed by the recorded actions; interrupted weather outputs may be
partial. This capture does not perform an exhaustive before/after audit of
every output file and does not claim those incomplete outputs are reusable.

## Captured reservation

The separate guard snapshot records PID **1772925** holding
`/opt/wx/gpu.lock`, state `reserved`, from **03:10:41.304616 UTC**, with an
expiry ceiling of **13:10:41.304601 UTC** on September 6. Its original launch
record names the bounded reservation program and records that the guard
itself signals no workloads. The captured state is historical; the expiry
is not a guarantee the reservation remains active until that time.

Original remote records:

```text
/home/orionh/HEXAPOD_runs/mkii_motion_recovery_v1/cpu_takeover_20260906T031357Z/takeover.json
/home/orionh/HEXAPOD_runs/mkii_motion_recovery_v1/cpu_cleanup_audit_20260906T030929Z.json
/home/orionh/HEXAPOD_runs/mkii_motion_recovery_v1/priority_20260906T031041Z/{launch.json,status.json}
```

`remote_inventory.json` preserves raw-file SHA-256 values, sizes and
modification times. `takeover.json` preserves every reverified argv/start
identity, signal action and completion result. These records do not prove
that a future weather process cannot be started.

## Reproduction

```sh
python3 artifacts/mkii_fourbar_2026-09-06/cpu_takeover_20260906T031357Z/verify.py \
  artifacts/mkii_fourbar_2026-09-06/cpu_takeover_20260906T031357Z \
  --out /tmp/hexapod_cpu_takeover_verification_new.json
```

The verifier checks original hashes, all 21 pre-stop identities, exact
signal/group coverage, exclusion of protected PIDs, service completion and
guard identity. `summary.json` is derived evidence. The CPU stop occurred
after the completed motion failure; no physics cause or remedy is inferred.
No remote write, signal, service change, launch, shared-note edit or
production-source change was made during preservation.
