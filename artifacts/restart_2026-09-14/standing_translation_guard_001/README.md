# Single-placement diagnostic operations guard

This is prepared source, not an executed experiment. Root must review and fill
one arm's bindings after the successor source and host are frozen. The included
templates have null hashes/paths and `root_review_complete: false`; they refuse
allocation. Each arm has an independent output and no prerequisite arm pass.

Only one robot at `origin` or `xy14_4` is permitted. This diagnostic cannot admit
standing, a 32-robot batch, training, physics or Stage 2. It cannot run PPO.
Existing numerical gates remain unchanged. If the native contract rejects quiet
behavior or contact evidence, retain the failed attempt and all acquired data;
do not relabel it as admission or make the other arm depend on a pass.

The guard reuses the successor host's verified source009 supervisor and deadline
adapter. Its `run_owned` function, 90-second AppReady check, 1200-second native
deadline, foreign-CUDA ownership monitor, raw-contact audit and exact container
cleanup remain unchanged. Only its preflight/resource callbacks add current
reservation checks. Preflight holds both compute locks, releases them before
the host reacquires them for the entire owned native lifecycle, and rechecks
the reservation under the host locks before container creation. After host
cleanup, the guard reacquires both locks and rechecks original inputs,
reservation, exact container exit and resources. It never releases the persistent
Spark reservation or restores unrelated services.

Reservation bindings use the new 14 September root, 32 user masks (including
DeepSeek), four system masks, new queue helper and current directory blocker.
Old unit-state assumptions and the released 11 September marker are rejected.
Original backups remain verified. The root-owned `slurmd.service` remained
active with zero observed children after stopping it required authentication;
see the referenced receipt. These controls are not an absolute block on every
producer. The existing startup and runtime foreign-CUDA monitor remains required.

## Review and CPU checks

```sh
python3 -B test_guard.py
python3 -B verify_bundle.py
```

The negative tests cover pending/old source and reservation bindings, modified
frozen bytes, admitting identities, arm/output mismatch, input overlap, both
compute locks, callback preservation and exact container cleanup. They contact
neither Spark nor Docker. `PROVENANCE.json` identifies unchanged parent sources.

## Dispatch interface for the sole root dispatcher

Create a separate root-reviewed guard bindings JSON outside this frozen folder
from the appropriate template. Fill the exact successor source/host manifest
hashes, remote paths, that arm's absolute output, and its native host bindings
path/hash. The native bindings schema is `canonical_placement_host_binding_v1`;
its hashes, coordination, arm and output must match the guard bindings, and
`native_dispatch_authorized` must be true. Set `root_review_complete` only after
review. Neither file may change while the attempt runs.

Run the wrapper with these required arguments (all variables are dispatcher-set
absolute paths or verified hashes; no value is inferred by the guard):

```sh
/usr/bin/python3 -B "$guard_dir/launch_guarded_diagnostic.py" \
  --bindings "$guard_bindings" --bindings-sha256 "$guard_bindings_sha256" \
  --guard-freeze-sha256 "$guard_freeze_sha256" \
  --placement "$placement" --output "$output" --preflight-only
```

Preflight creates no output and makes no native call. After that succeeds, use
one fresh systemd user unit with `RuntimeMaxSec=1320`, `TimeoutStopSec=180`,
`KillMode=process` and `PYTHONDONTWRITEBYTECODE=1`. Its ExecStart is the same
command with `--preflight-only` omitted. Its ExecStopPost is that same exact
command with `--cleanup-only` appended. Root supplies the unique unit name and
retains the dispatcher/transfer/launch receipts. SIGTERM/SIGINT creates the
owned attempt's `stop.request`; the reused supervisor handles its exact cleanup.

The cleanup fallback accepts only the current arm's reviewed output and the
single recorded `hexapod-reference-physics-<uuid>` container name/full ID. It
refuses an identity mismatch, stops only that immutable ID, and verifies exit.
It remains usable if source integrity failed, then records any reservation or
resource verification failure for owner review. Cleanup never starts a retry.

Archive each unit's stdout/stderr, `guard.json`, `guard_cleanup.json`, native
campaign/job receipts, raw observations and final manifests. A stopped or failed
attempt remains failed; a completed acquisition still confers no admission.
