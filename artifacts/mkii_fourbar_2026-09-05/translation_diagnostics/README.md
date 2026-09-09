# Matched single-robot placement diagnostics

At 2026-09-05 20:42 UTC, the origin and +6 m world-X diagnostics completed on
frozen source `d6d5863` with identical v5 asset, Kp30/Kd0.30, TGS128/1,
800 Hz physics, zero actions, 200 control steps and 3,200 recorded physics
substeps each. Neither run admits simulation training or hardware operation.

| Position | Settled peak applied torque | Maximum C-pin gap | Minimum supporting feet |
|---|---:|---:|---:|
| (0,0) m | 0.666658 N m | 0.134110 µm | 6 |
| (6,0) m | 0.680521 N m | 0.959709 µm | 6 |

The matched settled interval is 0.8–4.0 seconds. Source/runtime identities and
all delivered target samples are exactly equal. The applied-torque difference
is 0.0138621 N m. This single-axis translation **does not reproduce** the large
multi-environment standing jitter. Placement-associated noise in earlier traces
therefore does not establish world-distance precision as the cause. Y/diagonal
placement, batch/native constraint behavior and duration require distinct tests.
The prior 32-environment Kd0.30 log already reports 1.627909 N m and four-foot
support during 2.4–4.0 seconds, so duration alone cannot explain that difference.

`comparison_x6.json` contains exact metrics, primary-report and trace hashes,
source equality and interpretation. Original reports, NPZ traces, Docker logs,
source hash lists, supervisor state and launch records are preserved beneath
`origin/` and `translated_retry/`. The empty `admitted` files refer only to the
host resource-launch gate; report admission flags remain false.

## Interrupted first translated attempt

`translated_failed_inspect/` preserves the first +6 m attempt. It has no final
validator report and is not a physical pass or failure. Its host supervisor
failed with `docker inspect failed (exit 1)` and removed only its owned immutable
container ID. A temporary CPU-only SDK inspection container disappeared between
foreign-container inventory and GPU-configuration inspection. The captured
Docker lifecycle events put that removal at 20:33:37.559302100 UTC, followed
42.153 ms later by the supervisor stopping the owned diagnostic. See
`docker_events_inspection_race.txt`; timestamps are Docker timeNano values.
The successful retry uses exactly the same frozen source and settings, after
suspending temporary CPU readers. The supervisor race is fixed separately in
`d863663`, tolerating only a proven missing full-ID foreign container, with
all GPU and owned-container checks preserved.

This directory may gain separately named diagnostics. Published checksums are
immutable snapshots; later evidence receives a new manifest rather than editing
an earlier one. Per-run source archives remain on Spark in their original
output directories and are excluded from this compact local copy.
