# Completed motion-prefix replay — failure reproduced

The replay completed **32 environments × 2,500 controls / 40,000 physics steps
per environment** on 2026-09-06 at approximately 03:50:35 UTC. It reproduced
campaign 008's motion failure. `diagnostic_complete=true` records completed
data collection; **pass and simulation/hardware admission remain false**.
The owned container was removed and the supervisor exited 0 after recording
`diagnostic_complete`. This is not a validation pass or a trained policy.

The actual runtime manifest, ordered 32 reset positions, complete startup and
settled metric dictionaries, all 15 recorded individual response minima, and
the four motion peaks below match campaign 008 **exactly** in the primary JSON
reports. The replay retains the original 1,000-control standing history and
the first 15 individual motor tests through LR pushlever; it stops before the
remaining three individuals and all group tests.

| Reproduced driven metric | Recorded value |
|---|---:|
| Maximum C-pin separation | 0.213820967474 mm |
| Maximum raw motor demand | 85.207611083984 N m |
| Maximum applied torque | 5.5 N m |
| Maximum passive velocity relation residual | 221.326507568359 rad/s |
| Maximum relative C-pin velocity | 7.051413536072 m/s |
| Minimum loaded-foot count | 0 |

Individual LM and LR pushlever response minima are −0.005873203278 and
−0.190767854452 rad. These are each the minimum across 32 environments of
the positive-minus-negative end-hold means; they do not identify a joint-map
error. No environment reset, non-foot ground contact, or invalid physics sample
occurred. The existing closure, support and direction checks failed.

## Coverage and inference limits

All **40,000 physics updates and force writes** were counted. All 2,500 control
boundaries have compact telemetry in 31 NPZ files. Six detailed NPZ files cover
every one of the 4,800 physics steps in controls **[2200, 2500)**, all 32
environments and all 18 active motors, during LF/LM/LR pushlever tests. Earlier
controls have boundary telemetry and physical metrics, but no detailed trace.

The [offline analysis](analysis_README.md) and [machine-readable event data](analysis_analysis.json)
record PD inputs, torque clipping, native/cache velocity agreement and event
neighborhoods. Its CPU recomputation of float32 end-hold reductions differs by
up to 5.96×10⁻⁸ rad from recorded GPU minima; the primary replay and campaign
minima themselves match exactly. Interval-average position differences and
instantaneous native velocities are different quantities. The observed order
of contact/velocity/torque events does not establish a unique cause or justify
changing a motor limit. Further first-impact analysis lives separately in
[motion_prefix_analysis](../motion_prefix_analysis/README.md).

## Frozen inputs and storage

Run directory on Spark:

```text
/home/orionh/HEXAPOD_runs/mkii_motion_recovery_v1/prefix_replay_001/hexapod-fourbar-diagnose-20260906T032844Z-d73ff362
```

Source commit: `3011b0f505d7f492ce4203443caa59ac76204356`.
Functional SHA256: `9fa39026e79ca4d327d67f58c37b9e2f800700e627994b51b3640bb53d106089`.
The run's 306-file input snapshot has manifest SHA256
`4a65a78a6b1ca06982da802027ded027daf3319f2f85a91e7f87933a7c378172`
and archive SHA256 `c5fb6b4456fab47c66065370273ab2d147e568cbcad53addad43936661091c8c`.
This run snapshot is distinct from the source release's 295-path manifest.

This directory contains the retrieved report, supervisor, full container log,
offline analysis, retrieval hashes and an exact copy of the previously
preserved campaign 008 report (`campaign008_report.json`). The comparison copy
has SHA256 `c57a2f565d11b62213cca043aea4b26bdaa18d56cc2ad71848cf322793de751e`.
The original retrieved files have not been rewritten.

**The six detailed and 31 control NPZ files and the source archive remain on
Spark; their bytes are not included here.** `remote_inventory.json` preserves
every retained run-file path, SHA256, byte count and modification timestamp as
captured by a read-only remote inventory at **11:42:08 UTC**. Its 43 files total
**482,457,484 bytes**; the 37 raw NPZs account for **420,148,011 bytes**.
It is a recorded identity, not a claim
that a later offline invocation has reread remote bytes. The analyzer verified
all 37 NPZ hashes, dimensions and sample intervals on Spark before producing
the retrieved analysis. No raw trace was downloaded for this compact archive.

`SHA256SUMS` covers every local file except itself. From the repository root:

```sh
python3 artifacts/mkii_fourbar_2026-09-06/prefix_replay_001/verify.py
```

The verifier checks local bytes, remote-inventory consistency, all sample
intervals, report/source identity, cleanup, and the exact stated reproduction.
It reports zero raw files rehashed by default. To rehash all retained NPZ bytes,
run the same verifier with `--raw-dir <retained-run-or-downloaded-NPZ-directory>`.
The verifier is read-only, uses the standard library, and launches no workload.
Future analysis or interpretations belong in a new artifact directory.
