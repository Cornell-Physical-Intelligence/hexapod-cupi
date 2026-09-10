# Detailed native standing32: completed acquisition, rejected admission

The detailed direct-drive robot completed all **8,000 physics steps for 32 replicas** (1,000 controls, 20 simulated seconds). The unchanged native standing report accepted **11/32** replicas. Separately, the host failed its **600-second deadline** and exited with status 1. Both outcomes are preserved: finishing acquisition did not grant standing or training admission, and the timeout does not erase the recorded physical rejection.

This was a neutral standing screen with no policy loaded. No walking, hardware, PPO, or Stage 2 qualification follows from it. The model remains the corrected 19-body/18-joint detailed CAD, with the explicitly provisional, uncalibrated 48 V software effort envelope.

## Recorded result

These figures are aggregated from the byte-verified native [standing report](curated/run/standing/standing_report.json), also embedded in the original [remote terminal audit](terminal/audit.json). They are **not a new local numerical replay** of the contact stream or NPZ traces.

| Check | Recorded result |
|---|---:|
| Combined physical and quiet acceptance | 11/32 |
| Physical bounds | 11/32 |
| Quiet bounds | 25/32 |
| Replicas losing six-toe support after settling | 21/32 |
| Missing-support replica/substep observations | 74 total; 1–10 in each affected replica |
| Quiet SDK joint-rate RMS maximum | 0.099565849 rad/s; limit 0.03 |
| Quiet joint-position range maximum | 0.006698237 rad; limit 0.02 |
| Quiet planar excursion maximum | 0.208512 mm; limit 10 mm |
| Quiet heading excursion maximum | 0.011775246°; limit 2° |
| Requested and applied torque peak, all steps | 1.160323143 N·m |
| Requested torque saturation, nonfoot contacts, terminations/truncations | 0 |

The scored quiet window contains the last 800 controls (16 seconds). Quiet failures were replicas **2, 4, 7, 15, 18, 27, 30**, all of which also failed six-toe support. [SUMMARY.json](SUMMARY.json) contains every support count and the unchanged recorded thresholds. SDK joint rates and angle-derived motion remain distinct measurements; this static publication does not establish the cause of either failure.

The native [session](curated/run/standing/session.json) records all rows captured, one initial reset, and 8,000 complete steps. Its [state](curated/run/standing/state.json) says `status: completed`, `standing_pass: false`, and `training_allowed: false`. The original [campaign](curated/run/campaign.json) and [job](curated/run/jobs/standing.json) both retain `TimeoutError('Standing phase exceeded ten-minute bound')`. Native total wall time was 606.564 s; the controlled session recorded 585.272 s. These are different measured scopes, not replacement deadlines.

## Explicitly curated raw evidence

The full immutable remote inventory records **36 files / 4,969,155,341 bytes**. This public bundle contains only **24 selected raw files / 3,281,218 bytes**, each at most 2,000,000 bytes. They were fetched by a read-only SSH process and independently matched by byte count and SHA-256. No large raw file was downloaded or compressed locally.

The other **12 files / 4,965,874,123 bytes remain remote-only**:

- `run/standing/contacts.jsonl`: 4,620,466,111 bytes.
- `run/standing/control_trace.npz`: 39,318,700 bytes.
- `run/standing/substeps_000.npz` through `substeps_009.npz`: 306,089,312 bytes combined.

[REMOTE_RAW_INVENTORY.json](REMOTE_RAW_INVENTORY.json) preserves all 36 original paths, sizes and hashes from the audit. [RAW_SELECTION.json](RAW_SELECTION.json) gives the exact selected/remote-only partition and absolute remote paths. Run files reside under `/home/orionh/HEXAPOD_runs/canonical_direct_20260910/native_standing32_002`; pause files reside under `forecast_pause_010` in the same base directory. The audit authenticated the complete remote inventory twice at terminal observation. The local verifier checks that recorded inventory and selected bytes; it does not assert current remote availability.

**This bundle is not self-contained for full raw replay.** A contact, support or numerical trace re-analysis requires the excluded original bytes. The copied scorer is preserved for lineage, not executed by this publication verifier.

## Identity and cleanup

The recorded owner was `hexapod-canonical-native-standing32-002-20260910.service`, invocation `1a38495bda9f4fefa4e5585574aabc56`. The terminal audit verified its failed/inactive process state, MainPID 0, both exact owned container name/ID absent, unchanged inputs, and completed per-job restoration. The persistent HEXAPOD reservation was **not released**. This snapshot is not a live status probe.

Exact preparation is copied without changes: [source003](source/FREEZE_SHA256.json) (87 payloads), [host004](host/FREEZE_SHA256.json) (33), [guard32_002](guard/FREEZE_SHA256.json) (62), and [auditor002](auditor/FREEZE_SHA256.json) (8). [PROVENANCE.json](PROVENANCE.json) pins those freezes and the current root setup/dispatch/progress snapshots. The original terminal audit is preserved verbatim. No earlier admission or current failed outcome is rewritten.

## Portable verification

From the repository root:

```sh
python3 -B -S artifacts/mkii_updated_2026-09-10/native_standing32_rejected_002/verify_bundle.py
```

Only the Python standard library and files inside this artifact are used. It checks the outer payload map, all copied preparation freezes, the full recorded inventory and curated partition, every selected byte hash, native output seals against recorded hashes, and the acquisition/rejection/timeout/cleanup semantics. It aggregates the original report again and compares `SUMMARY.json`; it never imports or runs native physics, connects to Spark, or rescales thresholds.

Root owns any subsequent diagnosis, shared status/roadmap updates and Git publication under `docs/PROJECT_SITE.md`. This artifact and its bounded central update preserve failed evidence without authorizing a new policy run.
