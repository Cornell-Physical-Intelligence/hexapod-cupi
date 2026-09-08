# 1600 Hz nominal validation failure — preserved originals

The nominal v5 run completed all **1,000 standing and 2,400 driven controls** for 32 environments, with **108,800 physics substeps per environment** (3,481,600 environment-substeps). It failed the unchanged driven-support gate. It did not produce training admission, a PPO checkpoint, or a policy video.

The original report's sole error is `driven: unintended contact or missing support`. Its driven nonfoot-contact count is zero, while minimum support is zero, identifying the missing-support side of that error. Aggregate evidence does not identify the duration or exact frame of the support loss. There were zero terminations and zero truncations; this result should not be described as all robots overturning.

| Recorded quantity | Value |
| --- | ---: |
| Standing controls / driven controls | 1,000 / 2,400 |
| Physics / policy rate | 1,600 Hz / 50 Hz |
| TGS position / velocity iterations | 64 / 16 |
| Settled mean plate height | 0.1357128195 m |
| Settled minimum support | 5 |
| Settled maximum raw/applied torque | 0.8220795989 N·m |
| Driven minimum support | 0 |
| Driven nonfoot-contact environment-substeps | 0 |
| Driven maximum raw motor demand | 83.7286605835 N·m |
| Driven maximum applied motor torque | 5.5 N·m |
| Driven maximum instantaneous motor-envelope excess | 0 N·m |
| Driven maximum C-pin position gap | 0.0726735816 mm |
| Driven minimum nonfoot clearance | 22.0423192 mm |
| Driven minimum burst headroom | 0.0990690738 |
| Driven maximum passive velocity-relation residual | 82.3740921021 rad/s |
| Driven maximum relative C-pin velocity | 2.6425828934 m/s |
| Invalid numeric/contact samples | 0 |
| Terminations / truncations | 0 / 0 |

All 18 individual and 18 coordinated positive-minus-negative response entries are positive; the source records `driven_coordinate_pass=true`. Position closure stayed below the unchanged 0.100 mm limit, but that alone did not establish acceptable dynamics/support. Velocity residuals are observational telemetry, not newly invented acceptance gates. Raw demand is the pre-limiter demand, not physically applied torque; the applied envelope was respected.

The exact runtime is source commit `c2af43ca0f384a4c2c7ab8f1d627f309dc78a683`, functional identity `c33b4591e99f287579181469dcc132f659d1980200708475442dbe3ffcb7ecfc`, numerical recipe `mkii_fourbar_tgs_external_forces_1600hz_position64_128_velocity16_v6`, physical model `mkii_fourbar_v5`, and `coincident_flat_origin_v1` placement. No result is transferable to another numerical recipe by relabeling it.

## Supervisor and helper terminal states

The container exited zero, but the supervisor independently rejected its failed primary report and exited **1**. It recorded unchanged source identity and cleanup `removed_exact_id`. The campaign records a passed probe followed by failed nominal validation, with no retry or subsequent phase. Refined validation, scratch PPO and full PPO did not start in this campaign.

The capture follower ended `no_video`, reason `Campaign failed`. The continuation coordinator ended `original_failed`, with zero completed full updates, empty segment/launch history, and no active job. A read-only `/proc` snapshot after retrieval found the exact recorded campaign, follower and coordinator PIDs absent. This snapshot concerns those three original processes only; it makes no claim that the entire Spark is idle.

## Preserved bytes and provenance

- [Original validator report](raw/nominal/report.json), [supervisor](raw/nominal/supervisor.json), [container log](raw/nominal/container.log), [CPU asset audit](raw/nominal/cpu_asset_audit.json), source file manifest and original empty `admitted` barrier marker.
- [Original campaign record](raw/campaign.json), [host log](raw/run/host_001.log), original launch metadata, helper launch logs and terminal state files under `raw/run/`.
- [Independent extracted summary](independent_summary.json) retains full settled/driven metrics and all 36 signed-response values.
- [Retrieval metadata](retrieval.json) records exact source paths, sizes, hashes, inode/mtime, transfer time and a second remote verification **after local files were written**. All 15 copied originals match the second remote hashes; all sources remained unchanged across retrieval. Raw bytes were not reformatted.

The 62,132,248-byte source archive was deliberately left on the Spark and hashed remotely in both passes. Its SHA-256 is `e4279924b1b16a990f8c2b22ad44fe2e16225af8fd2037f5dc52fcf17f5e7274`; exact path:

```
/home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/campaign_001/fourbar-campaign-20260907T004524Z-8592e14a/nominal/hexapod-fourbar-validate-20260907T004734Z-56533deb/source.tar.gz
```

This local artifact therefore contains the raw failure and lifecycle evidence plus the source manifest/archive hash, not another copy of that 62 MB archive. `retrieve.py` documents the read-only transfer and refuses to overwrite an existing retrieval. `SHA256SUMS` covers every local artifact file except itself.

The user requested a pause while preparing a replacement single-leg URDF during this already-running retrieval. Only preservation/inspection of these existing results was completed afterward. No remote file was changed, no GPU job or experiment was run, and no repository commit or status edit was made. Work stops here pending the new URDF.
