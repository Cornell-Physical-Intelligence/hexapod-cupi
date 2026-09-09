# Physical MKII four-bar campaign, 5 September 2026 UTC

This is a separate physical-loop training lineage. It preserves the archived mock and corrected serial results. The new asset has 31 bodies, 30 articulation coordinates, six excluded closure joints and 18 RS05 drives. Only the active motors receive actions; passive knees and rods close physically in the simulator.

## Evidence at publication

**Campaign 004 completed its full 800 Hz nominal run and failed the original closure-accuracy bounds. PPO has not started.** Its source remains frozen at commit `ea05fe8`, and its directory is `/home/orionh/HEXAPOD_runs/mkii_fourbar_v1/campaigns/fourbar-campaign-20260905T044533Z-5465207c`. The earlier probe `probe/hexapod-fourbar-validate-20260905T044533Z-c2be22c1/report.json` passed with **one robot, 100 control steps and 1,600 physics substeps**. Settled peak torque was 0.66570187 N·m, all six pads supported the robot, non-foot ground contacts were zero, mean plate height was 0.13570817 m, and maximum closure separation was 0.59549 µm. Startup is separate: peak torque 1.69954038 N·m and maximum closure separation 1.49399 µm. That short pass did not establish full training admission.

The nominal run completed **32 environments × 1,000 standing plus 2,400 driven control steps**: **16,000 standing + 38,400 driven = 54,400 physics substeps per environment**, at 1.25 ms / 16 substeps per 50 Hz control step. TGS used 64 position / 1 velocity iteration with external forces every iteration. Maximum closure separation was **0.0003454488469 m**, exceeding 0.0001 m; passive-coordinate residual was **0.01900333166 rad**, exceeding 0.005 rad. Raw torque demand peaked at **33.77581024 N·m**, and applied torque peaked at **5.5 N·m**. All individual/group direction checks passed, resets and unwanted ground contacts were zero, and minimum support was one pad. The numerical failures stopped the campaign before training.

Focused diagnostics and a separate D6 v4 candidate are prepared. The [all-angle equivalence audit](closure_equivalence/README.md) supports retaining only the two independent transverse closure rows: the existing planar tree supplies the other three directions, to subnanometre authored precision. The candidate retains all original full pin-position, axis and passive-branch checks; no acceptance bound was relaxed. Neither the focused diagnostics nor the candidate has a GPU validation pass. Complete matching nominal/refined validation is still required before the 64-environment / 3-iteration scratch PPO and checkpoint/inference check, then a separate 512-environment / 1,000-iteration resume process.

Frozen release checks: **797 unit tests passed**, and CI passed for `ea05fe8`. The historical 112-file manifest verified against its immutable Git source; a separate **270-file 800 Hz manifest** verified that release. New diagnostic/candidate source requires separate checks and identity. Two package-definition files changed to package the versioned modules/data. No old checkpoint or archived manifest was rewritten.

The new diagnostic/D6 source passes **811 tests in 41.946 seconds**. Its separate **276-path manifest** has SHA-256 `1aa17aae7bc9a7a9503888fe8a84bfbb77f9d89401fce6d819a9e234eaa3f02f`. The planned isolated Spark source directory is `/home/orionh/HEXAPOD_runs/mkii_fourbar_diagnostics_v1/source`. These checks do not constitute a live diagnostic or candidate-asset pass.

The campaign 004 reservation, PID 1333865, was renewed at 04:56:24 UTC and **released on campaign failure at 05:12:38 UTC**; its record remains `/home/orionh/HEXAPOD_runs/mkii_fourbar_v1/priority_campaign_004.json`. The earlier guard PID 1232035 had been terminated after its successor queued. A separate diagnostic guard, PID **1364926**, started at **05:15:44 UTC** with a three-hour ceiling; its metadata is `/home/orionh/HEXAPOD_runs/mkii_fourbar_v1/diagnostic_priority_20260905T0515.json`. Shared coordination-file bytes were not changed during campaign 004's active phase. Neither reservation bypasses workload admission checks or creates an MPS quota.

## Earlier 5 ms evidence — preserved

The first live physical probe passed: `fourbar-campaign-20260905T034653Z-a83056cd/probe/hexapod-fourbar-validate-20260905T034654Z-91ce5326`. Its historical 5 ms / four-substep recipe completed 100 control steps / 400 physics substeps with one robot. Settled peak torque was 0.67483574 N·m, six pads supported the robot, non-foot ground contacts were zero, mean plate height was 0.13569902 m, and maximum settled physical closure error was 1.8253 µm. Startup torque peaked at 1.81281567 N·m and closure error at 65.9877 µm.

The first full nominal run failed driven closure: 0.246159 mm pin separation and 0.0114364 rad passive residual exceeded the original 0.1 mm / 0.005 rad bounds. All direction, torque-envelope, contact and reset checks passed. The 64/8 diagnostic also failed at 0.366053 mm / 0.016916 rad, with no torque-envelope, contact or reset failure. Those complete 5 ms runs each sampled 4,000 standing plus 9,600 driven physics substeps per environment.

The subsequent 5 ms TGS 64/1 recipe with external forces every iteration passed its short probe, then failed at the first tibia reversal after 1,252 driven control steps: closure reached 0.883917 mm and triggered a reset. These failures remain evidence, not training admissions. The [CAD inertia/stability diagnostic](solver_stability/README.md) motivates testing 1.25 ms without changing masses, gains or acceptance bounds; it does not prove the live solver's stability.

## Failed startup attempts are retained

| Attempt UTC | Outcome and repair |
|---|---|
| 02:16:37 | CPU audit rejected last-bit ARM/x86 derived floating-point differences. Added narrowly bounded numeric comparison while retaining exact structural and hash identity. |
| 02:26:34 | Config decorator imported as module. Corrected import path. |
| 02:30:15 | Premature runtime import loaded standalone USD before Kit, causing native startup failure. Split actuator configuration/runtime and made task environment export lazy; exact installed SDK CPU import now rejects any premature `pxr` load. |
| 03:21:23 | GPU admission blocked by weather workload; no container launched. |
| 03:38:44 | Kit, asset audit and environment setup succeeded; reset failed at the Warp API's int32 index boundary. Corrected SDK indices while preserving Torch gather indices; regression executes actual boundary methods. |
| campaign 03:45:00 | Host argument rejected a descriptive non-hex source-commit label before physics. Relaunched with explicit `uncommitted`; each run archives exact source bytes and hashes. |

`startup_attempts/` and `campaigns/` contain reports and raw logs. `supervisor.json` identifies the exact owned container, source manifest/archive hashes and cleanup outcome. Full source tarballs remain on Spark beside each report; earlier attempts 1–3 predate tarball capture and retain hash manifests only. Source checkout during this campaign is `/home/orionh/HEXAPOD_runs/mkii_fourbar_v1/source`; output directories are separate. Do not describe working-tree runs as clean-commit runs.

## Limits

The RS05 model uses vendor torque-speed and bounded overload tables with provisional 48 V, gains, cooldown/recovery, cooling and friction assumptions; it is not hardware identification. Current collision geometry is an approximation: the [independent collider-fit audit](collider_fit/README.md) found real pad surface gaps up to 6.38 mm and direction-dependent overcoverage. Initial flat-ground software training does not establish accurate rough-terrain contact, self-collision, a safe full joint workspace or hardware readiness. Sensor navigation and polygon coverage remain separate program streams.
