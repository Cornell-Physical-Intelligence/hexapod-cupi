# Physical MKII four-bar campaign, 5 September 2026 UTC

This is a separate physical-loop training lineage. It preserves the archived mock and corrected serial results. The new asset has 31 bodies, 30 articulation coordinates, six excluded closure joints and 18 RS05 drives. Only the active motors receive actions; passive knees and rods close physically in the simulator.

## Evidence at publication

The first live physical probe passed: `fourbar-campaign-20260905T034653Z-a83056cd/probe/hexapod-fourbar-validate-20260905T034654Z-91ce5326`. It completed 100 control steps / 400 physical substeps with one robot. Settled peak torque was 0.67483574 N·m, six pads supported the robot, nonfoot ground contacts were zero, mean plate height was 0.13569902 m, and maximum settled physical closure error was 1.8253 µm. Startup torque peaked at 1.81281567 N·m and closure error at 65.9877 µm. This short result does not grant full training admission.

The active campaign proceeds through 32-environment standing plus 2,400 driven control steps at nominal and refined solver settings; only matching passes admit a 64-environment / 3-iteration scratch PPO and checkpoint/inference check, followed by a separate 512-environment / 1,000-iteration resume process. Campaign JSON records actual progress. An incomplete campaign is not a successful training run.

Local release checks: 784 unit tests passed in 21.720 seconds. The historical 112-file manifest verifies against its immutable Git source; a separate 268-file manifest verifies this release. Two package-definition files changed to package the new versioned modules/data. No old checkpoint or archived manifest was rewritten.

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
