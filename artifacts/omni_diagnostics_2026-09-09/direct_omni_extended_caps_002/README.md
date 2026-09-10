# CAPS500: completed learning, quiet standing still fails

The fixed-budget normal-CAPS experiment completed 500 PPO updates and all six acquisition phases on the full C-study robot. Its final checkpoint is `c376a0a4eb04d54396b4fd6171fe173167767463245213cce7c2cc1d3a2877cf`. Strict actor, critic, normalizer, optimizer and deterministic-action reload passed. This is a completed experiment, **not a Stage2 pass**: all 48 quiet-stop trials still fail the unchanged gates.

Read the [complete numerical report](cpu_analysis/remote/analysis/REPORT.md), [machine-readable analysis](cpu_analysis/remote/analysis/report.json), and [terminal audit](curated/audit/remote_terminal_audit_001.json). The analysis independently replays the original raw training and stop data on Spark. It reports no forensic errors; every original campaign/cold input hash matches before and after analysis.

## What the result establishes

The learner collected 12,288,000 transitions in 12,000 controls across 1,024 replicas and applied 10,000 optimizer minibatches. Actual learning and finalization took 1,033.245 seconds, excluding simulator startup and the other acquisition phases. Seven decision checkpoints and all 500 ordinary saves remain recorded. This run starts afresh from original checkpoint1971…, with the original C geometry, 1.6 N·m applied cap, direct315/318 observations, 0.040 rad/20ms limiter and unchanged physical/scoring contracts.

The final constant-command results improve planar tracking in all 12 cases against this run's initial original policy. Yaw tracking worsens in five, and requested-torque saturation worsens in nine. These are descriptive differences, not statistical significance or all-direction qualification. The stop results retain joint motion, target changes and torque-limit requests. The maximum scored excursion of 1.972m includes a reset inside the quiet window; it must not be described as continuous physical drift. Every reset remains a failed trial.

The native receipt and terminal audit verify source004's599 files, all550 C assets, native005, host005, guard002, the original supervisor and all12 owned container names/IDs absent. The unit completed successfully under invocation `ca9455cefb554c7691e7f8b2ed3ac939`; pause063 restored the previously active weather timers. Historical smoke and the earlier launcher-only failure remain separate immutable evidence.

## Complete remote analysis, explicit local subset

The terminal inventory contains576 files totaling3,251,065,549 bytes. The [curated bundle](curated/README.md) publishes75 selected raw files totaling260,791,365 bytes, including all evaluation traces, seven decisions, final weights, training event/joint data, logs and receipts. All500 ordinary autosaves (2,189,882,500 bytes) and the required800,391,684-byte training trace remain hash-inventoried on Spark. They are distinct omission categories. **This public subset cannot replay the complete training analysis locally.** No missing raw input is represented as an omitted optional checkpoint.

The frozen analyzer ran through the prepared CPU executor using an existing NumPy interpreter, one CPU thread and no GPU visibility. Its [execution receipt](cpu_analysis/remote/execution_receipt.json) records7.532 seconds,1,470,640KiB peak child RSS, all three analysis outputs, unchanged complete input trees and `evidence_verified=true`. Original absolute input paths remain in its report; they were not rewritten to hide remote-only inputs.

`verify_bundle.py` checks every included byte, the nested curated/analyzer inventories, actual result identities and the recorded complete-remote inventory against the terminal audit. It does not contact Spark or claim that remote-only bytes can be reconstructed from this subset. Later physical comparisons, checkpoint selection and recording are separate evidence; no automatic next training run or gate change follows from completion.
