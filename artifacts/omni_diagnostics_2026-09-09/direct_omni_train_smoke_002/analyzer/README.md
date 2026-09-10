# Direct PPO result comparison

This CPU analyzer is bound to native003 contract freeze
`20fbcd40a869b39d8e3143a6c9715ec62ada5cc6211eaf473f60f8b7167678cb`,
598-file source `64144b2c2466f17c809554ef53f82b736b4199136cc821451d7ac2788752bd2e`,
and plan `9fb7d0b01480523972b4f1c17e717756d7dcf1c54b9caa436245671937de538c`.
It does not launch, train, mutate inputs or change acceptance gates.

```sh
.venv/bin/python3 tmp/direct_omni_train_analysis_002/analyze.py \
  --campaign /absolute/path/to/fetched_direct_campaign \
  --cold /absolute/path/to/direct_omni_cold_results_001/run \
  --output /absolute/path/to/fresh_review_directory
```

The output is `REPORT.md`, `report.json` and `INPUTS_SHA256.json`. All consumed
inputs are hashed before reading and checked again before writing the report.
The output must be new and outside both input trees. A missing/failed phase
produces an explicit partial-evidence verdict; it is not called successful.
Phase source/asset/runtime identity, accepted state hashes, exit status and owned
cleanup receipts are checked. This checks the fetched receipts, not a new remote
container or full asset audit; root owns that independent terminal audit.

For the smoke, the constant-command baseline is the actual cold004 run. It has
no moving-to-stop baseline. Final stop/quiet values are therefore absolute
measurements, never invented improvement ratios. For each pilot the analyzer
uses its own cold initial constant and stop phases against final phases.
Constant comparisons require identical commands, profile, reward weights,
observation timing/options and named joint order. A 0.03-slew run cannot be
pooled with the formal0.04 data.

Constant JSON summaries cover four replicas per direction; their raw joint
traces cover only one replica per direction. This limitation is explicit and
unrecorded per-replica values are never reconstructed. Each direction reports
planar/yaw error, requested saturation/peak, applied torque, nonfoot body count,
tilt and terminations, with named joint metrics and trace-derived target steps.
Every positive delta is labeled a measured regression, without inventing a
statistical significance threshold or an aggregate promotion score.

All48 moving-to-stop replicas are replayed from raw trace. The exact new
1600-control schedule and explicit XYZW→WXYZ conversion are checked; the final
ten-second quiet window is recomputed using the unchanged seven bounds. A reset
anywhere in the trial still invalidates the replica. The report preserves failed
bounds and all individual values. Adjacent-angle interval averages and raw SDK
joint velocities remain separate; neither replaces an existing gate. Nonfoot
contact environment-steps are derived from available pre-reset reward terms.

Training reports actual completed optimizer calls, controls/transitions,
per-row zero-command exposure, contact/failure counts, torque, target movement,
losses, learning rate, wrapper throughput and separately parsed native RSL
collection/learning times. It verifies every raw event ledger row against its
pre-reset trace, audit counts and checkpoint/decision hashes. Strict reload is
identified as the actual runtime's actor/critic/normalizer/Adam/action proof;
the analyzer checks its provenance and model hash but does not claim to rerun
the actor. No loss-only or median-only promotion is allowed. All telemetry is
50Hz endpoint data, not400Hz motor qualification or a claim that SDK rates are
physically unbiased.

Focused tests:

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=tmp/reference_residual_ppo_001/_deps:tmp/direct_omni_train_analysis_002 \
.venv/bin/python3 -m unittest discover \
  -s tmp/direct_omni_train_analysis_002 -p 'test_*.py' -v
```

Five tests cover actual cold self-comparison and a single-direction regression,
the full native synthetic stop rollout and an altered quiet verdict, a hidden
last-replica torque/event failure, input mutation, and exact RSL timing formats.
The synthetic rollout remains explicitly a software fixture; no new physical
result is included in this frozen analyzer preparation.

Successor002 changes only the source/native pins and human formatter handling of a missing reload receipt (`None`). Scorers, raw consistency checks, completed-campaign requirements and all physical gates are unchanged. A new test reads the actual failed smoke001 training receipt/trace: two observed optimizer updates and 1536 transitions remain visible, reload is false, and continuation is not admitted. That old run is a formatter fixture; it is not relabeled as the new source. Six tests pass.
