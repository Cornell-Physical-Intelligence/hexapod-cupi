# PPO repair 003: quiet standing is still unresolved

Two independent 50-update PPO trials improved tracking slightly but did not suppress the standing limit cycle. Neither earned continuation. Stage 2 remains incomplete; no new full qualification or walking video was claimed.

Every column below uses the same C robot, checkpoint origin, 12 scenarios, four replicas per scenario, seed 7057, 2–12 second settled measurement window, 1.6 N·m applied cap and **0.03 rad per 20 ms diagnostic limiter**. These are controlled C-study diagnostics, not the archived Stage2C formal measurement at 0.040 rad per 20 ms. Do not combine the two lineages in a performance table.

| Metric | Original checkpoint | A: lower exploration | B: added standing-action cost |
|---|---:|---:|---:|
| Median planar error (m/s) | 0.03951 | 0.03606 | 0.03619 |
| Median yaw error (rad/s) | 0.09178 | 0.08335 | 0.08431 |
| Median requested torque saturation | 5.744% | 5.676% | 5.531% |
| Standing joint velocity RMS (rad/s, four replicas) | 0.75014 | 0.73740 | 0.74017 |
| Terminations / truncations | 0 / 0 | 0 / 0 | 0 / 0 |

The medians do not establish quiet standing. The standing scenario's saturation rose from 5.75% to 5.85% (A) and 6.03% (B). In the recorded standing replica, approximately 90% of target steps still reached the limiter, and body drift over the settled window remained about 13 cm. The reward reduced raw action mean-square only slightly. [Full per-direction and trace measurements](results/README.md) retain all cases and distinguish one-replica traces from four-replica summaries.

![Standing targets still oscillate](results/standing_target_trace.png)

![Per-direction tracking and motor demand](results/direction_comparison.png)

Both trials preserve the learned actor, critic and observation-normalizer tensors exactly before training. They explicitly reset exploration standard deviations to 0.10 and Adam moments, set entropy coefficient to zero, and start at learning rate 0.00005 with the existing adaptive schedule. This is a documented fine-tune initialization, not an optimizer-state resume. Only B adds a standing-only mean-square penalty on raw sampled actions before clipping/slew, weight −2. The actor remains active. See [method and launch identity](METHOD.md), [the exact plan differences](results/sources/plan_a_to_b.patch), and [provenance checks](results/provenance_checks.json).

The experiment completed as user service `hexapod-omni-repair-pair-003-20260909.service`; forecasting timers were restored under pause 012 after exit. [Recorded campaign](results/repair_pair.json) and [restoration](results/forecast_pause_012/restored.json) are historical evidence; live job state belongs in [STATUS](../../../STATUS.md).

## Source and checkpoint preservation

[The result manifest](results/SHA256SUMS.json) binds the exact downloaded payloads, including both checkpoints and diagnostic traces. [Changed frozen source hashes](changed_source_hashes.json) bind the five source files retained in `frozen_changes/`. The full source manifests and training plans for A and B are in `results/sources/`; their code/asset hashes match, with only the intended training-plan reward difference.

The behavioral patch was ported onto main while preserving main's pinned-runtime bootstrap. Future main launches additionally reject differences in non-plan source hashes and nonfinite applied-torque maxima before interpreting a comparison. Those extra admission checks were CPU-tested after this run; they are not retroactively claimed as code in the frozen Spark snapshots. No previous checkpoint, source or acceptance gate was changed.

The next controller investigation should test a continuous body-twist reference with a bounded learned residual, including exact zero-command settling, reachability and stop/reversal continuity, before committing more PPO time. This remains a contingency until its geometry and simulator checks pass; terrain standing preparation can proceed independently.

Rebuild the numeric comparison and direction plot into a **new** directory with `uv run --with matplotlib python artifacts/omni_diagnostics_2026-09-09/repair_003/compare_results.py --output tmp/repair003_review_new`. The script refuses an existing output and does not rewrite the published result bundle.
