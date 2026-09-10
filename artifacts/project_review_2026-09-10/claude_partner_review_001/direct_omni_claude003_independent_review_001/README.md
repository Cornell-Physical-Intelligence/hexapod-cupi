# Independent check of the narrowed PPO audit

The actual 50-update evidence disproves the proposed stuck-learning-rate and missing-curriculum-reset explanations. Quiet target chatter is confirmed directly. No new runtime code defect is established by these three claims, and neither pilot is qualified.

| Actual evidence | Curriculum | CAPS |
|---|---:|---:|
| Update-end LR at floor | 14/50 | 13/50 |
| Median update-end LR | 1.5e-5 | 2.25e-5 |
| Maximum update-end LR | 5.0625e-5 | 5.0625e-5 |
| Eligible adjacent reset plateaus changed | 601/601 | 599/599 |
| Final quiet target increments at .04 rad bound, joint/control fraction | 87.48% | 87.54% |
| Median final quiet deterministic raw-action step RMS | .55940 | .55173 |
| Final quiet admission | 0/48 | 0/48 |

## Learning rate: not stuck at the floor

Both first updates end at 1e-5, but update 2 ends at 2.25e-5 in both pilots. The complete 50-entry paths are in `report.json`. They repeatedly rise and fall; the recorded maxima even exceed the initial 5e-5 slightly. Thus the claim that the entire pilots ran at one fifth of their declared rate is false.

These are **post-update** samples. The callback does not record every minibatch's KL or rate, so this report does not estimate the exact effective rate over all optimizer steps. The two-update smoke's floor endpoints cannot substitute for these actual pilot paths. A floor endpoint also does not prove no learning or a null comparison.

## CAPS: modest aggregate decline, no impossibility proof

The CAPS temporal loss averages .316994 over updates 1–10 and .312438 over 41–50: a **1.44% decline**. The first single value is .303884 and the last .312023, illustrating why one endpoint is a poor trend measure. The spatial loss's corresponding means are .00167116 and .00155829. Valid temporal-pair fraction averages .949245 across all updates. The curriculum branch reports zero auxiliary losses because it deliberately skips these actor calls; those zeros do not mean its policy is temporally constant.

Taking the square root of temporal MSE gives an aggregate RMS scale for policy means under the minibatch states and weights at which each loss was computed. It is not a quiet-only metric, per-joint 95th percentile, fixed-policy replay, or executed motor target. Joint-limit clipping, reset handling and the slew limiter intervene before execution. Changing observation normalization and the training state distribution further prevents interpreting the modest decline as a controlled identical-state comparison. Arithmetic on `.5 * sqrt(loss)` therefore cannot prove a 50-update convergence limit, a required 140-fold reduction, or that no smoothing at all was learned.

The quiet problem is nevertheless real. This review reads **all 48 deterministic final-stop traces** during 22–32 s at exactly zero command, excludes terminal pairs, and computes their actual raw-action and target increments. About 87.5% of executed joint/control increments hit .04 rad in both branches. CAPS raw-action step RMS is slightly lower than curriculum, while the existing quiet target p95 remains .04 in every replica. This independently supports “the tested CAPS allocation did not solve quiet chatter”; it does not support “CAPS cannot solve it in 50 updates.”

## Reset and command coverage: call and redraw confirmed

The exact source002 `HexapodEnv._reset_idx` contains an unconditional `self._sample_commands(env_ids)` at source line 2097. The inherited `OmniFlatEnv._reset_idx` calls it through `super()`. Dynamic dispatch reaches `BalancedOmniEnv._sample_commands`, which zeros commands/targets and invokes `direct_schedule.reset(ids)`. The `omni_evaluation` guard disables the parent's unrelated timer-based command sampler in rewards; it does not block this reset call.

The trace coverage matters:

- `training_trace.npz`: **1,200 controls × 1,024 replicas** for command, elapsed age, terminal flags and summary measurements.
- `training_joint_trace.npz`: **only replicas 0–7** for joint vectors and exploratory raw training actions. This is not full-fleet actor-mean evidence.
- `training_events.json`: all terminal IDs and pre-reset fields; matched exactly to the full trace.
- `final_stop/stop_trace.npz`: 1,600 controls × 48 replicas, including deterministic raw actions and executed targets.

All 256 dedicated quiet rows have exactly zero commands at every captured control in both pilots. Every observed post-reset next sample has zero command and .02 s elapsed age. The ordinary physical age increment agrees with .02 s within 4.96e-7 s. Schedule age itself was not logged and is separate from the simulator's initially randomized episode-length counter.

For nonquiet episodes, this review infers a motion vector only when an actual nonzero plateau was observed. An independent NumPy command-slew recurrence then checks over **1.17 million adjacent command pairs per branch**, agreeing within 1.49e-8. All 601 curriculum and 599 CAPS adjacent resets with visible plateaus on both sides change their motion vector. Another 329/330 resets lack a plateau on one side and are explicitly **not comparable**. Both yaw signs, pure turns, translations, combined twists and every one of 16 translation-bearing bins occur. The longest continuous nonzero command run is 439 controls, consistent with an 8 s motion target plus the finite slew to zero. This confirms meaningful resampling and mode coverage; it does not prove uniform finite-sample coverage or infer unobserved random draws.

## Next bounded PPO choice

Do not spend the next pilot merely “repairing” a nonexistent reset gate or forcing a larger learning rate on the assumption that it was stuck. The actual defect to target is zero-command oscillation before the limiter.

A defensible **proposal**, not an adopted source change, is a single matched 50-update ablation that strengthens temporal policy-mean regularization specifically for consecutive exact-zero-command pairs, while retaining the current moving-pair and spatial penalties, adaptive optimizer, schedule, physics, .04 software slew and all cold constant/stop gates. For example, an explicitly declared extra quiet-only coefficient of .9 would be an experiment, not a paper-derived optimum or a convergence promise. Log quiet and moving temporal losses separately, valid-pair counts and every minibatch's KL/rate. Keep the penalty before target clipping/slew so saturation does not hide the desired command variation or remove its gradient. Do not replace measured quiet gates with the auxiliary loss.

The existing final-stop evidence is sufficient to justify this targeted comparison. Compare all 48 unchanged quiet trials and all constant directions; reject a “success” that simply damages commanded movement or hides requested torque saturation. The minor reduction already observed does not justify an automatic unchanged continuation. A noise-response or action-history-feedback diagnosis may inform a later intervention, but neither is established as the cause here.

## Reproduction and provenance

```sh
python3 -B review.py --campaign-root /path/to/direct_omni_matched_pilots_001 --output /tmp/hexapod_claude003_review_replay
```

Only Python standard library and NumPy are imported. The copied final Claude result is the sole Claude output reviewed. Copied source bytes are bound in `INPUTS_SHA256.json`; the complete original source map and native freeze are retained beside them. `report.json` hashes every consumed raw file; `curriculum_reset_plateaus.json` and `caps_reset_plateaus.json` expose every eligible and ineligible episode comparison. Large immutable traces are referenced from the matched-pilot publication instead of duplicated. `replay.log` contains the actual successful numerical replay, not simulated or proposed results.

All work is read-only with respect to the physical runs, frozen runtime, checkpoints and repository. Publication must append a project-site evidence record under `docs/PROJECT_SITE.md`; a proposed record is included, and the earlier matched-pilot bundle remains immutable.
