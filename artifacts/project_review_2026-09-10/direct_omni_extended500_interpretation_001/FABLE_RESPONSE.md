The 10x budget shifted constant-command tracking by up to 11 mm/s and left the quiet failure unchanged. Separately, the two before/after tables use different baselines, which must be resolved before any regression claim is carried forward.

## What the data establish

**Quiet failure is budget-invariant.** Both checkpoints fail all 48 replicas on every bound, with a near-identical signature:

| Quiet metric | CAPS50 | CAPS500 |
|---|---:|---:|
| SDK joint velocity RMS, median, rad/s | 1.274 | 1.263 |
| Worst adjacent-angle interval RMS, rad/s | 1.502 | 1.478 |
| Joint position range, median, rad | 0.318 | 0.297 |
| Replicas with target-step p95 pinned at 0.0400 rad per 20 ms | 48 | 48 |
| Limiter active fraction, median | 0.863 | 0.868 |
| Per-env mean requested saturation, median | 0.105 | 0.111 |
| Quiet requested peak, Nm, against 1.6 applied | 6.99 | 5.82 |

With zero command, the executed target moves at the slew cap on roughly 86 percent of steps. A 2 rad/s cap at the provisional 4.4 Hz peak yields about 0.23 rad peak-to-peak, the same order as the 0.30 rad observed range. That is a consistency check only. It says the historical limiter bounded the amplitude of the executed oscillation, not that it set the frequency.

**The oscillation originates upstream of the limiter.** In the provisional receipt over 47 reset-free rows, raw action holds 75.9 percent of power in 2 to 5 Hz, executed target 64.2, actual q 54.4, all peaking at 4.4 Hz. The limiter and plant attenuate but do not remove it. Whether the cycle is sustained by observation feedback through the plant or is intrinsic to the actor is not separable from these traces.

**Constant tracking moved slightly.** Relative to CAPS50, planar error fell in 11 of 12 directions by 1 to 11 mm/s, yaw error rose in 6 of 12, and requested saturation rose in 9 of 12. Constant-case mean saturation rose from 10.12 to 10.47 percent while requested peaks fell from 7.11 to 6.72 Nm. Four replicas per direction and one seed pair support no significance statement.

**Optimizer facts have a limit.** Temporal CAPS loss fell from 0.317 to 0.242 while executed quiet metrics did not move, so the loss is not a proxy for the gate. Sampled PPO actor gradient norms of 4 to 10 dwarf quiet norms near 0.05, combined norms are clipped to 1, and PPO-quiet cosines sit near zero. These are sparse pre-Adam rows and do not measure the parameter update.

## Inconsistencies and rejected claims

- **Baseline mismatch, review required.** The per-direction table labeled 50 to 500 and the analyzer table share every "after" value but differ in every "before" value, for example stand planar 0.0307 versus 0.0379. The analyzer baseline appears to be a different checkpoint, probably the preserved original, but is unlabeled. Against CAPS50, arc_right planar regressed by 1.9 mm/s and turn_left yaw regressed slightly, which the analyzer's regression lists omit. Its zero termination delta for diagonal also cannot hold against CAPS50, which terminated a diagonal replica at 20.86 s. Label the baseline and regenerate the lists.
- **Reset-contaminated mean.** The CAPS500 quiet mean of 75.8 mm and maximum of 1.97 m come from environment 20 terminating on base contact at 30.24 s inside the scored window. That is a counted trial-reset failure, not drift. Use medians or the common 46-environment subset, 49.9 mm versus 34.3 mm.
- **Bound values absent.** The quiet bounds are not in the supplied evidence, so I can confirm that all 48 fail but not by how much.

Rejected as unsupported:

- Budget produced a better controller. Only small descriptive tracking deltas exist.
- Falling temporal loss means quieter execution. Refuted by unchanged quiet metrics.
- Gradient-norm imbalance caused the failure. Adam and clipping effects were not measured.
- The limiter, observation noise, or an 80 ms target filter is the cause or the cure. The filter test worsened saturation.
- 1.6 Nm and 0.04 rad per 20 ms describe the new robot, or any 18-action checkpoint transfers to it. Neither holds.

## Next bounded work: admit the new model, then diagnose

Three lessons carry over. The sensing floor was argued about but never measured. The rate limit dominated executed motion, so it must be declared and its effect measured. The quiet gate must be applied to a null policy before a learned one.

Decisive policy-free measurements on the canonical URDF:

1. **Declared envelope.** Record the sourced torque cap, rate limit, and gains in the selector before any trace is scored. Nothing inherited from the study.
2. **Suspended constant-target holds** per named joint: residual velocity RMS, position range, and requested torque over 10 s. This is the sensor and actuator floor. If it approaches the historical 1.27 rad/s, the quiet gate is unmeasurable on this stack.
3. **Suspended controlled excitation** at 2, 4.4 and 8 Hz with target rate at the declared limit: gain, phase and saturation of q against target. This shows whether the plant and limiter themselves favor the historical band.
4. **Settled contact** at the declared zero pose: summed normal force against 73.2 N, base settling, drift and contact chatter over 10 s, plus a bounded sensitivity sweep on the provisional motor-mass cylinders.
5. **Null-policy quiet check.** A hold controller with the declared gains under the configured observation noise must pass every quiet bound. Only then does a learned-policy failure carry meaning.

Conditional ranking for later control work:

1. **Sensing, action and physics diagnostics** first. The failure was budget-invariant and began in the raw action, so the observation-to-action loop is the unknown.
2. **Objective and architecture** second, only if the null policy passes and a fresh policy still oscillates. The decisive measurement is raw-action band power and executed target-step p95 under zero command at a fixed budget.
3. **More budget** last. A 10x increase changed no quiet metric.

Do not adopt the historical policy or its geometry on the new model, and route the baseline-label correction through the required site update before these numbers are quoted anywhere.
