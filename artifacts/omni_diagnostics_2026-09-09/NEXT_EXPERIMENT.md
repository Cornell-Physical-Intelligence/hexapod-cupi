# Evidence and next PPO experiment

**Superseded experiment proposal, 9 September 2026.** The 100-update continuation
specified below was executed and rejected: median requested saturation rose from
5.74% to 6.64%, standing motion barely changed, and forward yaw tracking worsened.
Its planned longer continuation did not run. Preserve this proposal as the reason
for that test; do not treat it as the next queued action. See the
[pilot outcome](repair_002/repair.json), [control comparison](comparison_report.json)
and canonical [STATUS](../../STATUS.md) for current control probes and next steps.

The previous policy's jitter is physical: the joints repeatedly change their targets and positions during a zero-velocity command, and the body's heading changes corroborate the reported gyro motion. Observation history, repeated reads and command encoding passed the diagnostic checks exactly. Reset contamination is not the explanation: standing had no resets.

Three probes used the same checkpoint, 12 command cases, four replicas per case, 12 seconds per case and the same seed. Each had a full standing admission for its exact input plan. Measurements were captured before automatic resets; steady summaries exclude the first two seconds of every episode. These probes diagnose behavior and do not qualify the policy.

| Probe | Requested torque saturation, median | Positive mechanical power, median | Base-contact terminations |
|---|---:|---:|---:|
| Original | 14.93% | 4.94 W | 4 |
| Target slew 0.03 rad / 20 ms | 5.74% | 2.30 W | 0 |
| External forces every solver iteration | 15.26% | 5.27 W | 4 |

Halving the target slew reduced requested saturation by about 61.5%, positive mechanical power by about 53.5%, and eliminated the four base-contact terminations in this short probe. Median tracking errors improved slightly. The solver option did not improve the measured behavior, so the next experiment retains the existing solver setting. This does not prove the solver option can never be useful with a different task or controller.

The median hides an important tradeoff: fast-forward planar error increased from about 0.041 to 0.089 m/s, and the left arc increased from about 0.040 to 0.058 m/s. Their yaw errors also worsened. The policy was trained with faster target changes, so the slower action response needs adaptation. These directions must recover during the short continuation; the smoother motor behavior cannot waive tracking failures.

Quiet standing still fails badly. In the recorded standing replica, slowing targets reduced aggregate joint velocity RMS from 1.44 to 0.74 rad/s, but a joint still traversed 0.275 rad (15.7 degrees), target steps frequently hit the new limiter, and the body drifted 16.8 cm over the ten-second steady window. Zero command must become a quiet, stable posture with feedback available for disturbances. Rigidly freezing the articulation would conceal the problem.

## Historical 100-update continuation proposal

Keep the existing 315/318 actor/critic observations, checkpoint, geometry, actuator limits, command sampling, PPO settings and qualification limits. Apply:

```json
{
  "target_slew_rad_per_20ms": 0.03,
  "reward_weights": {
    "stand_joint_velocity": -0.5,
    "stand_target_velocity": -0.15,
    "stand_posture": -2.0,
    "action_rate": -0.075,
    "saturation": -0.75,
    "torque_excess": -0.6,
    "worst_torque_excess": -0.2
  }
}
```

The new standing terms penalize mean squared physical joint speed and processed target speed only at near-zero commands. They are zero by default and never disable the actor. The other cost changes address the measured repeated target reversals and motor clipping. Costs use their documented existing units; no actuator cap or termination threshold is relaxed.

First evaluate the held checkpoint with these exact settings, then train 100 updates and repeat the short diagnostic. Continue only with measurable reductions in standing motion and motor demand while retaining or improving each direction's tracking and failure counts. A promising result must still pass the full held-out direction/transition suite and actual-video inspection. No update budget constitutes completion.

## Completion includes all directions and pathing

The user chose the accepted forward benchmark and current straight trial as the visual smoothness standard. Stage 2 remains incomplete until reverse, both strafes, diagonals, all translation bearings, both turn directions, gentle/tight arcs, S curves, fixed-heading curves, reversals and stopping look comparably smooth. Benchmark 1 remains frozen.

The comparison JSON contains per-direction joint speed, target increments, actual pose excursion and applied torque increments. Its proposed quiet-stand and stop-to-stand bounds are engineering targets, not measured hardware standards. They require a separate 30-second undisturbed standing trial and stop trials; disturbance recovery is tested separately so needed balance corrections are not penalized as jitter. Instrumented forward-reference metrics should establish the numerical smoothness comparison before declaring completion.

The full evaluator now captures physical state before resets and counts settling time per episode, while preserving every failure and existing acceptance gate. This corrects the prior evaluation's reset-transient contamination without treating it as a remedy for the real standing oscillation.

Positive mechanical power is not battery power. The short probes use nominal simulator physics and do not establish physical motor thermals or deployment readiness.
