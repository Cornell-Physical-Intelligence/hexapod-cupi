# Control probes: the learned policy causes the standing oscillation

The plant can stand quietly with active PD control and the existing 1.6 Nm actuator cap. The learned policy does not. Removing synthetic observation noise makes little difference, and the tested 80 ms position filter worsens motor demand and several directions.

All comparisons below use the **same original checkpoint**, C geometry, stance, runtime joint order, 0.03 rad per 20 ms slew limit, 2.5 ms physics step, seed, 12 command cases and four replicas per case. Measurements cover 12 seconds and exclude each episode's first two seconds. Identity checks and source-result hashes are in [comparison.json](comparison.json). The baseline is the exact-settings baseline from [repair_002](../repair_002/baseline/diagnostics.json), not the earlier 0.06 slew run.

| Learned-policy probe | Median requested saturation | Median planar error | Median yaw error | Median positive mechanical power |
|---|---:|---:|---:|---:|
| Baseline | 5.74% | 0.0395 m/s | 0.0918 rad/s | 2.30 W |
| Observation noise removed | 5.80% | 0.0393 m/s | 0.0916 rad/s | 2.30 W |
| 80 ms target filter | 8.08% | 0.0332 m/s | 0.1135 rad/s | 2.72 W |

All probes had zero terminations and truncations. The filtered policy's better median planar error does not outweigh its regressions: forward yaw error increased from 0.0889 to 0.1636 rad/s, right-arc yaw error from 0.1093 to 0.1717, and diagonal yaw error from 0.1032 to 0.1742. Every direction remains subject to its own checks.

## Standing isolates the cause

| Controller | Joint velocity RMS | Requested saturation | Physical heading-rate standard deviation |
|---|---:|---:|---:|
| Learned baseline | 0.7501 rad/s | 5.75% | 0.06986 rad/s |
| Zero normalized action, active PD | 0.00158 rad/s | 0% | 0.000026 rad/s |
| Learned, observation noise removed | 0.7492 rad/s | 5.80% | 0.06650 rad/s |
| Learned, 80 ms target filter | 0.7399 rad/s | 8.43% | 0.06466 rad/s |

The zero-action control holds nominal joint targets through the real simulated actuator; it does not freeze the body or articulation. It is a plant control, not a proposed walking controller. Its steady planar speed was 0.00022 m/s. This makes another speculative PD, solver or sensor-noise change a poor next use of the GPU.

Observation/history/command audits again returned zero discrepancies. These compact reports establish persistent physical motion but do not establish its new spectral peaks or accumulated displacement; no fresh trace files were downloaded for this review.

## Next bounded PPO intervention

Retain the original checkpoint, 0.03 slew, full observation noise, existing PD gains and actuator cap. Remove the unsuccessful filter. Run two **50-update** branches:

1. **Low exploration:** existing repair rewards, explicitly initialize training action standard deviation to 0.10 and set entropy coefficient to zero.
2. **Low exploration plus quiet-action cost:** the same setup, adding only a standing-only mean-square cost on the **unclipped sampled normalized action**, with weight −2.0. Keep the current command mixture so the extra cost is the isolated difference between branches.

Retain actor/critic weights and observation normalizers. Record the exploration and optimizer initialization explicitly; merely changing `init_std` in a config is insufficient if loading the checkpoint overwrites it. Neither branch disables the actor. Moving commands retain their learned control, while quiet-action regularization points toward the empirically quiet stance and still permits feedback corrections.

The prior [policy-loop analysis](../repair_002/policy_loop_analysis.json) found that the processed target hits its slew limit in about 90% of standing joint-step samples. Its new motion penalty therefore remained almost constant during the failed pilot. Penalizing raw action intent before clipping and slew avoids that plateau. Lower exploration should make small control improvements easier to measure; the checkpoint's saved distribution parameter had grown substantially above its original initialization.

Evaluate each branch with the same short diagnostic. Continue only one that improves standing motion and torque while preserving each direction. If neither improves, stop reward-only retries and prepare a smooth command-conditioned reference with bounded residual PPO, using the accepted forward benchmark as the reference. All numerical gates, sustained quiet/stop tests and all-direction visual review still apply before Stage 2 can be called complete.

These are single-seed causal screens, not hardware or terrain qualification. Positive mechanical power is not battery power.

Raw compact results: [zero action](zero/diagnostics.json), [no observation noise](no_noise/diagnostics.json), [filtered requests](filtered/diagnostics.json).
