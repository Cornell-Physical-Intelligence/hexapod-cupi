# Candidate003: bounded 50-update pilot rejected

The new target-velocity PPO actor did not learn useful locomotion in this pilot. It also lost the initially quiet stance. Do not continue this checkpoint with the unchanged setup or describe its video as walking success. Stage 2 remains incomplete.

The exact source is `fb5e472bb710daa50d1d7f16a6f3812d2ed845e2400fcc6769859b3fb0b9519e`, using formal `.04 rad/20 ms`, acceleration `8 rad/s²`, unchanged PD `30 / .6`, requested torque rating `1.6 N·m`, observation noise `1`, and no target filter. The `.03` diagnostic profile was not used or pooled. Source003 had already passed full standing, sampled exploration for 32 replicas over 20 seconds, and actual two-update RSL learning/save/load checks. That admission did not establish locomotion.

Training ran exactly 50 scratch PPO updates with a zero actor mean head, initial standard deviation `.005`, fresh Adam at `1e-4`, adaptive learning rate, and zero entropy bonus. Learned standard deviation was `.00456–.00509`. Initial and final evaluations used the same 12 direction cases, four replicas per case, and the same quiet/stop trials. All four observation/history/state audit errors were zero. Checkpoints:

- Initial: `bae4f415ad7c2e63b2dbe8cfb8a95d348cdfc0da89d3d79e64bdd9fac4377eaa`
- Final: `88f8e60a7b1536229f51524cdc646f06d53e4f33cabb74aa89ebde73d06f7247`

| Test | Initial | Final |
| --- | ---: | ---: |
| Quiet stand replicas passing | 48/48 | 0/48 |
| Stop-to-stand replicas passing | 48/48 | 0/48 |
| Quiet stand terminations | 0 | 33 |
| Stop-to-stand terminations | 0 | 9 |
| Quiet worst joint velocity RMS | .00889 rad/s | .14537 rad/s |
| Quiet worst joint position range | .00589 rad | .63175 rad |
| Quiet worst-joint requested saturation | 0% | 68.47% |
| Stop worst-joint requested saturation | 0% | 100% |

The short directional review also fails: final translation remains about `0.0000–0.0004 m/s` for `0.10–0.20 m/s` requests, with negligible turn response. Forward mean velocity is `−0.000218 m/s` for a `+0.10 m/s` request. Short-window requested saturation rose from zero to `1.54%` at stand and `1.39%` forward. Longer quiet trials expose much larger failures; short-window smoothness does not establish stable holding.

## Measured failure mechanism

Small deterministic velocity biases accumulate into substantial position drift. During the 10-second post-settle diagnostic window, maximum per-joint target range is `.048–.124 rad` across cases, including `.069 rad` at stand. Forward `revolute_3` has mean normalized action `+.00594` and standard deviation `.00202`, yielding approximately `+.119 rad` target drift while the body does not move forward. This is not clipping against the formal `.04` step budget: typical largest-joint target-step p95 is only `.00014–.00037 rad`.

Requested torque is explained by the actual PD target error: `30*(target−position)−.6*joint_velocity` matches recorded torque with RMS residual `.0016–.0058 N·m` across final traces. The evidence identifies accumulated target bias as the immediate cause, rather than a high-frequency action oscillation or an observation-history mismatch. Physical finite-difference and endpoint joint velocities are recorded separately in [analysis.json](analysis.json); reset transitions are excluded from target-step calculations. Full per-replica and named-joint evidence is retained in the source diagnostic/quiet JSON files.

The admitted low-noise exploration did not discover useful swing/contact behavior in 50 updates. Its airtime reward remains zero or negative. This does not establish that target-velocity control is impossible, but it gives no reason to spend another unchanged long run on this checkpoint.

## Next bounded experiment

Prepare a new lineage with a finite bounded position residual around an executable gait reference. A constant actor bias must produce a bounded offset, not an indefinitely integrated target. Initialize the residual at zero, retain feedback at zero command, and keep the quiet reference stationary when no movement is requested. Do not copy old checkpoint tensors into a different action/observation contract.

Before PPO, test zero-residual standing and one low-speed forward swing/stance sequence in full C-robot contact physics. The support-aware reference is a geometry seed that still needs measured pose/contact reconciliation; its ideal planted-foot result is not physical admission. Benchmark1 can inform useful swing/contact shape, but its archived torque violations prohibit treating it as an admitted teacher. Preserve forward/left/yaw command semantics and leave all-bearing, reverse, yaw, arcs, transitions and terrain extensions explicit. The root agent owns dispatch after reviewing the next source.

## Reproduction and provenance

`python3 tmp/omni_velocity_pilot_003_review/analyze.py` regenerates `analysis.json`. All ten downloaded input files were compared by SHA-256 against the remote originals; see [REMOTE_INPUT_SHA256.json](REMOTE_INPUT_SHA256.json). These are compact diagnostic evidence and a derived analysis, not new simulator output. The guarded campaign and restoration receipts are managed by the root agent. The honest video should expose the actual final policy and stop on any terminal event.
