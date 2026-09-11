# Solver diagnostic decisions and actual velocity evidence

The 32/4 single-robot candidate was genuinely rejected by the unchanged SDK quiet-rate gate. It completed all 8,000 controlled steps and passed the physical support/motor/nonfoot checks; it did not admit 32 robots. The selected next comparison is source005 32/0, with unchanged position iterations, geometry, PD derivative, material, dt and gates, plus read-only legacy-friction, link-velocity and filtered-floor-aggregate diagnostics. This receipt makes no claim that it has run.

## Actual evidence

`SINGLE_COMPARISON.json` consumes the hash-verified arrays of actual single native4 (source003, 32/1) and native5 (source004, 32/4), without copying raw data. Every one of the 8,000 interval-angle recurrences in each run is exact. The table below uses all 6,400 post-settle 400 Hz rows, distinct from the official 50 Hz endpoint quiet statistic.

| Quantity | 32/1 LM / RM tibia | 32/4 LM / RM tibia |
|---|---:|---:|
| Signed SDK mean, rad/s | −0.022955 / −0.023016 | −0.038729 / −0.039248 |
| SDK RMS, rad/s | 0.022977 / 0.023039 | 0.038749 / 0.039268 |
| Interval-angle RMS, rad/s | 0.001668 / 0.001830 | 0.001683 / 0.002226 |
| Position range, µrad | 41.33 / 52.65 | 48.03 / 52.61 |
| SDK minus interval integral over 16 s, rad | −0.36728 / −0.36825 | −0.61966 / −0.62798 |

The official maximum 50 Hz SDK quiet RMS increased from 0.023023 to 0.039215 rad/s. The large signed mean is a persistent discrepancy relative to the constrained position trajectory; calling it a specific solver residual or fictitious motion would require more evidence. [PhysX documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/Simulation.html) explicitly allows a difference between constrained pose increments and reported velocities. The existing derivative input and gate are retained.

The earlier 32-robot source003 result also contains actual short motion transients. `EVIDENCE.json` independently reduces the preserved analyzer002 result and reconstructs all 222 adjacent-event joint intervals exactly; it does not claim to reread the 4.969 GB remote raw run. Across 74 single-toe zero-resultant events, other feet carried at least 10.74675 N. All affected toes had 128 exact inactive tuples, but no event had all six feet or all affected-leg SDK velocities zero. A complete-articulation sleep signature is absent. The zero-force data alone do not distinguish separation from reporting or constraint behavior.

The event tibia interval rate reaches 1.3591 rad/s and endpoint SDK rate 2.6449 rad/s. RMS residual against the endpoint SDK rate is 1.0416 rad/s, versus 0.03361 against the average of adjacent SDK endpoints. That is consistent with substantial within-step acceleration; it does not prove an integrator formula. Finite differences are interval averages, not instantaneous endpoint velocities. A different joint has larger whole-window interval RMS than SDK RMS, so finite differences are not a uniformly cleaner replacement. No concrete application-level stale-read, index, frame or counter bug was established.

## Bounded decision table

| Actual outcome | Decision with existing gates |
|---|---|
| 32/4 single fails quiet, physical otherwise passes (observed) | Preserve rejection and skip its 32 launch. Run the separately reviewed 32/0 single diagnostic only under root's allocation. |
| New single fails source/API/counter/shape or native integrity | Preserve partial output and repair only the demonstrated implementation fault in a successor; do not infer physical pass from missing data. |
| New single fails support, nonfoot, clearance, motor or actual-motion bounds | Do not launch 32 or PPO. Use the actual first failure and new filtered aggregate/legacy readback to narrow the next comparison. No unconditional gain or solver sweep. |
| New single still fails SDK quiet RMS while angle excursions remain small | Preserve failure. Compare signed means, stored PD recurrence, legacy coefficient and independently copied contact channels; do not substitute finite differences or automatically clear friction. |
| New single passes every original gate | Admit only the same frozen source's 32-robot standing screen. A single pass does not predict replica success or identify hardware fidelity. |
| Same-source 32 completes but any replica fails | No PPO admission. Separate real angle/support transients from persistent SDK discrepancy using the full raw time series; scope any next native probe to the demonstrated failure. |
| Same-source 32 passes every original gate and terminal integrity | Root may bind a new PPO integration source to that exact standing source and actual1/32 receipts. No silent rebinding of frozen old preparation or automatic broad training. |

Zero/few TGS velocity iterations are a documented numerical diagnostic option, not a guaranteed repair. The D6-drive and tendon-specific statements on the [official limitations page](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/current_limitations.html) are not generalized to this robot.

Reproduction: run `analyze_events.py` for the exported event reduction and `compare_single_32_1_32_4.py` for the exact actual single-array comparison. Their source and input hashes are retained. `verify_bundle.py` checks artifact integrity only. Root owns publication and required `docs/PROJECT_SITE.md` updates; no living docs, GPU or tracked sources were changed here.
