# What the locomotion literature suggests changing next

The current reference is a useful physical and recovery test, but it gives PPO little authority to discover a faster gait. Across these 12 representative works, effective controllers either learn joint-position targets directly or can substantially modify a gait generator's timing and foot motion. None of the reviewed recipes establishes a requirement to retain our particular five-support, two-second-swing, ±0.02 rad correction combination.

This is a literature-based inference about the search space, **not evidence that our robot can safely execute a faster gait**. Keep the present frozen pilot, 1.6 Nm actuator cap, raw failure records, and existing acceptance verdicts intact. A different action or support contract requires a separately named experiment. The parallel hexapod/actuator review is responsible for the hardware-specific feasibility comparison.

## Breadth-first comparison

The linked [machine-readable matrix](source_matrix.json) contains authors, robot/actuation, observation and action details, reported compute, limitations, and primary URLs for every row. Missing costs or motor ratings are explicitly unverified.

| Work | Useful comparison with our branch | Reported training cost |
|---|---|---|
| [Learning agile and dynamic motor skills for legged robots](https://arxiv.org/abs/1901.08652) (2019) | Direct joint targets and learned actuator dynamics; TRPO, not PPO. | ~4 h locomotion; desktop CPU/GPU |
| [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://proceedings.mlr.press/v164/rudin22a.html) (2022) | Direct positions, massively parallel PPO, terrain curriculum. | <4 min flat / <20 min rough; RTX A6000 |
| [RMA: Rapid Motor Adaptation for Legged Robots](https://arxiv.org/abs/2107.04034) (2021) | History-based adaptation; original task mainly forward. | ~24 h base + 3 h adaptation; GPU unspecified |
| [Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior](https://proceedings.mlr.press/v205/margolis23a.html) (2023) | Body twist and selectable gait/style; stepping at zero speed is possible. | Not verified |
| [Policies Modulating Trajectory Generators](https://proceedings.mlr.press/v87/iscen18a/iscen18a.pdf) (2018) | Actor modulates generator and corrects output. | <1,000 rollouts in some cases; wall time unverified |
| [Learning Quadrupedal Locomotion over Challenging Terrain](https://arxiv.org/abs/2010.11251) (2020) | Actor controls leg frequency and foot residuals. | ~12 h teacher + 4 h student; RTX 2080 |
| [Learning robust perceptive locomotion for quadrupedal robots in the wild](https://arxiv.org/abs/2201.08117) (2022) | Phase/residual control with recurrent uncertain terrain input. | Not verified |
| [DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning](https://arxiv.org/abs/2301.10602) (2023) | Direct positions, short history, asymmetric critic. | ~1 h; RTX 3060 Ti |
| [Regularizing Action Policies for Smooth Control with Reinforcement Learning](https://ai.bu.edu/caps/) (2021) | Smooth the policy mapping directly; not a gait architecture. | Not verified |
| [Rapid Locomotion via Reinforcement Learning](https://arxiv.org/html/2205.02824v1) (2022) | Position PPO with expanding body-twist commands. | <3 h; RTX 3090 |
| [Extreme Parkour with Legged Robots](https://arxiv.org/abs/2309.14341) (2024) | Terrain teacher followed by depth policy; later-roadmap example. | <20 h; RTX 3090 |
| [Minimizing Energy Consumption Leads to the Emergence of Gaits in Legged Robots](https://arxiv.org/abs/2111.01674) (2021) | Speed/energy objectives can produce different gaits. | ~24 h; GPU unspecified |

Training durations are the authors' results under different simulators, hardware, batch sizes, robots and success criteria. They are not Spark ETAs. In particular, Rudin's cited timing used an **RTX A6000**, not an RTX 3090. Some papers train multiple stages or experts; their reported fast phase should not be mistaken for the entire development effort.

## Three actionable conclusions

### 1. Compare action authority before spending many more updates on the same small residual

A fixed swing clock and fixed support sequence cannot be replaced by PPO when its only output is a small position correction. More updates can improve compensation inside that envelope, but cannot reliably answer which gait family is best. Our measured static four-support result motivates a broader experiment; it does not prove a dynamically stable gait.

The most informative next architecture comparison is a **fresh stance-relative joint-position PPO** against a **phase/foot-placement-modulating residual controller**, using the same physical asset, torque cap, command goals and compute accounting. The latter has closer precedents in [PMTG](https://proceedings.mlr.press/v87/iscen18a/iscen18a.pdf), [Lee et al.](https://arxiv.org/pdf/2010.11251) and [Miki et al.](https://arxiv.org/html/2201.08117v1). A single direct-position comparator is the simpler first alternative if engineering time is the limiting resource.

This does not mean an unconstrained torque actor or a sudden action-scale increase. Preserve realizable target position/rate/acceleration and actuator limits, and make any new action authority observable. A nominal stance and PD loop already provide substantial structure. The common 0.04 rad per 20 ms target bound is a declared software profile; it is not a measured motor speed limit. A new dynamic-support training protocol must be explicit, while existing five-support screen results retain their original meanings.

### 2. Train commanded movement and quiet standing as distinct tasks with transitions

Use the requested body twist as the navigation goal. An actor-induced governor pause must not receive full goal-achievement credit merely because its admitted target becomes zero. Keep deliberate stopping, finite supported settling, and settled quiet separate. Compare reward components at stopped, accurately tracking, opposite and overspeed motion before allocating a larger pilot.

The literature supports progressive command/terrain difficulty and penalty curricula, rather than demanding that every untrained trajectory succeed. [RMA](https://arxiv.org/html/2107.04034v1) explicitly motivates increasing penalties gradually. [Rapid Locomotion](https://arxiv.org/html/2205.02824v1) and [Walk These Ways](https://proceedings.mlr.press/v205/margolis23a/margolis23a.pdf) show command-space expansion. These are useful design precedents, not permission to silently weaken a torque or acceptance gate.

For this robot, use recoverable, individually recorded training terminations; preserve true-terminal versus timeout bootstrap, reset history isolation and pre-reset evidence. Then assess fixed complete command trajectories independently: all legs, both translation and yaw signs, arcs, command changes, stop and at least ten seconds of quiet. Ten updates of 256 controls provide 51.2 simulated seconds per continuously active replica; recovery reduces usable time and phase coverage, so report both. A median across directions cannot hide one failed bearing.

Quiet standing requires explicit no-command data and retained feedback for disturbances. Action-rate penalties alone do not guarantee a quiet policy. [CAPS](https://ai.bu.edu/caps/) is a bounded alternative to test if jitter persists: regularize nearby-state and successive-state policy outputs, then check that disturbance correction and command changes remain responsive. Its quadrotor results do not predict hexapod energy savings.

### 3. Keep the terrain path compatible with a simpler successful locomotor

A fixed scripted gait is not necessary to prepare for terrain. [DreamWaQ](https://arxiv.org/pdf/2301.10602) illustrates short-history direct-position control with a privileged critic; [Miki et al.](https://arxiv.org/html/2201.08117v1) illustrates a recurrent perception interface; [Extreme Parkour](https://extreme-parkour.github.io/resources/parkour.pdf) illustrates teacher-to-depth transfer. Preserve body-twist commands and clear actor/critic state contracts now, then add terrain height/support uncertainty and a sensor-trained estimator in a separately qualified stage.

Do not label ideal simulator contact, velocity or terrain channels deployable sensing. Our SDK rate versus angle-difference discrepancy remains unresolved: retain both channels and interval/history validity. Mechanical power based on suspect rates cannot establish energy efficiency. [Fu et al.](https://arxiv.org/html/2111.01674v1) supports allowing gait choice to vary with speed and energy, but it does not validate our current power proxy or make insect-like appearance an optimization objective.

## Ranked comparison against the controller we already have

The archived direct-position PPO is the first comparator, not a discarded architecture. Its original checkpoint is `1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8`, with 315 actor / 318 critic observations. Read the preserved [repair003 method](../../../../artifacts/omni_diagnostics_2026-09-09/repair_003/METHOD.md) and [full results](../../../../artifacts/omni_diagnostics_2026-09-09/repair_003/results/README.md).

Both repair branches kept the old command distribution, ran only 50 updates, reset exploration to 0.10 and Adam moments, and retained the diagnostic 0.03 rad / 20 ms limiter. Branch B added a sampled raw-action standing cost. They modestly improved tracking but did not remove the limit cycle: standing joint-rate RMS stayed about 0.74 rad/s, roughly 90% of target increments reached the limiter, and the recorded replica still drifted about 13 cm. The raw-action cost changed mean-square intent only about 0.6% relative to branch A. This rejects those tested repairs; it does **not** prove direct-position PPO cannot learn quiet standing.

| Priority | Candidate and concrete change | Implementation effort | Short decision experiment |
|---|---|---|---|
| 1 | **Preserved 315/318 checkpoint, targeted new fine-tune:** explicitly balance zero-command, motion and stop transitions; retain its history/action interface. Test policy-mean temporal/spatial regularization as a separate ablation, rather than repeat lower exploration or sampled-action MSE alone. | Small for command curriculum; moderate for a verified policy-loss integration. Reuse checkpoint, runner and evaluation machinery. | Replay exact baseline first. Then equal-budget curriculum-only versus curriculum-plus-regularization if resources permit; keep optimizer/std choices explicit. Compare every direction, stop, target alternation, raw SDK and angle-derived motion, and 400 Hz torque. Stop if quiet gains erase locomotion or torque worsens. |
| 2 | **New modulated generator:** permit phase/frequency and foothold/foot-residual changes within verified actuation/support bounds. This directly addresses the fixed two-second wave's speed ceiling. | Substantial: new observable state/action contract, asynchronous contact/stop tests, exploration and checkpoint validation. Existing tensor reference is a starting point, not a completed learner interface. | Zero-residual and bounded exploratory contact proof first, then matched short learning. Require a measured gain in speed/efficiency at unchanged motor limits and retained stop/quiet; do not count a changed reference speed as learned improvement. |
| 3 | **Fresh direct-position PPO** with the same body-twist goal and deliberate stand/stop curriculum. Useful if inherited limit-cycle behavior resists the targeted fine-tune. | Moderate: much existing code reusable, but fresh initialization/exploration and more learning required. | Match the fine-tune's reward/action/dynamics when isolating initialization. No old-checkpoint resume claim. Compare compute-to-useful-motion and all-direction retention. |
| 4 | **Continue the fixed-wave tiny residual** only as a bounded learning/recovery and compensation experiment. | Lowest incremental integration effort, but structurally limited gait authority. | Finish the specific recovery repair and matched screens; do not promise this branch will recover the old policy's speed or discover new support patterns. |

Rank 1 is credible because the failed repairs did not test a command-balanced quiet/stop curriculum or CAPS-style policy mapping loss. It is a hypothesis, not a convergence promise. The actual noise-removal/filter probes and repair003 argue against another unchanged std/reward-only retry. Preserve existing history initially: appending new channels would invalidate the 315/318 checkpoint interface and confound the comparison.

The 0.03 diagnostic and 0.04 formal profiles stay separate. First reproduce the old policy in its exact source/dynamics. If testing a newer solver or telemetry contract, measure the original policy under those same new dynamics before attributing changes to training; retain both source identities. Never compare a newly filtered or re-timed policy against an unmodified baseline without labeling that intervention.

The current moving residual campaign completed nine updates and failed recovery during the tenth collection, before its decision checkpoint or matched screens. That is evidence of executable PPO, not successful policy qualification. Diagnose its concrete recovery condition independently of this architecture ranking. Do not allocate a long successor merely because optimization ran.

For all candidates, measure integrated 32-replica timing before increasing capacity. Report collection, optimization, evaluation and simulator time separately. A paper's large-batch speedup is not evidence that our reference, sensors or Spark configuration scales to 1,024 replicas. Keep faster-support work parallel and retain complete fixed evaluation trajectories; a ten-update checkpoint is a diagnostic decision point, not full Stage 2 completion.

## Review scope and unresolved details

This is a breadth-first review of twelve representative approaches, not a systematic meta-analysis or proof of an optimal architecture. All sources are primary papers, author pages or official code. No unverified third-party reproduction supplies implementation facts. Repository defaults are mutable and may differ from paper experiments. Exact motor ratings and wall-clock costs remain unverified where the matrix says so; no values were filled by analogy.

No robot model, frozen runtime, checkpoint, acceptance threshold or GPU workload was changed for this review. Current actual-recovery evidence remains separately frozen; this document is neither a run-approval receipt nor a Stage 2 completion claim.
