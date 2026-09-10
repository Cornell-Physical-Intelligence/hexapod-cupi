# Current project status

Updated 9 September 2026. **Highest priority: finish the user-selected C-study Stage 2 flat omnidirectional PPO controller. Stage 2 is not complete.** Terrain and perception implementation proceed in parallel. The separate physical four-bar program and its evidence remain preserved; C-study results do not qualify that mechanism or hardware.

## Read this first

- **Current geometry study:** 72.5 mm femur, 126 mm tibia, fixed coxa; full six-leg scaled mock with transferred CAD mass/inertia (8.2608 kg). This is the user's explicit study exception, not a replacement for the production asset or final motor/linkage packaging.
- **Completion standard:** every travel bearing, reverse/strafe/diagonals, both turns, combined arcs and path transitions must be comparably smooth to the accepted forward animation. Zero command must settle into quiet standing. Actual videos and per-direction tracking, stability and motor checks are required; a budget, reward or median cannot complete the stage.
- **Control:** RSL-RL PPO, body forward/left/yaw-rate requests, 315-value proprioceptive-history actor and 318-value privileged critic. The study retains its explicit 1.6 N·m applied motor cap. The physical-model actuator contract is separate.
- **Reference:** [Benchmark 1](artifacts/length_study_2026-09-09/benchmarks/benchmark_01_c_300/README.md) remains immutable. The [corrected five-path recording](artifacts/omni_flat_2026-09-09/omni_preview_001/rollout.mp4) uses actual policy actions and ideal simulator localization for its outer follower; it is not terrain or perception qualification.

## Latest verified experiments

| Work | Result | Implication |
|---|---|---|
| Original omnidirectional continuation | 1,425 additional PPO updates; final 0/154 static and 0/14 transition gates | Training budget finished; Stage 2 did not |
| Slower target change diagnostic | Requested saturation median 14.93% → 5.74%, positive mechanical power 4.94 → 2.30 W, base-contact terminations 4 → 0 | Useful intervention, but fast-forward/left-arc tracking and quiet stance still need work |
| Separate PhysX solver flag | No improvement | Not adopted |
| 100-update quiet/smooth reward pilot | Planar error 0.0395 → 0.0288 m/s; saturation 5.74% → 6.64%; stand joint velocity barely changed; forward yaw tracking worsened | Rejected; longer continuation did not run |
| Paired 50-update PPO trials | Standing joint RMS 0.7501 → 0.7374 (A) / 0.7402 (B); neither met the existing continuation screen | No further PPO allocated to these branches; quiet standing remains unresolved |
| Terrain fixture attempt 003 | All 30 fixtures passed actual Isaac mesh/ray/PhysX/contact checks | Fixture import admitted; full C terrain standing and walking still unqualified |
| Camera placement | Exact rays through a hashed serial-CAD snapshot (excluding physical four-bar motion): initial six-view concept 16.1% sampled visibility; expanded proposal 78.1%, worst case/sector 48.1% | Neither layout is ready to manufacture or proves safe terrain coverage |
| Perception CPU replay | Timestamped depth/point-cloud transforms, robot masking, uncertainty/age map and student patch; 14 targeted tests pass | Synthetic interface evidence; no real sensor, ROS or actor integration |

## Execution and next actions

**Paired trial `omni_repair_003` completed both branches**, with no terminations/truncations. A reset exploration/optimizer settings explicitly; B added a standing raw-action penalty. The mean controller still oscillates, and roughly 90% of recorded standing target steps hit the 0.03 rad/20 ms diagnostic limiter. Neither branch earned continuation. [Matched results, checkpoints, source identities and plots](artifacts/omni_diagnostics_2026-09-09/repair_003/README.md) preserve the negative result. Pause 012 restored both forecasting timers after the user unit exited.

**Full-C terrain entry has reached physical stepping but is not admitted.** Attempt 001 failed Mesh import; attempt 002 passed that point but lost dynamically attached controller settings during config copy. Attempt 003 resolved both, passed fresh flat standing, then returned a termination on every terrain control. Its post-step measurements are reset-contaminated and cannot be interpreted as steady support load. [Attempt 003](artifacts/terrain_readiness_2026-09-09/runtime/terrain_robot_smoke_003/README.md) preserves the failure. The installed SDK uses XYZW rotation tuples, while the adapter authored WXYZ; main now corrects that with an independent rotation test. The terrain agent owns a short pre-reset diagnosis and fresh corrected admission; no full-C terrain standing or walking pass exists. Pause 015 restored both forecasting timers and cleanup released the completed run's GPU ownership.

**Controller architecture work is proceeding on two bounded candidates.** The [frozen reference feasibility study](artifacts/omni_diagnostics_2026-09-09/reference_feasibility_001/README.md) rejects the first forward-waveform extension and global slowing; its 13 tests and independent metrics replay reproduce within floating-point roundoff. One agent is building stance-anchor/swing/stop handling; another is testing an observable, acceleration-bounded target-velocity action interface with a new 495/498 actor/critic schema. These remain CPU candidates pending full-robot admission and are not old-checkpoint continuations. Stage 2 remains open.
1. Complete the candidate action/reference checks, including executable reset/observation state and a bounded physics/runner smoke. Allocate a short new-lineage PPO pilot only after standing and exploration calibration pass. Compare 0.03 diagnostic and 0.04 formal target-rate settings separately; preserve every direction, motor limit and source identity.
2. Run sustained quiet standing and stop-from-motion checks, then the [46-case visual review](artifacts/omni_diagnostics_2026-09-09/REVIEW_RUNBOOK.md) and the full command/transition suite on promising checkpoints.
3. Validate the full admitted C robot on the now-checked terrain fixtures. Continue exact support queries, measured slope bounds, reset/curriculum manifests and sensor-map preparation in parallel.
4. Reconcile final detailed C CAD, payload mass, actuator calibration and camera/bracket coverage before hardware or terrain capability claims.

The user authorizes pausing the identified Forecasting-Pipeline/StormScope workloads for this work. Every job checks other GPU users, holds both existing job-scoped locks, records frozen source/configuration and owns only its exact container. Pauses 012–015 recorded restoration after their respective completed or failed runs. No other workload may be stopped. A five-minute task follow-up reviews meaningful changes; remote jobs do not depend on the desktop remaining connected.

**Git policy:** the user requests a commit and push for every verified meaningful step, including relevant Markdown updates. Merge with current `main`, preserve collaborators' work and immutable evidence, run relevant combined checks, and verify the pushed SHA. Do not publish unreviewed temporary files or machine secrets.

## Shared context

[Living plan](docs/PLAN.md) · [Current study and runtime contract](experiments/c_length_study/README.md) · [Terrain and sensing plan](artifacts/project_review_2026-09-04/TERRAIN_AND_SENSING_PLAN_2026-09-09.md) · [PPO evidence](artifacts/omni_diagnostics_2026-09-09/README.md) · [Terrain integration](isaaclab/hexapod_terrain/README.md) · [Mount study](artifacts/sensor_mount_study_2026-09-09/README.md) · [Perception replay](artifacts/perception_readiness_2026-09-09/README.md) · [Run history](HANDOFF.md).

The [preceding physical-program status](docs/archive/STATUS_2026-09-08_physical.md) is historical context. Its user pause pending a revised single-leg export and its failed physical admission remain true for that lineage; they are not the state of the separately authorized C-study experiments.
