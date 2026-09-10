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
| Terrain fixture attempt 003 | All 30 fixtures passed actual Isaac mesh/ray/PhysX/contact checks | Fixture import admitted; full C terrain standing and walking still unqualified |
| Camera placement | Exact rays through a hashed serial-CAD snapshot (excluding physical four-bar motion): initial six-view concept 16.1% sampled visibility; expanded proposal 78.1%, worst case/sector 48.1% | Neither layout is ready to manufacture or proves safe terrain coverage |
| Perception CPU replay | Timestamped depth/point-cloud transforms, robot masking, uncertainty/age map and student patch; 14 targeted tests pass | Synthetic interface evidence; no real sensor, ROS or actor integration |

## Execution and next actions

**Control diagnostic batch `omni_control_probes_002` finished all three comparisons**, with zero terminations, on Spark as `hexapod-omni-control-probes-002-20260909.service`. Each exact plan passed standing admission. Zero-action standing was quiet (joint velocity RMS 0.00158 rad/s versus policy 0.7501); removing observation noise did not fix the oscillation, and the 80 ms filter worsened tracking and saturation. Both interventions are rejected. The next bounded PPO trial should address the learned oscillating mean actions and exploration. Inspect live state before asserting another run is active:

```sh
ssh spark 'cat /home/orionh/HEXAPOD_runs/mock_length_study_20260909/omni_control_probes_002/diagnostic_campaign.json'
```

1. Use the control probes to choose the next small PPO correction. Preserve motor limits, action/observation identity and failed evidence. Recover every direction, not just the median.
2. Run sustained quiet standing and stop-from-motion checks, then the [46-case visual review](artifacts/omni_diagnostics_2026-09-09/REVIEW_RUNBOOK.md) and the full command/transition suite on promising checkpoints.
3. Validate the full admitted C robot on the now-checked terrain fixtures. Continue exact support queries, measured slope bounds, reset/curriculum manifests and sensor-map preparation in parallel.
4. Reconcile final detailed C CAD, payload mass, actuator calibration and camera/bracket coverage before hardware or terrain capability claims.

The user authorizes pausing the identified Forecasting-Pipeline/StormScope workloads for this work. Every job checks other GPU users, holds both existing job-scoped locks, records frozen source/configuration and owns only its exact container. Pause 011 recorded successful restoration after the diagnostic unit exited. No other workload may be stopped. A five-minute task follow-up reviews meaningful changes; remote jobs do not depend on the desktop remaining connected.

**Git policy:** the user requests a commit and push for every verified meaningful step, including relevant Markdown updates. Merge with current `main`, preserve collaborators' work and immutable evidence, run relevant combined checks, and verify the pushed SHA. Do not publish unreviewed temporary files or machine secrets.

## Shared context

[Living plan](docs/PLAN.md) · [Current study and runtime contract](experiments/c_length_study/README.md) · [Terrain and sensing plan](artifacts/project_review_2026-09-04/TERRAIN_AND_SENSING_PLAN_2026-09-09.md) · [PPO evidence](artifacts/omni_diagnostics_2026-09-09/README.md) · [Terrain integration](isaaclab/hexapod_terrain/README.md) · [Mount study](artifacts/sensor_mount_study_2026-09-09/README.md) · [Perception replay](artifacts/perception_readiness_2026-09-09/README.md) · [Run history](HANDOFF.md).

The [preceding physical-program status](docs/archive/STATUS_2026-09-08_physical.md) is historical context. Its user pause pending a revised single-leg export and its failed physical admission remain true for that lineage; they are not the state of the separately authorized C-study experiments.
