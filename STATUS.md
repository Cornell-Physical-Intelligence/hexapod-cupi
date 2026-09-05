# Current status

Last reviewed: 2026-09-05 19:29 UTC. **The first full nominal physical-model validation passed.** All 32 robots completed 1,000 standing + 2,400 driven control steps; all individual/group directions passed, minimum driven support was three feet, and maximum linkage gap was 0.081909 mm against 0.100 mm. No resets or non-foot ground contacts occurred; delivered torque remained inside the RS05 envelope. Campaign 006 has automatically advanced to the refined solver comparison. **No physical-model PPO has started yet:** both passes and their convergence comparison must succeed first. Frozen source is `9cd8d4c`. [Full nominal evidence](artifacts/mkii_fourbar_2026-09-05/campaign_006_ramped_targets/README.md) · [Prior full failure](artifacts/mkii_fourbar_2026-09-05/campaign_005_physical_mimic/README.md) · [Scheduling change](docs/MKII_MOTOR_TARGET_SCHEDULING.md).

[Mission](dar.md) · [Living plan](docs/PLAN.md) · [Prepared future runs](docs/NEXT_RUNS.md) · [Dated audit](artifacts/project_review_2026-09-04/URDF_VALIDATION.md) · [Leg test plan](artifacts/project_review_2026-09-04/LEG_TEST_STAND.md)

## Current program

Build no-RTK bounded-area coverage with a learned omnidirectional gait, vision/LiDAR/IMU navigation, and a generic survey payload interface. The deployment target is Jetson Orin Nano and 18 RS05 motors. Only the CAD is complete; single-leg testing precedes remaining-parts ordering. There is no assembled robot ready for full-body powered testing.

The target is a four-week integrated simulation prototype and six-week candidate to begin hardware transfer, from the resumed development campaign. A first field demo is conditionally targeted H+2–3 weeks, where H is assembled-robot readiness and is currently unknown. Those targets depend on passing the plan's gates.

## Blocking asset findings

The 19-link/18-joint serial URDF passes the performed static checks. Its modeled mass is 8.26081134 kg; installed masses, motor behavior, stops, linkage mapping and payload configuration are not physically verified.

The original deployed USD **fails all 19 link inertia tensor round trips**. A new versioned serial-v2 bundle repairs them from the URDF and passes the all-link gate locally, in Spark's standalone USD library and in Kit's own USD library. It is staged separately on Spark. The new nominal reset removes the approximately 8 mm penetration; the opt-in v2 task/runtime pair has explicit anatomical frames and CAD-specific named action mapping. Historical assets and task defaults remain unchanged.

The serial URDF welds pushrod/lever motion into the femur, so successful standing does not establish correct four-bar dynamics. The earlier approximately 0.5 mm cut-point residual came from stale reference frames: independent recovery of current CAD pin lines finds transverse discrepancies below 0.00073 mm and a 30 / 77.5 mm planar parallelogram. [Those recovered frames](artifacts/mkii_step2_2026-09-04/physical_fourbar_reference/README.md) now drive a separate physical USD with **31 rigid bodies, 30 articulation coordinates, six excluded revolute closures and 18 active motors**. The active joints are coxa, femur and push-lever; knees and rods remain passive. CPU checks preserve all masses/tensors/visuals/primitives and verify the prescribed closed branch, exact coordinate bridge and transformed limits. This is authored-geometry evidence, not live solver qualification.

Live startup, reset, named joint order, standing torque and contact checks now have [dated reports](artifacts/mkii_step2_2026-09-04/README.md). The hardened 32-environment / 1,000-control-step run passes with all 4,000 physical substeps sampled: settled computed peak 0.880025 N·m and six loaded pads throughout the settled window. Startup computed demand reaches 2.364431 N·m, above the imposed 1.6 N·m applied cap but below the manufacturer's 5.5 N·m peak. The [RS05 specification review](docs/RS05_SPEC_REVIEW.md) confirms **1.2 N·m continuous stall**, versus 1.6 or 1.8 N·m at 100 rpm under different cooling conditions. The inherited model already records 5.5 N·m but does not deliver that burst torque or model thermal derating. The 1.6 cap is not universally conservative. Driven joint/direction tests, motor-model qualification and per-collider contact checks remain open. A standing pass does not grant complete G0, training admission or hardware readiness. [The original audit](artifacts/project_review_2026-09-04/URDF_VALIDATION.md) and [step 1](docs/MKII_STEP1.md) remain historical evidence.

The new task uses a separate RS05 model with a bounded torque-speed/voltage envelope, continuous-load allowance, shared burst/recovery budget and phase-current proxy. Its implementation, parameters and resolved configuration are bound to the new runtime identity. It assumes 48 V; cooldown, gains, friction, inertia and installed cooling remain provisional. This is not a calibrated thermal model or a hardware CAN/current-protection implementation. Its **84 observations** include 18 motor burst-headroom estimates. The policy has 18 named motor actions, anatomical navigation coordinates and **no gait clock or prescribed tripod schedule**.

The first physical probe, using the **historical 5 ms / four-substep recipe**, passed 1 environment × 100 control steps with all 400 physics substeps sampled in campaign `fourbar-campaign-20260905T034653Z-a83056cd`. Its [report](artifacts/mkii_fourbar_2026-09-05/campaigns/fourbar-campaign-20260905T034653Z-a83056cd/probe/hexapod-fourbar-validate-20260905T034654Z-91ce5326/report.json) records six loaded pads and no non-foot contacts throughout settling, 0.674836 N·m settled applied/computed peak, 0.135699 m mean plate height, and 1.825 µm maximum settled closure-position error. Startup applied/computed peak was 1.812816 N·m and maximum closure error 65.99 µm. These are a two-second startup result, not the full standing/driven or convergence gate.

At 5 ms, the nominal **32-environment × 1,000 standing plus 2,400 driven control-step** run completed but failed numerical closure bounds: driven pin separation reached 0.246159 mm (limit 0.1 mm), and passive-coordinate residual reached 0.0114364 rad (limit 0.005 rad). All 18 individual/group direction checks passed, applied torque stayed within its envelope (peak 4.05806 N·m), and there were no resets or unwanted ground contacts. The separate 64/8 diagnostic also failed (0.366053 mm pin separation, 0.016916 rad passive residual), while completing all direction, torque, contact and reset checks. The subsequent 5 ms explicit-force 64/1 recipe failed at the first tibia reversal after 1,252 driven steps: closure reached 0.883917 mm and triggered a reset. Those historical full sequences used 4,000 standing plus 9,600 driven physics substeps per environment. Their failed evidence and earlier startup failures remain preserved.

**Completed campaign 004**, `fourbar-campaign-20260905T044533Z-5465207c`, used 1.25 ms physics / 16 substeps with unchanged 50 Hz policy control, gains, masses and acceptance bounds. Its startup probe `hexapod-fourbar-validate-20260905T044533Z-c2be22c1` passed **1 environment × 100 control steps / 1,600 physics substeps**. Settled support was six pads with no non-foot ground contact; peak applied/computed torque was 0.665702 N·m, mean plate height 0.135708 m, and maximum closure error 0.5955 µm. Startup peak torque was 1.699540 N·m and maximum closure error 1.4940 µm. The [reproducible inertia analysis](artifacts/mkii_fourbar_2026-09-05/solver_stability/README.md) motivated the smaller timestep without claiming it proved PhysX stability.

The nominal run completed **32 environments × 1,000 standing plus 2,400 driven control steps**, sampling **16,000 standing + 38,400 driven = 54,400 physics substeps per environment**. It failed: maximum pin separation was **0.345448847 mm** against the original 0.1 mm limit, and maximum passive-coordinate residual was **0.019003332 rad** against 0.005 rad. Raw torque demand peaked at **33.77581024 N·m**; applied torque peaked at **5.5 N·m**. There were no resets or unwanted ground contacts, all individual/group direction checks passed, and minimum support was one pad. Completing the sequence did not grant training admission.

Focused event traces will distinguish explicit motor feedback, contact loading and closure-constraint convergence. The separate **D6 v4 candidate** locks the two independent transverse closure directions; an [all-angle geometric audit](artifacts/mkii_fourbar_2026-09-05/closure_equivalence/README.md) shows that the planar tree already enforces the other three constrained directions, to subnanometre authored precision. It preserves the original full pin-position, axis and passive-branch checks. This is a mathematically equivalent planar-geometry candidate, not a GPU-validated fix. A matching full nominal and refined validation must pass before PPO; original bounds remain unchanged.

The [collider-fit audit](artifacts/mkii_fourbar_2026-09-05/collider_fit/README.md) also identifies approximation limits: sampled silicone surfaces lie up to **6.38 mm outside the complete tibia collider union**, including a **5.04 mm** discrepancy in a nominal lower contact band. The tested ±0.30 rad motor-target grid gives a foot-bottom bias of **−3.109 to +0.427 mm** relative to CAD. Broad structural shapes extend beyond their own link's CAD envelope, including about 54.8 mm for a body box and 29.1 mm for a coxa box in tested directions. These are sampled, per-link measurements. Pad/clearance refinement and self-collision qualification remain necessary for accurate terrain-contact claims. Hardware and rough terrain are not qualified; full G0 remains open.

## Repository and compute evidence

- Reviewed `main` at `81d7c6f`, including James's six-package refactor, contracts, tests, reward structure, sensor prototypes and new CAD task. Current main has no operational coverage planner or camera/LiDAR/IMU fusion stack. Existing terrain and sensor work is preliminary.
- Main's pinned Python 3.12/uv suite passed **520 tests**; all 112 entries of the archived pipeline manifest matched. The older local branch passed 288 tests. Those checks do not certify the imported physics or real hardware.
- Stopped the identified hexapod queue, wrapper, training container and log followers. The interrupted scratch CAD run was `q_crouch_s52`, near iteration 418/500. Existing checkpoints and logs were preserved. Follow-up inspection found no remaining hexapod training process or loaded hexapod service.
- The Spark project mirror `/home/orionh/HEXAPOD` is a bind-mounted directory, not a Git repository. The physical campaign uses `/home/orionh/HEXAPOD_runs/mkii_fourbar_v1/source`; the earlier serial source remains separate. Each bounded run has its own source hashes, supervisor, report and logs. Inspect live state before every launch; weather was stopped only under the user's subsequent explicit priority instruction.
- The new implementation preserves archived checkpoint/task behavior. Two packaging files listed by the old manifest, `packages/hexapod_core/pyproject.toml` and `packages/hexapod_env/pyproject.toml`, were revised for installation. The 112 archived lineage files are verified separately; the frozen historical manifest is not rewritten to certify the new task. Earlier [step-1 test evidence](artifacts/mkii_step1_2026-09-04/README.md) remains historical.
- Earlier weather stops occurred under the user's explicit instruction to prioritize the hexapod. Campaign 004's reservation, PID 1333865, was renewed at 04:56:24 UTC and **released when that campaign failed at 05:12:38 UTC**. Its record remains `/home/orionh/HEXAPOD_runs/mkii_fourbar_v1/priority_campaign_004.json`; the older guard PID 1232035 had been terminated after its successor queued. A separate diagnostic reservation, PID **1364926**, started at **05:15:44 UTC** with a three-hour ceiling, recorded in `/home/orionh/HEXAPOD_runs/mkii_fourbar_v1/diagnostic_priority_20260905T0515.json`. These are cooperative reservations, not MPS quotas; fresh workload admission checks remain required. Shared coordination-file bytes were not changed during campaign 004's active phase.
- Current compute instruction: **use full available Spark compute until another agent requests sharing through `/home/orionh/SPARK_COMPUTE_COORDINATION.md`**. Read the shared file before each new launch and at checkpoint boundaries during future long runs; the [handoff note](docs/SPARK_COMPUTE_COORDINATION.md) records the procedure. The earlier 60/40 split is only a starting preference if sharing is requested. No quota, MPS partition or shared long-training launcher is enabled; current short validation already uses exclusive admission and full available compute.

## Historical policy evidence

There is **no admitted CAD policy ready for hardware**. Spark has scratch CAD runs, making the old “nobody has trained CAD yet” statement stale, but those runs used the now-flagged USD and are not a corrected-asset baseline.

The archived mock's recorded best checkpoint remains:

```text
artifacts/phase2_recovery_stage2c_stable_forward/checkpoints/current_best/model_2.pt
SHA-256: a66a1c83e5b0675ade73d05538ae57d81451568ab45d9e8bcfd2edc67bb2e69a
```

It misses the moving-command yaw gates and highest-command deck gate. It is neither hardware-ready nor the required parent of the new CAD campaign. Stage2G's improvement claim still lacks an admitted artifact bundle; its tripod-phase reward is outside the requested new gait design. The prior metrics and unresolved evidence questions are preserved verbatim in [the archived status](docs/archive/STATUS-before-autonomy-review-2026-09-04.md). Historical gates and artifacts remain unchanged.

## Next work

Diagnose the v5 simultaneous-knee failure and qualify a physically justified candidate, then repeat the gated sequence: **1-environment × 100-step startup probe → 32-environment × 1,000 standing plus 2,400 driven control-step validation at nominal and refined solver settings → 64 environments × 3 scratch PPO iterations → a separate process resuming that verified checkpoint for 512 environments × 1,000 iterations**. At 1.25 ms / 16 substeps, each full validation samples 16,000 standing and 38,400 driven physics substeps per environment. No stage may use a prior failed campaign or short probe as admission. The larger workload remains contingent on smoke-run memory/throughput and checkpoint checks. During initial PPO, closure failure invalidates the run; ordinary falls remain learning events. PPO has not started, and terrain/hardware gates remain open.

Perception, navigation, runtime emulation and leg-test preparation proceed independently. Early policies are bootstrap candidates; measured dynamics or collider revisions require screening and sometimes retraining.

Open inputs are the hard deadline, accountable stream owners, full-robot delivery/assembly dates, sensor availability, survey footprint/stability and measured actuator/terrain bounds. The [decision record](docs/PLAN.md#9-new-context-and-decision-record) defines how new context changes the plan and which evidence must be re-established.

Frozen `ea05fe8` release checks: **797 tests passed**, and CI passed. Its immutable historical 112-file manifest and separate 270-file 800 Hz manifest both verified. The new diagnostic/D6 source now passes **811 tests in 41.946 seconds** and has a separate **276-path manifest**. Its planned isolated Spark source is `/home/orionh/HEXAPOD_runs/mkii_fourbar_diagnostics_v1/source`; these source checks are not GPU validation. [Solver stability analysis](artifacts/mkii_fourbar_2026-09-05/solver_stability/README.md).

## 2026-09-05 07:02 UTC continuation

The workload detector confused a blocked `/usr/bin/flock` weather wrapper with its not-yet-started Python command. The GPU was empty; kernel lock records proved the weather job was queued behind the hexapod reservation. The revised detector requires executable identity, blocked lock/inode evidence, stable PID and no children before treating a wrapper as waiting. Actual GPU processes still veto contention.

The user reaffirmed top priority. Reservation was renewed without GPU work being interrupted: queued weather wrapper1394883 was cancelled, old guard1364926 ended, and guard1407273 acquired the existing scheduler lock at07:02:04UTC. Its ten-hour maximum is17:02:04UTC; selected campaign completion/failure or an explicit release file releases it sooner. Status is `/home/orionh/HEXAPOD_runs/mkii_fourbar_diagnostics_v1/priority_20260905T0700/status.json`.

New source binds the selected immutable v3/v4 bundle through CPU audit, Kit audit, actual runtime, solver comparison and pre-learner admission. v3 remains the default; explicit v4 selection can undergo the full existing gates. Prior reports are preserved, including the historical diagnostic runtime label that saidv3 while its selected USD identifiedv4.

Release verification: **841 tests passed in45.428seconds**; the new279-file asset-binding manifest and archived112-file lineage checks pass. Both actual CPU audit reports bind to their selected v3/v4 bundle; this is not a live Kit pass.

## 2026-09-05 16:16 UTC — native physical coupling candidate

The complete D6 comparison `072121Z-16d523c4` failed; its primary report and compact trace analysis are preserved. The new [v5 candidate](docs/MKII_PHYSICAL_COUPLING_CANDIDATE.md) uses native bilateral PhysX constraints for the exact parallelogram coordinate relations while retaining all bodies, inertias, geometry and limits. It passes CPU asset checks and the local suite (**844 tests in 57.415 seconds**). It still needs live validation. No PPO has started.

## 2026-09-05 17:48 UTC — full v5 outcome and next diagnostic

Campaign `fourbar-campaign-20260905T165013Z-e9c5e5a6` completed all 32 × 3,400 control steps. The [preserved report](artifacts/mkii_fourbar_2026-09-05/campaign_005_physical_mimic/README.md) records the closure, support and group-direction failures. Raw requested torque peaked at 16.6328 N·m; delivered torque stayed within its envelope (5.5 N·m maximum). No resets or non-foot ground contacts occurred. A short probe had passed; it did not admit learning.

The bounded diagnostic `hexapod-fourbar-diagnose-20260905T174632Z-36232e95` uses eight robots, the same v5 asset and controller, 200 standing + 700 group-motion control steps, and 128/1 solver iterations. It records per-substep actuator terms, actual joint/body poses and contact forces; it cannot issue training admission. Priority guard PID 1465763 holds `/opt/wx/gpu.lock` with a maximum expiry of 2026-09-06 02:16:34 UTC. The shared note remains `/home/orionh/SPARK_COMPUTE_COORDINATION.md`.

## 2026-09-05 18:20 UTC — target scheduling experiment

The v5 128/1 group trace completed: maximum closure21.017 µm, all18 group directions positive, four zero-support samples of1.25 ms each at the second negative knee-target update. This shorter eight-environment sequence does not reproduce the full nominal history and cannot admit learning. The interpolation source is9cd8d4c, functional identity `c53071afdf320f6a9d6f91de09ddc6102de74df6f45a203e3a8166a8686548ca`.

Current diagnostic is `hexapod-fourbar-diagnose-20260905T181945Z-ec78eb1e`, supervisorPID1519950, ownedcontainera421e1ff02ab0b2013907339215b1425df344fd6432e4b58cc8cb59859244b1f. Source `/home/orionh/HEXAPOD_runs/mkii_ramped_targets_v1/source` is frozen. It uses200 standing +700 group-motion steps, eight environments, nominal64/1 and v5. Only delivery of active motor position targets changed; velocity feedforward remainszero.

## 2026-09-05 19:05 UTC — full campaign 006

The short scheduled-target diagnostic completed all 14,400 physics samples per
environment with no physical gate errors. Independent analysis also verifies all
18 group directions and 2,073,600 delivered target values. This is diagnostic
evidence, not training admission. The source passed 854 local tests and GitHub CI.

Full campaign `fourbar-campaign-20260905T185810Z-c36097a7` started at 18:58 UTC
under PID 1532490 in `/home/orionh/HEXAPOD_runs/mkii_ramped_targets_v1/campaigns/`.
Its new startup probe passed; nominal validation is running under supervisor
1534642 in `nominal/hexapod-fourbar-validate-20260905T185936Z-b545949f/`.
The campaign automatically proceeds to refined validation, qualification,
64-environment / 3-update scratch PPO and 512-environment / 1,000-update resumed
PPO only after each prior phase passes. The physical bounds remain unchanged.
An SSH timeout delayed the diagnostic-to-campaign handoff; a fresh connection
restored access. The running campaign does not depend on the local connection.
