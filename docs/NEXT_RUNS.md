# Prepared CAD simulation campaign

> **10 September 2026 — approved ground-truth training model:** the user visually approved the detailed direct-drive robot and requires its motor-weight-corrected URDF for **all future training**. [Canonical selection](../robot/active_model.json) and [full CAD/limit report](UPDATED_CAD_IMPORT.md) are authoritative. Do not launch further simplified C-study/mock or historical four-bar training. First prepare the new model-bound runtime, native SDF cooking and physical admission. Earlier campaign sequences below retain their dated lineage for reproduction; old checkpoints/task IDs are unchanged. Stage 2 smoothness and quiet-standing gates remain required.


## Detailed-model next steps

1. Verify `robot/active_model.json` and the corresponding `usd_002` nominal motor corrected source identities on the isolated native preparation directory.
2. Introduce a new task/runtime contract for the actual joint names, signs/zeros, per-leg yaw limits, +X tibia contact geometry and explicitly modeled actuator behavior. Preserve old task registrations.
3. Cook the exact SDF geometry in Isaac Sim 6.0.1 and qualify suspended travel, contacts/settling, self-collision handling, solver effort and observation/action mapping before a small learning smoke.
4. Only a matching native admission can authorize learning on this canonical model; record its source, asset, motor and checkpoint hashes. Then pursue the unchanged all-direction, smoothness and quiet-standing gates.

## Historical C-study continuation order (superseded)

This sequence records the prior C-study plan and no longer authorizes new simplified-model training. The later four-bar campaign sections also remain historical. Live outcomes and exact Spark ownership belong in [STATUS](../STATUS.md).

1. Bind the preserved 315/318 omnidirectional checkpoint to a newly named cold evaluation under the common 0.040 rad/20 ms formal limiter; completed execution is recorded in STATUS. Keep original solver 16/4 and external-force settings explicit; the historical 0.03 diagnostic and reference009 physics are separate comparisons. Require exact checkpoint/source/asset identity and fresh standing before actor playback.
2. Run a 32×24×2 positive-CAPS integration smoke with actual checkpoint reload and constant/stop evaluations. After successful integration, independently initialize matched 1024×24×50 curriculum and curriculum-plus-CAPS pilots from the original checkpoint. Use one immutable behavioral source with explicit branch/allocation selectors and fresh standing for each allocation. The curriculum covers standing, starts/stops, reversals, all bearings and combined translation/yaw. Mask reset-crossing temporal pairs and avoid treating intentional command transitions as sensor noise. Preserve quiet and useful tracking together; do not accept reduced motion as smoothness.
3. Prepare a separately named phase/foot-placement-modulating controller as the architecture alternative. The fixed wave's tiny residual and chosen swing clock do not establish a physical speed ceiling. Actual support, load transfer, joint/torque limits and full command response must admit added authority.
4. The failed reference learner's recovery receives a small zero-residual recorded-randomized-versus-canonical reset comparison. Keep 200-control recovery, root state, gains, 1.6 N·m and unchanged checks. Repeated full/mixed-row resets must be measured before renewed learning. Do not repeat the failed allocation unchanged. Intermediate checkpoints may preserve diagnostic learned weights but remain unqualified until cold screens pass.
5. Measure valid learning experience, throughput, failures and memory before allocating more replicas or updates. Use matched source/configuration and evaluation cases; record source/asset/checkpoint hashes and each limiter. Paper training times are not Spark ETAs.
6. Complete the full Stage 2 quantitative and visual matrix, then the final frozen terrain envelope and untouched qualification layouts with retained flat capability. Develop proprioception-only, privileged-terrain and causal noisy-map conditions in parallel; compare identical cases to separate controller and perception failures.

The [breadth-first paper review](../artifacts/project_review_2026-09-10/legged_policy_literature_001/REPORT.md) supports this comparison design. Its recommendations do not replace physical admission, existing numerical thresholds or visual benchmark quality. Live allocation state belongs only in [STATUS](../STATUS.md).

**Physical four-bar validation and training campaign implemented, 5 September 2026 UTC; physical acceptance remains pending and PPO has not started.** The user authorized continuing into training after the required checks. The new task has 31 bodies, 30 articulation coordinates, six physical closure joints, 18 active motors, an explicit bounded RS05 model and an 84-value observation contract without a gait clock. The earlier [serial standing reports](../artifacts/mkii_step2_2026-09-04/README.md) remain a separate baseline. [STATUS.md](../STATUS.md) records the latest probe and compute state; [PLAN.md](PLAN.md) defines the program.

## Why early full-body training is useful

Yes: once known model/import defects and the actuator acceptance gate are addressed, full-body training can start before the leg stand is built, with unmeasured parameters explicitly provisional. It can establish action/frame correctness, discover candidate omnidirectional coordination, expose reward exploits, exercise navigation and deployment code, and compare terrain observations. Those gains do not require pretending CAD is exact hardware.

A later dynamics update does not automatically make every policy useless. Keep a nominal baseline and plausible randomized variants. When measurements arrive, evaluate existing checkpoints on the new nominal model and its measured uncertainty range before deciding to fine-tune or retrain. Large geometry, linkage, control-mode, action-schema or observation changes can require a new lineage. Domain randomization is not a substitute for correcting a known inertia error.

## Campaign sequence and stop conditions

The immediate bounded sequence is **1 environment × 100 control steps → 32 environments × 1,000 standing control steps plus 2,400 driven control steps (9,600 driven physics substeps), repeated with nominal and refined solver iterations → 64 environments × 3 scratch PPO iterations → a separate process resuming the verified checkpoint for 512 environments × 1,000 iterations**. The larger run depends on smoke-run memory/throughput, checkpoint/inference checks, source identity and fresh GPU admission. No step is automatically admitted by the historical serial pass.

The driven count includes **18 individual motors × two signs × 50 control steps = 1,800**, plus **three motor groups × two signs × 100 = 600**. Each control step contains four 5 ms physics substeps. This paragraph describes the planned 5 September physical campaign. That lineage is now paused pending the revised leg export; its later support-gate failure and exited jobs are recorded in [the archived physical status](archive/STATUS_2026-09-08_physical.md). Corrected physical-model PPO has not started. [STATUS.md](../STATUS.md) records the separately authorized C-study execution.

The physical validator samples actual body-link pin positions/axes, primitive ground clearance and motor limits every 5 ms. Reset coordinates must match the physical contract, and the anatomical command-frame check must pass. During initial PPO, a numerical closure failure invalidates the whole run before reset can hide it; normal falls remain learning events. A successful bounded campaign establishes provisional flat-ground learning, not a qualified terrain or hardware controller.

| Phase | Question answered | Required evidence before proceeding |
|---|---|---|
| C0: source and asset reconciliation | What exact robot/configuration will run? | Baseline commit, clean isolated work area, CAD input hashes, Spark config comparison, versioned actuator parameters/hash and resolved configuration identity in the simulation/runtime manifest, and explicit artifact directory. Preserve all historical tasks/runs. |
| C1: offline import correction | Does USD preserve the intended physics? | Serial characterization: all 19 tensors, masses/COMs/units/joints/collisions and dependency hashes. Physical reference before training: dedicated 31-body / 30-tree-coordinate / six-closure audit, corrected passive joint frames, no mimic/passive drives, and an explicit 18-actuator adapter. Never apply a blind global quaternion inversion. |
| C2: short simulator acceptance | Does the actual mechanism behave correctly before learning? | Serial standing is an intermediate characterization. Before training, qualify the physical loop/adapter under one-mechanism and full-body tests: closure/branch/convergence, measured reset, all 18 active motor mappings and passive states, per-collider support/shaft/lever/rod contacts, substep torque, driven joint sweeps, anatomical direction/yaw and finite observations. Qualify the versioned motor's torque-speed/voltage, thermal, burst/recovery and phase-current limits; align applied clipping, rewards, faults and runtime semantics. Freeze the corresponding runtime manifest. |
| C3: small throughput/smoke run | Can this exact environment learn and save recoverable artifacts? | Profile a small batch then increase environment count within measured memory/throughput limits; complete a short learning cycle, checkpoint, evaluation and resume test. No fixed giant environment count before profiling. |
| C4: flat omnidirectional baseline | Can an unconstrained gait track and stop in all directions? | Stand, ±forward, ±lateral, diagonals, ±yaw and combined-command transitions; reject collapse, foot dragging, persistent harmful contacts, joint-limit exploitation or excessive sustained torque. |
| C5: bounded robustness and terrain | Which uncertainties matter and where does the policy fail? | Measured/provisional actuator variants, friction and payload bounds; progressively qualified slopes/steps; evaluate each change against a common suite. |
| C6: perception and deployment | Can the actor work using only available onboard observations? | Compare proprioceptive baseline, ideal-terrain teacher and realistic-terrain student; calibration/age/dropout/occlusion tests; exact deployment preprocessing, timing, action and recurrent-state parity. |
| C7: integrated candidate | Does the full system accomplish the mission? | Held-out polygon/obstacle/fault scenarios, independently scored trajectory/coverage, failure ledger, reproducible release bundle and hardware-transfer prerequisites. |

Stop an experiment early for nonfinite state, wrong physical command direction, mismatched assets/schemas, unstable startup, persistent infeasible torque/contact, corrupt checkpointing or an invalid observation source. A loss/reward curve alone does not admit a policy. Tune only on training/development cases, then screen finalists over multiple seeds on the held-out matrix.

The existing 32-environment/1,000-step standing procedure is a starting probe after repair, not complete acceptance. Record startup peak raw demand separately from applied torque and steady-state saturation. Passing a standing threshold cannot establish dynamic accuracy, leg-leg clearance, motor thermal duty or terrain capability.

The historical serial reports characterize an actuator capped at 1.6 N·m, although 5.5 N·m peak already appears in that URDF/config. Startup raw demand of 2.364431 N·m is below published peak. The [RS05 specification review](RS05_SPEC_REVIEW.md) distinguishes 1.2 N·m continuous stall from cooling-dependent rotating ratings and bounded bursts. The old scalar cap is not universally conservative and its manifest omits actuator parameters. Preserve those reports with that interpretation.

The physical task now has a separate versioned model and resolved motor identity: torque-speed/voltage limits, continuous-load allowance, shared burst/recovery budget and phase-current proxy. Its 48 V assumption, cooldown, gains, friction, inertia and cooling equivalence remain provisional. The budget is not a calibrated temperature model, and the proxy is not hardware CAN/current protection. Native qualification and later stand identification remain necessary.

The [collider-fit evidence](../artifacts/mkii_fourbar_2026-09-05/collider_fit/README.md) prevents a 1:1 contact claim: silicone samples lie up to 6.38 mm outside the complete tibia collision union, including 5.04 mm in a nominal lower contact band. The tested ±0.30 rad target grid gives foot-bottom bias −3.109 to +0.427 mm relative to CAD; broad structural boxes also introduce substantial per-link excess volume. Fit the pad exterior and structural clearance regions, then requalify the changed asset before relying on precise terrain contact. These sampled measurements neither prove full mesh coverage nor qualify self-collision or real pad compliance.

## Training design to resolve before a long run

- Add a new versioned CAD task/configuration where behavior or interfaces change; do not rename an existing task or retrofit a checkpoint's provenance.
- Define forward as anatomical -Y in the CAD frame and left as +X, with explicit transforms at the navigation seam. Resolve all 18 actions by joint name and record the observed articulation order.
- Use conservative initial velocity/acceleration envelopes derived from stance clearance, actuator capability and short checks. Increase them only after evaluation; archived mock command magnitudes are not new hardware requirements.
- Use tracking, stability appropriate to terrain, slip/contact, energy/actuation, limits and smoothness objectives. Log component values. No fixed tripod phase, gait clock or scripted swing sequence in the new learned policy.
- Include standing, starts/stops, reversals, turns and combined commands from the first tracking suite. Test all directions at the actual navigation sensor visibility limits.
- Bind an explicit versioned motor configuration instead of inheriting the archived actuator. Parameterize mode/gains, friction/backlash, effective transmission, torque-speed/voltage, thermal and burst/recovery limits, phase current, latency/jitter and payload dynamics. Do not make 5.5 N·m an indefinite effort allowance. Unmeasured ranges are labeled provisional and physically motivated; no invented identification result.
- Isolate initial curriculum questions rather than changing reward, geometry, gains and randomization together. Reserve compute for finalists and reproducibility checks, not just one long training trajectory.

## Leg-stand update loop

Build the confirmed fixture as a fixed rail, passive vertical carriage and the three actuated joints, with the actual hip mount, moving mass, rail friction and ground contact. Use [the stand protocol](../artifacts/project_review_2026-09-04/LEG_TEST_STAND.md) for synchronized encoder/carriage/force and motor logs.

Fit actuator/linkage/fixture parameters on one subset of trajectories and validate on held-out loads and motions. Compare predicted and measured joint angles, carriage displacement, force, timing and temperatures within explicitly selected error tolerances. The stand constrains body motion and cannot validate whole-body balance or six-leg load redistribution.

For every model revision, run the same old-checkpoint/new-model screen and publish its delta. Fine-tune candidates that remain stable but lose performance; retrain when the representation or dynamics changed substantially. Recheck full-body torque, geometry, self-collision and support limits after mass or mechanical revisions. Freeze the model/calibration versions used by each field release.

## Spark readiness and experiment provenance

Spark execution is authorized. Inspect current unrelated GPU/container workloads before each launch and use the versioned supervised launcher and lock protocol. Access through Tailscale and live Isaac startup are verified, but current load, disk headroom and software health must be rechecked at launch. A stopped project queue does not guarantee the GPU is idle indefinitely.

Use an isolated, explicitly mounted source/run directory. The existing Spark mirror is not a Git checkout; do not run a blind `git pull` there or overwrite another task's changes. Save the effective source and configuration manifest before launching. Determine concurrency from actual memory/load; never compete with an unrelated job just because overall compute access is broad.

Each run bundle records source commit and any diff, task ID, all USD layer/CAD hashes, package/container versions, resolved config and randomization ranges, motor assumptions/calibration version, command/observation/action contracts, named joint mapping, seed, timing, checkpoint hashes, evaluation cases and failure outcomes. Use unique run labels, frequent recoverable checkpoints and an explicit resume test. Keep log/manifest generation deterministic and exclude secrets.

Use the new physical campaign's `isaaclab/deploy/run-mkii-fourbar`, `isaaclab/validate_mkii_fourbar.py` and `isaaclab/train_mkii_fourbar.py` with an isolated source and unique report directory. The physical source is `/home/orionh/HEXAPOD_runs/mkii_fourbar_v1/source`. The earlier `validate-mkii-v2` and `train_mkii_v2.py` apply to the serial lineage and cannot certify or train this physical task. Preserve its source/checkpoint identities and resume in a separate process.

The physical model and 30-coordinate / 18-motor adapter are implemented from the [recovered CAD pin frames](../artifacts/mkii_step2_2026-09-04/physical_fourbar_reference/README.md). Its active push-lever coordinate uses a recorded phase bridge and per-leg defaults/limits; passive reset coordinates close the loops without mimic drives. The nominal reset height is 0.142970 m. The 1-environment startup pass is preliminary; full standing, driven and solver-convergence acceptance remains pending. Current live results and retries belong in [STATUS.md](../STATUS.md).

Packaging changes are limited to the two old-manifest-listed core/env `pyproject.toml` files; 112 archived lineage files remain independently verified. Preserve the old manifest and historical reports instead of relabeling them as the physical task's release evidence.

Historical four-bar allocation instruction, superseded for new training: use **full available Spark compute**, until another agent requests sharing through `/home/orionh/SPARK_COMPUTE_COORDINATION.md`. Read that file before each new launch and at checkpoint boundaries during future long runs; the [coordination note](SPARK_COMPUTE_COORDINATION.md) holds the handoff procedure. The former 60/40 split is a starting preference if sharing is requested. Short acceptance remains exclusive and already uses full available compute. No quota or shared launcher is enabled; measure throughput and memory in a paired pilot before relying on overlapping long jobs.

The user explicitly prioritized the present hexapod campaign over weather. An owned guard reserves the weather launcher's existing `/opt/wx/gpu.lock` cooperatively until campaign exit, for at most three hours. Preserve workload admission checks and the shared-file handoff; this reservation does not authorize arbitrary changes to other jobs. Exact process and probe state is recorded in [STATUS.md](../STATUS.md).

The earlier reference-only recovery proposal remains a separate infrastructure lineage. Its successful admission cannot guarantee every later timeout reset, and its fixed gait is not the project's selected final architecture. Any successor preserves requested-goal motion rewards, zero-bootstrap true terminations, final pre-reset critic values for time limits, and selected/unselected history isolation. Formal motor/contact, cold-policy, quiet and Stage 2 gates remain unchanged. See the C-study sequence above for the current experiment design.


## Historical 2026-09-10 C-study direct-PPO proposal (superseded)

The then-proposed allocation was one quiet-priority temporal-smoothing ablation, initialized afresh from original315/318 checkpoint1971…, with the current19-body/18-joint C asset and unchanged direct source/physics/action contracts. Native004 is preparation only until its frozen source, CPU regressions and fresh same-source smoke are admitted. Root remains the only guarded Spark dispatcher; this document does not launch or admit a job.

- Valid consecutive exact-zero command pairs: temporal coefficient1.0. Moving pairs:0.1; spatial:0.1. Preserve the current all-valid denominator and done/command-change masks. Quiet pairs already make up68.34% of observed valid pairs, so the change explicitly increases their objective priority.
- Pilot:1024 environments ×24 controls ×50 updates, with original checkpoint, fresh optimizer/std protocol, adaptive LR, deterministic acquisition phases and original common0.040 rad/20ms limiter. New smoke32×24×2 must exercise this actual branch and strict reload before the pilot.
- Record mode-separated temporal losses, sparse actor-gradient decomposition and per-minibatch KL/rate. Verify diagnostics leave optimizer/normalizer/RNG and learning results unchanged when the objective is held equal.
- Retain all-direction/yaw/arc and quiet-stop comparison, exact checkpoint hashes and full failure evidence. A longer unchanged budget remains scientifically valid; this short ablation is chosen to discriminate objective priority quickly, not to claim convergence or impossibility.

See the [actual matched pilots](../artifacts/omni_diagnostics_2026-09-09/direct_omni_matched_pilots_001/REPORT.md) and [verified review/proposal](../artifacts/project_review_2026-09-10/claude_partner_review_001/README.md). Never reuse the failed preview finalizer without its separately verified successor.


For the quiet-priority allocation, use the [frozen native004/host003/guard004 preparation](../artifacts/omni_diagnostics_2026-09-09/direct_omni_quiet_priority_preparation_001/README.md) and require the complete matching source003 smoke before pilot admission. Keep root as sole dispatcher. The user explicitly prioritizes PPO over all weather activity: defer identified weather blockers while preserving outputs and exact recovery/ownership controls. Live state belongs in STATUS; this paragraph does not queue or launch a pilot.


## Historical 2026-09-10 C-study budget proposal (superseded)

Historical context only; this does not authorize another simplified-model launch. The [completed stronger-quiet comparison](../artifacts/omni_diagnostics_2026-09-09/direct_omni_quiet_priority_pilot_001/README.md) supersedes the immediate50-update proposal above. Prepare one normal-CAPS1024×24×500 run from original checkpoint1971…, unchanged objective/physics/gates and the complete fresh optimizer/std protocol. Use native005/source004 and require its own successful32×24×2 integration; older-source smoke cannot admit it. Root alone dispatches through a separately bounded guard.

Retain all500 ordinary saves remotely, exact file hashes, final and decision1/10/25/50/100/250/500; collect14 sparse gradient rows across those milestones. The versioned host must enforce1800 seconds for training,600 for other phases and90 for startup. A500-update result still requires exact reload, fresh direction/stop evaluation and unchanged numeric/visual gates. No automatic next branch, exact-resume claim or Stage2 promotion follows from budget completion. Current execution remains only in STATUS.


For the bounded500-update source004 experiment, use the [host005/guard002 successor](../artifacts/omni_diagnostics_2026-09-09/direct_omni_extended_caps_retry_002/README.md), retaining both successful smoke004 and failed001 ownership/restoration admission. Read-only preflight must execute the actual installed supervisor setup on Spark. Select a final checkpoint for the prepared diagnostic recording only after full campaign/reload/evaluation evidence. Full training raw may remain remote with exact inventories and CPU analysis; explicit omissions must not be called local replay. Current execution belongs only in STATUS.


## 2026-09-10 — first canonical native inspection scope

Prepare one headless native import of the exact motor-mass-corrected USD through a new source/asset identity. The initial scope is zero-gravity, no ground, no drives or target/reset writes: inspect actual19-body/18-joint state, full inertias/limits and native SDF shapes, retain SDK warm-up and eight explicit2.5ms samples. Authoring readback alone cannot prove native cooking. Keep the authored self-collision settings; constraint correction remains possible in zero gravity. Bound actual application readiness and the total inspection, preserve failed outputs, and reverify all sources/dependencies after exit. This is preparation for native physical admission, not admission or learning.

The successor to [inspection001](../artifacts/mkii_updated_2026-09-10/native_inspection_001/README.md) must use the installed public tensor API rather than the obsolete private import that failed before model loading. Use new source/host/guard/output names and bind the original failure's exact cleanup/restoration. Keep the same asset and eight-step scope. Declare the initialization barrier and any unavailable cooking counter explicitly; no unknown value may masquerade as zero pending work. A successful import still precedes actuator, contact and supported-standing admission.

[Inspection002](../artifacts/mkii_updated_2026-09-10/native_inspection_002/README.md) preserves the partial native identity result and the compiled SDF signature failure. The next isolated source uses a supported string pattern and retains exact returned-path/count validation against all 153 instances. No tolerance, collider or mass changes follow from that API correction. Do not treat the successful initial readback as an eight-step or contact pass.

## 2026-09-10 — after passive import, admit explicit actuation and support

Bind [native003](../artifacts/mkii_updated_2026-09-10/native_inspection_003/README.md) and the [corrected actuation design](../artifacts/mkii_updated_2026-09-10/actuation_design_001/README.md). Prepare one floating zero-gravity coordinate/effort experiment with unchanged model detail, no floor or position drives, signed0.005Nm pulses for20ms and20ms coast. Keep the0.05rad/2rad/s diagnostic bounds and preserve all raw channels. The old larger pulse/coast proposal was rejected before execution.

In parallel, prove SDF distance/gradient channel layout, local frames and toe geometry against source meshes. Then qualify the provisional neutral servo on one cold robot and32 replicas with the new distal+X contact classifier. Only the matching actuator/support result can admit fresh32×24×2 PPO integration and strict reload. Choose larger batches from measured canonical-model throughput and memory. Existing omnidirectional smoothness, quiet-standing and actuator gates remain unchanged; hardware transfer still requires identification.

The [historical comparison and independent/Fable review](../artifacts/project_review_2026-09-10/direct_omni_extended500_interpretation_001/README.md) inform logging and action-state design. No further C-study GPU training follows from that result. Suspended excitation, gravity/support settling and motor-envelope tests require the subsequent explicit runtime and justified actuator settings; CAD zero is not automatically a supported stance.


## 2026-09-10 — supported standing after native effort checks

Bind the [completed detailed coordinate/effort evidence](../artifacts/mkii_updated_2026-09-10/native_actuation_001/README.md) without altering its source or asset. The native generalized matrix uses the root-COM velocity form; retain both raw conventions. Run the separately prepared read-only source-mesh/SDF query, then qualify actual toe contact and supported standing with the declared provisional servo. Query acquisition and its diagnostic verdicts are separate: a completed query can still reject its declared channel/frame/shape assumptions.

For the new standing screen, identify the plate-frame root origin explicitly and initially retain a 0.055 m minimum height against the nominal 0.076611 m stance. Record this geometry-bound choice rather than asserting equivalence to the historical base frame. Keep quiet drift, heading, joint motion, target steps, torque, saturation, no-reset and nonfoot-contact gates unchanged. Derive the distal +X contact patch from the real rounded-square tibia mesh before dispatch. One-robot standing precedes actual 32-replica memory/throughput and the fresh PPO integration; no prior C-study capacity or actor is transferred.


The source-shape query's measured tuple binding is now recorded in the [separate interpretation](../artifacts/mkii_updated_2026-09-10/native_sdf_query_001/README.md). Future queries must explicitly decode gradient-first/distance-last and keep original acquisition verdicts intact. Proceed to supported standing on the unchanged geometry. For the new full-substep nonfoot/clearance requirement, use the clipped triangle boundary rather than a filtered vertex subset; six-toe support remains a post-settling check.
