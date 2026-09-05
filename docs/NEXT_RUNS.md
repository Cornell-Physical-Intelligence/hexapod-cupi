# Prepared CAD simulation campaign

**Status: corrected serial standing validation implemented, 2026-09-04 (later runs on 5 September UTC). No training runs are queued.** The user subsequently authorized Spark execution. C0 and the serial-conversion portion of C1 have CPU and Kit-native evidence; the bounded standing checks now have [live reports](../artifacts/mkii_step2_2026-09-04/README.md). Physical linkage dynamics, driven joint/direction checks and the [RS05 actuator contract](RS05_SPEC_REVIEW.md) still keep C2/G0 open. [Step 1](MKII_STEP1.md) records the earlier repairs; [PLAN.md](PLAN.md) defines the program.

## Why early full-body training is useful

Yes: once known model/import defects and the actuator acceptance gate are addressed, full-body training can start before the leg stand is built, with unmeasured parameters explicitly provisional. It can establish action/frame correctness, discover candidate omnidirectional coordination, expose reward exploits, exercise navigation and deployment code, and compare terrain observations. Those gains do not require pretending CAD is exact hardware.

A later dynamics update does not automatically make every policy useless. Keep a nominal baseline and plausible randomized variants. When measurements arrive, evaluate existing checkpoints on the new nominal model and its measured uncertainty range before deciding to fine-tune or retrain. Large geometry, linkage, control-mode, action-schema or observation changes can require a new lineage. Domain randomization is not a substitute for correcting a known inertia error.

## Campaign sequence and stop conditions

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

These reports characterize an actuator capped at 1.6 N·m, although 5.5 N·m peak already appears in the URDF/config. Startup raw demand of 2.364431 N·m is below published peak. The [RS05 specification review](RS05_SPEC_REVIEW.md) distinguishes 1.2 N·m continuous stall from cooling-dependent rotating ratings and bounded bursts. The present torque-speed/thermal approximation is unqualified, its scalar cap is not universally conservative, and the current manifest omits actuator parameters. Preserve this baseline; use a new motor/task contract before training the physical four-bar.

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

The importer, v2 CAD configuration, runtime manifest and bounded `isaaclab/validate_mkii_v2.py` standing validator are implemented. Use `isaaclab/deploy/validate-mkii-v2` as documented in [Operations](OPERATIONS.md#11-corrected-serial-cad-v2-live-validation); it preserves per-run source hashes and checks the explicit JSON outcome even when Kit exits zero. A passing serial standing report does not clear the full C2 gate. The future `isaaclab/train_mkii_v2.py` wrapper has a CPU-only `--dry-run`; it provides neither physics acceptance nor checkpoint admission. Never launch the historical validator as evidence for v2.

The next physical-model task is concrete: the [recovered CAD pin frames and implementation plan](../artifacts/mkii_step2_2026-09-04/physical_fourbar_reference/README.md) define a 31-body physical four-bar model, six excluded closure joints and an explicit 30-coordinate / 18-actuator adapter. The current full assembly resolves the earlier stale cut-frame discrepancy without another CAD clarification. First qualify one passive mechanism under driven motion, then the full body; do not substitute animated mimics for solver evidence or silently reuse the serial stance and motor-angle contract.

Use **full available Spark compute now**, until another agent requests sharing through `/home/orionh/SPARK_COMPUTE_COORDINATION.md`. Read that file before each new launch and at checkpoint boundaries during future long runs; the [coordination note](SPARK_COMPUTE_COORDINATION.md) holds the handoff procedure. The former 60/40 split is a starting preference if sharing is requested. Short acceptance remains exclusive and already uses full available compute. No quota or shared launcher is enabled; measure throughput and memory in a paired pilot before relying on overlapping long jobs.
