# Prepared CAD simulation campaign

**Status: offline step 1 implemented, 2026-09-04. No training runs are queued.** The subsequent execution request authorized the offline asset/configuration repairs and animated inspection. C0 and the serial-conversion portion of C1 now have CPU evidence; physical linkage fidelity and C2 remain open. [Step 1](MKII_STEP1.md) records changes and limits; [PLAN.md](PLAN.md) defines the program.

## Why early full-body training is useful

Yes: once known model/import defects are fixed, full-body simulation can start before the leg stand is built. It can establish action/frame correctness, discover candidate omnidirectional coordination, expose reward exploits, exercise navigation and deployment code, and compare terrain observations. Those gains do not require pretending CAD is exact hardware.

A later dynamics update does not automatically make every policy useless. Keep a nominal baseline and plausible randomized variants. When measurements arrive, evaluate existing checkpoints on the new nominal model and its measured uncertainty range before deciding to fine-tune or retrain. Large geometry, linkage, control-mode, action-schema or observation changes can require a new lineage. Domain randomization is not a substitute for correcting a known inertia error.

## Campaign sequence and stop conditions

| Phase | Question answered | Required evidence before proceeding |
|---|---|---|
| C0: source and asset reconciliation | What exact robot/configuration will run? | Baseline commit, clean isolated work area, CAD input hashes, Spark config comparison and explicit artifact directory. Preserve all historical tasks/runs. |
| C1: offline import correction | Does USD preserve the URDF's intended physics? | Correct converter or explicit versioned post-import authoring; all 19 tensors reconstruct in the link frame, masses/COMs/units/joints/collision geometry checked, all referenced layers hashed. Never apply a blind global quaternion inversion. |
| C2: short simulator acceptance | Does the articulation behave correctly before learning? | 18 named joints/six feet, resolved order and mappings, nonpenetrating reset, standing startup and settled torque/contact logs, per-joint sweeps, anatomical direction/yaw tests, finite observations and compatible runtime manifest. |
| C3: small throughput/smoke run | Can this exact environment learn and save recoverable artifacts? | Profile a small batch then increase environment count within measured memory/throughput limits; complete a short learning cycle, checkpoint, evaluation and resume test. No fixed giant environment count before profiling. |
| C4: flat omnidirectional baseline | Can an unconstrained gait track and stop in all directions? | Stand, ±forward, ±lateral, diagonals, ±yaw and combined-command transitions; reject collapse, foot dragging, persistent harmful contacts, joint-limit exploitation or excessive sustained torque. |
| C5: bounded robustness and terrain | Which uncertainties matter and where does the policy fail? | Measured/provisional actuator variants, friction and payload bounds; progressively qualified slopes/steps; evaluate each change against a common suite. |
| C6: perception and deployment | Can the actor work using only available onboard observations? | Compare proprioceptive baseline, ideal-terrain teacher and realistic-terrain student; calibration/age/dropout/occlusion tests; exact deployment preprocessing, timing, action and recurrent-state parity. |
| C7: integrated candidate | Does the full system accomplish the mission? | Held-out polygon/obstacle/fault scenarios, independently scored trajectory/coverage, failure ledger, reproducible release bundle and hardware-transfer prerequisites. |

Stop an experiment early for nonfinite state, wrong physical command direction, mismatched assets/schemas, unstable startup, persistent infeasible torque/contact, corrupt checkpointing or an invalid observation source. A loss/reward curve alone does not admit a policy. Tune only on training/development cases, then screen finalists over multiple seeds on the held-out matrix.

The existing 32-environment/1,000-step standing procedure is a starting probe after repair, not complete acceptance. Record startup peak raw demand separately from applied torque and steady-state saturation. Passing a standing threshold cannot establish dynamic accuracy, leg-leg clearance, motor thermal duty or terrain capability.

## Training design to resolve before a long run

- Add a new versioned CAD task/configuration where behavior or interfaces change; do not rename an existing task or retrofit a checkpoint's provenance.
- Define forward as anatomical -Y in the CAD frame and left as +X, with explicit transforms at the navigation seam. Resolve all 18 actions by joint name and record the observed articulation order.
- Use conservative initial velocity/acceleration envelopes derived from stance clearance, actuator capability and short checks. Increase them only after evaluation; archived mock command magnitudes are not new hardware requirements.
- Use tracking, stability appropriate to terrain, slip/contact, energy/actuation, limits and smoothness objectives. Log component values. No fixed tripod phase, gait clock or scripted swing sequence in the new learned policy.
- Include standing, starts/stops, reversals, turns and combined commands from the first tracking suite. Test all directions at the actual navigation sensor visibility limits.
- Parameterize motor mode/gains, friction/backlash, effective transmission, strength/voltage/thermal bounds, latency/jitter and payload dynamics. Unmeasured ranges are labeled provisional and physically motivated; no invented identification result.
- Isolate initial curriculum questions rather than changing reward, geometry, gains and randomization together. Reserve compute for finalists and reproducibility checks, not just one long training trajectory.

## Leg-stand update loop

Build the confirmed fixture as a fixed rail, passive vertical carriage and the three actuated joints, with the actual hip mount, moving mass, rail friction and ground contact. Use [the stand protocol](../artifacts/project_review_2026-09-04/LEG_TEST_STAND.md) for synchronized encoder/carriage/force and motor logs.

Fit actuator/linkage/fixture parameters on one subset of trajectories and validate on held-out loads and motions. Compare predicted and measured joint angles, carriage displacement, force, timing and temperatures within explicitly selected error tolerances. The stand constrains body motion and cannot validate whole-body balance or six-leg load redistribution.

For every model revision, run the same old-checkpoint/new-model screen and publish its delta. Fine-tune candidates that remain stable but lose performance; retrain when the representation or dynamics changed substantially. Recheck full-body torque, geometry, self-collision and support limits after mass or mechanical revisions. Freeze the model/calibration versions used by each field release.

## Spark readiness and experiment provenance

On a future execution request, inspect current unrelated GPU/container workloads first and follow the project's supervised launcher and lock protocol. Access through Tailscale was verified in this review, but current load, disk headroom and software health must be rechecked at launch. A stopped project queue does not guarantee the GPU is idle indefinitely.

Use an isolated, explicitly mounted source/run directory. The existing Spark mirror is not a Git checkout; do not run a blind `git pull` there or overwrite another task's changes. Save the effective source and configuration manifest before launching. Determine concurrency from actual memory/load; never compete with an unrelated job just because overall compute access is broad.

Each run bundle records source commit and any diff, task ID, all USD layer/CAD hashes, package/container versions, resolved config and randomization ranges, motor assumptions/calibration version, command/observation/action contracts, named joint mapping, seed, timing, checkpoint hashes, evaluation cases and failure outcomes. Use unique run labels, frequent recoverable checkpoints and an explicit resume test. Keep log/manifest generation deterministic and exclude secrets.

The new importer, v2 CAD configuration and runtime manifest are implemented. A versioned simulator acceptance implementation and physical linkage decision are still required. The future `isaaclab/train_mkii_v2.py` wrapper has a CPU-only `--dry-run`; it does not clear C2, authorize training or provide checkpoint admission. Never launch the historical validator as evidence for v2.
