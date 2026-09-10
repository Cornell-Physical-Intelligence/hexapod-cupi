# Quiet standing and visual completion review

These review modules are separate from the frozen repair source. They are CPU-tested; their first Isaac invocation must be a smoke check. Add their dispatch hooks only to a new frozen source. Existing running jobs and source manifests must remain unchanged.

## Quiet standing and stopping

Call `omni_quiet_review.evaluate_quiet_review(env, runner, plan, output, checkpoint_sha)` when `plan.omni.quiet_stand_review` is present, before the ordinary diagnostic/full-evaluation dispatch. Use a dedicated plan with `evaluation_num_envs: 48`, remove `omni.diagnostics`, and set:

```json
{"quiet_stand_review": {"seed": 27057}}
```

The evaluator runs 32 seconds of undisturbed standing, scoring the final 30 seconds. It then runs 12 command scenarios in parallel for eight seconds, requests zero velocity, allows two seconds after the ramped command reaches zero, and scores at least ten further seconds. The total simulation time is 53 seconds. Every termination or truncation anywhere in a trial invalidates its replica, including failures before its scored window.

Proposed simulation bounds apply individually to every joint and environment: joint velocity RMS ≤0.03 rad/s; joint travel range ≤0.02 rad; target-step 95th percentile ≤0.002 rad per 20 ms; requested torque saturation ≤0.5% for each joint; applied torque ≤1.60001 Nm; planar excursion ≤1 cm; heading excursion ≤2 degrees. These are explicit engineering targets requiring hardware calibration later. Disturbance recovery is a separate trial so legitimate balance corrections are not treated as failure to stand quietly.

Outputs are `quiet_stand.json`, runner-compatible `evaluation.json`, and two compressed traces. They explicitly identify their review type and never declare Stage 2 complete.

## All-direction visual review

Call `omni_visual_review.record_visual_review(env, runner, plan, output, checkpoint_sha)` when `plan.omni.visual_review` is present. Rendering uses one full robot at the established 20 ms control period. Use a fresh plan with:

```json
{"visual_review": {"seed": 37057}}
```

There are 46 cases totaling 334 seconds: 16 translation bearings at each of 0.10 and 0.20 m/s; both signs of 0.20 and 0.40 rad/s turns; gentle/tight arcs; both strafing-arc turn directions; S curve; fixed-heading curve; direction reversals; and 32 seconds of zero-command standing. Independent trials reset only at labeled boundaries.

Direct command cases have no position-feedback correction, so drift and unintended motion remain visible. Only the S and fixed-heading path cases use the explicitly labeled ideal-simulator-pose follower. The learned policy always controls the joints and actual contact physics determines body motion. Ground blue/gold/orange annotations show the desired path/travel, body heading and actual trail.

For the first rendering smoke, an explicit subset is supported:

```json
{"visual_review": {"cases": ["bearing_0_0.10", "turn_+0.20", "quiet_stand"]}}
```

Subset recordings set `all_cases_recorded: false`. Full recordings still require visual inspection against the accepted forward benchmark and current straight trial; merely rendering a clip cannot pass smoothness. The five-minute clip has a per-case frame index in its JSON, so any individual case can be extracted later without repeating GPU work.

Outputs are `rollout.mp4`, `video.json`, `visual_review.json`, per-case progress JSON and a compressed per-joint trace keyed by case and time. Final Stage 2 review additionally requires all existing numerical tracking/contact/torque gates and the sustained quiet/stop evaluator.
