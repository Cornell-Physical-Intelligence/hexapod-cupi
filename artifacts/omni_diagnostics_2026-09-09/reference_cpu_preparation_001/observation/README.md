# Wave002 + residual002: executable CPU observation contract

This is a **CPU-only, versioned contract**, with **740 actor values and 743 critic values** computed from the field definitions. It has no learner, checkpoint loader, physics admission or deployed estimator. Useful measured walking and the full physical gates remain prerequisites for integrating PPO. No production or pinned runtime file is changed.

The binding is frozen wave002 (`ef92745d…a56b`) and residual002 core (`fbf65749…6e40`). `source_contract.json` records the full hashes, exact 5 mm wave configuration, leg names and named canonical stance. A 7 mm successor is deliberately rejected under this identity even though its state API is unchanged. Its adoption requires a separately reviewed binding and tests. The older 525/528 proposal, 675/678 physics wrapper, and previous velocity policies are incompatible.

## State and timing

The actor contains five chronological 63-value sensor/action/requested-command frames, five history-valid flags, episode age, and 419 current values. The current fields contain executable target and residual position/velocity; measured body and toe motion; contact points, classification, validity and age; the original swing and separate landing coefficients/timings; flight, descent, contact-loss and confirmation counters; landing candidate point and preload; desired motion versus measured pose; bounds, anchors and stop/quiet state. No current decision deadline or retained trajectory is replaced by a phase label. The exact contiguous slices and provenance are generated in `spec_formal_004.json` and `spec_diagnostic_003.json`.

The sensor frame is gyro×0.25, projected gravity, next requested forward/left/yaw×[5,5,2.5], named joint offset from the frozen nominal stance, joint velocity×0.05, and the preceding normalized bounded residual goal. The previous reference request is also encoded separately: it is controller state used to detect a stop transition, and may differ from the next requested command. Known controller/reference state receives no invented sensor noise. The caller supplies one already-sampled measurement; rereading a step returns the same observation, and a changed same-step input is rejected rather than advancing history or silently redrawing noise.

The API is `ObservationBuilder(joint_names, named_nominal, num_envs, limits=...)`, then explicit `reset(ids, fresh_episode_ids, reset_times)`, then `build(packets)` with one complete packet per replica. The first sample is step zero. Subsequent samples must be exactly 20 ms apart; missed or reordered samples fail. A partial reset clears only those replicas. Reusing an episode identity, continuing a terminal sample, or submitting an old episode fails. Validation is transactional: one invalid row does not mutate any replica's history. Returned arrays and metadata cannot mutate cached observations or profile limits.

Each packet binds its source identities, runtime joint order, completed measurement time, current reference output and current executable controller state. Reference position/velocity plus residual position/velocity must reconstruct the total target; the reference output and float32 emitted target must agree. The nominal stance is checked by name. Runtime permutations are accepted only when every joint field and the declared order match; the schema identity records that order.

Formal 0.04 rad/20 ms and diagnostic 0.03 profiles have separate limits and schema identities. They are not pooled or described as motor speed ratings. The three critic-only values are explicitly supplied true navigation-frame linear velocity. In an instrumented simulator where the actor also receives true body velocity, this critic field is redundant; the separation allows a later, explicitly qualified estimator/teacher comparison without silently feeding the actor privileged critic values.

## Coordinate and provenance contract

Points are translated then rotated into the current body frame. Velocities, preload vectors and higher polynomial coefficients are rotated only. Both initial and desired rotations are relative to the measured body, avoiding quaternion sign ambiguity. Flight baseline and peak heights are encoded relative to current body height along explicit world up; the gravity vector and body rotation must agree. Tests apply common translation, yaw and arbitrary coordinate-frame rotations, transforming world up and scalar height origins consistently. The output remains unchanged. This is coordinate covariance of the encoder, not a claim that the flat-world generator supports tilted terrain or arbitrary changes to physical gravity.

The accepted measurement provenance is explicitly one of `simulator_truth_instrumented`, `synthetic_fixture`, or `future_estimator_unqualified`. Every output still states `deployment_qualified=false` and `policy_training_allowed=false`. Selecting an estimator label does not qualify hardware, contact estimation, odometry, uncertainty or timing. Desired reference state is always labeled `virtual_desired_motion_not_prescribed_physics_pose`. No body pose or contact is commanded by this builder.

Fresh contact classification is distinct from availability of a contact point. All six contact estimates must be valid and no older than 40 ms. A confirmed contact requires a valid finite point. An airborne foot may have no point; the explicit representation is a finite zero sentinel with `contact_point_valid=false`. The builder rejects NaNs rather than inventing support. The fixture exporter performs this documented storage conversion only where the original point-valid mask is false; it retains that mask. A production adapter must make the same distinction and must independently qualify the 40 ms freshness contract.

## Tests and reproduction

```sh
PYTHONDONTWRITEBYTECODE=1 uv run python -B -m unittest discover \
  -s tmp/reference_policy_observation_002 -p 'test_*.py' -v
```

The self-contained fixtures include an actual settled reference002 sample initialized through wave002, and one explicitly synthetic landed sample after replaying the real failed swing prefix. Both use an explicitly shifted encoder time origin. The synthetic fixture is a controller-state test, not evidence of physical landing. `build_fixtures.py` records provenance and refuses to overwrite an existing file; recreating it requires the archived frozen source/trace and a fresh `--out`.

`export_spec.py` validates a fixture and computes the slices, implementation hash, source/config hash, profile and schema identity. It refuses existing output. For example:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tmp/reference_policy_observation_002/export_spec.py \
  --profile formal_004
```

Tests cover both trajectory modes; frame covariance; observable differences in command-filter, flight, touchdown, landing, preload and stop state; partial resets and stale history; same-step repeat behavior; runtime joint permutation; strict missing/malformed/nonfinite/source/config rejection; contact freshness; and separation of privileged critic input. Later integration still needs the actual simulator/deployment adapter, source verification at process startup, latency/noise validation, full gait success, and a newly admitted learner/checkpoint lineage.
