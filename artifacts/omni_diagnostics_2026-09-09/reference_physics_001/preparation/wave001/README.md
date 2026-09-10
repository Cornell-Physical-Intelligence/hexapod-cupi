# Measured-contact wave reference 001 — CPU preparation, 2026-09-10

This prepares the next **zero-residual full-robot physics test**. It is not a walking policy, a physics admission or a terrain result. All 23 ideal-motion/contact cases pass the reference geometry and executable budget checks. The first actual test remains 0.005 m/s forward with one swinging leg and five measured supporting feet. A projected support polygon cannot establish the 1.6 N·m torque requirement; the physics adapter must independently stop on excessive requested torque, nonfoot contact, lost support, invalid reference or termination.

Runtime files are exactly `wave_reference.py`, `serial_geometry.py`, `wave_math.py`, and `geometry/{candidate_c_reference.json,f050_t060.urdf}`. The geometry inputs are byte-identical copies of the frozen C benchmark inputs; no CAD was edited. The exact named URDF FK/IK and joint signs are retained. There is no dependency on the previous prototype's directory at runtime. NumPy, SciPy and PyTorch are required. Older prototypes remain unchanged.

## API and physical-state contract

```python
generator = WaveContactReference(observed_articulation_joint_names)
initial = generator.reset(settled_measured_snapshot)
result = generator.step(measured_snapshot, [0.005, 0.0, 0.0], dt=0.02)
if not result["valid"][0]:
    # Stop the bounded screen and preserve telemetry; never emit an invalid IK target.
    raise RuntimeError(result["failure_reason"])
env.set_reference_targets(result["q_ref"], result["v_ref"], result["a_ref"], result["valid"])
env.step(zero_residual_action)
```

`q_ref`, `v_ref` and `a_ref` are NumPy arrays shaped `[1,18]` in the explicitly supplied runtime order. `valid` is bool `[1]`. Position differences and their differences define the executable velocity and acceleration; they are the same knots checked by the residual wrapper. The prototype uses the existing zero motor-velocity-feedforward semantics. Invalid output has `None` targets and latches its reason until a new reset. Nothing writes the robot's root pose or measured joints.

Snapshots are single-replica batches from the physics adapter. Required fields are time; measured root-link position, raw **XYZW** quaternion and matching body-to-world rotation; legacy body-frame linear/gyro measurements; measured joint positions; currently emitted joint targets and executable target velocity; actual named soft joint limits; measured toe reference-point position/velocity; actual contact points and their validity; distal/shaft/coxa/femur/base contact flags; and optional terminal flags. Time must match the generator's current time exactly at 50 Hz. The next emitted knot is at `time + 0.02`. The legacy body linear velocity is the SDK COM tracking signal; it is never combined with link-origin pose to reconstruct a foot velocity. Measured toe velocities come from the physics adapter's consistent link transform/velocity calculation.

Reset requires at least five foot-only contacts and a stationary emitted joint target. It preserves that target **exactly**, including the PD preload that supports body weight. It does not replace it with the deflected measured joint position. Initial virtual desired pose equals measured pose; virtual reference anchors are `R_measured * FK(q_emitted) + position_measured`. Actual measured toe/contact anchors remain separate. Their difference and the initial named `q_emitted - q_measured` offset are logged. A 1 mm measured-FK agreement check detects wrong toe points or names; declared 0.15 rad joint and 25 mm point-offset bounds reject an excessive initial mismatch.

The reference uses the intersection of actual named soft limits and the frozen geometry's 95% soft limits, reserving an additional 0.02 rad for the finite residual. It checks 1.75 rad/s reference velocity and 6 rad/s² reference acceleration before emitting a target. The separate residual controller reserves 0.25 rad/s and 2 rad/s², so total targets remain within 2 rad/s and 8 rad/s² under its independent checks. These are experiment parameters, not measured RS05 motor-speed limits.

## Wave/contact transitions

The explicit order is `lf, rr, lm, rf, lr, rm`. A swing lasts 2 seconds, followed by at least 0.30 seconds of confirmed-contact hold; one full cycle takes about 14 seconds. The world-frame quintic/Hermite swing has matched endpoint position, velocity and acceleration and a 5 mm C2 lift bump. Proposed landing placement predicts the filtered desired body motion to the midpoint of the next stance. Active swings keep their endpoints when commands change. This first prototype holds each leg's initial virtual toe-plane height; it is flat-ground only.

Only one leg may swing. The other five must have measured distal contacts and finite contact points, with the measured serial-model COM at least 25 mm inside their projected polygon. Real force distribution is not inferred from that geometry. Actual planted-toe drift over 20 mm and desired-versus-measured body position error over 35 mm reject the reference.

The swinging leg must exhibit **two consecutive** no-contact samples; separated contact glitches do not count as flight. Touchdown requires three contact samples at or after the planned end, measured toe speed below 0.04 m/s, and position within 12 mm of the planned endpoint after accounting for the previously recorded preload offset. Unexpected early touchdown rejects the test. If planned swing time expires without a contact, the admitted body target goes to zero explicitly; missing touchdown after 0.60 seconds rejects the test. A timed trajectory end alone never counts as a step.

At accepted touchdown, the virtual anchor remains exactly at the C2 trajectory endpoint. The actual measured toe anchor and new reference-minus-measured offset are latched separately. This preserves joint-target continuity without pretending that a deflected physical foot equals its target. There is no instantaneous anchor correction or hidden body-pose adjustment. Larger discrepancies need a separately designed bounded reconciliation procedure and currently fail closed.

All requested, admitted target, filtered admitted and actual measured twists are returned. The default command governor preserves twist ratios and explicitly exposes any derating; the first forward request of 0.005 m/s has factor 1. The current CPU envelope is 0.005 m/s translation and 0.015 rad/s yaw, with constant arcs tested at yaw ±0.01. That envelope is kinematic only. Abrupt transitions, untested combinations and higher speeds may still fail.

## Stop and observable state

A requested zero immediately suppresses new liftoffs. Any existing swing finishes and must receive a measured touchdown. The critically damped desired-motion filter then approaches zero. Mode is `stopping_reference_motion` until no leg is swinging, command magnitude is at most 1e-6 and command-rate magnitude at most 2e-5 in the declared component units. The terminal knot then sets command and rate exactly to zero; joint P/V/A bounds still apply. `reference_quiet_hold` describes only the reference, not a claim that the robot is quiet.

`stop_requested_time_s` and `reference_quiet_time_s` expose finite stop latency. It is 5.54 seconds for the forward fixture and 5.34–6.12 seconds across moving CPU cases. The physics adapter therefore uses a longer stop window so at least ten seconds of measured quiet behavior can be graded after finite reference stop and settling. Zero body command never disables residual balancing feedback or PD control; zero residual is imposed only for the first reference-only physical proof.

Output state exposes gait order/index, mode/current leg, consecutive-flight and contact-confirmation counters, complete swing coefficients and timing, reference/measured anchors and preload offsets, neutral toe coordinates, desired pose/original rotation, command-filter velocity/rate, joint limits, dwell timer, counts and stop times. The future policy must observe the relevant generator state in addition to the residual controller state. The current physics wrapper's 675/678 schema is not yet a complete learned-actor contract; no old checkpoint is accepted.

## CPU evidence and next gate

Nine tests exercise exact initial targets/preload and reversed runtime order; a full forward wave plus finite stopping; missing touchdown; motion without observed flight; separated contact glitches; measured support/pose failures; actual tracking failure; hard rejection of over-budget references; and explicit arc-preserving derating.

The 23-case study uses four seconds standing, 24 seconds commanded motion, then 12 seconds stop. Its measured body follows the desired motion ideally and its contacts are synthesized from toe height. It covers stand, 16 translation bearings, both pure yaw directions and four constant arcs. All 23 pass; maximum reference velocity is 1.092 rad/s and acceleration 2.023 rad/s². The smallest support-geometry margin is 88.0 mm. Forward reaches 11 confirmed synthetic touchdowns with no speed derating and no new liftoffs after stop. **None of these contacts or motion measurements came from Isaac physics.**

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tmp/omni_reference_wave_001 -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 tmp/omni_reference_wave_001/run_cpu_study.py
```

`cpu_report.json` preserves per-case results and source hashes. The next bounded physical proof must preserve the full C model, actual measured contact transitions, torque cap and first-failure evidence. It must compare actual displacement and integrated velocity against the original command, rather than crediting the virtual desired trajectory. Terrain, perception, toe/pad collision geometry, force balance, dynamic robustness and sustained actual quiet standing remain independent gates.
