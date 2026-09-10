# Single-leg stand: measurements for the simulator

Prepared 4 September 2026. The user confirmed a vertical rail: the hip/body mounting carriage moves up/down and is otherwise constrained, reproducing the leg's body-to-coxa mounting relationship. This specifies instrumentation and experiments, not commands to operate the motors. **The highest-value deliverable is synchronized raw data from commanded motion, actual joint/link motion, foot forces and carriage motion.**

## Exact stand model

Model a fixed world/rail, a passive vertical prismatic joint, a carriage carrying the normal body-side coxa mount, and the existing coxa/femur/tibia chain. Retain all three motorized leg joints with their actual URDF transforms; the rail adds one translational degree of freedom. The carriage cannot roll, pitch, yaw or translate sideways. Represent its measured mass/COM, rail resistance and travel, with the foot contacting an instrumented ground plate. No full-body ballast or fictitious body rotation should be inserted without matching the real fixture.

Place a force sensor **under the foot plate**, and a linear encoder/position sensor along the rail. An inline carriage load cell can additionally measure transmitted load, but does not replace foot-force measurement when identifying contact behavior. Use adjustable, measured carriage ballast to sweep supported load. Account for carriage, fixtures and the leg's own mass before choosing ballast; simply putting one-sixth of the whole robot's mass on top of the leg double-counts part of the load.

This stand can identify vertical support, compliant settling, loaded joint response, compression/push-off and repeat-contact dynamics. It cannot establish free-body attitude stability, six-leg load sharing or lateral traction limits by itself. The bearings carry off-axis forces and moments, which must not be silently treated as zero.

## What to instrument

| Priority | Measurement | Why it matters |
|---|---|---|
| Essential | Every motor's commanded position/velocity/feedforward torque, control mode, gains and limit settings | Reproduce the exact deployed actuator interface, rather than an ideal position controller |
| Essential | Timestamped actual position/velocity, reported torque/current where available, temperature, voltage and fault state | Fit tracking response, delay, saturation, friction and derating; identify missing/stale feedback |
| Essential | Independent force measurement at the foot contact or known load path | Calibrate real load/force against motor estimates and model predictions |
| Essential | Rail/carriage position, carriage mass and rig geometry | Separate fixture dynamics from the leg; compare predicted foot and body motion |
| Essential | Independent joint/link or foot-position reference, initially calibrated video with markers | Check encoder zero/sign, linkage relation, compliance and backlash that motor telemetry may not expose |
| Strongly preferred | Three-axis foot force, or normal force plus a separately measured shear force | Distinguish support load from friction/slip and evaluate off-axis forces |
| Useful | IMU on the moving hip carriage; supply current/voltage logging; temperature near housing and ambient | Capture vibration, electrical loading and warm-up drift |

A calibrated one-axis load cell can start the vertical-load study. A three-axis force measurement is more informative for walking. A six-axis force/torque sensor is useful but is not a prerequisite for the first useful dataset. Select capacity from the stand's calculated maximum load and overload margin; select bandwidth from the dynamics being identified. Do not buy a sensor solely because its advertised maximum sampling rate is high.

Record motor frames at their actual available rate, preserving acquisition/receive timestamps and packet sequence. For force and rail position, target **at least 200 Hz for initial slow response tests**, preferably 500–1,000 Hz if sensor bandwidth and the data-acquisition system support it. These are acquisition design targets, not required motor command rates. Run the intended 50 Hz policy interface as one condition while observing its response faster where possible. Never upsample slow telemetry and call it higher-bandwidth measurement.

Use one monotonic timebase or calibrated clock offsets. Aim for measured synchronization uncertainty well below the delay being estimated; around 1 ms is a useful initial target. Log raw samples and actual timestamps rather than only averaged CSV rows. Host receipt time can include USB/CAN buffering and is not automatically the physical sampling time.

RobStride's [official sample](https://github.com/RobStride/Python_Sample) exposes position, velocity, torque, temperature and fault feedback for supported control modes. Confirm fields/units against the exact RS05 firmware and mode. Treat reported torque as an estimate until compared with an independent force/torque measurement; do not equate it to a load-cell reading.

## Test sequence

### 0. Characterize the fixture

Measure moving and fixed masses, rail direction, attachment transforms, travel, force-sensor placement and any springs/counterweights. With the leg disconnected or otherwise isolated from the rail force measurement, measure carriage resistance in both directions at several slow speeds. Document stiction, hysteresis, preload and any sensor cross-talk. Use masses/forces traceable to a reference to check force zero/scale.

For this **vertical hip carriage**, the force plate under the foot and carriage position measure vertical support/compliance and push-off behavior. The rail supplies lateral reactions and prevents body roll/pitch; these constraints belong in the stand simulation. Measure rail resistance over its useful travel in both directions. Any counterweight or spring also belongs in the fixture model.

### 1. Kinematics, zero/sign, limits and compliance

Move one joint at a time through a mechanically approved range using slow trajectories. Record all three motor states and independent link/foot positions. Repeat in both directions, including reversals near several working poses. Compare measured forward kinematics against the URDF; check motor-to-joint sign, zero, scale, cross-coupling and hard/soft limits.

For the tibia parallelogram, directly measure the motor/lever angle against the knee/link angle. Verify the ideal unit-ratio relation rather than assuming it from the mimic model. If the measured mapping differs, fit a monotonic mapping and the corresponding velocity/torque transformation. Measure loaded and unloaded deflection separately from kinematic calibration; otherwise a flexible leg can look like the wrong linkage geometry.

### 2. Static force map across stance

At a grid of mechanically feasible femur/tibia/coxa poses, apply known loads and record force, joint angle, command effort and temperature. Repeat increasing and decreasing loads. This exposes gravity-model error, torque estimate bias, friction, hysteresis and structural compliance. Include the intended nominal stance and plausible alternatives, not just the CAD zero pose.

For scale, the modeled 8.2608 kg assembly weighs about 81.0 N. Equal vertical sharing would be 13.5 N per leg with six supporting legs, 27.0 N with three, or 40.5 N with two. These are reference cases, **not a prescribed gait or peak test-load recommendation**. Real support is uneven and dynamic; the completed robot/payload will alter the load. Choose the actual test envelope from structural calculations and measured motor limits.

With all required forces known, compare external joint load to `J(q)^T F`, adding link gravity, inertial and fixture contributions as appropriate. A single normal-force channel cannot recover all three joint torques when unmeasured rail/shear reactions are present. Force/angle data is more reliable than deriving load from commanded torque alone.

### 3. Loaded actuator response

At several poses and loads, apply small bounded steps, smooth ramps and progressively faster sinusoidal/swept-frequency trajectories in the intended motor control mode. Begin with low amplitude and bandwidth and remain within the established rig limits. Repeat both motion directions.

Measure command-to-response delay and jitter, rise/settling time, overshoot, steady-state error, velocity/torque saturation, deadband/backlash, direction-dependent friction and gain sensitivity. Change one factor at a time initially. Repeat representative conditions at lower supply voltage and after normal warm-up. This produces the actuator model and realistic randomization ranges for Isaac.

### 4. Foot/contact and loaded cycling

Use the actual foot pad and replaceable representative contact surfaces. Cycle feasible stance and swing-like motion under known loads; vary frequency and contact orientation within the stand's scope. Measure force versus deflection, touchdown transients, contact duration, slip and energy/temperature over repeated cycles. A rigid one-axis fixture cannot identify every tangential contact property, so record which forces are constrained rather than measured.

Use supported compression/loading tests first. Drops/impacts require a separately designed test and instrumentation; they are not necessary to collect the first actuator-identification dataset. Do not infer a continuous-duty rating from a short peak-load success. Motor electronics temperature is not necessarily winding temperature.

### 5. Hold out a validation set

Fit parameters using only part of the data. Replay **different** trajectories, poses, load levels and warm/cold conditions through a simulator containing the same rail fixture. Compare joint trajectories, carriage motion, measured forces, delay and electrical/thermal indicators that the model represents. Publish time histories and residuals, not just one aggregate score.

Only after the fixture model matches held-out tests should the calibrated actuator/contact parameters transfer into the full six-leg simulation. The rig's rail friction and artificial body constraints stay in the fixture model. Full-body balance, terrain perception, body collisions and six-leg load redistribution still require their own tests.

## Files the software team needs

One run directory should include:

- `metadata.json`: CAD/URDF version, motor serial IDs/firmware/mode, gains and limits, calibration IDs, sensor transforms, rail orientation/masses, load, surface, supply settings, ambient/start temperatures and exact trajectory ID.
- Timestamped motor commands and raw feedback, with motor identity, sequence and fault fields.
- Timestamped foot force, rail position and independent motion reference; calibration units and time offsets.
- The exact input trajectory plus synchronized video where useful.
- A short run note identifying slips, clipping, stops, missing data and any hardware change.

From this, Codex can build the stand's simulation, fit motor/linkage/contact parameters, generate comparison plots, quantify uncertainty, and update the full robot's dynamics and evaluation suites. Data collection and physical operation remain with the test team. The broader sim-to-real literature specifically identifies actuator modeling and latency as important transfer factors: [Tan et al., RSS 2018](https://roboticsproceedings.org/rss14/p10.pdf).

**First useful session:** fixture measurement + encoder/linkage check + a few static loads + small loaded response trajectories, all synchronized. This is more actionable than a long endurance video without force and timing data.
