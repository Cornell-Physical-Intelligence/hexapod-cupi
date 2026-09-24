# RS05 ratings and simulation assumptions

You use a provisional motor model in the approved direct-drive simulator.
Read catalog ratings as separate operating limits; the 1.6 N·m software cap
does not establish continuous-stall capability or hardware calibration.
The specification review below dates to 4 September 2026.

## Manufacturer specifications and revision differences

| Quantity | Published value | Qualification |
|---|---|---|
| Peak output torque | 5.5 N·m | Peak capability; duration and operating state matter. |
| Rated output torque while rotating | 1.6 N·m at 100 rpm with a 70 × 70 mm aluminum plate; 1.8 N·m at 100 rpm with a 150 × 150 mm plate | Different published cooling conditions, not proof of a hardware revision. |
| Continuous stall torque | 1.2 N·m in the vendor overload table | The 1.6 N·m rotating rating is not a continuous-stall rating. Installed cooling remains to qualify. |
| Rated / allowed supply voltage | 48 V / 15–60 V | The allowed range does not promise the 48 V speed/torque curve at every voltage. |
| No-load speed | 480 rpm ±10% | Approximately 50.27 rad/s; this is not rated loaded operating speed. |
| Rated loaded speed | 100 rpm ±10% on the website | Approximately 10.47 rad/s. |
| Reduction / nominal mass | 7.75:1 / 191 g | Website mass tolerance ±10 g; the CAD override uses the nominal mass. |
| CAN bitrate / encoder | 1 Mbit/s / 14-bit magnetic, two encoders | This does not establish end-to-end command/feedback timing. |

Sources: [RS05 product page](https://robstride.com/products/robStride05) and the manufacturer's [July 13, 2026 specification index](https://github.com/RobStride/Product_Information/blob/main/README.md). Record the relevant hardware/firmware revision and cooling arrangement before adopting a rated value. Catalog specifications describe an actuator; they do not measure the assembled robot's thermal or electrical performance.

The manufacturer's [RS05 English manual, filename 260713](https://github.com/RobStride/Product_Information/blob/main/Product%20Literature/RS05/RS05User%20Manual260713.pdf) specifies the 70 mm plate on PDF pages 10–12 (printed pages 8–10). Its internal revision history still lists initial release on November 25, 2025. The [series specification, filename 260713](https://github.com/RobStride/Product_Information/blob/main/%E7%81%B5%E8%B6%B3%E6%97%B6%E4%BB%A3RS%E7%B3%BB%E5%88%97%E4%BA%A7%E5%93%81%E8%A7%84%E6%A0%BC%E4%BB%8B%E7%BB%8D%28260713%EF%BC%89.pdf) specifies the 150 mm plate on PDF page 27. The README dates the catalog July 13, 2026; filename dates alone do not establish a changed motor.

### Burst and thermal conditions

The rendered overload tables were checked directly: English manual PDF page 12 and series specification PDF page 29. Representative published points are:

| Operating condition | Published torque and duration |
|---|---|
| Stalled | 1.2 N·m rated; 1.6 N·m for 175 s; 5.5 N·m for 1 s |
| Rotating, 70 mm aluminum plate | 1.6 N·m rated; 5.5 N·m for 3 s |
| Rotating, 150 mm aluminum plate | 1.8 N·m rated; 5.5 N·m for 3.7 s |

The manual gives 25°C ambient and a 145°C winding-temperature test constraint. Its 135°C board/default protection thresholds are a different quantity. Ambiguous 180°C and “6nm at100rpm” references require manufacturer clarification. The stall table does not restate cooling. These tables do not establish warm-start, repeated-burst or cooldown behavior for this robot.

The catalog's 48 V torque-speed points include 1.6 N·m at 450 rpm and 5.5 N·m at 70 rpm (PDF page 28). These describe the speed/load performance curve, not continuous thermal permission. A peak torque, no-load speed and cooling-dependent rated point cannot be combined into one unlimited operating corner. The maximum phase current is 11 A peak ±10%; it is not battery-line current. The 15–60 V supply range does not establish a full torque-speed curve at every voltage.

## Current direct-drive implementation

Use [`motor_force`](../locomotion/env.py) and [`EnvConfig`](../locomotion/env_config.py)
for the maintained implementation. At each 400 Hz physics step, the controller
computes `12 * (target - position) - kd * velocity`, using the per-joint `KD`
values in `env_config.py`. It clips that demand against a provisional 48 V
speed curve and the 1.6 N·m cap. The curve reaches zero at 480 rpm. The native
readback checks zero implicit drive and armature; this controller applies effort.

You observe requested and applied torque through
[`force_metrics.py`](../locomotion/force_metrics.py). Demand above 1.6 N·m and
speed-dependent clipping are distinct quantities. The controller has no measured
voltage, current, temperature or burst-recovery state. A 5.5 N·m URDF field
cannot authorize unlimited peak operation. Vendor torque and speed refer to
motor output; do not multiply them by the internal reduction ratio.

The approved mass correction uses 191 g per motor with estimated housing
inertia. It does not identify rotor/reflected inertia. See the
[mass ledger](../robot/hexapod_mkii_updated_v1/MASS_INERTIA.md).

## Hardware work still required

Use the [direct-drive leg stand](LEG_STAND_HARDWARE.md) to measure encoder
sign/zero, gains, friction, latency and effective inertia. Record commanded and
measured joint state with force, bus voltage, current and temperature. Validate
fitted parameters on held-out loads and trajectories. Characterize installed
cooling and burst recovery before giving the controller peak authority.

The program lead confirmed a conductive metal mount as a design assumption. Measure its
thermal response; neither vendor cooling plate establishes this assembly's
rating. Battery voltage and temperature limits remain open. Give a changed
actuator model a new identity and repeat native admission and motion checks.

The [original review](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/3ccfd4a12aa7b95347c0884ac3ca466cad97b8aa/docs/RS05_SPEC_REVIEW.md)
retains the earlier standing interpretation and four-bar `rs05_v2` follow-up.
Those implementations and results retain their historical scope.
