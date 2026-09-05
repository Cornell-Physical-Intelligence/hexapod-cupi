# RobStride RS05 specification and simulation review

Reviewed 4 September 2026 locally / 5 September UTC after the project lead challenged the torque interpretation. **RS05 peak output torque is 5.5 N·m.** The [manufacturer product page](https://robstride.com/products/robStride05) confirms this. The 2.364431 N·m startup demand in the standing run is below that published peak; describing it as exceeding the motor's peak capability would be incorrect.

The verification miss was failing to reconcile manufacturer specifications with the inherited actuator approximation before interpreting the run. The repository already records 5.5 N·m, but its actual applied-torque cap is 1.6 N·m. Passing that capped serial baseline establishes neither accurate burst behavior nor a calibrated RS05 model. The earlier run reports remain unchanged and their interpretation is corrected here.

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

## What the current code actually does

The v2 task inherits the archived `ROBSTRIDE_RS05_CFG` through [the articulation builder](../packages/hexapod_env/hexapod_env/assets/articulation.py) and [v2 task config](../packages/hexapod_env/hexapod_env/tasks/mkii_v2/config.py). The [shared config](../packages/hexapod_env/hexapod_env/asset_cfg.py) and [frozen actuator contract](../packages/hexapod_core/hexapod_core/actuator.py) contain these distinct settings:

| Setting | Actual value / effect | Evidence status |
|---|---|---|
| `effort_limit` | 1.6 N·m applied ceiling | Deliberate scalar cap in this baseline, not the RS05 peak. |
| `saturation_effort` | 5.5 N·m | Intercept in Isaac Lab's assumed DC-motor speed envelope. It does not override the lower applied cap. |
| `effort_limit_sim` | 5.5 N·m | Separate solver effort bound; it does not make the explicit actuator deliver 5.5 N·m. |
| `velocity_limit` | 480 rpm | Used as a no-load endpoint by the DC model, although the frozen contract calls it nominal velocity. |
| `velocity_limit_sim` | 528 rpm | Numerical allowance; not an independent guaranteed motor capability. |
| Gains / armature / joint friction | Kp 30, Kd 0.6, armature 0.0007 kg·m², static/dynamic friction 0.01, viscous 0.002 | Inherited assumptions, not identified on the leg stand. |
| Voltage, phase current, motor/driver temperature, accumulated heat, cooling | No physical state/model in this actuator | Actual derating and burst recovery are unmodeled. |

The installed Isaac Lab 3.0.0 source on Spark was inspected at `/home/orionh/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py`. `IdealPDActuator.compute` retains raw PD demand and separately clips applied effort. `DCMotor._clip_effort` uses, with joint speed v and no-load speed v₀:

```text
upper(v) = min(5.5 × (1 − v/v₀), 1.6)
lower(v) = max(5.5 × (−1 − v/v₀), −1.6)
applied  = clip(raw PD demand, lower(v), upper(v))
```

Consequently, raw computed torque is **not capped at 5.5 N·m**. The inherited contract's prose implying such a raw-demand ceiling is inaccurate; the 5.5 setting shapes clipping, not raw demand. Its claim of approximately one second at stall also requires the actual overload curve and conditions, not a universal timer.

This simple envelope allows 1.6 N·m up to approximately 340.36 rpm in the motoring quadrant before tapering to zero at 480 rpm. That is a derivation from the SDK settings, **not a verified continuous torque-speed map**. A low scalar torque cap alone does not make the whole dynamic model conservative at all speeds, voltages or temperatures.

## Meaning of the standing result

[The hardened standing run](../artifacts/mkii_step2_2026-09-04/standing_003/report.json) measured 0.880025 N·m settled computed peak and 2.364431 N·m startup computed peak, with delivered effort capped at approximately 1.6 N·m. Its `startup_raw_rating_exceeded` flag means above this 1.6 baseline. It is not a peak-capability fault flag. The test proves standing under its stated approximation; changing the cap/model changes the dynamics and requires a new run.

The run does not test delivered 5.5 N·m bursts, a current controller, thermal recovery, the physical four-bar or a real CAN driver. Its 0.880025 N·m settled maximum is below the published 1.2 N·m stall rating, but the 20-second test does not prove thermal endurance or matching cooling. The 1.6 cap must not be described as guaranteed continuous-stall-safe. Existing raw artifacts and historical contracts remain traceable; revised physical behavior belongs in a new actuator/task version.

The URDF itself already gives all 18 active joints an effort limit of 5.5 N·m and a velocity of approximately 50.27 rad/s. Its nominal 191 g motor mass does not identify rotor/reflected inertia. The manufacturer torque/speed values are output-side values: do not multiply them by the internal 7.75 ratio again. The current manifest pins geometry and action mapping but omits actuator parameters; current CPU asset integrity therefore does not certify the motor configuration. Runtime currently emits position targets and has no implemented hardware CAN/current/temperature protection path.

## Required next actuator model and evidence

1. Pin manufacturer document and hardware/firmware revisions. Separate rated torque, peak torque, no-load speed, rated operating point, solver allowances and numerical gains in the new contract; give each value a source or an explicit provisional label.
2. Bind an explicit versioned RS05 actuator configuration to the physical four-bar task. Include its parameters/hash in the simulation/runtime manifest; it must not silently inherit the archived model. Use active motor coordinates at the lever, with passive-joint state retained separately.
3. Model admissible torque against output speed, supply voltage and thermal state. Peak authority needs a bounded burst/recovery model supported by the vendor curves and leg-stand data. Rate penalties, applied clipping, episode fault criteria and runtime limits must share those semantics. Separately measure raw demand, actual clipping (computed versus applied), demand above the applicable continuous envelope, burst exposure and thermal/current budget. The current metric named saturation means raw demand above 1.6; at speed, clipping can occur below that value. Raw PD demand is a diagnostic, not measured delivered motor torque.
4. Identify gains, latency, encoder sign/zero, friction, effective/reflected inertia and saturation on the vertical leg stand. Log synchronized commanded/actual position, velocity, motor current, bus voltage and driver/winding temperature with carriage motion and foot force. Distinguish phase-current units from battery current. Validate on held-out loads and trajectories.
5. Re-run torque-speed, transient/thermal, reset/standing and directional tests on that new model, then screen candidate policies. Full-body load redistribution, battery sag and enclosed-motor cooling still require assembled-robot evidence.

The project lead confirmed that the motor housing will conduct heat directly into a metal mount, with useful heat removal expected. Use this conductive path as a design assumption; its thermal resistance, heat capacity, interface and ambient rejection remain provisional until measured. The mount is not automatically equivalent to either vendor plate. Battery voltage/cell count and allowable joint temperature remain open inputs. These cannot be recovered from a visually accurate URDF. This review corrects the specification interpretation; it does not relabel unmeasured actuator dynamics as 1:1.
