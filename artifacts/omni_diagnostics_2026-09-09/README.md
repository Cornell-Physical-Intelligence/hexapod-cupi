# Omnidirectional PPO motor diagnostics

These short controlled probes diagnose the previous policy; they are not qualification runs.

| Probe | Terminations | Median saturation | Planar error | Yaw error | Positive power |
|---|---:|---:|---:|---:|---:|
| Original controller | 4 | 14.93% | 0.0409 m/s | 0.1067 rad/s | 4.943 W |
| Slew 0.03 rad/20 ms | 0 | 5.74% | 0.0395 m/s | 0.0918 rad/s | 2.301 W |
| External forces each iteration | 4 | 15.26% | 0.0426 m/s | 0.1093 rad/s | 5.267 W |

![Scenario comparison](comparison.png)

![Standing trace](standing_motor_trace.png)

Raw reports and compressed traces retain runtime joint names, checkpoint hashes, per-episode age, all termination causes and requested versus applied torque.

## Completion requires smooth movement and quiet standing

The accepted forward animation and current straight trial are visual references. Every bearing, reverse/strafe/diagonal, both yaw signs, both arc directions, path bends, reversals and stops require review. Passing a median or a forward clip cannot complete Stage 2.

The JSON contains direction-by-direction joint velocity, target increments, pose excursion and applied torque increments. Quiet-stand acceptance bounds are proposed there for a separate 30-second undisturbed test, plus stop-to-stand and disturbance recovery; this 12-second probe cannot establish those gates.
