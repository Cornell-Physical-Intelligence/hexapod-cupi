# Refined 128/16 comparison: overturn during driven motion

This complete-validation attempt failed at approximately **2026-09-06 23:48 UTC**.
It is not a completed validation, numerical admission, or PPO run. The supervisor
removed its exact container and the GPU process exited.

The run used source `5d476d42546bf4f5c84ca39b8c224c8f68f1454b`, functional
identity `d6fbd6e9b5cf499dd8603613d6977ff226f443fb15f149bbe320fc11e15d5b54`,
the unchanged physical v5 asset, 32 coincident-origin environments, 800 Hz
outer physics, 128 position/16 velocity iterations, Kp30/Kd0.30, and the RS05
5.5 Nm peak envelope. Its [launch](../rsl501_release/refined_launch_001.json)
followed the user's explicit instruction to take full training priority.

All 1,000 standing controls completed. The log records positive responses for
the first 16 individual motor checks. During the **right-middle tibia pushlever**
test, at positive 0.04 rad offset/control index42, all32 environments terminated
with the `upside_down` reason. No truncations or other termination reasons were
reported. The remaining individual and grouped tests did not complete.

| Measurement | Settled standing | Driven prefix through failure |
| --- | ---: | ---: |
| Peak raw demand | 0.680117 Nm | 83.598953 Nm |
| Peak applied torque | 0.680117 Nm | 5.500000 Nm |
| Maximum C-pin separation | 0.000209 mm | 0.060937 mm |
| Minimum loaded feet | 6 | 0 |
| Maximum passive velocity relation residual | 0.010888 rad/s | 1,006.274353 rad/s |
| Maximum relative C-pin velocity | 0.000950 m/s | 30.848927 m/s |

Geometric closure alone stayed below its bound, while motion became unstable.
Increasing position iterations from the earlier64/16 configuration did not
establish a usable numerical recipe. These window maxima do not reconstruct
the causal sequence at the failure; no detailed event trace was captured here.
They support testing a smaller outer timestep before further iteration increases.
Raw demand is the controller's request, not motor-delivered torque.

Coverage is42,288 captured physics substeps per environment:16,000 standing
and26,288 driven. `driven_steps` is1,642 because the failure occurs inside the
next `step()` before its completed-control counter increments. All16 samples
of that failing control were captured and drained before the reset was reported.
This is preserved failure coverage, not the requested full54,400-substep run.

The original [report](report.json), [supervisor](supervisor.json),
[container log](container.log), CPU asset audit and source manifest are unchanged.
[capture.json](capture.json) records SHA-256 values independently read on the
Spark and matched after copying all six files. The source archive remains
remote and was not copied by this retrieval. Supervisor exit status1 rejected
the result even though the container process exited0; the report explicitly
records `pass=false` and `ValueError: Unexpected environment reset`.

The next candidate is a separately named1,600 Hz outer-step recipe with the
same20 ms policy period and motor-target endpoint limiter. No threshold, motor
limit, geometry or historical report is changed to admit this failed run.
