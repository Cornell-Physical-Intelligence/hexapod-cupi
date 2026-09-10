# Batched wave005: qualified flight, explicit unloading and measured stop

This version ports the exact frozen wave005 scalar reference to batched Torch
state. It preserves the frozen wave004 parent unchanged under `parent/` and the
wave005 scalar oracle unchanged under `oracle/`. It changes no physics, actuator,
reference path, support margin, landing correction limit or quiet gate. There is
no actor, checkpoint loading or PPO admission.

`BatchWave005(joint_names, num_envs, device)` accepts the same complete named
float64 measured snapshot as the parent. Call `reset(snapshot, selected_mask,
fresh_episode_ids)`, then `step(snapshot, requested_forward_left_yaw, dt=.02)`.
All dynamic state remains device tensors; source checks occur at construction,
and numerical/contact/rate checks remain active every step. Invalid rows latch
failure and emit no executable target; selected fresh resets leave other rows
unchanged. The actual-position float32 rounding recurrence remains explicit.

The new `unloading` mode distinguishes force-free micrometre rebounds from
qualified flight. Qualification requires two consecutive off-contact samples
and at least 2 mm measured lift within that run. An unqualified contact return
clears the run and peak, preserving its raw sample/run count, optional return
time, run length and lift. Qualification must occur by the existing swing apex.
After qualification, every returned contact still faces the same apex/descent,
speed, 12 mm endpoint/excursion and provisional landing/support checks. No true
airborne return is ignored. Stop completes the current swing and then reaches
finite reference quiet hold without new liftoffs.

The state includes a new optional-return validity bit, its time/lift, four raw
unloading counters and a ninth mode. `flight_count` now means the current raw
consecutive off-contact run; `flight_seen` is the qualification latch. This is a
new state/schema binding, incompatible with wave004 and its 838/841 observation.
The original vertical and horizontal polynomial clocks, landing P/V/A, anchor
preload, support counters and commanded/desired motion are retained.

## Evidence

Thirteen CPU tests pass. The inherited mixed asynchronous six-row test, partial
resets, named-order/precision cases and provisional landing/contact-loss tests
continue to match the new scalar oracle. Additional tests distinguish repeated
unqualified rebounds, early qualified obstacle return and the finite apex
failure deadline across mixed rows.

The complete actual009 trace replays all 2,200 post-settle controls: 11 confirmed
steps, all six legs, and final reference quiet hold. Maximum differences from
the recorded emitted reference are 2.7e-15 rad in position, 1.4e-13 rad/s in
velocity and 1.3e-11 rad/s² in acceleration. At control1037 the RR rebound remains
unqualified unloading with two raw force-free samples and one return. Old003
insufficient clearance still rejects at the apex; old004 still rejects the
12 mm landing bound; old008's final rebound is preserved as an unqualified run.
The accepted physical result belongs to the separately audited scalar source009
screen. This CPU replay is not a new tensor/GPU physical admission.

`ACTUAL009_REPLAY_REPORT.json` records the complete-prefix comparison.
`source_contract.json` binds source009 and exact raw trace/reference-state hashes.
`STATE_SCHEMA.json` documents all typed fields. Timing is local CPU-only and
includes dynamic validation and full state output; it excludes physics,
observation and PPO, and gives no Spark/GPU throughput estimate.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s tmp/reference_tensor_wave_005_001 -p 'test_*.py'
```
