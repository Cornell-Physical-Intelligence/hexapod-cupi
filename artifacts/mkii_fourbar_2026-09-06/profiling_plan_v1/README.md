# Physical-substep profiling plan

This is a read-only source analysis and a plan for one future bounded
diagnostic validation. **No timing measurements, runtime changes or GPU jobs
were performed for this artifact.** The reported roughly 0.8 s per 20 ms
simulated control step is a motivation supplied by the current investigation,
not a result independently established here.

The first experiment should measure where that wall time goes while retaining
all sixteen physics updates, all 31 native contact sensors, every physical
metric, and the existing per-control drain/guard. Do not benchmark a faster
version with monitoring removed and present it as training throughput.

## What the inspected code establishes

The captured `DirectRLEnv.step` takes the non-backend-decimation branch:
`_apply_action → scene.write_data_to_sim → sim.step(render=False) → optional
render → scene.update`, repeated sixteen times. The project's `SubstepHook`
calls `PhysicalMetrics.capture` immediately after each original scene update,
before post-control dones/rewards/reset processing. Validation drains afterward;
the training guard instead drains and checks at the sixteenth callback, before
any reset. Those different drain positions must be distinguished in reports.

`scene.write_data_to_sim` invokes articulation actuator processing: gathering
commands/state, `RS05V2Actuator.compute`, its budget model, then Warp target and
telemetry kernels and the native force write. `_apply_action` alone is **not**
the actuator cost. `PhysxManager.step` calls native `simulate` and
`fetch_results`; its inclusive wall time should be separated from those two
calls and their CUDA activity.

`scene.update` advances articulation timestamps and joint acceleration, then
calls each sensor's `update`. With the inspected default
`lazy_sensor_update=True`, sensor `update` launches a timestamp/mask kernel;
acquisition occurs when `.data` is read. `PhysicalMetrics.capture` first reads
lazy articulation pose/velocity properties, then computes closure/clearance and
reads every contact sensor. Consequently an undivided “metrics time” includes
native state/contact acquisition as well as arithmetic.

There is a concrete repeated-acquisition path: the inspected PhysX
`ContactSensor.data` always calls `_update_outdated_buffers`. The base method
always calls `_update_buffers_impl` with its GPU mask. The backend immediately
calls native net-force acquisition, plus filtered force, pose and contact-data
acquisition for feet, **before** executing masked Warp update kernels. A false
mask does not skip those native calls in the inspected Python implementation.

Expected counts for a normal no-reset control step are therefore:

| Call | Code-derived count per control step | Why |
| --- | ---: | --- |
| Physics step / original scene update / metric capture | 16 each | 800 Hz physics, 50 Hz control |
| Sensor `update` timestamp path | 496 | 31 × 16 |
| Contact `.data` and native net-force calls | 595 | 496 metric reads + 50 done reads + 49 reward reads |
| Each foot filtered-force, pose and contact-data acquisition | 144 | 6 × 16 metric reads + 6 × 4 fields × 2 done/reward calls |
| Metric drain | 1 | Sixteen captured rows remain mandatory |

These are **static expectations to confirm with counters**, not observed call
counts or evidence that any call is expensive. Count additional render/debug,
initialization, reset or diagnostic accesses separately. `env._contact_state`
reads `.data` four times per foot, and dones/rewards repeat contact arithmetic.
The metrics method also builds a new device `foot_mask` every physics substep.
Both are concrete optimization candidates, with unknown timing importance.

Finally, `drain` performs `torch.stack(pending).cpu().tolist()` on sixteen rows
of eighteen reductions, then updates Python windows. Its wall time can include
waiting for previously queued GPU operations. It must not all be attributed to
copy bandwidth or Python bookkeeping. Validation also performs host reads of
termination counts and observation/reward finiteness every control step.

## Instrumentation regions

Use fixed preallocated region IDs, `perf_counter_ns` for host durations, and
nested NVTX/`record_function` ranges in a separately hashed diagnostic fixture
or diagnostic source copy. Every wrapper calls the original implementation
exactly once and returns its original result. Bind instance-specific wrappers;
do not alter SDK files on the host. If a fine-grained metric block requires
annotations inside its body, review that diagnostic copy against the exact
original statements. Such a run is diagnostic evidence, not a PPO admission
for altered executable code.

| Region | Exact boundary / contents |
| --- | --- |
| `control_total` | Complete guarded `env.step`, validation drain and host checks; report also `env.step` alone. |
| `target_schedule` | `_apply_action`, including target interpolation and target buffer write. |
| `actuator_total` | `_robot._apply_actuator_model`; nest `RS05V2Actuator.compute` and `RS05V2BudgetModel.step`. Keep command gathering, native force write and staging kernels separately attributable inside `scene.write_data_to_sim`. |
| `physx_total` | `sim.physics_manager.step`; nest native `_physx_sim.simulate` and `fetch_results`. Also count/wrap `wait_for_playing` separately. |
| `render_and_app` | Any actual `sim.render`, visualizer and app-pump calls; count zero when absent instead of assuming headless eliminates them. |
| `scene_update_original` | Original scene update only, excluding the project's callback. Nest articulation data update/joint acceleration and 31 `sensor.update` calls. |
| `metrics_state_acquisition` | First body pose/quaternion, link linear/angular velocity and joint-velocity reads in capture. Nest lazy articulation `get_link_transforms`, velocity/position getters and kinematic-update calls. |
| `contacts_acquisition` | Each sensor's `_update_buffers_impl`, with fixed body-name ID. Nest `get_net_contact_forces`, `get_contact_force_matrix`, body `get_transforms`, `get_contact_data`; separately identify Warp processing kernels. |
| `metrics_closure` | Six pin position/axis comparisons, pin point velocities, twelve passive velocity relations and passive position residual. |
| `metrics_clearance` | The 171 primitive transforms, support-height calculations, ground subtraction and foot/nonfoot reductions. |
| `metrics_reductions` | Force reductions/stack, height and actuator telemetry gathering, finite checks and eighteen-value row construction. Exclude nested acquisition time. |
| `drain_stack`, `drain_transfer`, `drain_python` | Stack the existing sixteen rows; CPU transfer/materialization; Python list/window aggregation. Preserve exact values and aggregation order. |
| `post_control` | Dones, rewards, observations, host validation checks, reporting and reset handling; count repeated contact acquisitions under their real caller. |

Report inclusive and exclusive host time with call counts. Do not add nested
inclusive totals together. Charge hidden SDK queries to acquisition, then
report the caller that triggered them; otherwise metrics and sensor time will
be double-counted.

## Timing method and bounded experiment

1. At startup, record actual runtime type/source paths inside the owned
   container. `ContactSensor` and `Articulation` imports are factories in this
   SDK; instrument the resolved **PhysX** instances. Hash those sources and
   compare them with the records below. Record `lazy_sensor_update`, actual
   sensor update periods/flags, backend/device, solver recipe, scene layout,
   body/joint order, reset state and source/asset/motor identity.
2. Use one 32-environment, 128-control-step standing diagnostic on the same
   selected model/solver/layout. Steps 0–31 warm kernels/caches and are still
   fully monitored. Steps 32–63 provide an uninstrumented steady baseline;
   steps 64–95 are traced; steps 96–127 provide a post-trace baseline. Preserve
   the original no-reset standing behavior and all 2,048 physical captures.
   State/thermal drift means those windows are not perfect A/B repetitions;
   record that limitation. This short run does not replace full admission.
3. Start with low-overhead host ranges/counters and a CUDA/API timeline through
   Nsight Systems if available on the installed ARM stack. Verify supported
   CLI/capture-range options before launch. Capture the marked steady window,
   not import/shader/startup work. Use Torch CPU/CUDA profiling only as a
   fallback and state if native PhysX/Warp activity or streams are missing.
   Disable stack/shape/allocation tracing initially; enable it only for a
   smaller second window if launch/allocation overhead is implicated.
4. **Do not synchronize after each region or sensor.** That changes overlap and
   can manufacture the very bottleneck being measured. CPU timers measure
   submission plus any genuine blocking. Correlate native CUDA/API events and
   NVTX ranges across all observed streams. Torch-stream CUDA events can time
   known Torch work, but cannot alone time PhysX/Warp work on other streams.
   Drain's existing blocking transfer provides the normal per-control
   completion point; only use a documented whole-window device completion
   boundary when needed to close a trace, and report its overhead separately.
5. Preserve physics-substep IDs and per-sensor acquisition counters. Assert
   exactly sixteen captures per control and one complete drain, including
   during tracing. Preserve native row mappings, collision groups, six foot
   filters/contact-point streams, and all original physical guards. Do not
   trigger extra sensor `.data` reads merely to inspect profiler state.
6. Compare baseline/traced/post-trace median, p90, p99, total wall time and
   maximum. Report observed profiler slowdown. If instrumentation overhead or
   thermal/state drift prevents useful attribution, shorten the traced window
   or use lighter tracing; do not silently subtract an assumed overhead.

The guarded launcher retains its normal reservation, workload checks, source
integrity and exact-process cleanup. One future profiling job is enough for
initial attribution. Do not launch it alongside a full qualification/PPO job.
After a measured improvement, repeat the same diagnostic. Only then consider
a separately bounded 64/128/512-environment scaling check, one at a time.
No 512-environment PPO ETA follows from the current 32-environment anecdote.

## What to decide from the trace

- **Native PhysX/fetch dominates:** preserve timestep, constraints and solver
  settings while measuring GPU occupancy and waiting versus useful kernels.
  Review only then whether batch size or scene/collision cost is responsible.
- **Repeated acquisition dominates:** test a versioned cache keyed to the
  actual physics-substep ID, acquired once per existing native sensor and
  invalidated on scene update/reset. Reuse the same timestamp's snapshot for
  done/reward consumers. Never reuse contact data across physics substeps,
  collapse 31 sensor bindings, or add a GPU-to-host mask check on every read.
- **Python/kernel-launch arithmetic dominates:** first precompute invariant
  masks/indices; consider vectorizing six closure loops and fusing pointwise
  reductions only after an exact numerical-equivalence check. Keep every
  sample and its population/precision unchanged, including float64 sums.
- **Drain appears dominant:** distinguish preceding GPU wait, transfer,
  conversion and aggregation before changing it. Keep the existing per-control
  guard completion boundary and fail-before-reset behavior.
- **Actuator dominates:** inspect small-kernel/temporary allocation costs in
  its existing explicit model. Any fusion must preserve per-substep burst
  integration, clipping, thermal assumptions and telemetry, not simplify the
  motor physics to improve throughput.

Deliver the trace plus JSON/CSV containing source hashes, step/sample/call
counts, inclusive/exclusive host timings, observed GPU kernels/streams and
idle intervals, drain wait/transfer split, physical metrics and memory/clock
conditions where available. A GPU utilization percentage alone cannot identify
which region is limiting performance.

## Source evidence inspected

Read-only inspection completed 2026-09-06 approximately 23:46 UTC; local HEAD
was `711bffa264cbb441fbfca1fad6cc6d9e5091068e`. The existing captured
`DirectRLEnv` source is in
[`rsl501_startup_review/installed_sources`](../rsl501_startup_review/installed_sources/lab_isaaclab_isaaclab_envs_direct_rl_env.py.txt).
Additional source text was read from Spark's SDK directory without importing
Isaac Sim or touching its running processes. The future profiler must verify
the container resolves these same bytes rather than assuming the host tree is
authoritative for an arbitrary image.

| Source, relative to repository | SHA-256 |
| --- | --- |
| `isaaclab/validate_mkii_fourbar.py` | `2007a7d84d9c51888a67df8bec7292924b2e57411d9312e6e5607c713617725c` |
| `isaaclab/train_mkii_fourbar.py` | `d12e8ff695a4a8d896484b624c30f85a63ea016d1fe78c3075abaa9b0e50410d` |
| `packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/env.py` | `ead9ec2a829dcebfaef4f940992fd18bda3bb056ff68d79f7a6878aac229ed34` |
| `packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/config.py` | `61e328464c48ff0f00bfa595ec3af42d2ae35e4fb3c32ac98c88177f2d5fedf2` |
| `packages/hexapod_env/hexapod_env/actuators/rs05_v2_runtime.py` | `3456545bf68baebd45317a20933dd08800b7de817e93440868b19db0b490a04c` |
| `packages/hexapod_env/hexapod_env/actuators/rs05_v2_model.py` | `c905c84e4a34688d5a93e0668df6d087a9484a32c803f95f03b897950d5efb15` |
| Captured `lab_isaaclab_isaaclab_envs_direct_rl_env.py.txt` | `f75cb355938e4fc0c07d87d143cd255e4bd7ccb5ddcc5a45812a517de6947bde` |

Paths below are relative to `/home/orionh/IsaacLab/source/` on Spark:

| SDK source | SHA-256 |
| --- | --- |
| `isaaclab/isaaclab/sensors/sensor_base.py` | `618ce99c742ac35c2fcbdbe29649930fb228bf16d460277636cd7d02daf19b18` |
| `isaaclab/isaaclab/scene/interactive_scene.py` | `caf1c0646707d6c5a3c28a81020cab741987d96e74f3e43290f0584e221afe40` |
| `isaaclab/isaaclab/scene/interactive_scene_cfg.py` | `b518a9781d40824933710b6e2a1408a3d1313c3510edff7cc2510f769b37f5e6` |
| `isaaclab/isaaclab/sim/simulation_context.py` | `819536a201b525774b4bc04d6bd9a9e3fc11397961b4881e9083eb4502ee958a` |
| `isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py` | `09507db17049bcadd6a48f5992bc20f79b019356c13f1ba3ef1527b2e4188b2c` |
| `isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py` | `47d96553282b389493e69f691b8dcbfeb98997a5a80b88c2571c60ba8b58cb6d` |
| `isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py` | `43a31992353ad3a500024d0ec908d5b253ada7a0aac9a1cd2261466e10bda4ff` |
| `isaaclab_physx/isaaclab_physx/physics/physx_manager.py` | `79852394dafa48aeb5e9cdb39072dde27c9ab35f6b587a17ce931318ab44fd4a` |
