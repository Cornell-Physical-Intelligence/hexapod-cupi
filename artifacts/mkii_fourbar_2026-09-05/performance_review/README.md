# Contact-sensor performance review

Read-only review on 2026-09-05 of the installed Spark Isaac Lab source at
`/home/orionh/IsaacLab`, Git revision
`ffff603eafc6b74264a5261cc0183d6a65390d78`. No simulation was launched, no
functional source was changed, and no runtime speedup was measured for this
review. The current campaign retains its frozen sensor implementation.

The first optimization candidate is reducing repeated `.data` getter calls.
Reducing the 31 physical contact providers to seven requires additional
implementation and parity evidence because the stock resolver does not
preserve this robot's nested body paths. Reducing them to two also relies on a
filtered multi-body case whose SDK documentation and manual example disagree.

## Installed source references

These are exact installed paths and line numbers observed during the review.
They identify the reviewed SDK revision, not a moving online documentation page.

| Reference | Installed path and lines | Relevant behavior |
| --- | --- | --- |
| S1 | `/home/orionh/IsaacLab/source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py`, lines 58–67 | Documents filtered reporting as one sensor body against one or more partners; warns against filtering multiple feet together. |
| S2 | `/home/orionh/IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_cfg.py`, lines 67–86 | Describes filtered force matrices and repeats the single-sensing-body restriction for PhysX. |
| S3 | `/home/orionh/IsaacLab/source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py`, lines 288–337 | Resolves the first parent match, collects descendant body leaf names, reconstructs view paths from that parent plus leaf names, and checks the resolved body count. |
| S4 | `/home/orionh/IsaacLab/source/isaaclab/isaaclab/sim/utils/queries.py`, lines 368–449 | Resolves expressions against the source/first environment and returns destination expressions for the matched prims. |
| S5 | `/home/orionh/IsaacLab/source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py`, lines 126–131 and 374–447 | Every `.data` access invokes buffer updating; the update implementation fetches net forces and, when enabled, filtered forces, poses and contact-point data. |
| S6 | `/home/orionh/IsaacLab/source/isaaclab/isaaclab/sensors/sensor_base.py`, lines 194–213 and 393–402 | Scene updates launch timestamp bookkeeping; outdated-buffer updating calls the implementation without a host-side early return, then launches flag/timestamp bookkeeping. |
| S7 | `/home/orionh/IsaacLab/source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor_data.py`, lines 74–142 | Tensor shapes include body and partner dimensions; `contact_pos_w` is an average contact position, not the individual contact manifold. |
| S8 | `/home/orionh/IsaacLab/source/isaaclab_physx/test/sensors/check_contact_sensor.py`, lines 101–108 | Manual example configures multiple ANYmal feet against one global ground despite the documented restriction. This example was not executed in this review. |
| S9 | `/home/orionh/IsaacLab/source/isaaclab_physx/test/sensors/test_contact_sensor.py`, lines 263–294, 832–856 | Reviewed automated filtering tests use separate sensing bodies; contact-position tests check shape and loaded/unloaded behavior. They do not establish the proposed six-foot nested-body configuration. |
| S10 | `/home/orionh/IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py`, lines 626–650; `/home/orionh/IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene_cfg.py`, line 80 | Scene update iterates registered physical sensors. Lazy sensor updating defaults to true. |
| S11 | `/home/orionh/IsaacLab/source/isaaclab/isaaclab/utils/warp/proxy_array.py`, lines 122–148 | `.torch` caches a zero-copy tensor view; it does not preserve a past sample when the underlying buffer changes. |

## Supported behavior and unresolved cases

Unfiltered contact buffers support multiple sensor bodies, with net-force shape
`[num_envs, num_bodies, 3]`. That capability alone does not make a broad subtree
expression correct for the physical MKII hierarchy.

In S3, the initializer collects descendant leaf names and then constructs
`first_resolved_parent/(name1|name2|...)`. The bodies in this robot have different
nested parents. A static check against the existing
`configs/mkii_fourbar_v3_kinematics.json` body paths produced:

| Candidate common parent | Descendant body names collected | Reconstructed direct-child body paths that exist |
| --- | ---: | ---: |
| `Geometry` | 31 | 1 |
| `Geometry/body` | 31 | 6 |

These ordinary broad patterns would fail the initializer's count check; they
are not a configuration-only route to one provider for all 31 bodies. No stock
single-sensor expression resolving all 31 nested bodies was validated here.
A custom provider/resolver that preserves full relative paths would need a
separate implementation review and live mapping/parity check. Do not flatten or
duplicate physical bodies merely to simplify sensor selection.

For filtered foot reporting, S1 and S2 explicitly require one sensing body per
environment. S8 contradicts that guidance for multiple feet against a shared
global ground. The implementation allocates `[num_envs, num_bodies, partners, 3]`
arrays and does not explicitly reject `num_bodies > 1`, but those facts do not
establish correct partner association or values. Treat a six-foot, one-ground
filtered provider as **unverified**, not as established unsupported behavior or
an established optimization. Keep six individual filtered foot providers until
an isolated comparison proves equivalence on the installed backend.

## Repeated getter work

The current task's
`packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/env.py`, lines 142–146,
reads each foot sensor's `.data` separately for `force_matrix_w`,
`contact_pos_w`, `pos_w` and `quat_w`: 24 getter invocations per
`_contact_state()` call for six feet.

S5 and S6 show that each getter enters the native-fetch/Warp-update path even
when the sensor's environment mask is already current. The mask prevents
unwanted buffer updates; it does not remove the surrounding native API
invocations and kernel launches. Reading `.data` once per foot within that
sampling pass, then accessing its four fields, is therefore the first bounded
candidate. This would reduce those getter invocations from 24 to six without
changing the number of physical providers.

Do not cache a sample across physics steps without explicitly refreshing it at
each step. Zero-copy views alias mutable buffers (S11). Historical comparisons
must clone the necessary sample or finish their reductions before the next
update. A future named-view adapter must register each physical provider only
once with the scene; registering all per-body aliases as sensors would repeat
updates and undermine the intended reduction.

## Static call-count estimate

The following counts assume lazy sensor updating, one read of each provider's
`.data` per physics substep, the current pose/contact-point settings, no debug
visualization and no reset. They exclude additional reward/termination reads,
articulation work, the physical solver, motor calculations and metric kernels.

An unfiltered provider schedules one timestamp kernel, one net-force kernel
and one outdated-flag kernel, plus one native net-force fetch. A filtered foot
provider additionally fetches the force matrix, pose and contact-point buffer,
and schedules pose-splitting and contact-point unpacking kernels (S5–S6).

| Provider arrangement | Sensor Warp launch invocations per substep | Native sensor fetch invocations per substep |
| --- | ---: | ---: |
| Current: 25 unfiltered single-body providers + six filtered foot providers | `25×3 + 6×5 = 105` | `25×1 + 6×4 = 49` |
| Hypothetical: one 25-body unfiltered provider + six unchanged filtered foot providers | `1×3 + 6×5 = 33` | `1×1 + 6×4 = 25` |

The second row requires the unresolved full-path provider work above. It keeps
31 sensed bodies rather than duplicating six feet in an additional 31-body
provider. These are source-derived invocation counts, **not measured GPU work,
latency reductions, throughput, or a predicted end-to-end speedup**. Larger
batched kernels still process all bodies and environments.

## Profiling and parity gate before changing providers

1. Preserve the current source/asset identity and run archive as the baseline.
   After the active campaign, use an isolated diagnostic with the same seed,
   environment count, actions, initial pose, motor parameters and solver recipe.
   Begin with 32 environments, warm up compilation/allocation, and measure a
   fixed short window such as 100 policy steps. Record the precise window.
2. Attribute CPU time, native API calls and CUDA/Warp launches separately to
   physics stepping, sensor updates/fetches, physical metric collection and
   policy work. A profiler trace or narrowly scoped timing wrappers may be used
   in that diagnostic. Synchronize at measurement-window boundaries, not after
   every kernel. Report instrumentation overhead and exclude warmup separately.
3. Compare the local getter-caching candidate first. Require unchanged tensor
   values, named bindings, finite/invalid classifications, reset behavior,
   termination decisions and physical acceptance results. Keep every 1.25 ms
   physics sample and 16 samples per 20 ms policy transition.
4. Only if measured sensor overhead warrants it, evaluate a full-path batched
   unfiltered provider alongside the original providers in a short parity
   diagnostic. Require all 31 unique body names and all environment/body
   indices to resolve exactly; shuffled body order must not change meaning.
   Preserve net forces, the six foot force matrices, poses, contact centroids,
   no-contact NaNs and partial-environment resets. Check transient contacts at
   early substeps, not just the policy endpoint. Do not mix performance numbers
   from this dual-provider diagnostic with the final reduced-provider timing.
5. Treat six-foot filtered batching as a separate experiment because of the
   SDK inconsistency. Check one loaded foot at a time, multiple loaded feet,
   translated environments and the shared ground association. Do not infer
   correct filtering from array shapes alone.
6. Retain the same contact capacity, force thresholds, motor envelope, closure,
   ground-clearance and sampling gates. Preserve source identities and failures;
   requalify any adopted implementation. Scale the final measured comparison to
   the intended training environment count only after parity passes.

This review authorizes no provider replacement and makes no change to the
current campaign's physics, sensor coverage or acceptance criteria.
