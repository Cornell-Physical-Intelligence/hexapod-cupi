# Finite position residual: first physics contract

This002 successor changes only writable storage in the environment wrapper and adds its regression. A review reproduced an outside-inference reset failure after001 stored `tanh`/cast outputs from an inference rollout. The wrapper now allocates ordinary writable action, target and slew-telemetry tensors; controller math and every limit remain byte-identical. Frozen001 is preserved. Twelve tests pass, including execution of an inference rollout followed by a reset outside inference mode.

This isolated prototype replaces the failed velocity-action integrator with a **finite position residual around an executable reference**. It does not load a PPO checkpoint or train an actor. The next admissible experiment is zero-residual standing and a low-speed one-leg-at-a-time crawl in full C-robot contact physics. No GPU run has been performed by this preparation.

Candidate003's measured failure was accumulated target bias: tiny actor velocity outputs moved the targets `.048–.124 rad` over ten seconds without useful locomotion, and longer quiet trials terminated. A constant action in this prototype instead converges to the fixed offset `radius*tanh(action)`. The reference and residual have separate position/rate/acceleration budgets. Invalid reference geometry or discrete rates fail before the next target is emitted; the controller does not hide a bad gait by clipping or slowing its reference.

## Explicit initial profile

```python
options = dict(
    profile='formal_004',
    residual_radius_rad=.02,
    residual_velocity_rad_s=.25,
    residual_acceleration_rad_s2=2.,
    total_acceleration_rad_s2=8.,
)
```

The common formal `.04 rad/20 ms` permits total target velocity `2 rad/s`; the residual reserve leaves the reference `1.75 rad/s` and `6 rad/s²`. The separate diagnostic `.03` profile leaves `1.25 rad/s` and the same `6 rad/s²`. They must have different source/plan identities and separate results. Neither target profile is a measured motor speed limit. The `.02 rad` residual amplitude is an explicit proposed experimental bound, not a learned policy or physical admission.

Every reference joint must stay at least `.02 rad` inside the runtime named joint limits. The residual is bounded to that radius, `.25 rad/s` and `2 rad/s²`; therefore total targets stay inside the combined P/V/A limits. The radius is a position offset, not another target slew limit. A zero residual emits the reference exactly. Existing motor PD gains, torque rating, termination/contact gates, physical asset and noise settings are retained by the environment configuration.

## Executable interface

After the pinned C-study bootstrap and AppLauncher, use:

```python
from reference_residual_env import (
    configure_reference_residual_physics, ReferenceResidualPhysicsEnv,
)

configure_reference_residual_physics(cfg, options, reward_overrides)
env = ReferenceResidualPhysicsEnv(cfg, evaluation=True)
env.omni_diagnostic_enabled = True

# Required before every step, in actual env._robot.joint_names order.
env.set_reference_targets(q_ref, v_ref, a_ref, valid)
observations, reward, terminated, truncated, extras = env.step(zero_residual)
```

`q_ref/v_ref/a_ref` are `[num_envs,18]`; `valid` is an explicit boolean `[num_envs]`. The reference generator must establish geometry, support and measured-contact transition validity. The controller independently checks reference joint margins, control-step velocity and acceleration. Supplied derivative metadata is retained separately from recomputed discrete derivatives. The generator must provide fresh references after every reset and every step. Stop on terminal events; do not continue with stale gait phase or anchors.

`env.reference_residual_target` contains Torch target/reference/residual P/V/A, raw action, finite residual goal, applied float32 position targets and cast error. `env.reference_residual_sample` captures these as immutable CPU arrays in the reward hook before automatic reset. When diagnostics are enabled, the ordinary pre-reset sample includes the new fields, raw Isaac XYZW quaternion and its explicitly derived WXYZ counterpart. The historical absolute-position filtered-target fields are removed. The wrapper sends position targets and **zero joint-velocity feedforward**, retaining current direct-PPO motor semantics. Executable derivative metadata does not silently become a new motor drive.

The controller state is `env.reference_residual_controller`. Its `reference_position` initially equals the exact canonical reset target. A settled measured joint position generally differs from that commanded target under load. Initializing a reference directly to the measured position would erase static PD preload and create a target jump. The contact-aware generator must explicitly reconcile measured FK anchors with the executed targets, record any measured initialization preload, and preserve the first target continuously. The wrapper never writes a desired body pose or teleports measured joints.

## Observation and checkpoint boundary

The physics wrapper has a 135-value frame and five-frame history (`675` policy, `678` critic). It retains the original 63 sensor/command/action values and appends 72 noiseless controller-known values, ordered:

1. Total position target minus nominal position, 18 values.
2. Total target velocity divided by the total limit, 18 values.
3. Residual position divided by its radius, 18 values.
4. Residual velocity divided by its limit, 18 values.

This exposes every persistent state of the target controller. Reference P/V are recoverable by subtraction. **It is not a complete learned actor contract:** the external generator's phase, contact state, admitted command and desired-body state still need a reviewed observation representation. The class rejects `evaluation=False`, provides no PPO runner, and the checkpoint helper rejects every load. No old `315/318` or `495/498` actor is silently transferred. Forward/left/yaw commands remain available through the inherited command interface; the first forward-only smoke does not define a forward-only eventual policy.

## First physical experiment and stop conditions

The root agent owns scheduling and source/asset freezes. The terrain agent prepares the full-C read-only asset audit, fresh 32-replica standing admission, and one-robot low-speed reference smoke. The sensor agent prepares a new measured-contact-aware wave generator. These preserve frozen support002 geometry evidence and candidate003 sources.

Start with five-foot support and only one swing leg. Record requested and admitted twist, actual body twist/displacement, commanded and measured foot contacts, support timing, touchdown mismatch, target P/V/A, requested/applied torque and all terminal events. Abort reference emission on invalid geometry/contact reconciliation, stale data, violated P/V/A/joint margin, or physical failure. Zero-residual lag must be zero before target casting. No relaxation of the `1.6 N·m` rating or quiet/contact gates is authorized. If the robot cannot unload a leg under the cap, record that result before trying another support pattern or speed.

Only an actual useful, stable physical reference could justify a later bounded residual PPO pilot. Quiet hold, all bearings, reverse, both turns, arcs and stops remain required before Stage 2 completion. Terrain/perception remain future extensions; successful CPU geometry is not success on real contact dynamics.

## CPU verification

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tmp/omni_reference_residual_002 -p 'test_*.py' -v
```

Twelve tests pass: constant-bias convergence instead of windup, exact zero-residual reference emission, invalid/stale reference rejection before target emission, separate profiles, named limits/reset consistency, observable-state recovery, 4,000 noisy float32 steps, inference-mode storage, return feedback at a zero residual goal, and execution of the actual environment hooks against explicit CPU SDK doubles. The hook tests cover the 675/678 schema, repeated observation stability, torque-finiteness rejection, raw XYZW capture, no inherited absolute action mapping, and an actual wrapper inference rollout followed by a reset outside inference mode. They do not replace the next Isaac runtime admission.
