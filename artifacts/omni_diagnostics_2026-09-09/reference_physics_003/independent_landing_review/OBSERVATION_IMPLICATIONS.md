# Landing successor: actor contract implications

The reference physics screen has no actor. The provisional 525/528 observation plan is therefore not a checkpoint contract, and this landing change must not silently turn it into one. Do not begin residual PPO until the final generator state and an executable encoder agree.

The draft adds `landing_blend` and `awaiting_landing_support` modes. Preserve the original swing alongside the new landing trajectory: its endpoint and timing are used to validate early contact. A phase label alone does not reconstruct the two trajectories.

In addition to the existing controller, command, contact and anchor fields, the future actor must either observe or deterministically reconstruct:

- Original swing coefficients, start/elapsed time, duration and lift, retained through landing. These are already proposed fields, but must not be overwritten with landing coefficients.
- Landing coefficients, elapsed time and duration. The endpoint follows from the polynomial; if a different endpoint representation drives a decision, expose that as well.
- The latched candidate measured contact point and the landing trigger state. Use the current estimated body frame, with correct translation of position coefficients and rotation-only transformation of higher coefficients.
- Measured flight baseline and peak height, or an equivalent complete representation. Peak-minus-baseline alone cannot reconstruct the subsequent descent-from-peak check. Express heights relative to the same gravity/local-estimator frame rather than leaking absolute global coordinates.
- Descent-confirmed state, consecutive flight/contact/loss counters, and each controller mode. Contact validity, age and confidence remain explicit. A missing sample cannot become a confirmed contact.
- The current preload used by the landing endpoint rule. The existing reference-minus-measured anchor state may reconstruct it before touchdown; demonstrate that reconstruction and retain the distinction between initial preload and any newly admitted landing preload.
- Any future bounded-recovery deadline, abort latch, remaining dwell or command-admission state that changes stop behavior.

The original endpoint consistency error, endpoint correction magnitude and measured lift may be derived rather than duplicated if the encoder proves they match the controller. Known trajectory state receives no synthetic sensor noise. Measured pose/contact/velocity inputs retain declared estimator provenance, latency and validity. If simulator truth supplies them, the lineage remains an instrumented teacher, not a deployable student.

Stopping during landing must preserve its finite phase and contact-confirmation state. A zero twist suppresses new liftoffs, allows the active bounded landing to finish, then waits for physical support confirmation and finite reference quiet. A time limit without contact is a rejection, not quiet completion. The actor must still be able to apply bounded balance corrections during zero movement commands.

Before any learner, add tests for state serialization/reconstruction across the swing-to-landing and landing-to-hold transitions, contact bounce and missing measurements, stopping at every landing stage, same-step observation reads, partial resets, and common-frame translation/yaw invariance. Version the final actor and critic widths after these fields stabilize; old 525/528, 675/678 and previous velocity-policy checkpoints remain incompatible.
