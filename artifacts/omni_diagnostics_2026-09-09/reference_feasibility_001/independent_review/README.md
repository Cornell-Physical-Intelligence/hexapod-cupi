# Independent review: omnidirectional reference contingency

The prototype is useful kinematic evidence, but it is not ready to replace the controller. The next bounded step should establish a rate-feasible, support-aware reference-only crawl and compare its actual torque, slip and quiet stopping before adding a learned residual. Do not spend another long PPO run trying to discover whether this reference is executable.

## What the independent review confirms

The frozen prototype correctly handles the forward/left/yaw convention, arbitrary constant SE(2) twists, named serial-C joint chains, the forward reference special case, and explicit rejection of invalid IK. Its documentation accurately separates kinematics from motor/contact qualification. The 50.265 rad/s URDF speed check is a geometric screen bound, not a loaded-motor or target-slew admission.

The transition figures in the first report—0.1393 rad per 20 ms and 0.0952 m/s predicted stance slip—belong to its **100 mm stance-travel** configuration. They must not be attributed to the separate 60 mm static screen.

I independently ran the same 44-second transition sequence with **60 mm travel, 20 mm lift, 0.65 duty, minimum cadence 0.8 Hz, and command-filter omega 4 rad/s**. [Exact results](cpu_review.json) show:

| Transition | Invalid IK leg-steps | Largest target increment, rad/20 ms | Largest requested joint rate, rad/s | Predicted stance-reference slip, m/s |
|---|---:|---:|---:|---:|
| Forward | 0 | 0.09929 | 5.0333 | 0.03044 |
| Reverse | 0 | 0.10401 | 5.2646 | 0.07231 |
| Left strafe | 0 | 0.11783 | 6.0000 | 0.05206 |
| Left strafe + yaw | 0 | 0.14508 | 7.3186 | 0.02087 |
| Forward + right yaw | 0 | 0.13714 | 6.8442 | 0.09864 |
| Pure yaw | 0 | 0.11036 | 5.4598 | 0.06182 |
| Stop | 0 | 0.08750 | 4.3401 | 0.02862 |

The current 0.03 rad/20 ms processed-target bound permits at most **1.5 rad/s** of position-target motion. This is a software rate budget, not the motor's nameplate speed. Even perfectly continuous foot curves can violate it. Post-hoc clipping would change both stance motion and swing touchdown timing, invalidating the unmodified reference's no-slip argument.

At the neutral pose alone, exact-URDF Jacobians require maximum planted-foot joint rates of 1.7667 rad/s for 0.20 m/s forward, 1.7568 for 0.20 m/s left, 0.9073 for 0.40 rad/s yaw, and 2.5907 for 0.20 m/s forward plus 0.40 rad/s yaw. These are only necessary instantaneous stance conditions; swing return is considerably more demanding.

For the present polynomial, swing `dg/dphase = 1-S'(u)/(1-beta)`. At beta=0.65, the peak return factor is `1.875/0.35-1 = 4.357`: some swing feet move more than four times the stance foot speed even before Jacobian amplification or lift. Shorter excursion raises cadence and does not remove this rate burden. Lowering duty reduces that factor but also reduces double-support overlap; the kinematic sensitivity is not a balance recommendation.

## Important Benchmark 1 contract difference

The historical `ReferenceGaitEnv` calls the parent action processor, then **overwrites `_processed_actions` with its reference plus residual**. Consequently the reference bypasses the parent's processed-position slew limiter. It additionally supplies matched joint-velocity targets. Its saved derivatives peak at **7.6958 rad/s at 1.3 Hz**. The current direct-joint controller instead has a 1.5 rad/s processed-target budget and zero joint-velocity target.

Therefore reproducing the accepted forward reference path does not demonstrate equivalence to today's actuator input contract. The forward video remains the visual quality standard. Keep its files and its qualification failures unchanged; do not claim it proves a new reference with the current limiter will work. Explicitly version any later velocity-feedforward experiment, and derive feedforward from the *final executable target trajectory*, never from a fast reference that was subsequently clipped.

## Bounded next approach

1. **Use coherent global time scaling as a diagnostic floor.** Scale the complete path clock, body twist, cadence, horizontal foot path and lift together. Expose requested, admitted and achieved twist separately. Give the reference a rate budget below 1.5 rad/s so feedback retains room. For illustration, a 1.2 rad/s reference budget divided by the 60/20 mm screen's 14.342 rad/s maximum gives a scale at most 0.0837 before extra sampling/acceleration margin: only about 0.0167 m/s for a 0.20 m/s request. This is an honest low-speed geometry test, not full-speed Stage 2 success. Sensor agent's `continuation_001` owns this experiment and may use a different explicit reserve.
2. **Do not ship global scaling as the navigation governor.** A variable time scale must satisfy acceleration as well as velocity: `q_ddot = alpha² q'' + alpha_dot q'`. It can make emergency stopping very slow. The useful next governor should inspect a short foot/support horizon, admit a reachable body twist and swing timing, and smoothly reduce the admitted twist before rate, joint-margin or support limits are hit. Keep all three twist components coherent when preserving an arc. A limiter reaching saturation should trigger re-planning, not silently produce a different gait.
3. **Fix stance anchors and stop support before residual PPO.** A foot in stance should retain its latched odometry/world anchor, expressed using an explicitly declared body-motion estimate. Re-plan swing endpoints with matched position and velocity. Stop should suppress the next lift-off, safely finish or re-plan feet already in swing, establish support, then hold the reference while feedback remains active. Returning every grounded foot to the neutral pose by shrinking an amplitude causes slip. Do not leave swing feet suspended or teleport joints. If simulator body truth is used for a temporary teacher reference, label that dependency; a deployed controller needs the corresponding estimator and uncertainty handling.
4. **Change the residual's temporal interface, not merely its penalty.** A fresh arbitrary `q_ref + 0.12*action` each 20 ms can recreate the same oscillation. Prefer bounded residual target velocity with an explicit acceleration bound and integrated target state, or low-rate residual trajectory knots with smooth interpolation. Include the executable reference, residual target/velocity, phase/support and command-governor state in the observation contract. Give planned reference and feedback a shared rate budget; allow unused reference budget to remain available during quiet stance. Position and velocity feedforward must be mutually consistent after all constraints. A joint-velocity residual is an initial implementation candidate, not yet a selected architecture.
5. **Only then run a short reference-only full-robot admission.** Check fresh static standing; representative forward, reverse, strafe, diagonal, both yaw signs and a combined arc; then stop/reversal support. Require no reset, unchanged 1.6 N·m applied cap and existing torque/contact gates, low processed-target limiter activity, measured slip, quiet joint velocity and target motion. Report inability to reach the requested speed. Add a new PPO actor only after the reference itself earns that allocation, with new action/observation/checkpoint lineage and explicit state visible to the learner.

This path retains the eventual terrain architecture: the command remains body twist, footholds and clearance can become terrain/contact aware, and the learner supplies bounded corrections instead of inheriting a fixed unseen clock. A fixed tripod is a temporary flat-ground coordination prior, not a terrain gait constraint. Unknown footholds, slip and estimator uncertainty remain explicit inputs and failure conditions.

## Alternative if the reference remains too slow or fragile

The smallest principled direct-PPO architecture comparison is a **stateful, acceleration-bounded joint-target-velocity action**. The actor proposes velocity, a bounded acceleration process changes it smoothly, and the position target integrates that velocity inside the same joint/rate/torque limits. The executable target and target velocity become observation state. This removes the current ability to request alternating distant absolute positions every step; the existing post-hoc action filter probe did not test such a policy trained with its own filter state and action semantics.

This alternative preserves freedom to discover terrain support timing but still requires new training and can drift if its learned velocity does not settle. Quiet-stand and stop gates remain essential. It is a more defensible limited comparison than adding an RNN or simply increasing the old action-rate penalty: observation history audits already passed, reduced observation noise did not remove the loop, and two controlled low-exploration updates barely changed physical standing motion. Those findings do not prove recurrent policies cannot help; they do not identify recurrence as the missing mechanism.

Neither reference nor action architecture is selected by this review. No GPU, Git, production runtime, benchmark or frozen prototype file was changed.

## Review of the separately frozen time-governor continuation

I also reviewed `continuation_001/time_governor.py` and its report. Its constant global clock correctly scales command, phase, foot velocity and joint velocity by alpha, and command acceleration by alpha squared. Independent reruns passed all ten original prototype tests and all three time-governor tests.

The continuation's 65% duty / 60 mm / 10 mm configuration admits only 0.0180 m/s maximum translation and 0.0360 rad/s yaw; its stop-to-near-neutral criterion takes 21.58 seconds. The 50% duty / 5 mm lift sensitivity admits 0.0301 m/s and 0.0602 rad/s, with 13.86 seconds settling. Remaining predicted stance slip is 0.00859 and 0.02001 m/s respectively, substantial relative to those admitted speeds. These results support the rejection of global clock scaling as the finished navigation/stop controller, while retaining it as a useful feasibility diagnostic. The lower-duty case is not recommended for a heavy robot without support/contact admission.

Two integration constraints deserve explicit tests if this code advances:

- `advance_governed` documents a fixed scale but does not store/check it across calls. A runtime wrapper must enforce that contract or include the alpha-rate term in acceleration bounds when changing scale.
- A nominal feedback rate reserve does not constrain an additive position residual. With 0.12 rad position scale, independent Gaussian action standard deviation 0.10 and a 20 ms action interval, sampling alone produces expected position-target increment RMS `0.12*sqrt(2)*0.10/0.02 = 0.849 rad/s`, already above the reserved 0.25 rad/s. The new policy must be trained through an actual stateful residual rate/acceleration process. Post-hoc clipping or setting a smaller standard deviation does not establish that process.

The controller choice remains open. The strongest next design candidate is **contact/stance-aware reference timing with a jointly budgeted residual target velocity**, validated reference-only first. If its usable speed or stop corridor remains inadequate, the direct acceleration-bounded target-velocity PPO comparison is the cleaner alternative. Neither route should resume the current absolute-position actor unchanged.
