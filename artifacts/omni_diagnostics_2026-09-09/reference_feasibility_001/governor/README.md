# Separate global-clock feasibility continuation

**A limited crawl reference can satisfy the sampled position-rate and point-workspace checks; the current Stage 2 command range cannot be claimed.** The same curve is slowed coherently rather than clipping individual joints. The resulting explicit velocity derating, long stops and remaining stance-slip make this a diagnostic result, not an architecture selection or dynamic admission.

The first prototype remains frozen in `../FREEZE_SHA256.json`. Its 100 mm and 60 mm cases are preserved separately. This directory contains only the subsequent governor, three tests, two parameter choices and their own report/traces.

## Coherent scaling

Let the virtual prototype trajectory be `q(tau)`, with virtual body twist `c(tau)` and phase `phi(tau)`. Choose one scale `a`, fixed over a declared sequence, and let `tau=a*t`:

```text
q_physical(t) = q(a*t)
qdot_physical = a*qdot_virtual
c_admitted(t) = a*c_virtual(a*t)
command_acceleration_physical = a^2*command_rate_virtual
phase_rate_physical = a*phase_rate_virtual
foot_velocity_physical = a*foot_velocity_virtual
```

Thus forward/left velocity, yaw, phase and lift motion all share the same clock. The body follows the same ideal geometric curve over a longer duration if it perfectly tracks the admitted twist. Slowing only phase or silently clipping IK joints would not preserve this relation.

The 0.03 rad/20 ms target limit corresponds to 1.5 rad/s. The prototype allocates 1.25 rad/s to the reference and reserves 0.25 rad/s for feedback, then adds a 10% margin to the **sampled** maximum reference derivative:

```text
a = min(1, 0.90*1.25 / max_sampled_virtual_joint_rate)
```

This is a sampled screening bound, not a proven continuous-workspace certificate. The integrated actor would still need a verified combined target limiter; the reserve alone does not enforce how fast a learned residual moves. `advance_governed` rejects invalid IK rather than silently selecting clipped joint angles. It exposes both `admitted_command_target` and filtered `admitted_command`; these must not be graded as though they were the original request.

## Results

Both parameter choices were checked over 245 commands × 512 phases × six feet, plus a separate continuous reversal/arc/stop sequence and independent physical 20 ms stepping. No individual-joint slew projection was applied.

| Quantity | 65% duty, 60 mm travel, 10 mm lift | 50% duty, 60 mm travel, 5 mm lift |
|---|---:|---:|
| Sampled invalid leg/phase or transition points | 0 | 0 |
| Maximum virtual reference joint rate | 12.497 rad/s | 7.470 rad/s |
| Global clock scale | 0.09002 | 0.15060 |
| Admitted maximum translation, from requested 0.20 m/s | **0.0180 m/s** | **0.0301 m/s** |
| Admitted maximum yaw, from requested 0.40 rad/s | **0.0360 rad/s** | **0.0602 rad/s** |
| Largest joint increment in the physical transition sequence | 0.01125 rad | 0.01084 rad |
| Largest joint rate in that transition sequence | 0.5626 rad/s | 0.5419 rad/s |
| Maximum planar command acceleration in that sequence | 0.00238 m/s² | 0.00667 m/s² |
| Maximum yaw-command acceleration in that sequence | 0.00595 rad/s² | 0.01665 rad/s² |
| Stop to near-neutral/quiet reference | **21.58 s** | **13.86 s** |
| Maximum predicted stance-reference world slip | **0.00859 m/s** | **0.02001 m/s** |

The maximum reference rate over the constant-cycle screen becomes 1.125 rad/s by construction, below the 1.25 rad/s reference allocation. The separate transition sequence used at most 0.10 m/s and 0.30 rad/s **virtual** commands, which explains its lower measured peak. Do not confuse its 0.011 rad steps with the worst constant-cycle case.

The command screen spans 16 bearings at 22.5° spacing, three translation speeds and five yaw values. This is not a proof over every intermediate bearing, transition or disturbance. The 50% duty case has no nominal double-support interval and only 5 mm lift; neither reduced stance duty nor reduced clearance is qualified for this heavy robot. These are sensitivity choices, not recommended gait settings.

## Stopping and command error remain explicit

This governor reduces the target magnitudes by approximately **91.0%** and **84.9%**, respectively. The result cannot satisfy a request for 0.20 m/s or qualify the existing full-speed Stage 2 suite. A future planner must accept a declared feasible twist/timing envelope rather than treating this derating as tracking success.

The stop test begins after a turn and requires all point feet within 0.2 mm of neutral, reference joint rates below 0.01 rad/s and admitted command components below 1e-4. This is reference settling, not measured robot stopping or actual quiet balance. Global time scaling also stretches the filter response. For a steady command followed by zero, the ideal critically damped filter has remaining translational path length bounded by `2*v_virtual/4`; at the screened maximum this is 0.10 m. The corresponding yaw integral is 0.20 rad. Uniform time scaling preserves those geometric integrals while lengthening the clock time. No collision or support corridor has been checked for that stopping path.

The remaining slip is substantial relative to the admitted speeds. The command/filter-dependent trajectory changes its stance anchors during transitions; merely running it slowly does not eliminate that structural problem. A contact-latched stance and endpoint-matched swing/settling design remains a prerequisite before this could become a convincing terrain residual-policy baseline.

## Independent review and limits

The PPO agent's independent [60 mm/20 mm review](../../ppo_repair_003_preparation/reference_review/cpu_review.json) found all its transition IK points valid, but maximum increments of 0.14508 rad/20 ms and predicted slip of 0.09864 m/s. Therefore the original transition problem was not caused only by the 100 mm workspace failures. Its review also found that Benchmark 1 overwrote `_processed_actions` after the inherited slew limiter and supplied reference velocity feedforward: matching that waveform does not imply compatibility with the direct-omni 0.03 rad/20 ms limiter.

No test here evaluates torque, motor heating, support forces, foot/shaft collisions, real body tracking, estimated contacts, sensors or terrain. Slowing a reference reduces rates but cannot remove infeasible static support load. The current direct PPO job is unchanged. Observation/action/checkpoint incompatibilities remain those documented in the first prototype.

## Reproduction

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tmp/omni_reference_prototype/continuation_001 -p 'test_time_governor.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 tmp/omni_reference_prototype/continuation_001/study.py
```

Three tests independently check constant-twist stance consistency under the shared clock, geometric path preservation and explicit command derating, and the rate reserve/invalid-input behavior. `report.json` includes sources and the first-freeze hash; its two compressed traces preserve the physical 20 ms measurements. All files remain in ignored temporary storage; no Git, CAD, GPU or training action was taken.
