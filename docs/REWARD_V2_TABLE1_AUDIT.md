# Reward v2: Table I audit

Issue #18's first bounded task: audit the current reward against
[Liu et al., Table I, p. 3](https://arxiv.org/pdf/2511.03167v1#page=3) for
formulas, units, signs and weights; resolve the printed positive tracking
exponent against the intended decreasing reward; define command-scaled
tracking as a separate adaptation; resolve the stationary-reward criterion.
This document is that audit. It records findings and proposed resolutions. It
does not implement a new reward schema; per the issue, that follows this
audit's review and an assigned owner/reviewer.

Sources: Table I of arXiv:2511.03167v1 (re-read at 400 DPI from the source PDF
to confirm exact printed symbols), and the current reward implementation in
[`locomotion/task.py`](../locomotion/task.py) (`measured_reward`,
`REWARD_VERSION = "canonical_quiet_quadratic_log_tail_v2"`).

## Table I, exactly as printed

```text
Task r^g
  Linear velocity   1    * exp( ||v_t,xy - v_t,xy^des||_2 / 0.15 )
  Angular velocity  0.5  * exp( ||omega_t,z - omega_t,z^des||_2 / 0.15 )

Style r^s
  D Score           1    * max[0, 1 - 0.25*(d_t^score - 1)^2]

Penalty r^l
  Linear velocity (vertical)   -1     * v_t,z^2
  Angular velocity (roll/pitch) -0.08  * ||omega_t,xy||_2
  Joint torque                  -2e-6  * ||tau||_2
  Joint acceleration            -1.5e-7 * ||q_ddot||_2
  Action rate                   -0.01  * ||a_t - a_t-1||_2
  Collisions                    -0.05  * n_collision
  Joint torque limits           -0.05  * ||max(|tau_t| - tau^limit, 0)||_2
  Joint velocity limits         -0.5   * ||max(|q_dot_t| - q_dot^limit, 0)||_2
  Contact force                 -0.1   * ||max(|f_t| - f^limit, 0)||_2
```

`||.||_2` on a vector is the (unsquared) Euclidean norm. On the scalar yaw-rate
error it reduces to absolute value. Style is out of scope for Reward v2 (Step
5 builds the AMP discriminator that produces `d_t^score`); it is listed here
only so the full table is on record.

## Finding 1 — the printed task-tracking exponent has no minus sign

As printed, `exp(||error||/0.15)` *grows* without bound as tracking error
grows. That contradicts "task tracking reward" and every other legged-RL
tracking reward in the literature (Gaussian/exponential kernels that peak at
zero error and decay outward, e.g. Rudin et al.). This is almost certainly a
typesetting error in the paper, not an intended reward.

**Proposed resolution:** implement it as a decreasing reward,
`exp(-||error||/0.15)`, so it equals 1 at zero error and decays toward 0 as
error grows. Declare this substitution explicitly in the reward's declaration
metadata (the way `locomotion/task.py` already declares
`quiet_cost_tail`/`reward_version` adaptations) rather than silently "fixing"
it — this is exactly the ambiguity the issue asks to be resolved and recorded,
not quietly assumed.

## Finding 2 — squared error vs. unsquared L2 norm

The paper's task-tracking terms divide an **unsquared** norm by a fixed scale
(`||error||_2 / 0.15`). The current implementation instead squares the error
and divides by a variance-like parameter:

```python
"linear_tracking": config.linear_tracking_weight * torch.exp(
    -(velocity[:, :2]-commands[:, :2]).square().sum(-1) / config.linear_error_variance)
```

`.square().sum(-1)` is `||error||_2^2`, not `||error||_2`. This is a
*different functional form* from Table I (a squared-exponential/Gaussian
kernel vs. the paper's linear-in-norm exponential kernel), not just a
different constant. Both are legitimate reward shapes independently, but they
are not the same formula, and the difference changes the reward's sensitivity
profile near zero error (Gaussian kernels are flatter near zero; the paper's
form has a sharp corner at zero error since `d/dx exp(-x/0.15)` does not
vanish at `x=0`).

Several penalty rows have the same squared-vs-unsquared discrepancy: `tilt`,
`roll_pitch_rate`, `vertical_velocity` (squared in both — matches), `effort`
vs. `Joint torque` (paper: raw, unsquared, un-normalized `||tau||_2` in N*m;
current: normalized *mean-square* fraction of the 1.6 N*m cap — a materially
different quantity, not just a rescaling).

**Proposed resolution:** Reward v2's task-tracking and penalty terms should
match the paper's literal functional form (unsquared L2 norm divided by its
declared scale, or hinge-then-L2-norm for the limit-violation rows), not
reuse the existing variance-based Gaussian forms. Where a term is kept in its
current (non-Table-I) form for a stated reason, declare it as a named
adaptation rather than presenting it as a reproduction.

## Finding 3 — weight mismatches against Table I

| Term | Table I weight | Current weight | Current field |
| --- | ---: | ---: | --- |
| Yaw tracking | 0.5 | 0.3 | `yaw_tracking_weight` |
| Vertical velocity penalty | 1.0 | 0.05 | `vertical_velocity_weight` |
| Roll/pitch angular-velocity penalty | 0.08 | 0.01 | `angular_xy_weight` |

None of these are declared anywhere as intentional robot-specific
adaptations (7.47 kg vs. the paper's 25.5 kg, different actuator envelope).
They read as pre-existing, independently tuned values from before this repo
adopted the paper as a reference (`REWARD_VERSION` is
`canonical_quiet_quadratic_log_tail_v2`, a name with no reference to Table I
at all). **Reward v2 needs to either match these weights or explicitly declare
and justify the departure** — undeclared mismatches are exactly what this
audit exists to surface.

## Finding 4 — terms in Table I that are missing from the current reward

- **Joint acceleration** (`-1.5e-7 * ||q_ddot||_2`): no equivalent term
  exists. The closest current term, `target_motion`, penalizes joint-*target*
  rate of change (a policy-smoothness proxy), not measured joint
  acceleration.
- **Joint torque limits** (hinge penalty beyond `tau^limit`): the current
  `effort` term is a smooth normalized mean-square torque cost, not a
  hinge-at-the-limit penalty. There is no term that specifically fires only
  when a joint exceeds its torque limit.
- **Joint velocity limits** (hinge penalty beyond `q_dot^limit`): the current
  `quiet_joint_rate` term penalizes joint speed, but only during zero-command
  ("quiet") holds — it is a stillness term, not a general velocity-limit
  penalty active during motion.
- **Contact force** (hinge penalty beyond `f^limit`, on **foot** contact
  force): the current `nonfoot_contact` term penalizes **non-foot** body
  contact force (an explicit safety/collision guard against the frame or legs
  touching the ground). These are opposite in subject (foot vs. non-foot) and
  serve different purposes; neither substitutes for the other.
- **Collisions** (`-0.05 * n_collision`, a count): no direct counterpart. The
  current `nonfoot_contact` term is a combined event-indicator-plus-force
  penalty on a specific contact class, not a generic collision counter.

## Finding 5 — "command-scaled tracking," as the issue names it

The paper's tracking scale (`0.15`) is a fixed constant regardless of
commanded speed. This repository's command bank tops out at 0.05 m/s forward
(`speed_levels_mps = (.025, .05)`), a small fraction of whatever speed range
the 25.5 kg paper robot was commanded at (unstated in the paper text
extracted so far). A fixed 0.15 scale tuned for a faster robot would make
*any* achievable cupi velocity error look tiny in relative terms, flattening
the tracking reward's ability to discriminate near-target performance at
cupi's speeds.

The current implementation already encodes a per-robot, per-axis error scale
(`linear_error_variance = 9e-4`, `yaw_error_variance = 4e-2`) rather than the
paper's one-size-fits-all `0.15`. **Proposed resolution:** keep a per-robot
tuned tracking scale as the declared "command-scaled tracking" adaptation
named in the issue, but derive it from the paper's functional form (Finding
2) rather than the variance-based Gaussian form, and record the chosen scale
value and its justification explicitly rather than inheriting the existing
unlabeled constants.

## Finding 6 — the stationary-reward criterion

Table I has no dedicated "stationary" reward row: a zero-velocity command is
just the ordinary tracking reward with `v^des = 0`, in principle relying on
the tracking and penalty terms alone to produce stillness. This repository
instead has two dedicated quiet-only terms (`quiet_joint_rate`,
`quiet_target_motion`) that activate only when the held command is exactly
zero, with their own scales and a custom log-tail cost shape
(`quiet_cost_tail`) that has no Table-I counterpart at all.

Issue #18's own acceptance criteria require demonstrated "stop settling and
sustained stillness," which is a stronger, more explicit requirement than
Table I's implicit reliance on tracking-toward-zero. **Proposed resolution:**
define the stationary-reward criterion as retaining the existing quiet-only
terms as a declared additive adaptation layered on top of the Table-I-derived
task and penalty core, rather than dropping them to match the paper exactly
— they exist to satisfy a project requirement the paper's reward was never
designed to guarantee.

## Terms with no Table I counterpart to keep regardless

`height` (root-height tracking to the reference stance) and the
`ProximityGuard`/termination-penalty machinery are simulation- and
robot-specific safety/stability scaffolding absent from the paper entirely.
Nothing here suggests removing them; they should simply be declared as
adaptations, not silently presented as part of a "paper-faithful" reward.

## Summary: what Reward v2 should contain

1. Task tracking (linear, yaw) rebuilt to the paper's functional form —
   `exp(-||error||/scale)` — with the sign in Finding 1 resolved and a
   declared, justified tracking scale per Finding 5 (not necessarily 0.15).
2. Penalty terms rebuilt to match Table I's functional forms and weights
   (Findings 2–4) for vertical velocity, roll/pitch rate, joint torque,
   joint acceleration (new), action rate, torque-limit hinge (new),
   velocity-limit hinge (new, distinct from the existing quiet-only term),
   and foot-contact-force hinge (new, distinct from the existing
   non-foot-contact term).
3. A retained, explicitly declared set of cupi-specific adaptations: the
   quiet-stationary terms (Finding 6), the height term, non-foot-contact
   safety term, and termination penalty — none of which come from Table I.
4. No style term yet — that is Step 5, gated on the AMP discriminator.

Every substitution above is proposed, not committed: per the issue, this
audit needs an assigned owner and reviewer, and CPU-check formulas (see
`locomotion/tests/test_reward_v2_table1_formulas.py`) before any change lands
in the live training reward in `locomotion/task.py`.
