# Geometry screening without independent PPO

**Latest user decision: proceed with C (72.5 / 126 mm) and train a full-robot walking policy ASAP. Candidate C validation/training is authorized; earlier confirmation holds below are historical. A remains excluded.**


**Current status: Stage 1 complete; Stage 2 awaits the user's candidate confirmation.**
The user authorized this strategy, then requested a smell test before proceeding
past Stage 1. The CPU screen covered 58 geometries, including nine boundary and
intermediate probes. Review the six candidates, figures, methods and limits in
[STAGE1_CANDIDATE_REVIEW.md](STAGE1_CANDIDATE_REVIEW.md). No controlled dynamic
walking trial or hardware recommendation is implied by the mechanical estimates.

Stance audit correction: the minimum-clearance and static-torque pose pruning
missed lower postures with useful motion headroom. See
[STANCE_AUDIT.md](stance_audit_all_low_poses/STANCE_AUDIT.md). C has a refined
108 mm clearance / 1.387 N·m prescribed-motion candidate at 0.20 m/s. Before
treating the original shortlist as a full ranking, compare body-height bands
and stance widths explicitly across the remaining geometries; do not prune
only by static torque. The user has not confirmed Stage 2.

Recommendation on 2026-09-09: keep the 49-policy sweep held. Screen the full
femur/tibia grid with a common parameterized walking controller, tune its small
parameter set fairly for each geometry, and reserve full learned policies for
the promising regions and controls. The stages below describe the investigation;
only Stage 1 has been completed. Coxa remains fixed.

## Why change the first screen

The baseline completed 500 PPO updates but its recorded forward speed was
0.0313 m/s against a 0.20 m/s command. This confounds morphology with exploration,
stance selection, rewards and learning convergence. Its 64-environment
evaluation completed only the 0.10 m/s condition, then failed at the next reset
with an inference-tensor mutation error. Do not rank the partial evaluation.
The local evaluator now wraps the entire evaluation, including resets, in
torch.inference_mode; that correction still needs an Isaac end-to-end rerun.

## Stage 1: mechanical feasibility over the complete design space

Compute foot workspace, joint range/headroom, clearance and support margins
over whole candidate stance trajectories. Solve contact-force balance with
friction and unilateral contact constraints for three-, four- and five-leg
support sets, including gravity of the articulated links. Estimate joint loads
with Jacobians and inverse dynamics, not equal-load assumptions as a rejection
test. At the current 8.26 kg weight, equal load sharing would be approximately
27 N per supporting leg for three contacts versus 13.5 N for six; this is an
illustration, not a substitute for force/moment equilibrium.

Only reject physically infeasible cases on established constraints. Failure
of one trajectory, stance or local optimizer is not proof of an impossible
geometry. Retain uncertain cases for alternate poses and gait families.

## Stage 2: short, controlled dynamic trials in Isaac

Use inverse kinematics to turn smooth foot trajectories into joint targets.
The body remains floating and motion must result from torque-limited motors
and contact physics. No prescribed base velocity, kinematic foot pinning or
unlimited position drives. Include body attitude/height feedback and contact
feedback as needed. Check controller correctness first on the baseline plus
two contrasting feasible geometries before batching the grid.

Compare tripod, ripple and wave schedules. Search the same bounded parameters
and give the same optimization budget to every geometry: stance width, body
height, stride length, swing clearance, timing and duty factor. A single fixed
set of joint angles or one gait tuned only to the baseline is not a fair test.
Use the same absolute target speeds, payload, ground-clearance requirements,
terrain and actuator model. Begin with the existing 0.10/0.20/0.30 m/s points;
report feasibility as a speed/clearance envelope when a common target is not
achievable. Benchmark this pilot before promising a sweep runtime.

Rank only trials that actually travel at the requested speed and remain within
stability/contact/motor constraints. Compare energy per achieved distance,
per-motor RMS and peak torque, time at saturation, torque-speed margin, body
roll/pitch/heave, drift, slip, support margin and disturbance recovery. Mechanical
work is only an energy proxy: static motor loading may have zero mechanical
power while consuming electricity and producing heat. Keep motor loss/thermal
proxies separate until a measured electrical model is available.

## Stage 3: refine and verify

Retain roughly 4–6 promising, distinct candidates plus the original mock and
the actual production CAD reference. Keep multiple tradeoff regions and some
near-miss candidates to audit controller bias. Refine between grid points and
extend a boundary when trends justify it. A hard percentage boundary must not
hide a credible solution; actual packaging, workspace and actuator limits
constrain the search.

Verify finalists with accurate CAD geometry, per-part masses/COM/inertias,
rigid motor/mount dimensions and the real linkage transmission. The existing
fixed-inertia mock study remains a sensitivity experiment. Then compare
stronger controllers or learned residual corrections, repeated seeds and
robustness tests. A well-performing classical controller may already meet the
walking objective; independent scratch PPO is not a requirement for every
geometry. Findings remain conditional on the tested gait/controller families.

## Supporting primary research

- Ha et al., Joint Optimization of Robot Design and Motion Parameters:
  https://roboticsproceedings.org/rss13/p03.pdf
- Brodoline et al., Shaping the energy curves of a servomotor-based hexapod
  robot: https://www.nature.com/articles/s41598-024-62184-y

These support optimizing design with motion parameters and distinguishing
electrical losses from mechanical power. Their measured outcomes are not
performance predictions for this robot.
