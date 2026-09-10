# Leg-length decision protocol

**Latest user decision: proceed with C (72.5 / 126 mm) and train a full-robot walking policy ASAP. Candidate C validation/training is authorized; earlier confirmation holds below are historical. A remains excluded.**


Status: investigation in progress. **No hardware length recommendation yet.**

**Latest decision: the all-size PPO screen below is superseded and held.**
The user authorized the staged mechanical / controlled-gait / finalist approach
in [CONTROLLED_GAIT_SCREEN.md](CONTROLLED_GAIT_SCREEN.md), then explicitly asked
to review Stage 1 candidates before Stage 2. The 58-size CPU mechanical screen
is complete. See [STAGE1_CANDIDATE_REVIEW.md](STAGE1_CANDIDATE_REVIEW.md) for the
six candidates and limitations. Wait for explicit candidate confirmation before
controlled walking trials or further training. The original experiment below
is retained as history, not authorization to resume the 49-policy campaign.

The user requested a hardware decision report with images, a serious search for
missed solutions, and exploration beyond the initial bounds where warranted.
This document records the decision process before walking results are available.

## What the first experiment measures

49 femur/tibia combinations at 50, 60, 70, 80, 90, 100 and 110 percent of the
archived mock's 145 mm femur and 210 mm tibia. Coxa stays fixed. Each URDF has
the current CAD robot's anatomically registered masses and COM-centered inertia
tensors and the RS05 actuator model. The study varies mesh length, knee position
and longitudinal COM position; it deliberately holds mass and inertia fixed.

These are **scaled mock meshes**, not edited Onshape production parts. Motor
geometry, joints, fasteners, mounting holes, linkages and material quantities
must be modeled properly before making a manufacturing recommendation.

Every size has reset candidates selected using the same static search. The
candidate search retains at least 0.20 rad of target-motion headroom inside
the mock's soft limits, 5 mm nonfoot clearance, 70 mm root height, and predicted
hold torque no greater than 1.3 N m. Those predictions are not admission tests.
The default CAD asset and curriculum are unchanged.

## Matched first screen

The guarded Spark campaign runs the unchanged full standing gate for each size:
32 environments, 1,000 control steps, finite observations, correct articulation
and contact sensors, no unexpected falls/truncations or nonfoot ground contact,
and post-settling torque saturation fraction no greater than 0.5 percent.
Failed reset poses get the other precomputed candidates; a size with no passing
pose is recorded as rejected rather than trained through the failure.

Admitted sizes receive scratch PPO with 256 environments, 500 updates of 24
steps, seed 57, 400 Hz physics, 50 Hz policy, and the same rewards/network/motor
model. This is 3,072,000 environment transitions per size. Fixed mass and
friction keep this first screen interpretable. Baseline runs first for adapter
verification; the remaining sizes use a reproducibly shuffled order.

Each policy is evaluated with seed 7057 at 0.10, 0.20 and 0.30 m/s anatomical
forward commands, 64 environments and 20 seconds per speed. The first two
seconds are omitted from steady-motion averages, but their falls count.
Results include forward error, sideways drift, yaw rate, body tilt/vertical
motion, nonfoot contact, contact-point foot slip, computed torque duty above
1.6 N m, and positive mechanical joint power. Power is not battery energy;
it excludes motor/drive losses and regenerative details.

The initial shortlist uses the Pareto frontier across tracking error, tilt,
slip, torque duty, and mechanical power, subject to no falls/nonfoot contact
and at most 0.5 percent saturation at every test speed. Each speed must also
have mean absolute forward error <= max(0.03 m/s, 25% of the command); a policy
that stays still cannot qualify just because its power use is low. It does not invent a
weighted desirability score. If no size meets those conditions, report that
and investigate the failure rather than call the least-bad policy optimal.

## Work required after the first screen

1. Check convergence and learning curves before treating training failure as a
   morphology limitation. Retry rejected sizes with a justified stance/controller
   correction under consistent, recorded rules. Keep equal-budget comparisons.
2. Repeat promising candidates and close competitors with at least five training
   seeds. Retain the 100/100 baseline. Use episode-level confidence intervals,
   worst-case outcomes, and compute-normalized comparisons; do not bootstrap
   correlated timesteps as independent trials.
3. Refine femur and tibia around every separated promising region, including
   asymmetric ratios. Use a finer grid and off-grid samples. If an improvement
   points toward a boundary, test beyond the original 50–110 percent range in
   controlled increments; the user explicitly authorized pursuing overlooked
   or out-of-bound solutions. Coxa remains fixed. Enforce actual packaging,
   linkage and workspace feasibility instead of an arbitrary percentage box.
4. Compare unconstrained walking with insect-inspired coordination where useful.
   Visual resemblance alone is not evidence of efficiency, stability or optimality.
   Test whether reward shaping or a narrow action neighborhood caused the odd
   gait before attributing it to length. Include standing, starts/stops, reversing,
   turning and lateral walking; the first forward-only screen is insufficient.
5. Rebuild detailed candidate CAD with rigid motors/holes/bearings/fasteners and
   correct four-bar geometry. Recompute mass, COM and tensors by part, preserving
   the 191 g motor override. Verify joint centers, clearances throughout travel,
   self/inter-leg collision, range of motion, foot workspace and assembly fit.
   Retest with the CAD-consistent mass model, not only fixed transferred tensors.
6. Check numerical robustness with a smaller physics timestep/more solver
   iterations and alternate defensible contact settings. Check mesh/primitive
   collision sensitivity. A simulator exploit is a rejected result.
7. Evaluate finalists across friction, payload, mass/COM uncertainty, actuator
   strength/delay, speed, slopes, uneven ground and disturbances. Check per-motor
   RMS/duty/peak demand and torque-speed limits, rather than fleet averages.
   Establish mechanical loads and thermal margins from available hardware data;
   missing measured stops/thermal data remain explicit limitations.

## Final writeup and images

Produce a self-contained report under this artifact directory with dimensional
drawings and side-by-side CAD views of the recommended candidate(s), baseline,
and close alternatives; actual Isaac screenshots and gait/contact sequences;
the tested parameter map including boundary extensions; convergence/multi-seed
plots; motor-load and stability comparisons; rejected alternatives and reasons;
exact run/source/checkpoint identities; manufacturing changes and tolerances;
and a clear statement of what remains to validate on hardware.

Recommend a robust region/tolerance range when the evidence cannot distinguish
nearby dimensions, rather than imply false millimeter precision. State the
evidence supporting implementation and any remaining blockers. No finite
simulation search can certify a global optimum over every possible design,
terrain, controller and modeling assumption.

## Current execution

Remote campaign: `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/walking_campaign_003`.
Frozen source: `training_source_v3` alongside it. The CPU coordinator waits for
the shared GPU, takes both per-job locks, checks `nvidia-smi` and Docker before
each actual job, and releases locks afterward. It never stops unrelated work.
Its 48-hour bound prevents an unattended infinite queue. `stop.request` in the
campaign directory stops only this campaign. Infrastructure failures stop for
inspection. Source hashes and matching standing admission are checked before
training. No winner is inferred from a merely queued or incomplete run.

Forecasting GPU services are now paused with explicit user authorization;
monitor/verify/publish continue. A48hour remote timer restores the two GPU
schedules, with earlier restoration after hexapod GPU work finishes.
