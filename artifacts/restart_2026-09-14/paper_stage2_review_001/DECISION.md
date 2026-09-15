# Stage 2 on the corrected robot

James explicitly directs continuing until the existing Stage 2 milestone is
satisfied on the selected mass-corrected model, with paper-informed PPO and a
video of its actual Isaac Sim rollout. This is the same behavior/acceptance
target, not permission to relabel historical weights or weaken numerical gates.

The first Fable 5.1 maximum-effort invocation returned proposed tool calls as
text, not a usable review. Those calls were not executed. Its response and
status are preserved. The second invocation returned an actual review, session
`557db31f-d549-428b-b5e9-79bdf46b1d18`, in `REVIEW_002.md`; full model usage is in
`RESPONSE_002.json`.

Root adopts the corrected-robot PPO/AMP architecture as an isolated experiment:
five proprioceptive frames, supervised velocity estimator, privileged critic,
transition discriminator with least-squares loss and input gradient penalty,
and real native rollouts with terminal-state timeout bootstrapping. Robot
geometry, mass, inertia, joint limits, explicit 400 Hz PD, 50 Hz control,
1.6 N·m speed-dependent ceiling and 0.040 rad/control target limit stay bound.

The original zero inspection stance cannot track the requested vertical foot
lift inside the knee's lower limit. A proposed walking neutral of
`[0, -0.30, 0.40]` radians per leg reserves travel. Exact original mesh extrema
determine its initial height. This is a new declared stance requiring native
validation; it does not rewrite any failed zero-stance result. The model-specific
IK prior has 1,260 transitions and a maximum target increment of 0.037510 rad.
It is a kinematic adaptation, not the paper authors' whole-body trajectory
optimization dataset. Yaw stance uses a first-order body-velocity approximation.

Root accepts the review's applied-target observation correction, native replay
to measure actuator/contact feasibility and produce realized AMP references,
and in-episode command changes including stops. Discriminator fit and held-out
behavior must be measured; CPU tests cannot establish either. Network size or
reference diversity changes need their own recorded result and source identity.

Root will first validate this new native environment and stance at one and
32 replicas with the original complete 400 Hz contact/clearance checks and
standing thresholds. The full eight-arm floor/layout matrix is deferred:
the earlier origin/translated failure already has two exact repeats, and
the immediate question is admission of the new walking configuration. If this
configuration fails, an analytic-plane fixture is a declared successor test,
not a change to the robot or acceptance thresholds. The prior mesh failures
remain preserved. No standing success, learner allocation or Stage 2 success
is claimed by this design record.

The review's proposed replay/smoke thresholds are engineering suggestions, not
new project acceptance limits. In particular, its 2% replay saturation screen
does not replace the existing 0.5% Stage 2 threshold. Physical caps, non-foot
contact, achieved motion and saturation will be reported separately. Same-source
native admission, actual learning, held-out evaluation and human visual
acceptance remain separate evidence. The active goal ends only at the existing
Stage 2 requirements, not at a checkpoint save or video recording.
