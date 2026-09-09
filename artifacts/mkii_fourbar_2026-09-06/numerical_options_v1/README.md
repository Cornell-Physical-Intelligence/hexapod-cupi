# Numerical options after coincident-origin validation

This is a decision note, not an admission report. No runtime, acceptance gate,
asset, or GPU process was changed for this review. Finish the current complete
128/16 validation before selecting a new candidate.

The references below explicitly use **PhysX 5.6.1**, **PhysX 5.8.0**, and the
**ovphysx 0.5.10** stability guide. The actual PhysX binary/build inside the
Spark Isaac Sim installation was **not independently verified in this review**;
these sources explain documented solver behavior and API limits, not an exact
source audit of that binary.

## Observed context

The preserved [64/16 nominal result](../coincident_nominal_failure_001/README.md)
completed 32 environments, 1,000 standing controls and 2,400 driven controls.
It failed only the existing 0.100 mm C-pin separation bound, reaching
0.111171183 mm. All 36 individual/grouped signed motor checks passed; driven
support remained at least three feet; maximum raw and applied torque were both
4.423688889 N m. The maximum passive-coordinate relation error was
0.001514434814 rad. That run used source `ae4f388`; the current corrected
runner source `5d476d4` requires its own qualification identity.

The supervising agent reported the current refined standing-1,000 prefix at
128/16: peak raw torque 0.6801174283 N m, C-pin gap 0.208616 micrometers,
six loaded feet, and mean height 0.13572130259 m. Against the old nominal
standing values, torque differs by approximately 0.04368466 N m and height by
15.565 micrometers, both within the existing comparison bounds. This is
reported prefix context, not an independently retrieved final report or formal
cross-source admission. Driven motion had started; no complete refined result
or physical PPO completion was available when this note was prepared.

Do not reject a prefix just because its refined cumulative torque peak is
still below the nominal tolerance interval: later samples can increase that
peak into the interval. Conversely, an already excessive cumulative peak
cannot decrease. Final paired reports remain the decision evidence.

## Why final velocity iterations do not guarantee positional closure

PhysX documents that contacts and limits are processed after mimic constraints;
the solver can preserve those later constraints at the mimic's expense. Hard
mimics can require many position iterations. This supports incomplete
mimic/contact convergence as a hypothesis for the remaining gap, without
establishing the cause of a particular sample. The current explicit RS05 torque
controller is not PhysX's implicit spring-drive example. [PhysX 5.6.1
articulations and mimic joints](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html#mimic-joints)

The mimic formulation permits discretization and rounding residuals and uses
position-error bias to counter accumulated drift. Increasing correction can
also add energy. The 1.514 mrad maximum relation error over a 77.5 mm bar gives
roughly 117 micrometers, consistent in scale with the measured gap. These are
independent maxima; this is not a reconstruction of the same event. [PhysX
5.6.1 mimic mathematics](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_downloads/9489689e1019677308b6f6cbb82bc5db/mimicJoints.pdf)

TGS position iterations are internal substeps. Final velocity iterations solve
unbiased constraints for the last substep, addressing a different objective
from geometric correction. Increasing their number therefore does not ensure
zero C-pin gap. Internal iterations are also not interchangeable with complete
smaller simulation steps. In this repository, RS05 PD effort is recomputed once
per outer 1.25 ms physics step; extra TGS iterations redistribute that effort
without rerunning the controller. Full contact and articulation calculations
are not all refreshed at each internal iteration. [PhysX 5.8.0 solver and
force-application description](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/Simulation.html#constraint-solver)

## Decision branches after the complete refined result

| Refined outcome | Recommended next action |
| --- | --- |
| Materially smaller C-pin gap, all physical checks pass, and torque/height remain stable | Evaluate a newly named **127/254 position-iteration pair**, retaining 16 velocity iterations, 800 Hz, Kp 30/Kd 0.30, the same motions and every existing bound. Both new runs must pass; proximity of 127 to 128 is not a guarantee. |
| Little closure improvement, worse residual, or renewed impact/torque spikes | Prefer a new **1,600 Hz outer timestep / 32 substeps per 20 ms policy step** candidate over blind iteration escalation. Preserve gains, motor limits, target endpoints, measured physical durations and gates; update all time-dependent contracts together and qualify the new source. |
| Closure passes but torque or height diverges | The paired experiment is not sufficiently converged. Inspect the affected window and contact/velocity evidence before choosing between the two candidates above. Do not promote the refined run by itself. |

PhysX's articulation API permits position counts **1–255** and velocity counts
**0–255**. Promoting 128 to nominal while retaining the current exact doubling
would request invalid 256. A 127/254 pair preserves the doubled-resolution
contract; silently capping 256 to 255 would not. Confirm the installed SDK's
scene settings and actual articulation readback for any new recipe. [PhysX
5.6.1 iteration-count API](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/classPxArticulationReducedCoordinate.html)

The repository's `numerical_recipe()` and qualifier currently bind 64/16 and
128/16, the same complete runtime except position iterations, exact ordered
reset positions, both complete physical passes, and existing torque/height
comparisons. Changing the recipe requires a new numerical identity and release
manifest plus fresh paired evidence. Old failed reports cannot be relabeled.
The current old-nominal torque of 4.423688889 N m yields a diagnostic refined
range of approximately **4.21304–4.65651 N m** under the existing 5%/0.05 N m
rule, separately for raw and delivered peaks; this calculation does not bypass
source or physical-pass checks.

## Controller and physical-model alternatives

Keep **Kd 0.60** as a separately named control experiment, not a numerical
repair. It may reduce pre-impact motion but increases response to velocity
excursions. [Campaign 006](../../mkii_fourbar_2026-09-05/campaign_006_ramped_targets/README.md)
passed nominal 64/1 with 14.32394 N m peak raw demand, while its
[128/1 refined standing phase](../../mkii_fourbar_2026-09-05/campaign_006_ramped_targets/README.refined_stop.md)
failed torque convergence. That history does not establish Kd 0.60 convergence
for today's coincident layout and sixteen final velocity iterations.

The NVIDIA stability guide supports smaller timesteps, sensible gains and
inertias, and physically justified compliance as possible remedies. Adding
unmeasured mimic compliance, artificial passive drives, inflated inertia, or
arbitrary joint-speed/torque restrictions would change this robot model. Do not
use those changes merely to cross the existing acceptance threshold. Actual
compliance, armature and actuator behavior belong to measured hardware
identification and a separately validated physical model. [ovphysx 0.5.10
stability guide](https://nvidia-omniverse.github.io/PhysX/ovphysx/0.5.10/guides/articulation_stability.html)
