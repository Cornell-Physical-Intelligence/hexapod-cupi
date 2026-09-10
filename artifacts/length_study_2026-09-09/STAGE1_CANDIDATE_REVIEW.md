# Stage 1: candidates for your smell test

**Latest user decision: proceed with C (72.5 / 126 mm) and train a full-robot walking policy ASAP. Candidate C validation/training is authorized; earlier confirmation holds below are historical. A remains excluded.**


**Stage 2 is held pending your confirmation. No hardware length has been selected.**

**Stance audit update:** The original pose selection missed lower alternatives.
C also passes a refined prescribed-motion check at about 108 mm belly clearance
with 1.387 N·m peak at 0.20 m/s. The tall poses below are examples, not required
postures or a steady-walking ranking. See [the stance audit](stance_audit_all_low_poses/STANCE_AUDIT.md)
for the corrected interpretation and images.

User review: A (58 / 105 mm) is excluded because its femur is too short to fit the motors. B–F remain under consideration. C is the recommended lead for further testing; this recommendation is not user approval to start Stage 2.

Screened 49 original sizes plus 9 boundary/intermediate probes at the same 8.2608 kg mass and 1.6 N·m continuous torque limit. Coxa remained fixed. The full path search evaluated 1934 sampled trajectory configurations across four requested speeds.

![Candidate profiles](/Users/andreboufama/Documents/CUPI/HEXAPOD/artifacts/length_study_2026-09-09/stage1_candidate_profiles.png)

| Candidate | Femur / tibia (mm) | Peak at 0.20 m/s (N·m) | Margin below 1.6 | Belly clearance (mm) | Positive mechanical work (J/m) | Role |
|---|---:|---:|---:|---:|---:|---|
| B · f050_t050 | 72.5 / 105 | 1.34 | 16% | 118 | 29.4 | Compact candidate |
| C · f050_t060 | 72.5 / 126 | 1.38 | 14% | 142 | 24.7 | Balanced candidate |
| D · f050_t080 | 72.5 / 168 | 1.35 | 16% | 183 | 25.4 | More ground clearance |
| E · f050_t100 | 72.5 / 210 | 1.45 | 10% | 226 | 20.9 | Long-tibia energy comparison |
| F · f055_t060 | 79.75 / 126 | 1.47 | 8% | 137 | 29.4 | More femur room; less torque margin |

These figures are inverse-dynamics estimates for prescribed motion with optimized contact-force sharing. They are not measured motor current, battery energy, validated walking speed or a stability guarantee.

C (72.5 / 126 mm) is the recommended lead for the next test. Compared with B, it provides about 23 mm more nominal belly clearance and 16% less estimated positive mechanical work per distance, with nearly identical worst-motor RMS torque. D is the closest competitor: about 41 mm more clearance than C, slightly lower peak torque, and similar work demand. The screen has not measured whether either gives steadier real body motion. E retains the long-tibia comparison; F is a longer-femur packaging fallback to check. A is excluded by the user for motor packaging. Rigid motor, shaft, bearing and linkage fit remains unverified for the remaining candidates. Do not manufacture from these stretched meshes.

![Full torque map](/Users/andreboufama/Documents/CUPI/HEXAPOD/artifacts/length_study_2026-09-09/stage1_torque_map.png)

## What was checked

- All original 49 URDFs and 9 generated probe URDFs retain their exact recorded hashes, transferred mass properties, fixed coxa and original mock joint conventions.
- A common grid of femur/knee postures, six/three/four/five-foot support, 3D force/moment balance, unilateral contact, a conservative friction pyramid (coefficient 0.8), joint range and actual mesh extrema.
- Tripod/ripple/wave schedules; duty factors 0.65/0.80/0.90; 60 and 100 mm stance travel; 20 mm foot lift; 0.05/0.10/0.20/0.30 m/s targets. Each geometry uses the same pose-selection rules.
- Periodic leg linear/angular acceleration, transformed inertia tensors, 0.0007 kg·m² joint armature, 0.01 N·m Coulomb friction and 0.002 N·m·s viscous friction. Candidate speed demand stays within the motor model’s full continuous-torque region.
- At least 10 mm COM-to-support-edge margin, 5 mm nonfoot clearance, and no more than 2 mm point-contact/pad discrepancy. Candidates were rechecked at 256 phase samples after the 64-sample screen.
- Independent tests check inverse kinematics against exact forward kinematics for all six legs, Jacobians against finite differences, gravity torque against potential-energy derivatives, and force/moment/friction constraints. All 304 CPU tests passed.

## What the screen does not establish

The body trajectory is prescribed for inverse dynamics; a controller has not yet demonstrated that motion. Contact force transitions, impacts, slip, disturbance recovery, feedback tracking, inter-leg/self-collision and thermal endurance remain for Isaac and hardware checks. Mock joint stops and the production four-bar transmission also need reconciliation.

Mass and transferred link inertia were held fixed across lengths. This separates a length sensitivity study from a true CAD redesign; final masses, COMs and inertias must be regenerated from rigid parts. Mechanical work omits motor/drive heating and other electrical losses.

No sampled motion passed at 0.30 m/s. The original 145/210 mm mock baseline had a passing sampled case only at 0.05 m/s. These are findings about the tested motion family, not upper bounds on either design’s capability. The actual production CAD URDF is a separate control and is not equivalent to the mock baseline.

The search still benefits from shorter femurs near its lower boundary. The minimum manufacturable femur is unresolved; the boundary probes must not be mistaken for a certified optimum. Different duty factors, longer strides, alternative body motion and additional postures may change the ranking.

## Review decision

Please identify the candidate dimensions that look mechanically plausible and any you want ruled out. Stage 2 controlled walking trials will wait for that confirmation. All GPU training remains held.
