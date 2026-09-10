# Actual009 completed the declared slow reference screen; velocity fidelity remains unresolved

The fresh32×1000 standing run passed the existing physical and all32 quiet checks. The single-robot wave completed2400controls:24s of requested0.005m/s forward motion, followed by a stop with12.46s of measured quiet. Eleven qualified measured flight/landing events cover all six legs. No reset, nonfoot contact, or post-settle requested saturation occurred. This is a bounded zero-residual reference screen, not a learned walking policy, Stage2 pass, terrain admission or production setting adoption.

The original50Hz progress gate exactly replays:115.242225mm actual forward displacement versus111.075433mm integrated forward displacement, with4.372873mm3D disagreement over the24s movement window. It passes the unchanged5mm gate. Independent400Hz integration over those same physical endpoints instead gives5.102579mm link disagreement and5.091215mm COM disagreement. That is a material diagnostic miss relative to5mm, not a replacement verdict silently applied to the original screen. Across the entire44s post-settle motion-and-stop interval, the400Hz link disagreement grows to7.078234mm. The run must not be described as proving velocity/position consistency at400Hz.

The400Hz wave requested torque peaks at1.443101764Nm on `revolute_4` at11.4225s, while the50Hz samples peak around1.25420Nm. Every post-settle400Hz requested sample remains below1.6Nm. Standing peaks at1.483903289Nm after settling. The initial retained actuator sample remains63.703545Nm for standing and63.659622Nm for wave, with applied clamp near1.6Nm; actual early-settling updates peak3.961889Nm and2.883783Nm respectively. The initial retained getter is not proof of a newly applied63Nm impulse. The real sampled startup overloads remain excluded only by the already-declared simulation settling interval; hardware startup is unqualified.

New actual400Hz joint angles confirm the unresolved rate bias. During the16s standing interval,481/576joint-environment pairs differ by more than0.01rad between reported-rate integral and angle delta. The worst pair is env6/`revolute_2_1`: actual angle range0.000721216rad, angle delta−0.000697136rad, but reported-rate integral+0.409699616rad. The interval angle-difference RMS is0.000951096rad/s versus reported0.025609239rad/s. This rules out large unobserved400Hz joint excursions for that pair; it does not identify the native solver cause.

The bias is also present in the origin-near single robot. During24s of motion, `revolute_2_3` changes−0.139693260rad but its reported rate integrates−0.526645672rad. During final quiet, `revolute_2_6` changes−0.000000238rad while its rate integrates+0.065986861rad over12.46s. The unchanged quiet scorer still passes: worst reported joint RMS0.005296966rad/s and50Hz joint range0.000005126rad. Angle-derived interval rates are an independent diagnostic, not substituted actor observations, instantaneous velocity measurements, or favorable quiet metrics.

All8001standing and19201wave sample counters, timestamps and8-substep/control layouts passed exact coverage checks; every eighth joint position, joint velocity, torque, root-link position, root-link quaternion and root-link velocity equals its ordinary pre-reset row. The root-link/COM velocity identity was also checked. No extra physics steps or state changes were made by this review. A matched-origin initial-state experiment is the next proposed discriminator; neither coordinate precision nor a particular native mechanism is established by this evidence.

All32raw payload hashes matched the root's terminal audit. That audit binds926unchangedsource files,550unchangedadmitted assets, exact owned-container absence and pause035 restoration. These are historical terminal checks, not a current GPU-idle claim. The preparation/source and raw results remain unchanged.

Reproduce outside the raw result directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B substep_replay.py --run /path/to/reference009/run --output /fresh/substep_replay
PYTHONDONTWRITEBYTECODE=1 python3 -B extra_review.py --run /path/to/reference009/run --output /fresh/extra_measurements.json
PYTHONDONTWRITEBYTECODE=1 python3 -B coverage_review.py --run /path/to/reference009/run --output /fresh/coverage.json
```

NumPy and Torch are required because the unchanged quiet scorer imports its existing diagnostic dependency; these replay commands do not create an Isaac application or use a GPU.
