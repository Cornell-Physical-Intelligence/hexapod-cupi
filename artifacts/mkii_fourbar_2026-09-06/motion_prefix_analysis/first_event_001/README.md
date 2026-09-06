# First contact and motor-clipping sequence

CPU-only analysis of the frozen prefix replay. No physical admission or unique root-cause claim. The complete hashed source/trace identities and exact samples are in `first_event.json`; the 4096-row CSV retains all 838 original fields for all 32 environments over samples `[35984,36112)`.

## Measured order

The first clipping is in environment 24, initially at (−3,−5)m, during the LF lever negative phase that begins at sample 36000 (control 2250). Its LF femur remains commanded to −0.25rad throughout this event.

| Sample | LF pad bottom (µm) | LF foot force norm (N) | Femur pre → post speed (rad/s) | Raw / applied torque (Nm) | C gap (µm) |
|---:|---:|---:|---|---|---:|
| 36048 | 4643.642 | 0.000 | 0.701987 → -0.083608 | -3.741161 / -3.741161 | 0.089 |
| 36076 | 146.803 | 0.000 | -3.178817 → -2.900687 | 1.613779 / 1.613779 | 0.536 |
| 36077 | 2.950 | 125.554 | -2.900687 → 0.068408 | 1.643404 / 1.643404 | 9.519 |
| 36078 | -3.615 | 753.864 | 0.068408 → 18.707371 | 0.732495 / 0.732495 | 8.022 |
| 36079 | 194.442 | 0.000 | 18.707371 → 17.330900 | -4.906051 / -4.705057 | 0.492 |
| 36082 | 1511.348 | 0.000 | 14.704377 → 13.419758 | -5.569253 / -4.984757 | 0.500 |
| 36086 | 4152.595 | 0.000 | 9.815410 → 8.630290 | -5.927520 / -5.326364 | 0.509 |

At 36078, the net LF foot-force×dt is approximately 0.94233N·s. This is a reported aggregate, not an individually resolved contact impulse or an effective-mass estimate. The foot penetrates only about 3.62µm in the reconstructed collision-sphere geometry. The LF passive velocity residuals become−1.72 and+1.35rad/s while position closure stays near 8µm. The femur position change corresponds to 1.2495rad/s interval-average motion, while the reported endpoint velocity is 18.7074rad/s. These are different quantities; a velocity-level solver correction can appear after the position integration.

Raw equals applied torque before clipping. At 36079 the new speed lowers the provisional limit to 4.70506Nm, reducing requested braking from 4.90605Nm by 0.20099Nm (4.10%). Burst headroom is 0.6283, so this first reduction is not budget exhaustion. The force/velocity event precedes clipping; clipping cannot initiate it retroactively.

## Subsequent first impacts

The separately preserved `../secondary_impacts_001/` evidence confirms the same ordering in environments 9 and 16. At 36225, environment 9 has LF foot force 5363.41N and femur speed−2.969→57.043rad/s while raw=applied0.6361Nm. At 36226 its raw braking demand−17.2687 Nm receives zero applied torque because the provisional symmetric envelope reaches zero above 480rpm. Environment 16 likewise jumps−2.338→58.664rad/s at 36421 with applied−0.2731Nm and foot force 2962.72N, followed by zero applied braking against raw−18.4931 Nm at 36422.

The data demonstrate braking authority being reduced or removed after the contact/velocity jump. They support a possible reinforcing feedback mechanism; no counterfactual run establishes the size of any amplification. Both overspeed examples retain roughly 0.59 burst headroom. The symmetric speed envelope is a provisional model, not an identified four-quadrant regenerative-braking characteristic.

## All 32 environments, detailed LF/LM/LR window

19 environments clip at least once and8 exceed the model 480rpm point. The first five clipping environments are24,9,0,16,28. Peaks below are independently located, and do not necessarily occur at the first clipping or at one shared time. Outcomes are not monotonic with distance: (5,5)m stays below 2.65Nm, while (1,3)m reaches 85.21Nm. This batch alone does not isolate numerical placement from every other contact/solver sensitivity.

| Env | Reset XY (m) | Peak raw (Nm) | Peak C gap (µm) | First clipping sample |
|---:|---|---:|---:|---:|
| 0 | [5.0, -5.0] | 84.046204 | 213.821 | 36408 |
| 1 | [5.0, -3.0] | 3.964350 | 39.810 | none |
| 2 | [5.0, -1.0] | 5.073289 | 15.898 | none |
| 3 | [5.0, 1.0] | 5.380150 | 47.179 | 39235 |
| 4 | [5.0, 3.0] | 3.686852 | 20.925 | none |
| 5 | [5.0, 5.0] | 2.640317 | 20.538 | none |
| 6 | [3.0, -5.0] | 43.378185 | 112.059 | 39984 |
| 7 | [3.0, -3.0] | 19.244768 | 183.240 | 37799 |
| 8 | [3.0, -1.0] | 6.806722 | 18.008 | 39973 |
| 9 | [3.0, 1.0] | 84.323227 | 103.669 | 36226 |
| 10 | [3.0, 3.0] | 5.900606 | 133.386 | 39911 |
| 11 | [3.0, 5.0] | 85.033485 | 41.000 | 39445 |
| 12 | [1.0, -5.0] | 2.773508 | 13.564 | none |
| 13 | [1.0, -3.0] | 5.931184 | 11.791 | 37808 |
| 14 | [1.0, -1.0] | 26.193165 | 60.094 | 37760 |
| 15 | [1.0, 1.0] | 7.105961 | 37.044 | 37633 |
| 16 | [1.0, 3.0] | 85.207611 | 180.295 | 36422 |
| 17 | [1.0, 5.0] | 84.297264 | 42.710 | 39258 |
| 18 | [-1.0, -5.0] | 3.758850 | 12.873 | none |
| 19 | [-1.0, -3.0] | 7.276834 | 11.541 | 37812 |
| 20 | [-1.0, -1.0] | 5.057548 | 56.392 | none |
| 21 | [-1.0, 1.0] | 84.847397 | 119.399 | 39499 |
| 22 | [-1.0, 3.0] | 28.283379 | 69.664 | 37779 |
| 23 | [-1.0, 5.0] | 83.816940 | 78.062 | 39616 |
| 24 | [-3.0, -5.0] | 5.927520 | 25.935 | 36079 |
| 25 | [-3.0, -3.0] | 4.284086 | 55.560 | none |
| 26 | [-3.0, -1.0] | 2.859616 | 23.540 | none |
| 27 | [-3.0, 1.0] | 3.585392 | 29.706 | none |
| 28 | [-3.0, 3.0] | 13.543472 | 9.149 | 36446 |
| 29 | [-3.0, 5.0] | 3.671200 | 61.204 | none |
| 30 | [-5.0, -5.0] | 3.416238 | 88.799 | none |
| 31 | [-5.0, -3.0] | 3.251154 | 32.727 | none |

The precise timing within each solved physics step is not observable here. Contact force, velocity correction and constraint response in that step are coincident measurements, so this evidence does not assign an internal solver cause. Native individual contact patch impulses and a controlled counterfactual would be needed for that stronger conclusion.
