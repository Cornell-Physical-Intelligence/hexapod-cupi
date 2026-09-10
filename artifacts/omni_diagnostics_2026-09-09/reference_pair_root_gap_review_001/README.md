# Paired001 root position / reported velocity diagnosis

The original 24 s movement check remains **rejected: 5.579826 mm exceeds 5 mm**. The same endpoints with all 400 Hz samples produce a **6.979295 mm** float64 trapezoid discrepancy. This review establishes neither a native PhysX cause nor a correction. It does not change a gate, controller, source, asset or physics setting.

All 33 terminal raw payloads were independently matched against the recorded remote audit. All 2,400 control endpoints match the 400 Hz position, quaternion, root-link velocity, joint position/rate and torque arrays exactly. All 19,201 physics indices, simulation counters and SDK timestamp increments were checked. The original source's float32 reductions and recorded result are preserved in [report.json](report.json), alongside explicitly labeled float64 diagnostics.

## What the measurements say

The movement interval is **6.00–30.00 s**, control trace indices **299–1499** inclusive, physics indices **2400–12000** inclusive. The older generic observer's 4–28 s window is not used. World forward initially points almost along −Y. Each vector below is **position displacement minus integrated root-link velocity**, in world X/Y/Z millimetres, with the same interval endpoints.

| Fixed interval | 50 Hz X / Y / Z (mm) | 400 Hz X / Y / Z (mm) |
|---|---:|---:|
| Hold, 4–6 s | −0.0106 / −0.0259 / +0.0744 | −0.0110 / −0.0214 / +0.0748 |
| Motion, 6–12 s | −0.8863 / −0.3054 / −0.1735 | +0.2485 / −1.3841 / −0.2240 |
| Motion, 12–18 s | +0.0133 / −2.3929 / +0.1225 | +0.5732 / −1.6712 / −0.0711 |
| Motion, 18–24 s | −0.4784 / −0.3430 / −0.1594 | +0.7196 / −1.7546 / −0.0793 |
| Motion, 24–30 s | −0.4283 / −2.2470 / +0.1730 | +0.6282 / −1.8135 / +0.0068 |
| Whole motion, 6–30 s | −1.7797 / −5.2883 / −0.0375 | +2.1695 / −6.6233 / −0.3676 |
| Stop to reference quiet, 30–35.9 s | −0.0770 / +0.6325 / +0.1087 | +0.1273 / −0.5889 / −0.0282 |

The exact additive decomposition is `error50 = error400 + (integral400 − integral50)`. Sampling contributes **[−3.9492, +1.3350, +0.3302] mm** to the original result. Norms are not additive. The 400 Hz Y discrepancy accumulates with the same sign in all four fixed motion blocks and all eight physics-substep phases. The mean within-substep residual is approximately **[+0.0904, −0.2760, −0.0153] mm/s**. Mode-grouped additive contributions are retained; 1,029 `paired_motion` controls contribute most of the residual, with 160 contact-hold controls contributing as well. This is not evidence for one bad landing instant.

The source reports 504 scored quiet samples as 10.08 s. Their first and last measurement endpoints are **37.94–48.00 s**, spanning 10.06 s; the distinction is retained. Actual maximum planar excursion is only **0.1433 µm**, yet the 400 Hz world Y velocity integral over those endpoints is **+0.4797 mm**. This is a reported-rate discrepancy, not a claim of physical quiet drift. Reference quiet begins 35.90000000000001 s, **5.9 s after stop**, followed by the source's excluded settling interval.

Left/right/Simpson quadrature and bounded ±20 ms lag sensitivities are retained as diagnostics, never as alternative admission results. They do not establish a passing replacement. Float64 recomputation of the original 50 Hz result changes its norm by only about **51 nm**; arithmetic precision cannot explain the rejection.

## Frame and timestamp checks

The copied installed primary code and hashes are in [PRIMARY_SOURCE_RECEIPT.json](PRIMARY_SOURCE_RECEIPT.json). No Isaac application or GPU was launched to read it.

- [PhysX articulation data](installed_primary/physx_articulation_data.py) pulls actor-frame root pose from `get_root_transforms()` and COM world velocity from `get_root_velocities()`, refreshed under the same simulation timestamp. Root-link velocity applies the explicit COM lever-arm conversion. Quaternions are XYZW.
- [Conversion kernels](installed_primary/physx_shared_kernels.py) implement the rigid-body offset transformation. Recorded `v_com − v_link = omega × (p_com − p_link)` agrees within **1.22e−8 m/s**.
- Rotating reported body velocity into world reconstructs COM world velocity within **7.42e−9 m/s**. The navigation channel deliberately uses body-expressed COM velocity; the original displacement gate already uses matched **root-link position and root-link world velocity**.
- The COM-only 400 Hz discrepancy is **6.970579 mm**. Switching the point of reference therefore does not resolve the result.

These checks establish consistent Python frame conventions and exported endpoint/timestamp alignment. They do not instrument native solver integration, prove instantaneous velocity fidelity, or establish that the separately observed joint-rate bias causes this root discrepancy.

## Decision boundary

No physics change is supported as a demonstrated fix by this trace. Preserve the rejection and use the next bounded feedback-residual evaluation to determine whether actual motion and the position/rate gap improve under the **unchanged cold evaluation gates**. A finite-difference interval velocity may remain an additional diagnostic channel; replacing the instantaneous SDK channel or the original gate would be a separate, unproven change. This review proposes no new solver sweep and no further lag/frame search.

The actual paired controller completed 11 pairs / 22 measured landings, retained its physical support and torque limits (400 Hz post-startup peak **1.3970318 N·m**), and passed its final quiet checks. Those useful results coexist with the failed movement-consistency gate; they do not constitute paired gait, PPO, omnidirectional or Stage 2 admission.

## Reproduction and integrity

The full raw run and frozen physical source are referenced, not duplicated. Supply their reconstructed directories explicitly:

```sh
python analyze.py --raw-root /path/to/reference_pair_motion_results_001 \
  --source /path/to/source_pair_motion_001 --output /path/to/fresh_review
python verify.py
```

Analysis requires NumPy and SciPy. `verify.py` uses only the Python standard library. [report.json](report.json) binds all raw hashes and exact scoring/telemetry source hashes; [decomposition_arrays.npz](decomposition_arrays.npz) retains cumulative vectors and all eight phase means. [analysis.log](analysis.log) records the completed local execution. The raw terminal result and frozen upstream bytes remain unchanged.
