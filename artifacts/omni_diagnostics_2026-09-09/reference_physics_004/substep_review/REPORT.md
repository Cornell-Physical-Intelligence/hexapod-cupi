# Actual 004 velocity/position measurement diagnosis

The observed discrepancy is real in the recorded readbacks and persists at400 Hz. It is not explained by sampling only the eighth substep, confusing COM and link frames, or shifting velocity by one or two physics steps. This does **not** identify every native solver contribution or justify replacing the existing5mm gate. Actual 004 remains rejected for its separate12mm landing-correction bound; its complete walking/stop gates were never reached.

All ordinary control samples exactly equal the corresponding eighth physics sample. The standing trace is byte-identical to002/003 (`cff36fb1d78e066c7d3f5ec2d4b922ea2a7f6206c2640c2b78db757279f11d8e`), providing direct evidence that the new observer did not change this entire physical trajectory at control boundaries. Every settled consecutive velocity sample changes; simulation counters advance by one and SDK timestamps by2.5ms. This excludes a simple frozen buffer and inconsistent sample spacing. It does not independently read the proprietary native tensor backend twice or prove that no native implementation bug exists.

Independent replay computes integrals in float64 from the raw float32 samples, so final low-order digits differ slightly from the runtime diagnostic's float32 reduction. All32 standing environments are retained, and the full wave prefix is retained. Standing control boundaries200–1000 cover16 s; wave boundaries200–625 cover8.5 s, beginning at the actual pre-motion pose. No reset or truncation occurred.

| Interval | Actual forward pose change | Integrated400 Hz link velocity (trapezoid) | 3D discrepancy400 Hz | 3D discrepancy50 Hz |
|---|---:|---:|---:|---:|
| Standing, worst environment 3 | +0.003338mm | −9.680739mm | 9.705556mm | 9.862308mm |
| Wave, environment 0 | +37.311430mm | +27.398624mm | 9.948908mm | 10.552729mm |

Left/right400 Hz quadrature and ±1/2-substep shifts leave essentially the same discrepancy. COM gives9.684292mm standing and9.922118mm wave. The directly recorded rigid-body identity `v_link = v_COM + omega × (p_link − p_COM)` holds within1.49e−7m/s over all standing samples and3.73e−9m/s over the wave, including startup. The measured frame transform is therefore not a plausible source of the approximately10mm error.

The400 Hz settled maximum computed torque is1.46308756Nm standing and1.14622474Nm wave, compared with control-sampled1.46297991/1.14518774Nm. No settled substep exceeds1.6Nm. This is a stronger temporal observation for this bounded run; it is not a completed walking qualification.

Startup is explicitly **not hardware qualified**. The all-recorded standing peak is **63.7035446 Nm** at initial sample0 (time0, control−1), environment14, `revolute_2_6` (LR tibia), with applied torque−1.600000024 Nm. The wave initial sample similarly reads63.6596222 Nm at environment 0, `revolute_2_3` (RF tibia). These are the initial retained actuator-telemetry buffers before the first observed physics update; the getter does not recompute demand. Their earlier initialization history was not recorded, so they cannot be presented as an observed post-reset physics impulse or silently discarded. The first eight actual substeps report only about8.94e−6 Nm demand.

There is also real measured early settling saturation: the standing peak after an actual physics update is3.9835351 Nm at40ms, environment29, `revolute_1_5` (LM femur); the wave peak is3.0228059 Nm at40ms, environment 0, `revolute_1_4` (RF femur). Applied torque is clipped at1.600000024 Nm. The measured excess samples are32.5–62.5ms standing and32.5–47.5ms wave. These lie within the original4-second settling exclusion and do not change its declared result. They remain a separate startup concern, not evidence of an all-time1.6 Nm requested-torque limit. `startup_torque_review.json` preserves exact indices and first-sample values; `review_startup.py` reproduces it.

The actual resolved environment uses TGS,16position/4velocity iterations, stabilization disabled and `enable_external_forces_every_iteration=False`. The installed IsaacLab manager emits the matching noisy-velocity warning, while its configuration documentation recommends considering the option when velocity updates are noisy. `installed_solver_readback.json` preserves exact installed paths, source hashes and relevant definitions; preparation separately binds the concrete PhysX data getters and link/COM kernel.

NVIDIA states that constrained-body pose increments need not match returned velocity: geometric bias affects pose integration, while returned/carried velocity is produced by the velocity solve. It also describes TGS internal position-iteration substeps, which are distinct from the eight observed external physics steps. This is a documented reason why increasing external sampling does not restore a strict identity. [PhysX5.6.1 solver documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/RigidBodyDynamics.html#solver-iterations)

NVIDIA's newer steady-state discussion identifies nonzero reported joint velocity with once-per-frame external forces and describes distributing forces over internal TGS iterations. That is a motivated hypothesis for part of this observation, not direct proof about this complete contact-rich robot. [PhysX steady-state discussion](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/Simulation.html#tgs-steady-state-velocity-and-position-discrepancy)

Stabilization is a distinct optional damping mechanism and is disabled here, so it should not be named as the established cause. The documentation also cautions that stabilization can impair physical fidelity for robotics. [Omni Physics stabilization](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/simulation_control/simulation_control.html#stabilization)

Root authorized a separate standing-only005 comparison changing only external-force timing, with exact same targets/startup/gates and all400 Hz evidence. It can test that option's effect while keeping the prior older-policy negative result visible. It does not automatically adopt the option, waive5mm consistency, resume walking or train a policy. If measurement semantics later require a new metric, that must be an explicit separately versioned method with the legacy discrepancy still reported; a finite-difference velocity integrated from the same position samples would be a tautological check and is not an independent validation.

Reproduce with NumPy, writing outside raw evidence:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B review_substeps.py --run /path/to/reference_physics_results_004/run --output /fresh/review
```

`replay/review.json` preserves per-environment vectors, all integrators, timing-shift diagnostics, largest substep residuals, frame identities and named motor extrema. `freshness_evidence.json` records timestamps/counter/change fractions. All interpretations above are limited to these admitted measurements and source readbacks; the exact native force/geometric-bias decomposition remains unobserved.
