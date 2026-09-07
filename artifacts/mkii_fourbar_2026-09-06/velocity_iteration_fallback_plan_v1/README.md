# Separately versioned one-velocity-iteration fallback

**Proposed experiment only; no runtime edits or GPU launch.** This review reads source commit `c2af43ca0f384a4c2c7ab8f1d627f309dc78a683`. If the current 1600 Hz / 16-velocity-iteration recipe fails, the next controlled candidate should keep its model, timestep and controls and test **one TGS velocity iteration**. It requires a new numerical recipe, isolated source release and fresh admission. This document does not claim that candidate will be stable or faster.

## Why this is a defensible next branch

The actual profiler's loaded plugin is Omni PhysX **110.1.13**, internal build `c38f7d1`, SHA-256 `a66e7338758473c3361e7d81730a2fab3c1f746329e4e99541bf7e7834ab4b35`, with a compiled PhysX **5.9.0** serialization marker. The [version review](../physx_version_review_v1/README.md) preserves the runtime image, loaded-library mapping, static-file evidence and matching public-source revision. The public tag is a release-family reference, not proof of an exact mapping to the internal build.

NVIDIA's matching 110.1 guide recommends approximately one TGS velocity iteration and notes that applying external forces every iteration helps convergence. It explains that excessive acquired velocity can occur in complex collision cases. This supports changing velocity count as a controlled numerical experiment; it does not establish our root cause. [Official solver guidance](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/simulation_control/simulation_control.html#physics-solver).

The warning about more than four iterations refers to a historical change: before PhysX 5.3, TGS silently converted requested velocity iterations above four into additional position iterations; newer versions honor the requested counts. Therefore sixteen requested iterations must not be interpreted as one or four effective iterations in the current build. The warning is not an instability verdict. [Pinned SDK changelog, v5.3.0-105.1](https://github.com/NVIDIA-Omniverse/PhysX/blob/517a0073715120e114ee055b63b26c95e00d9039/physx/CHANGELOG.md#L1207).

Hard mimic constraints interacting with contact and stiff actuation, particularly on light links, need careful numerical tuning. NVIDIA supports reducing the outer timestep for complex closed loops; increasing TGS position iterations is not equivalent to reducing that outer timestep. Adding mimic compliance would alter the assumed mechanism response and should be a separate model investigation. [Official mimic guidance](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/rigid_bodies_articulations/articulations.html#articulation-mimic-joint-compliance), [official stability guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/guides/articulation_stability_guide.html#reduce-simulation-timestep-for-complex-systems-and-closed-loops).

The existing 800 Hz results do not isolate velocity count: both used 16. [64/16](../coincident_nominal_failure_001/README.md) completed the trajectory with only its 111.171 micrometre closure excursion exceeding the 100 micrometre limit; [128/16](../refined_rsl501_failure_001/README.md) overturned during a right-middle pushlever command despite a smaller maximum pin gap. They also have different source identities and are not a clean causal experiment. Closure alone cannot establish stable motion. The current 1600 Hz result must be preserved before drawing any new comparison.

## Minimal implementation delta, if selected

| Setting | Current v6 | Proposed fallback |
| --- | --- | --- |
| Outer physics timestep | 0.000625 s | unchanged |
| Decimation / policy interval | 32 / 0.020 s | unchanged |
| Solver / external forces every iteration | TGS / true | unchanged |
| Nominal / refined position iterations | 64 / 128 | unchanged |
| Velocity iterations in both runs | 16 | **1** |
| Numerical recipe ID | `mkii_fourbar_tgs_external_forces_1600hz_position64_128_velocity16_v6` | new unique ID, e.g. `mkii_fourbar_tgs_external_forces_1600hz_position64_128_velocity1_v7` |

The proposed ID is a planning label, not an implemented contract. Check that it remains unused before implementation. In a new source version, edit only `SOLVER_VELOCITY_ITERATIONS` and `NUMERICAL_RECIPE_ID` in `packages/hexapod_core/hexapod_core/fourbar_v1.py` for the runtime behavior. Its `numerical_recipe(1/2)` already changes position count only. `config.py`, `apply_numerical_recipe`, the actual resolved manifest and the qualifier consume these constants; they need no weakened or parallel override path. Leave the existing task ID intact.

Retain physical v5 geometry, masses, inertia, closed-loop construction, exact placement, contact material, 31 isolated native contact sensors, Kp30/Kd0.30, explicit RS05 torque/speed/thermal envelope, 48 V assumption, initial pose, action amplitude and the 0.040 rad/20 ms endpoint limiter with linear substep target schedule. Do not add artificial damping, mass, armature, compliance or speed caps in the same experiment. Preserve dtype and reduction order, every-substep acquisition/capture, 32-sample draining and the before-reset training guard.

`tools/mkii_training_contract.py` hashes the functional source and selected assets. A changed numerical contract therefore needs a new source manifest, validation reports, admission and checkpoint lineage. Do not edit frozen source manifests, reuse current-v6 admission, relabel an old one-iteration report, or resume a checkpoint across contracts. `compare_admitted_runtime` must continue comparing the complete typed manifest.

## Tests requiring deliberate updates

`isaaclab/tests/test_mkii_fourbar_numerical_recipe.py` contains the selected-recipe literals:

- Update the expected velocity count in `test_nominal_and_refined_change_only_position_iterations`, `test_validator_applies_exact_recipe_without_modifying_timing_or_limits` and `test_config_uses_explicit_backend_and_iteration_constants` from 16 to 1. Keep the assertion that the nominal/refined recipes differ **only** in position count.
- Change the application test's initial fake velocity count from 1 to 16 (or another wrong value), so the test still proves assignment rather than accidentally starting at its desired value.
- Update the expected recipe ID in `test_final_velocity_recipe_rejects_coherent_old_one_pass_evidence`; rename the test if helpful. Its `stale_backend` negative case must become 16 rather than 1. Add a coherent old v6 report with 1600 Hz, 32 decimation and 16 velocity iterations and prove it is rejected even when its old declared/runtime/count fields agree internally.
- Preserve old 800 Hz / 1-, 4- and 16-iteration rejection. Sharing a velocity count with an old report must not make its source/timestep/recipe acceptable. Add stale actual 16 to the declared-versus-resolved mismatch cases.
- Explicitly reject `True` and `1.0` wherever the contract requires integer velocity count 1, both in the declared recipe, actual resolved simulation and `[position, velocity]` list. Python equality alone conflates these values. Retain all existing strict scalar-type and finite-value guards.

Most fixtures derive the recipe dynamically. Run focused numerical-recipe, qualification-placement, campaign, supervisor, training-admission, training-guard, metrics, target-schedule and RS05 controller-profile tests, then the repository's canonical CPU suite `uv run python -m unittest discover -s isaaclab/tests`. Do not mechanically replace every literal 16 or 1: generic comparison fixtures in `test_mkii_fourbar_training_guards.py` intentionally use historical/fake manifest values, while 32 substeps and all timing expectations remain unchanged. Ensure a new-v7 nominal combined with an old-v6 refined report fails qualification, even if both otherwise claim success.

Update the live selected recipe in `docs/MKII_FOURBAR_TRAINING.md` only when implemented, preserving historical artifacts. `STATUS.md` remains the run owner's execution record. These documentation updates must not describe this untested proposal as admitted.

## Gates remain unchanged

No physical acceptance threshold requires an edit. The strict numerical-identity checks simply select the new recipe through the core constants.

- Both CPU and Kit asset audits and runtime binding pass; actual layout remains 31 bodies, 30 coordinates, 18 active motors. Live reset error stays at most 5e-6 rad and the anatomical frame check passes.
- Complete standing and driven tests without reset/termination/truncation; every physical substep is captured. Closure separation remains at most 0.0001 m; axis chord bound remains `radians(0.1)`; passive-coordinate residual at most 0.005 rad; applied torque at most 5.50001 Nm and motor-envelope excess at most 1e-5 Nm; zero invalid samples.
- Minimum plate height remains 0.055 m and nonfoot clearance 0.001 m. Outside startup, no nonfoot contact and at least one supporting foot are required. Raw demand and passive/pin velocities remain observed diagnostics; do not invent retrospective thresholds or confuse requested torque with delivered torque.
- Qualification requires both **32-environment, 1000-standing + 2400-driven-control** passes at 64/1 and 128/1 with identical actual placement and counts. The manifests may differ only in position iterations. Settled and driven mean-height deltas remain at most 0.001 m; applied and raw-demand peak deltas remain at most `max(0.05 Nm, 5% of refined peak)`.

At 1600 Hz, a complete run contains 108,800 physics substeps per environment. A short standing probe cannot admit PPO or exercise the previously failing driven prefix. If a reduced reproduction is used for diagnosis, label it diagnostic and still execute both complete validation runs before admission. Log contact/support history, attitude, raw/applied torque and passive/pin velocities along with geometric closure; keep warnings and all failure evidence.

## Controlled execution sequence and limits

After the current experiment completes, the owner may select this fallback and freeze its new release. Use the existing bounded campaign sequence: 1-env/100-control probe; complete nominal and refined validations; qualification; 64-env/3-iteration scratch PPO; separate-process resume to 512-env/1000-iteration PPO. A failed phase stops the sequence. Preserve source/asset binding, resource ownership, per-phase lock handling, timeout, shutdown and capture-follow-up requirements; external wrappers must bind the new exact source and wrapper hashes instead of reusing v6 labels.

A passing fallback would establish only the existing narrow flat-ground simulation admission. It would not demonstrate timestep convergence, all-terrain performance, a physically calibrated motor, collision-free workspace or hardware readiness. Measure elapsed time/throughput for matched work; fewer velocity iterations do not by themselves guarantee faster end-to-end operation, especially when sensor acquisition dominates. Only after stability is established should position-count reduction or other efficiency changes form a separate controlled experiment.

## Review and provenance

This directory is new. Only planning/evidence files were written; runtime, tests, existing artifacts, acceptance gates and Spark processes were not changed. `review_metadata.json` records the inspected source-file hashes and re-verification of the existing PhysX review. That review's `verify.py` and every `SHA256SUMS` entry passed. No fallback tests or live simulation were run because no fallback implementation exists yet.
