# Candidate recipe change checklist

Prepared from the physical-task source in this checkout on 2026-09-06. This is
an implementation checklist, not an applied change, admission report, or reason
to interrupt the current 128/16 qualification. Its baseline is the 800 Hz,
64/128 position-iteration pair, 16 velocity iterations, Kp 30 / Kd 0.30 recipe
in the RSL-RL 5.0.1 compatible source. Other agents' efficiency changes must be
included in the eventual frozen source identity if selected for the candidate.

The numerical rationale and versioned primary references remain in
[README.md](README.md). The iteration limit cited there is from the **PhysX
5.6.1** API; the TGS explanation also references **PhysX 5.8.0**. Neither document
establishes the exact PhysX binary build installed on Spark.

## A. Isolated 1600 Hz candidate

Keep the controller profile, motor envelope, geometry, constraints, contact
settings, placements, perturbations, policy/reward settings, and every admission
bound unchanged. Change the outer simulation timestep and its associated motor
update cadence together. Keep the nominal/refined pair at **64/16 and 128/16**
when testing this option; changing iterations at the same time would combine
two interventions.

### Required functional edits

Paths in the tables are relative to the repository root.

| File | Exact change or requirement |
| --- | --- |
| `packages/hexapod_core/hexapod_core/fourbar_v1.py` | Set `PHYSICS_DT_S = .000625` and `DECIMATION = 32`. Keep `POLICY_DT_S = PHYSICS_DT_S * DECIMATION`, which must equal `.02`. Assign a new, unique `NUMERICAL_RECIPE_ID`, for example `mkii_fourbar_tgs_external_forces_1600hz_final_velocity16_v6`, after checking that the label is unused. Keep solver type 1, position count 64, velocity count 16, and external forces every iteration. |
| `packages/hexapod_core/hexapod_core/rs05_v2.json` | Append `.000625` to `provisional.supported_physics_dt_s`. **Without this, the motor contract rejects startup.** Keep `provisional.physics_dt_s = .005`: it is the historical default used by other task lineages. Do not change vendor data, thermal assumptions, gains, or friction/armature. |
| `packages/hexapod_core/hexapod_core/rs05_v2.py` | Update the hardcoded supported-timestep error in `validate_operating_assumptions` to include `.000625`, preferably format the already-defined supported tuple. Validation must continue rejecting unlisted/nonfinite timesteps. |

The motor manifest hashes both the JSON and implementation source, and includes
the selected timestep. Thus the supported-list change changes motor/source
identity even though the vendor envelope is unchanged. Preserve earlier motor
manifests and admissions; generate new ones from the selected source.

### Already derived correctly; verify without duplicating constants

| File or component | Existing dependency and required result |
| --- | --- |
| `packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/config.py` | Simulation `dt`, contact sensor `update_period`, and the explicit motor factory timestep derive from `contract.PHYSICS_DT_S`; task `decimation` and `render_interval` derive from `contract.DECIMATION`. They must resolve to `.000625`, 32, and a 20 ms render/control interval. Update the explanatory 800 Hz / sixteen-update comment. |
| `.../tasks/mkii_fourbar_v1/env.py` | The startup guard checks simulation/motor timing and `self.step_dt`. The environment passes `contract.DECIMATION` explicitly to `MotorTargetRamp`, starts one ramp per policy action, and writes a ramp target in every `_apply_action`. Verify 32 writes and 32 physical state/metric updates per 20 ms action. |
| `.../tasks/mkii_fourbar_v1/target_schedule.py` | The ramp already supports any positive integer substep count. Its generic default of 16 may remain because the physical task passes 32 explicitly. Preserve exact last-endpoint assignment, partial reset semantics, soft limits, and zero desired velocity/feedforward torque. |
| `.../tasks/mkii_fourbar_v1/math.py` and `packages/hexapod_runtime/hexapod_runtime/fourbar_adapter_v1.py` | The endpoint pipeline must remain `.040 rad / .020 s` with action scale `.30`. Do not apply the endpoint limiter 32 times per action. The maximum individual target increment becomes `.00125 rad`; the segment still advances at at most 2 rad/s. |
| `packages/hexapod_env/hexapod_env/actuators/rs05_v2.py` | Preserve the historical `.005` defaults. The physical task explicitly supplies its `.000625` timestep. |
| `.../actuators/rs05_v2_model.py` and `rs05_v2_runtime.py` | The budget already receives the selected timestep. Consumption/recovery/exposure integrate physical time; remaining budget is divided by `dt` when deriving its instantaneous limit. One actuator/budget update belongs to each outer physics step, **not** each policy action or inner TGS iteration. No formula change is indicated. |
| Reward, command timer, episode, PPO configuration | Rewards and command holds use the unchanged policy `step_dt`. The 20 s training episode, 24-step rollout, PPO discount/GAE, and optimizer schedule keep their current meanings. No rescaling is required. |
| `isaaclab/validate_mkii_fourbar.py`, `train_mkii_fourbar.py` | Hooks check actual timestep and exact decimation. Metric accumulation, denominators, and required coverage use the imported decimation. Keep these checks and all thresholds intact. |

At Kp 30, the target-induced proportional increment is at most **.0375 N·m per
outer step**, versus .075 N·m at 800 Hz. This is only the contribution of changing
the target; it is not a bound on total requested torque. The explicit PD samples
measured velocity twice as often. This is a new numerical/controller sampling
recipe, not proof of calibrated motor dynamics or implemented hardware timing.

### Durations and counters that must remain consistent

The current validator's `--steps 1000` denotes policy steps, not physics steps.
Its episode allowance `(args.steps + 2401) * .02 + 1` remains correct.

| Measurement | Policy steps | Physical duration | Physics steps at 800 Hz | Physics steps at 1600 Hz |
| --- | ---: | ---: | ---: | ---: |
| Standing startup, first fifth | 200 | 4 s | 3,200 | 6,400 |
| Standing settled | 800 | 16 s | 12,800 | 25,600 |
| Standing total | 1,000 | 20 s | 16,000 | 32,000 |
| Each individual motor/sign segment | 50 | 1 s | 800 | 1,600 |
| All 18 motors, both signs | 1,800 | 36 s | 28,800 | 57,600 |
| Each grouped direction segment | 100 | 2 s | 1,600 | 3,200 |
| Three groups, both signs | 600 | 12 s | 9,600 | 19,200 |
| Complete driven window | 2,400 | 48 s | 38,400 | 76,800 |
| Complete standing + driven qualification | 3,400 | 68 s | 54,400 | 108,800 |
| One PPO rollout per environment | 24 | .48 s | 384 | 768 |
| Final inference probe | 100 | 2 s | 1,600 | 3,200 |
| Fifteen-second video | 750 | 15 s | 12,000 | 24,000 |

Physics-step counts above are shared simulation steps, each covering every
environment. One 32-environment full qualification is therefore **3,481,600
environment-physics steps** at 1600 Hz. A 512-environment, 1,000-iteration full
PPO phase still collects **12,288,000 transitions**, but executes **393,216,000
environment-physics steps**, excluding final inference. Wall time will not
necessarily double: measure the existing short probe/scratch throughput before
estimating the full job. Do not shorten physical durations or sampling coverage
to fit the previous estimate.

### Qualification, launcher, diagnostics, and video details

- `tools/qualify_mkii_fourbar.py` checks matching source/asset/motor identities,
  placements, full standing/driven completion, the selected nominal/refined
  recipes, and convergence. Keep these conditions. Update its `admission_scope`
  text stating “continuous 1.25 ms” to the new 0.625 ms sampling, or derive it from
  the contract; otherwise the admission prose will falsely describe the run.
- `isaaclab/deploy/run-mkii-fourbar` derives expected diagnostic coverage from
  the report's decimation and validates the declared and actual recipe.
  `run-mkii-fourbar-campaign` uses policy-step counts: probe 1 environment × 100
  steps; nominal/refined each 32 × 1,000 standing + 2,400 driven; scratch 64 × 3
  iterations; full 512 × 1,000 iterations. These counts need no timestep edit.
  Keep timeout/resource controls; adjust bounded launch arguments only from
  measured throughput and within launcher limits.
- `isaaclab/diagnose_mkii_fourbar.py` uses policy markers 2,200–2,500 for its
  motion-prefix trace. At 1600 Hz the interval becomes physics steps
  **70,400–80,000**, or **9,600 samples**, and the whole 2,500-control prefix
  becomes 80,000 physics steps. Preserve its existing placement restriction.
  Historical `motion_prefix_analysis` prose and event indices describe 800 Hz
  captures: do not rewrite those tools/reports or compare old raw physics-step
  indices as if they denote the same times. If a new diagnostic is needed,
  create a versioned analysis with timing read from the new report.
- The frozen `full_body_overlap_tools/live_full_body_overlap.py` derives action
  boundaries from the contract and accepts 32–512 physics steps divisible by
  decimation. If repeating its previous **.32 s** window, use **512 physics
  steps** at 1600 Hz instead of 256. Both satisfy the host's multiple-of-16
  check and the runtime's multiple-of-32 check. Preserve old evidence. Whether
  prior layout evidence remains sufficient is a source/admission question, not
  a reason to relabel an old report as a new run.
- `policy_capture_tools_v2/record_admitted_policy.py` calculates frame count
  from `cfg.sim.dt * cfg.decimation`, encodes at `1 / raw.step_dt`, steps once
  per frame, and uses the physical coverage guard. Fifteen seconds remains
  **750 frames at 50 fps**. Do not double frames or create a timelapse.
  `capture_common.py` validates duration as an integer number of policy steps;
  its `.02` default remains valid. Capture must use the new admitted source,
  matching checkpoint, and restored runner configuration.
- The existing campaign capture follower binds source hashes, campaign PID/start
  identity, and final checkpoint evidence. Create a new binding/output directory
  for a new campaign; do not retarget or overwrite a running/frozen follower.

### Focused CPU checks before the existing GPU sequence

1. Update **current-recipe expectations**, not acceptance bounds, in
   `isaaclab/tests/test_mkii_fourbar_numerical_recipe.py`: `.000625`, 32, new
   recipe ID; simulation, motor, contact, rendering, and startup guard defaults.
   Preserve negative tests for older recipes and add coherent old `.00125`/16
   evidence plus mismatched motor/simulation timing as rejection cases.
2. Extend `test_rs05_v2_actuator.py` supported-timestep/manifest expectations and
   `PhysicsTimeScalingTests` to `.000625`. Test equal physical durations for
   peak-budget consumption, exhaustion, recovery, and exposure; partial final
   budget must not overdraw. Keep the historical `.005` default assertions.
   Existing explicit `.00125` fixtures in controller-profile tests remain valid
   supported configurations and must not be globally replaced.
3. Exercise 32-substep endpoint interpolation and partial reset in
   `test_mkii_fourbar_target_schedule.py` / `test_mkii_fourbar_adapter.py`.
   Contract-derived tests should follow automatically; retain standalone
   generic-default tests. Check equal endpoints at common physical times,
   unchanged `.040 rad` endpoint limit, and exactly 32 writes per action.
4. Run `test_mkii_fourbar_training_guards.py`,
   `test_validate_mkii_fourbar_metrics.py`,
   `test_mkii_fourbar_velocity_telemetry.py`,
   `test_mkii_fourbar_validation_prefix.py`, and launcher/capture contract tests
   to verify complete coverage, counters, timing, and source rejection. Frozen
   historical fixtures are not candidates for blanket numeric replacement.
5. Run the canonical CPU suite once after the selected implementation and
   release changes. The completed RSL 5.0.1 CPU API/resume probe need not be
   recreated merely because physics sampling changes; exercise it again only
   if the runner/normalization/checkpoint API code changes.

No new broad diagnostic campaign is required before the standard short probe,
fresh full nominal/refined pair, admission, scratch/resume, and full PPO
sequence. Those existing stages still decide readiness. The pair must use one
frozen source and differ only in position-iteration multiplier.

## B. Exact minimal 127/254 alternative

This is an alternative to A, not a combined change. Retain **800 Hz / 16
substeps / 20 ms**, 16 velocity iterations, existing motor timestep support,
Kp 30 / Kd 0.30, and all physical/validation settings.

1. In `packages/hexapod_core/hexapod_core/fourbar_v1.py`, change only
   `SOLVER_POSITION_ITERATIONS` from **64 to 127** and assign a unique recipe
   ID identifying 127/254 at 800 Hz (for example
   `mkii_fourbar_tgs_external_forces_800hz_position127_final_velocity16_v6`).
   `numerical_recipe(2)` then produces **254/16**. The task configuration and
   validator already derive their iteration counts from this function/constant.
2. Update the nominal/refined count and exact recipe ID expectations in
   `test_mkii_fourbar_numerical_recipe.py`, including its configuration assertion
   currently expecting 64. Add coherent old 64/128 evidence rejection; keep all
   timing assertions at `.00125`, 16, and `.02`. Other tests using actual
   contract-derived reports should adapt without numeric edits. Preserve valid
   historical fixtures; inspect any failure rather than replacing every 64.
3. Add or verify a CPU assertion that both recipes' position counts are within
   **1–255** and velocity counts within **0–255**. A fail-fast range guard in
   `numerical_recipe` is useful hardening, but is not necessary for the two
   edits above to produce valid 127/254. If added, test the actual rejection
   path for 256. Do not silently clamp **128 × 2 = 256** to 255: that changes
   the stated refinement contract. The documented limit is from
   [PhysX 5.6.1 `PxArticulationReducedCoordinate`](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/classPxArticulationReducedCoordinate.html).
4. Make the shared release/documentation changes below and collect a **fresh
   127/16 nominal plus 254/16 refined pair**. The in-progress 128/16 run can
   inform the decision but cannot substitute for 127/16 under the source and
   multiplier contracts. No motor JSON, actuator, timing, scheduling, qualifier
   sample-period prose, or video changes are required for this alternative.

## Release and documentation changes shared by both alternatives

- Add a new pipeline manifest; do not regenerate the published
  `isaaclab/deploy/mkii_fourbar_v1_rsl501_compat_pipeline.sha256` in place.
  Update `CURRENT_MANIFEST` in `tools/check_pipeline_lineages.py`, preserve the
  previous manifest through `CURRENT_EXTRA_PATHS`, update the explicit manifest
  selection in `.github/workflows/tests.yml`, and document the new current
  release in `docs/PIPELINE_LINEAGES.md`. Check the resulting manifest and
  archived lineage with the existing checker.
- Update the active recipe/counts in `docs/MKII_FOURBAR_TRAINING.md`; preserve
  its historical sections. `docs/MKII_MOTOR_TARGET_SCHEDULING.md` explicitly
  describes source `9cd8d4c`, so preserve its historical sixteen-increment
  description and append/link the new cadence if A is selected. Root updates
  `STATUS.md` and `HANDOFF.md` with observed state and the selected source.
- Freeze all selected source edits together, then generate/verify identity and
  manifests. Use a new source snapshot and output directories; never mutate
  the currently running snapshot. Strict checkpoint/admission matching means
  a prior-source checkpoint cannot authorize this recipe's resume. Start the
  existing scratch-to-full sequence from this recipe's own admission.
- Neither option requires regenerating CAD, URDF, USD, masses, inertia, joint
  limits, or collisions. Neither establishes hardware transfer or all-terrain
  reliability. Do not alter the 0.1 mm closure bound, torque/envelope limits,
  support conditions, nominal/refined convergence thresholds, or motion set.
