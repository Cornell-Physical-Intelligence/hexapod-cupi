# Pipeline source lineages

The Stage2 mock and the physical four-bar task are separate releases. Source
integrity checks preserve both. They do not admit training or validate robot
dynamics; the physical validation reports and checkpoint contract remain
separate requirements.

`isaaclab/deploy/stage2_pipeline.sha256` remains byte-for-byte frozen, with
SHA256 `19fc816cf9c53a79be8e14831daa58a12eba3f7f07c5fca80d03f2dc947ccda1`.
Its 112 entries are verified from the immutable Git source commit
`81d7c6f2a43c7de99f32cd6bb1b7efb0f54874df`. The checker creates an isolated
Git archive, hashes every listed file from that archive, and checks the archived
manifest itself against the unchanged local copy. It neither checks out nor
executes archived source.

```sh
uv run python tools/check_pipeline_lineages.py historical
```

The current package configuration has two explicit, separately versioned
changes: `packages/hexapod_core/pyproject.toml` includes the RS05 v2 JSON data,
and `packages/hexapod_env/pyproject.toml` includes the actuator and task
subpackages. Requiring these current files to have their historical hashes
would remove needed packaging. The historical source and its old package
configuration remain recoverable at the pinned commit.

The original four-bar manifest `isaaclab/deploy/mkii_fourbar_v1_pipeline.sha256` remains preserved with source commit `3dc3fa0c9d0f9d05dd625c00f599ca456e94779c`. It describes the first 32/4-iteration implementation, whose driven solver test subsequently failed. Its bytes are included in the new release manifest.

The 5 ms TGS manifest `isaaclab/deploy/mkii_fourbar_v1_tgs_pipeline.sha256` is also preserved at source `ae1a023`. Its driven reversal failed; the current release changes physics to 1.25 ms while preserving 20 ms policy control and every physical acceptance tolerance.

The 800 Hz manifest `isaaclab/deploy/mkii_fourbar_v1_800hz_pipeline.sha256` remains
preserved at source `ea05fe8`. Its full driven test completed but exceeded the
original closure bounds. The new diagnostic release preserves that physical
configuration, adds event traces, and separately authors a D6 candidate that
removes redundant closure constraints. Diagnostic completion never admits PPO.

The first diagnostic manifest `isaaclab/deploy/mkii_fourbar_v1_diagnostics_pipeline.sha256`
is preserved at source `1c8f1cd`. Its first trace completed but Kit teardown exited
before the outer report writer ran. The lifecycle revision persists the report
before teardown and tests a nonreturning native close; model parameters and
physical gates are unchanged.

The diagnostic lifecycle manifest remains frozen at source `6734539` (the
`addcd8d` portability fix changed tests only). The asset-binding release records
the selected v3/v4 bundle consistently across CPU audit, Kit audit and live
runtime, requires equality through qualification and learner creation, and
distinguishes a proven blocked scheduler lock waiter from an active producer.
Model dynamics and physical acceptance thresholds are unchanged.

The asset-binding release is preserved at `0d1ceab`. Its complete D6 group
diagnostic failed the original closure and support checks. The next candidate
uses native bilateral PhysX mimic constraints for the parallelogram relations,
with all masses, geometry, actuator settings and measured residual limits
preserved. v3 remains default; v5 needs explicit selection and live admission.

The physical-mimic release remains frozen at `cfe0cf5`; test-only `ffa44f6`
keeps that functional identity. Its full nominal campaign failed closure,
support and group response. The next controller candidate preserves all
physics, assets, gains, caps, action endpoints and acceptance gates, but delivers
each motor position endpoint gradually over the sixteen physics substeps.
No velocity feedforward is added. This requires fresh physical admission and
cannot reuse the preceding reports or checkpoints. The earlier physical-mimic
manifest is retained byte for byte.

The scheduled-target release remains frozen at `9cd8d4c`. Its first full nominal
validation passed, but the refined solver developed standing torque chatter and
failed convergence. The next named controller candidate is
`mkii_pd_damping_030_v1`: Kp stays 30 N m/rad and Kd changes from 0.60 to
0.30 N m s/rad. Base RS05 JSON/defaults, voltage assumption, motor envelope,
overload budget, armature, CAD, contact model, timing, target ramp and gates
remain unchanged. The instantiated controller profile and gains are bound into
the motor/runtime manifest. This uncalibrated software-gain experiment needs
fresh physical admission; no preceding admission or checkpoint can be reused.
Both earlier manifests remain byte-identical.

The Kd 0.30 release is frozen at `a3081dd`. Both short standing checks passed
their individual physical gates, but their matched torque comparison failed.
The next release adds diagnostic-only world-XY translation and standing-only
traces. It leaves the task, gains, robot, solver recipe and all physical gates
unchanged. Diagnostics must report the requested and actual terrain origins
and reset root positions, with complete physics coverage and both admission
flags false. No validation/training option can request this translation.
The earlier Kd 0.30 manifest remains byte-identical.

The placement diagnostic release remains frozen at `d6d5863`. Matched one-robot
origin and +6 m X runs completed without reproducing the larger-batch torque
jitter; neither admits training. The next release explicitly filters collisions
between environments on both CPU and GPU. This manually populated task bypassed
Isaac Lab 3 automatic filtering, while the installed PhysX cloner disabled
GPU environment IDs. An authored USD verifier rejects missing or nonreciprocal
isolation groups and records their count-independent semantics in the runtime
identity. Backend collision response still needs a separate live overlap test.

The host supervisor also tolerates the narrow race where a foreign container
listed by full immutable ID disappears before its GPU configuration inspection.
Only an exact Docker missing-object response is accepted; unknown inspection
errors, GPU-capable foreign containers and unrelated GPU processes still block.
Owned-container identity and cleanup checks are unchanged. These changes require
a new source identity and fresh admission; all earlier manifests are retained.

The explicit collision-isolation release remains frozen at `d863663` and
completed its normal eight-robot GPU diagnostic with unchanged standing
behavior. Matched single/batch traces show that the standing transient can be
reproduced by one robot at the problematic world position. The trace contains
a 4.425369 rad/s mimic velocity residual after a contact impulse, despite small
coordinate error. The next targeted numerical recipe adds four final velocity
iterations instead of one, retaining 64/128 position iterations, 800 Hz physics,
50 Hz control, the same PD profile, assets and every acceptance tolerance.
Its ID is `mkii_fourbar_tgs_external_forces_800hz_final_velocity4_v4`. Actual
configured and resolved iteration counts remain checked at runtime and in
reports; old 128/1 evidence cannot qualify it. The earlier collision-isolation
manifest remains byte-identical. This is an unproven numerical candidate until
new diagnostics and the full admission campaign pass.

The four-final-velocity candidate remains frozen at `83a9bca`. Its short standing
trace reduced peak C-pin velocity mismatch from about 0.363 to 0.020 m/s, passive
velocity residual from 4.425 to 1.105 rad/s, and torque from 0.968 to 0.735 N m.
Some previously quiet locations worsened, so neither convergence nor full
qualification is established. The next controlled candidate uses sixteen final
velocity iterations, with ID
`mkii_fourbar_tgs_external_forces_800hz_final_velocity16_v5`. Position iterations,
controller, assets, timing and existing acceptance bounds remain unchanged.
It must reproduce the same eight world positions before a longer campaign.

This release also adds observational velocity-constraint measurements at every
physical substep: peak and RMS passive-coordinate velocity residual and C-pin
relative velocity, including angular lever-arm motion and explicit sample
populations. Full validation, diagnostics and the training physical guard retain
these values. No new velocity acceptance threshold is silently introduced.
Earlier one/four-velocity evidence cannot admit this new source. The published
four-velocity manifest remains unchanged; an unpublished telemetry-only draft
was superseded before release.

The sixteen-velocity source `c804169` and its released manifest remain frozen.
Its eight-world diagnostic is complete, and a matched 32-world standing pair
is running separately. The next release strengthens qualification without
changing robot dynamics: both full reports must contain exactly matching,
finite, ordered actual reset-root positions. Convergence descriptions derive
from the validated numerical recipes. The existing 0.05 N m / 5% torque
comparison now also checks raw pre-envelope demand; equal 5.5 N m clipped
peaks must not conceal divergent controller requests. This is a solver
sensitivity screen, not an absolute raw-demand cap. Velocity telemetry remains
observational and must be reviewed for the known constraint-velocity issue.
No prior short standing report grants admission to the new source. The
published sixteen-velocity manifest is retained unchanged in this release.

The placement-convergence source `fd34f66` remains frozen. Its full campaign
was interrupted by an explicit shared-compute request before completing
standing validation; it produced no nominal final report or PPO checkpoint.
The user subsequently granted exclusive Spark takeover. The next release
changes only coordination handling and release bookkeeping: the supervisor
reads exactly one canonical share-control line, records its semantic state
separately from raw note hashes, and ignores ordinary prose edits for control.
Explicit or invalid control still yields validation immediately and latches
the existing 120-second checkpoint grace for training. Initial and pre-barrier
control checks, unrelated-workload gates and exact-container cleanup remain.
This does not enable concurrent GPU sharing, alter robot dynamics or admit
prior incomplete runs. The old manifest and source are not rewritten.

The coordination-control release remains frozen at `1239159`. Campaign 008
completed every nominal standing and driven step but failed closure, support
and motor-direction checks. The motion-prefix diagnostic adds an exact
32-environment replay through the LR pushlever test, with every physics-step
metric, per-environment phase means and detailed traces around controls
2200–2499. It preserves physical parameters, prior motion history, placements
and acceptance gates, and cannot admit training. The preceding manifest remains
byte-identical; the diagnostic extension has its own source identity.

The motion-prefix source remains frozen at `3011b0f`. Its completed replay
reproduced the campaign 008 motion failure with identical runtime/reset layout
and matching recorded aggregate peaks. The opt-in `coincident_flat_origin_v1`
candidate changes environment placement before scene construction while keeping
the physical model, controller, solver recipe and acceptance bounds unchanged.
Its runtime identity records the named layout; reports separately retain every
environment's authored transforms, native sensor/link mapping, collider/mimic
ownership and actual initial reset readback. The original grid remains the
default and retains its existing runtime manifest. The candidate is not admitted
by its implementation or by an overlap diagnostic: full matched nominal/refined
physical validation remains required. External overlap/video tools are separate
artifact scripts with their own hashes, outside the frozen runtime identity.

The coincident-layout release remains frozen at `ae4f388`. Its full nominal
finished but missed the unchanged closure-position bound; no PPO was admitted.
The next release fixes two independently reproduced RSL-RL 5.0.1 API issues:
the trainer now applies Isaac Lab's deprecated-config adapter before runner
construction, and clones only inference-created model buffers before strict
checkpoint reload. It preserves buffer values, parameters, optimizer state,
normalizer settings and all physical parameters/bounds. Actual CPU runner
learning, checkpoint roundtrip and further learning in both the same and a
fresh runner are separate API evidence, not robot training. The new source
identity requires its own physical qualification; prior reports do not admit it.

The RSL501 compatibility source remains frozen at `5d476d4`. Its full 128/16
comparison overturned all 32 robots during the right-middle pushlever test,
despite staying within the geometric closure bound. The next numerical recipe
uses 1,600 Hz outer steps and 32 updates per 20 ms policy interval, retaining 64/128
position iterations and 16 velocity iterations. The RS05 contract explicitly
supports the 0.625 ms timestep; gains, physical geometry, torque/burst limits,
action endpoints, and every physical acceptance bound remain unchanged.
This changes controller/contact refresh frequency and requires a new identity
and fresh full paired qualification. Previous 800 Hz reports cannot admit it.

The initial preparation snapshot `8c1ade7` and its manifest preserve a single
regression-fixture failure: the test still expected an unapplied runtime after
the patch was installed. The next fixture revision explicitly selects the
installed-candidate comparison and retains all eight strict equivalence tests.
This is a test-lifecycle correction; physical parameters and gates are unchanged.

The measurement implementation reuses the original per-frame pin offsets and
batches passive velocity and force-norm arithmetic. Closure point/axis
contractions, acceptance reductions, all 31 native sensor reads, every substep
and the guard before reset remain intact. A frozen complete-row fixture using
the installed SDK quaternion math checks current capture against the previous
implementation; those regression inputs are included in this release manifest.

`isaaclab/deploy/mkii_fourbar_v1_1600hz_metrics_pipeline.sha256` covers **all 112 historical
paths** at their current hashes, every source/configuration/asset path in the
four-bar runtime identity, and the current launcher, CI workflow, lineage
checker, lineage regression tests, this document and workspace dependency
files. Verification requires the complete expected path set: missing entries,
extra entries and changed hashes all fail. Changes to any historical path
outside the two named packaging files fail, even if someone generates a new
manifest from that change.

```sh
uv run python tools/check_pipeline_lineages.py current
```

CI fetches full history and runs both checks independently. A raw
`sha256sum -c isaaclab/deploy/stage2_pipeline.sha256` remains appropriate when
reproducing the **matching historical checkout**. It is not the current
four-bar release check.

To create the new manifest after the source and asset files are finalized:

```sh
uv run python tools/check_pipeline_lineages.py generate
uv run python tools/check_pipeline_lineages.py historical
uv run python tools/check_pipeline_lineages.py current
```

Generation first verifies the historical lineage and refuses to overwrite an
existing destination. Future published releases need a new manifest filename
passed with `--manifest`, and a reviewed CI pointer update. Keep prior manifest
bytes and their source commits available. Do not regenerate a published
manifest to hide an unexpected difference.

Each owned Spark run also stores `source.SHA256SUMS` and `source.tar.gz` with
their digests in `supervisor.json`. That preserves the actual isolated run
snapshot, including uncommitted files, even if the staging directory is later
updated. A release manifest and a per-run archive serve different purposes;
neither replaces the live physical acceptance evidence.

## 9 September: integrated C-study priority release

`isaaclab/deploy/stage2_c_priority_20260909_pipeline.sha256` is the new current-check destination after integration onto main. It adds the separately isolated study and terrain tools plus the reviewed CI/dependency changes to the complete repository source identity. The previous 1,600 Hz metrics manifest, all archived paths and every production package remain byte-for-byte preserved. This source release is not new physical-model admission.

CI checks this filename explicitly with `uv run python tools/check_pipeline_lineages.py current --manifest isaaclab/deploy/stage2_c_priority_20260909_pipeline.sha256`. The pinned C-study runtime separately checks all 16 vendored files against its fixed tree digest. The original historical Stage2 check remains unchanged. New frozen study sources must include that runtime and manifest as documented in `experiments/c_length_study/README.md`.

### Documentation-contract follow-up

The first integration CI run caught two failures in an inherited documentation fixture that required a pending sensor-photo confirmation and another long-run permission request. The user has since confirmed Mid-360/D455 ownership and authorized parallel work. The updated fixture checks the correct 360 × 59-degree Mid-360 identity, confirmed inventory and separate standing/qualification context. Sensor physics, hardware-accounting tests and simulator gates are unchanged.

The current release check now targets `isaaclab/deploy/stage2_c_priority_20260909_docs_contract_pipeline.sha256`; it additionally covers this changed fixture and preserves the preceding integration manifest. CI and the default checker select this new filename.

### Paired PPO repair publication

`isaaclab/deploy/stage2_c_repair003_20260909_pipeline.sha256` is the current source-check destination for the isolated paired fine-tune tools, explicit initializer, raw-action instrumentation, launcher tests and deploy dispatch. It also binds the result hash manifest and read-only report generator. The completed Spark sources are retained separately with their exact hashes; main preserves its verified vendored bootstrap and adds stricter paired-source/nonfinite-torque preflight checks for future runs. Earlier source manifests and all production packages remain unchanged. CI and the default checker select this new filename.

### Terrain-entry adapter and reference feasibility

`isaaclab/deploy/stage2_c_terrain_entry_20260910_pipeline.sha256` is the new source-check destination. It adds the guarded two-phase full-C standing launcher, type-preserving Mesh reference, configuration deep copy and installed XYZW rotation convention, plus their regression tests. The frozen reference feasibility payload map, safe replay helper and terrain attempts 002/003 result maps are bound separately. The C-study runtime, production packages, robot inputs, acceptance gates and earlier manifests remain unchanged. CI and the default checker select this new filename. Runtime results require their own source and admission identities; the CPU reference studies do not admit a controller.


### Contact completeness and explicit quaternion telemetry

`isaaclab/deploy/stage2_c_contact_telemetry_20260910_pipeline.sha256` is the new source-check destination. Shared pre-reset capture preserves installed SDK XYZW and explicitly converts the WXYZ field used by quiet scoring; an independent 10° yaw regression rejects the unchanged 2° heading bound. Terrain point/friction sensors reserve at least 128 contact records per prim, and the guarded host rejects incomplete contact/friction data reported in the completed log even when raw metrics pass. Two new test modules, the reset-diagnostic/terrain-004 evidence maps and frozen velocity-candidate-001 source/result maps and replay helper are included. Prior manifests, checkpoints, runtime and acceptance gates remain unchanged.

The active velocity comparisons keep their own immutable source, including the candidate-specific quaternion conversion. This main release does not alter those running sources or transfer a previous admission. CI and the default checker select the new filename; later terrain capacity validation still needs fresh source and standing evidence.


### Research-poster publication release, 10 September 2026

The preceding contact-telemetry manifest remains byte-for-byte preserved. Publishing `tools/project_site.py` expands the integrated source inventory, so the new release is `isaaclab/deploy/stage2_research_poster_20260910_pipeline.sha256`. The checker’s current default and CI select it; its coverage also retains the preceding manifest. No robot runtime, C-study source, physics, checkpoint or acceptance gate changes. The first poster commit passed all code/robot tests and deployed Pages, then correctly failed release coverage because the new builder was missing. The separate failure/check record is `artifacts/project_review_2026-09-10/poster_release_check_001/README.md`.


### Causal sensor transport release, 10 September 2026

`isaaclab/deploy/stage3_sensor_transport_20260910_pipeline.sha256` selects the integrated transport correction, its immutable preparation/review/integration evidence map and the prior poster manifest. The checker default and CI select this new428-file release;112 historical paths remain verified. The evidence map binds the portable regression tests and exact old-source fixture as well as the adopted runtime. The previous poster manifest remains byte-for-byte unchanged. This source inventory change does not modify the separately frozen Spark C-study run or qualify terrain/perception.
