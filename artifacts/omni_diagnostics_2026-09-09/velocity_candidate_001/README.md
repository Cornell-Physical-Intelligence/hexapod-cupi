# C-study target-velocity candidate 001

Frozen CPU-reviewed source and first full-robot standing admission, 10 September UTC / 9 September Eastern. This is a new controller experiment, not a promoted walking policy. See [STATUS](../../../STATUS.md) for subsequent execution and decisions.

The C robot retains its 55.4535 mm joint-center coxa distance (47.7183 mm horizontal), 72.5 mm femur, 126 mm tibia, 8.26081134 kg mass and 1.6 N·m applied cap. The admitted study stance is femur 40° / tibia 120°. Manufactured packaging and the physical four-bar model remain separate qualification work.

The actor commands integrated target velocity with an explicit experimental 8 rad/s² acceleration bound and 0.04 rad/20 ms target budget. Its target position/velocity are observable in a new 495/498 actor/critic contract. Zero actor mean and initial Gaussian standard deviation 0.05 are tested first; no old checkpoint is resumed. Zero body command keeps policy feedback available. This target parameterization does not guarantee physical smoothness or imply measured motor speed/acceleration capability.

The six executable files, 25 CPU tests, noise study and original review are preserved verbatim in [candidate](candidate/README.md). The root independently reran all 25 tests and four guarded-launcher tests. Replay the candidate tests with:

```sh
uv run python -B artifacts/omni_diagnostics_2026-09-09/velocity_candidate_001/replay_cpu.py
```

The replay uses a temporary copy under the repository's `tmp/`, preserving the original relative layout and frozen bytes. The archived builders/launchers document the exact dispatch; their temporary source paths are provenance, not a portable automatic launch command.

Source `omni_velocity_source_001` was assembled from main `bf7a5e84294d0e80394c666f1d3d8581c7b96bfa`, the pinned 16-file study runtime and the previously admitted full-C assets. All 909 source payload hashes matched on Spark through Tailscale. Source manifest SHA: `54aeaa187f6d3067a4c522f435995d4d17077424f1f901ae99b9035ff5aa254a`; executable identity: `7d6cda50f026467981fe43edd342fb6cc72e9d775aa334f23c1ce43e075d2388`; plan: `6d4ad9cbba881fa061886261a3644525194ef7dfafe680af0a9baca55a275c29`.

Fresh standing passed all 32 robots × 1,000 controls: zero terminations/truncations, zero post-settle nonfoot contacts or requested saturation, peak settled requested torque 1.3148 N·m and mean settled root height 0.129507 m. [Admission](dispatch/remote/flat_admission.json) SHA: `9ae4dc3eca6ff0ad778ab0e282320c88487c38e4d2e56e11e3a248e8c7279226`. This validates the bounded standing setup only.

The dispatched user unit is `hexapod-omni-velocity-probe-001-20260910.service`, with results under `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/omni_velocity_probe_001`. Its next phase checks deterministic standing, sampled exploration and, only if those pass, two stand-only PPO updates with strict save/reload. No walking run starts automatically. Each phase holds both shared locks and owns one exact container; pause 018 arms restoration of the previously active forecasting timers. The published launch and standing records are historical snapshots; they do not claim the unit is still running or that the later probe passed.

Candidate snapshots explicitly convert installed Isaac XYZW quaternions into the quiet review's WXYZ convention. Independent 10° yaw regression catches a legacy shared-telemetry false heading pass. Existing quiet gates are unchanged; historical traces require explicit schema correction before heading reanalysis. Stage 2 remains incomplete.

## Completed probe: rejected before PPO

[Complete calibration](results/run/probe/calibration.json) passed every zero-mean replica. Over its short three-second post-settle quiet window, worst joint RMS was 0.008689 rad/s, target motion was zero, requested saturation was zero and maximum planar excursion was 0.0000314 m. This is a short initialization check, not the required sustained quiet-standing qualification.

Sampled standard deviation 0.05 was correctly applied, but only one of 32 replicas passed the motor/contact calibration. Joint-specific saturation reached 100% in the worst case; one termination occurred, with zero post-settle nonfoot contacts. The guard rejected the candidate before runner smoke or any PPO update. This is an exploration failure, not evidence that the zero-mean controller jitters. No walking checkpoint was created.

Root independently matched all 20 result payload hashes against Spark. [Post-run verification](results/run/post_run_verification.json) records all 909 source files and 550 asset files unchanged, owned containers exited, both locks free, no CUDA process, and both forecasting timers restored through pause 018. [Result hashes](results/SHA256SUMS.json) preserve logs, calibration, terminal state and restoration. Subsequent fixes require a new source/run label and a new matching admission; no automatic continuation is allowed from this rejection.
