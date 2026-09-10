# Bounded PPO repair comparison 003

Status: **completed and reviewed; neither branch selected for continuation**. See the [results report](results/README.md), [identity checks](results/provenance_checks.json), and [forecasting restoration receipt](results/forecast_pause_012/restored.json).

This is an isolated preparation and evidence directory. It does not modify the working branch or any previous Spark source. The comparison remains **unqualified**, regardless of its short-screen result.

The original direct-joint controller oscillates during standing. Active-PD zero-action controls held nearly still; observation-noise removal did not resolve the oscillation and the tested target filter regressed several directions. This experiment isolates a lower-exploration fine-tune and an additional reward on the raw standing action intent.

Both branches start independently from checkpoint `1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8`. They keep the 315/318 observation architecture, all learned actor and critic weights, and both observation normalizers. The loader checks those tensors exactly. It then initializes the scalar Gaussian standard deviations to 0.10. Training uses entropy coefficient 0, a fresh Adam optimizer at initial learning rate 0.00005, and the existing adaptive learning-rate schedule. The checkpoint iteration counter is retained. This is an explicit fine-tune initialization, not an optimizer-state resume.

Each branch receives exactly 50 updates with 1,024 environments and seed 157. Dynamics remain noise scale 1, target filter off, target slew 0.03 rad per 20 ms, and the existing 1.6 N·m actuator cap. The existing command distribution and repair rewards remain unchanged. Branch A has zero extra reward weight. Branch B adds weight −2.0 to the mean square of the sampled raw normalized action when planar command magnitude is at most 0.03 m/s and yaw command magnitude is at most 0.05 rad/s. This is measured before action clipping and target slew; the actor remains active.

Both branches run fresh standing admission, a 48-environment 12-second matched original-checkpoint diagnostic baseline, the 50-update fine-tune, then identical diagnostics of the trained checkpoint. Invalid checkpoint identity, observation audits, applied torque above 1.60001 N·m, resets, missing evidence, nonfinite metrics, unrelated CUDA work, or a stop request halt the campaign. Every direction is screened; aggregate improvements alone do not justify continuation. No extra training, qualification, or video is automatically allocated.

## Source and run identity

- Remote run: `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/omni_repair_003`
- Frozen sources: `omni_repair_source_003_a` and `omni_repair_source_003_b` under the same parent.
- Verified base: `omni_control_probe_source_002_no_noise`; its complete existing source manifest was checked before copying. All 16 historical `isaaclab/hexapod_rl` files are unchanged.
- Service: `hexapod-omni-repair-pair-003-20260909.service`; one-hour runtime bound with normal job-scoped GPU locks and identity-based cleanup.
- Forecast pause: `forecast_pause_012`; both forecasting services were inactive and the GPU was idle before launching. Previously active timers restore on service exit, with an independent 75-minute fallback.
- [Source manifest identities](source_build.jsonl) and [changed file hashes](changed_source_hashes.json).

## Validation and publication

Six focused CPU tests cover strict options, wrong-checkpoint rejection before resume, real saved tensor preservation, optimizer state handling, corrupted normalizer detection, and the unclipped standing-only reward. The [installed RSL CPU smoke log](installed/cpu_smoke.log) verifies the actual installed checkpoint API and deterministic inference without using the GPU. Its first attempt found a CPU map-location issue; the final helper explicitly loads to `runner.device` and the complete installed smoke passes.

For publication, port the three small behavioral diffs in [patches](patches/) into the current main versions. **Do not replace main's whole legacy files:** main's C-study runtime bootstrap and its vendored 16-file runtime must remain intact. Add `tools/omni_repair_training.py` and the new pair coordinator separately. The new frozen deploy wrapper adds only an `omni-repair-pair` case dispatching to the coordinator. The two new training-plan objects are also retained in `results/sources/`, and the builder records their exact hashes. Preserve existing release manifests and create a new release manifest/CI pointer after root's integration checks.

This experiment is not a Stage 2 completion decision. Quiet stand, stop transitions, all bearings, both turn directions, combined arcs and path transitions must still satisfy their full numerical and visual checks against the accepted forward standard.
