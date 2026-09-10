# Full-C opposing-pair load-transfer diagnostic001

Ready for an independently reviewed bounded physics run; no physics result exists in this preparation. This tests whether LM and RM can unload together while LF/LR/RF/RR support the actual full C robot. It is a proposed four-support diagnostic, not a walking gate, a PPO run, Stage2 completion or production adoption.

The exact parent is reference009 (926-file map `04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e`). The new source has934 payloads: eight new runtime files plus changed lineage metadata; every original runtime byte is unchanged. In particular, there are no matched-origin reset methods. The pair generator/scorer/helpers are copied exactly from frozen `reference_load_transfer_001` (`6cd797bee322d84328a6bbf9ff4aff289c57af45fbca27bc425d5e8d7a83e334`); shared geometry, math and residual core are byte-verified against source009. Existing `tools/wave_reference.py` remains unchanged. `SOURCE_BUILD.json` gives all hashes and reconstruction changes.

The inherited `solver_comparison` protocol name describes the retained reference009 configuration lineage. It does not authorize this campaign's phases. `PAIR_PROTOCOL` independently declares the only two actual host phases: a fresh32×1000 original physical-and-quiet standing admission, then a single1×1300 pair test. No old standing receipt is reusable. The pair mode binds the exact fresh standing identity and receipt hash; all32 standing replicas must pass the unchanged quiet scorer. No wave, origin or training mode is dispatched by this host.

## Timing, physical settings and target continuity

The fresh pair environment uses the original randomized physical reset, zeroes episode age, and runs the same200 controls: a2s bounded C2 target transition from the actual randomized target to the exact named float32 canonical C40/120 target, followed by2s hold. It never writes body/root/joint state beyond the inherited normal environment reset. The frozen generator reset receives the last actual pre-reset row and must preserve the emitted canonical target exactly.

Then1100 controls (22s) perform2s baseline,3s smooth7mm LM/RM raise,2s unload hold,3s smooth return and12s quiet, including2s return settling plus the full10s measured quiet window. All12 corner-joint targets remain fixed. The generator supplies joint-position targets through the existing exact zero-residual controller, retaining its named soft limits and formal target P/V/A budgets. No target clipping, body-pose prescription or reference time scaling is introduced.

The full robot, accurate mass/inertia, named articulation lookup,1.6Nm RS05 requested torque bound, actuator snapshot, contact buffers128, distal versus shaft classification, and SDK XYZW telemetry conversion remain from source009. Physics remains TGS16/1, external-forces-every-iteration true, stabilization false, dt.0025×8. Authored solver roots must read back16/1; absent unauthored per-link overrides are distinguished from conflicting values. The optional native iteration getter is still unverified, not guessed.

## Distinct proposed criteria

During all22 diagnostic seconds, all four corner contacts must remain valid, projected COM support margin must be at least50mm, body displacement at most15mm, body angle at most.10rad, corner drift at most10mm and reported corner slip at most.020m/s. There must be at least1 contiguous second in the unload hold with both actual middle toes at least2mm clear, at least2 force-free samples per foot and each normal force at most1N. Measured mean vertical force balance must be within5% of the8.26081134kg weight. Return requires six measured supports within.6s with three stable confirmations, followed by the original measured10s quiet scorer.

Every diagnostic400Hz requested torque sample, including its initial sample, must be at most1.6Nm; applied torque must be at most1.60001Nm. An immediate hook checks all eight substeps after each control before another target is emitted. The pre-reset row is saved before endpoint, finite, terminal and proposed measurement checks. A post-step check applies the frozen measured body/support/slip bounds to every completed row, including the final row. Exact reference lag and cast-error guards remain. Contacts are classified at50Hz; complete400Hz collision classification is not claimed.

The startup and diagnostic observers are separate, sequential contexts that restore the original scene.update callback. Their raw counters and timestamps remain unmodified. Diagnostic time0 corresponds to physical time4s; its8801 rows cover all1100 controls with actual angles, reported rates, link/COM pose/velocity and torques. The generic observer's postsettle prefix is only a relative-index summary, not a settling exclusion for the pair scorer. The original startup trace remains separately available and is not hardware-startup qualification.

A completed proposed diagnostic does not resolve the known joint-rate versus angle-increment discrepancy, prove a faster gait, qualify learned rate observations, or change existing wave gates. Raw measured forces and torques can inform a separate next gait decision.

## CPU/source verification

`tests_complete.log` records10 passing tests: the nine independently reviewed driver/host checks plus one complete1100-control synthetic wiring test using the actual frozen generator, residual core and scorer. It verifies target pairing, physical time4.02..26s,8801 substep indices/torque endpoints, final measurement checks and the full10s quiet window. Synthetic kinematics and contacts are not Isaac evidence. The frozen pair owner's16 core/scorer tests were previously reviewed separately and are not relabelled as new physics.

`INTEGRATION_PREFLIGHT.json` records all934 source hashes/pinned16 runtime verification, rejection of actual reference009 admission and rejection of a failed-quiet synthetic receipt. The source preflight itself creates no GPU process, output campaign or actual admission.

Focused commands (from repository root):

```sh
PYTHONDONTWRITEBYTECODE=1 uv run python -B -m unittest discover -s tmp/reference_pair_physics_adapter_001 -p 'test_pair*.py'
```

`build_source.py` and `check_integration.py` are one-shot preparation scripts. They reject existing outputs; do not rerun them inside the frozen preparation.

## Root-owned launch interface and expected outputs

After the current job exits, root verifies both locks, live workloads, exact owned-container absence and the previous forecast restoration. Root alone binds a fresh pause record, source and outer guard. Suggested new remote source/output names are `reference_pair_source_001` and `pair_load_transfer_001`; they are not launch claims.

The exact host invocation, under the root-owned guard and environment, is:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B /home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_pair_source_001/tools/launch_pair_physics_spark.py --source /home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_pair_source_001 --output /home/orionh/HEXAPOD_runs/mock_length_study_20260909/pair_load_transfer_001 --isaaclab /home/orionh/IsaacLab
```

The guard must set source/tools in PYTHONPATH for host imports, verify the complete934-file map, preserve readonly source and study mounts, and restore only explicitly paused forecasting workloads on every preflight failure, job exit or fallback. Each phase has a600s bound and90s AppReady deadline, with45s startup traceback and unbuffered output. Two phases allow at most1200s plus cleanup/host overhead; root should bind an outer bound and later restoration fallback rather than extending either phase silently.

Expected files are `campaign.json`, `jobs/{standing,pair}.json`, both contact-overflow audits and full logs; `standing/admission.json`, resolved environment/solver readback, trace and400Hz evidence; `pair/startup_reference.json`, `pair/pair_reset.json`, `pair/startup/{trace.npz,physics_substeps.npz,physics_substep_review.json}`, `pair/{trace.npz,reference_states.json,post_step_measurements.json,physics_substeps.npz,physics_substep_review.json,state.json}` and resolved pair environment/solver readback. Failures preserve available `partial_trace.npz`, rejected reference, partial recorder data and exception reason. Both original study hash maps, source checks and exact container IDs are retained even on terminal failure. Unknown Docker inspection/cleanup status remains owner-review uncertainty; it is not treated as absence.

`state.json` and `campaign.json` distinguish proposed diagnostic criteria from physical/Stage2/terrain qualification. The frozen scorer retains `external_source_asset_provenance_verified=false` because provenance is independently checked by the host/outer audit. Root should publish raw evidence and those external audits without rewriting the scorer's stated scope.
