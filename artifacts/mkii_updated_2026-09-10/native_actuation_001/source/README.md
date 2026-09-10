# Canonical native coordinate/effort diagnostic — source002

This is a new bounded native source for the approved 7.466088235kg,19-body,18-direct-joint robot. It requires the exact completed native003 inspection as an immutable input. It does not load an actor, create a ground plane, enable position drives, choose servo gains or admit standing, policy training or hardware operation.

The parent import/API/inertia/limits/FK/scene/SDF checks remain intact, including the actual110.1.13 string SDF factory and all153 exact returned paths. The explicit numerical change is `enable_external_forces_every_iteration=True`; its native scene attribute must read back true. Asset geometry, self-collision, joint travel, friction, armature, zero implicit-drive gains/maximum and other physics configuration are preserved. This choice is new Phase B provenance, not a rewritten Phase A result or a claim that it resolves the old SDK velocity discrepancy.

## Exact experiment

After eight import-prefix steps, run40 zero-effort substeps. At reset boundaries, check each of18 named joints at−0.01/+0.01rad, including root/state/unused-target buffer readback and all-link FK. Reset operations must not increment the physics counter. These are state-reset cases inside one native world; solver-cache clearing or a new cold world per case is not claimed.

For each named joint/sign, state-reset to canonical zero, run40 baseline steps, apply **±0.005N·m for8×2.5ms**, then apply zero for8×2.5ms. The source performs **2056 experiment steps plus8 import steps=2064**,36 coordinate checks,36 pulse cases and74 logged state resets. No physics state or pose is written during a pulse/coast interval. External torque is cleared before every reset and during finalization.

The initial/per-case zero-input baseline requires |q|≤0.0001rad and raw |dq|≤0.001rad/s. Every experiment sample retains the unchanged diagnostic limits |q|≤0.05rad, |SDK dq|≤2rad/s, |angle difference/dt|≤2rad/s and |external command|≤0.1N·m. These bounds define a small units/sign experiment, not a motor speed/torque rating. Offending samples are saved before rejection, and uncertain getter failures preserve acquired fields in `failed_partial_step.json`.

The full24-coordinate floating rigid-link matrix is reconstructed from the exact model. Both root-origin and root-COM-velocity coordinate forms are saved and compared against the native generalized mass matrix before using it. The response prediction uses **M⁻¹τ**, not the diagonal. The selected pulse/coast predicts at most0.008968rad and0.286976rad/s across all joints/cases; the preserved rejected draft predicted0.127704rad with its longer coast. `DESIGN002_FREEZE_SHA256.json` binds that independently tested correction. Actuated-joint measured displacement must be0.5–1.5× the local constant-matrix prediction. This broad predeclared units/sign discriminator is not nonlinear/contact controller qualification.

## Force and rate semantics

Each400Hz row contains requested external torque, software-applied torque, **pre-step native actuation input**, post-step native input buffer, projected joint reaction, full incoming child-frame joint wrench, named q/raw SDK dq and independent interval-angle rate. The external force setter is documented in the actual110 source as independent of implicit PD. Therefore native maximum-drive-force zero is preserved; it is not mislabeled as a measured motor torque limiter. The software experiment limits the external command explicitly. Native reaction wrench includes constraints and need not equal the command. Post-step input-buffer behavior is recorded rather than silently reclassified as delivered hardware effort.

There is no sensor-noise injection, synthetic feedback replacement, torque-speed actuator model, learning reward, thermal model or checkpoint. Position/velocity target buffers are reset and read back only as an API-consistency check with all drive gains still zero. Near+π scalar limits remain bound by the import contract; these small coordinate cases do not claim a native collision-free full-travel sweep.

Process/host RSS/high-water/available/cache observations and CUDA free/total are retained at experiment boundaries. CUDA allocator metrics do not attribute memory to SDF meshes. Larger replica allocation requires actual canonical1→32 shared-mesh memory and throughput evidence; C-study capacity is not transferred.

## Invocation and acceptance

```sh
python3 -B -S run_actuation.py --asset /asset --admission /admission --output /output/actuation --preflight-only --device cuda:0 --headless
/workspace/isaaclab/_isaac_sim/python.sh /inspection/run_actuation.py --asset /asset --admission /admission --output /output/actuation --device cuda:0 --headless
```

`/asset` is the unchanged exact nine-file asset. `/admission` is native003's inspection output directory, matching `PHASE_A_SHA256.json` in full, not a fabricated success receipt. Both mounts and the new source are read-only. A source-bound host using the proven600s/90s ownership supervisor is separate and root-owned. This bundle has no dispatch or service-control code.

`actuation_contract.py` is standard-library-only and file-relatively loads the unchanged parent inspection contract. Host APIs are `verify_inputs(args)` and `validate_result(directory,identity)`. The latter checks the post-exit native error stream, immutable outputs, exact counts, and independently replays every raw clock/command/interval-angle/reset/response receipt. Both admission flags remain false on success. Incomplete acquisition is retained as failure; no automatic servo/standing/PPO continuation follows.

The output seal occurs before `SimulationApp.close()` can exit the process. As in native003, native-error/log files remain explicitly live through close and the host must validate/hash them after exit. Failure summaries distinguish completed raw rows from actual physics steps if a getter fails after stepping.

## CPU validation

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python3 -m unittest discover -s tmp/updated_native_actuation_002 -p 'test_*.py'
python3 -S tmp/updated_native_actuation_002/verify_bundle.py
```

Nine focused tests execute the actual2056-step driver on a deterministic raw-view fixture and test complete naming/reset/clock/force schedules, separate reactions, actual003 admission mutation, duplicated raw clocks, stale reset targets, first getter failure and explicit nonfinite evidence. The fixture is not native floating-body physics and is not represented as an Isaac pass. Draft logs retain a repaired list-absolute-value error and a fixture import-cache error. Final frozen tests and exact source/native API receipts accompany this bundle.

Native first-run uncertainties remain the compiled setter/getter behavior, explicit effort persistence, native floating matrix convention and constraint effects after state reset. Each produces bounded raw evidence or rejection. No alias substitution, solver fallback, weakened bound, collider reduction or unchanged retry is authorized by this source.

Source002 preserves unlaunched source001 e12c206b4ef7e4f4333f44e7d45282e820b500a65eacde024a9cb329ce346b20. Only failure evidence changes: native force mismatch/set/get failures append exact requested/returned data, and failed reset getter chains preserve prior acquired fields. Repeated failures during zero-force cleanup append rather than overwrite. Successful native calls, schedule, bounds, action values, physics, entrypoint and contract are unchanged. `tests_successor.log` records all nine focused tests.
