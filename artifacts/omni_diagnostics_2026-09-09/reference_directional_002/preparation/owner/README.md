# Three previously unmeasured directional cases002

This is a fresh bounded full-C reference screen for left strafe, left turn and a forward-right arc. It uses unchanged wave005, zero policy residual and the original physical, contact, torque, motion-consistency and quiet limits. It does not train PPO, admit the full omnidirectional envelope, qualify terrain or change the production model.

Reverse001 already ran all 2,400 controls and is **rejected by its original 50 Hz 5 mm position/velocity consistency gate** (5.875744391 mm). Its separate NumPy-boolean finalization error prevented a terminal state save. Both failures remain immutable; this successor does not repeat or admit reverse. The included byte-identical `reverse001_replay.json` is a separate posthoc review, not a repaired original receipt. Its parent review freeze is `eeab7f65062d0545c4c4757a94adfdfe4eb3d2c9c6b1024fc738022ed72a4714`.

| Case order | Forward m/s | Left m/s | Yaw rad/s |
|---|---:|---:|---:|
| left_strafe | 0 | 0.005 | 0 |
| left_turn | 0 | 0 | 0.015 |
| forward_right_arc | 0.0035355339059327372 | −0.0035355339059327372 | −0.010 |

One fresh exact-source 32×1,000 standing admission must pass physical and all 32 unchanged quiet gates. Each independent seed0 case is then 1×2,400 controls: 4 s startup, 24 s declared motion, 20 s requested-zero stop. The startup retains the original random physical reset, a 2 s bounded C2 target transition to canonical C40/120 and 2 s hold. The final stop must contain the existing measured ≥10 s quiet window after reference stop and its 2 s settling interval. The host stops on first rejection; later cases remain unmeasured.

## Exact delta and failure evidence

Parent directional001 source map: `373c9ea08406f6f3593279fa9fda24badf56ae86bd938e140dba7d734b5ee919`. New map: `7c64602cae91478aebd4b03adcc79f56084db7529fd1c54c0439d061075a1c58`. `SOURCE_BUILD.json` binds all 930 payloads and exactly five changed paths: directional contract, scalar metrics, runtime finalization, host description/result count, and lineage. All other 925 parent payloads are byte-identical. No runtime files are added or removed.

The metric change is solely casting `yaw_required` to a built-in bool. Runtime JSON saves recursively normalize NumPy arrays/scalars while retaining `allow_nan=False`. Failure export now preserves the original exception even if partial trace/reference export also fails. If a computed state still contains an unsupported or nonfinite value, finalization emits a strict `failed` receipt with the original exception/traceback, the complete invalid-state representation, serialization error, null gate and `gate_serialization_valid=false`. It cannot turn malformed evidence into admission. File-I/O failures still propagate. Actual traces, old source and failed001 state are not modified.

No target, physics, control-loop or scoring threshold changes are included. The host still uses the approved exact-container cleanup and lock implementation; only the case descriptions and declared-success count change.

## Preserved contracts

Wave005 SHA is `8bacf0c61f7fdc6d43d9f9ebdb5adc643b026495bb303ec3c18157d121c78893`. Its governor/filter, 2 s swing, 7 mm lift, horizontal timing, qualified liftoff, bounded landing, preload and stopping state remain unchanged. The contract binds the frozen 846/849 observation-compatible state and runtime joint-order schema; this scalar runner emits no device actor packet or PPO update.

The exact full selected C study geometry/accurate properties, 550 asset payloads, pinned 16-file runtime, named joints, RS05/PD settings, TGS16/1/external-force setting, random reset, contact capacities, original telemetry and 400 Hz observer remain unchanged. SDK quaternion is XYZW; the existing quiet scorer receives the correctly converted WXYZ channel. Body forward is −Y, left +X, yaw +Z.

Requested, admitted target, filtered reference and actual measured navigation twist remain separate raw channels. Translation requires both signed command-axis position progress and signed measured velocity at least half the request. Turning requires both signed actual heading progress and reported body gyro at least half the request. Pure turn requires no translation, with the same explicitly proposed ≤10 mm planar excursion bound. All six legs must have qualified actual ≥2 mm flights and landings; at least five distal supports, no forbidden contact/reset, zero residual lag, original finite/torque limits and measured quiet are required. The original 50 Hz full-interval world position/velocity consistency bound stays **5 mm**. Complete 400 Hz pose, rates, joint angles, torques and parity/counters are retained as independent diagnostics; a favorable diagnostic cannot replace a failed original gate. Known joint-rate/angle bias remains unresolved, and this is not hardware-startup admission.

## Verification and dispatch

`tests_001.log` records 16 passing CPU tests. They include all inherited metric/host checks adapted to the declared three cases, explicit reverse rejection by the new CLI contract, strict full passing/rejected JSON payloads for every case, the actual001 NumPy gate failure, unsupported/nonfinite fallback, and the actual exception handler when reference export also fails. These are synthetic/read-only regression results, not new physics. `INTEGRATION_PREFLIGHT.json` verifies all 930 source hashes and pinned16 runtime, all three named identities, rejection of actual directional001 standing admission and rejection of failed-quiet synthetic admission.

```sh
PYTHONDONTWRITEBYTECODE=1 uv run python -B -m unittest discover -s tmp/reference_directional_adapter_002 -p 'test_*.py'
```

The build/preflight scripts are one-shot and reject existing output. Do not rerun them inside frozen preparation. Root owns transfer, complete remote source verification, workload/coordination checks, both locks, prior-job cleanup/restoration, a new forecast pause and dispatch. Suggested distinct remote names are `reference_directional_source_002` and `reference_directional_002`:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B /home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_directional_source_002/tools/launch_directional_physics_spark.py --source /home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_directional_source_002 --output /home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_directional_002 --isaaclab /home/orionh/IsaacLab
```

The outer guard puts source/tools on PYTHONPATH and binds the exact source map/host. There are at most four phases, each bounded to 600 s, with 90 s AppReady deadline, 45 s traceback and unbuffered output. Readonly source/study mounts, exact owned IDs, unknown-inspect cleanup uncertainty, source/asset audits on failures and independent forecast restoration remain mandatory. Root supplies a finite outer deadline and restoration fallback. No GPU dispatch is claimed here.

Expected evidence: fresh standing admission/readbacks, per-case state, full/partial pre-reset trace, requested/reference/actual channels, complete reference states, startup and resolved configuration, 400 Hz trace/review (8,001 standing samples; 19,201 per complete directional case), container/job logs, contact-overflow audits, source/study hashes and terminal campaign status. Preserve actual results in a new wrapper; never rewrite this preparation or reverse001 evidence.
