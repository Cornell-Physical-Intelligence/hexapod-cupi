# New C-study target-velocity candidate 001

This is a new, unqualified controller lineage for the admitted serial C study robot. It is ready for a guarded Spark standing admission and integration smoke. It has **not** trained or run in Isaac yet. Stage 2 remains incomplete, and no old checkpoint may be resumed into this controller.

The first useful experiment is **formal_004, acceleration 8 rad/s², zero actor mean, Gaussian action std 0.05**, with the original 1.6 N·m actuator cap, Kp 30, Kd 0.6, observation noise 1, and position filter disabled. The acceleration value is an explicit experimental controller setting, not a measured motor capability. Root owns packaging, workload admission and all GPU dispatch.

## Executable action and observation contract

`velocity_action.py` maps 18 normalized actions to desired joint-target velocities in the articulation's observed joint-name order. It then bounds the next velocity by acceleration and conservative joint-limit braking reserve, and integrates the target position. It emits position, velocity, acceleration and intervention flags. The full persistent state is target position and velocity; a joint limit cannot accumulate an invisible position error or unlimited velocity demand.

At each control interval, `q_next = q + dt * v_next` and `abs(v_next - v) <= a * dt`. These are discrete target-knot bounds. A zero-order-hold position drive is retained; there is no claim of continuous acceleration/jerk at motor commands. Position-target velocity feedforward stays zero, matching the direct-PPO PD semantics. Physical joint velocity, torque, slip and stability must be measured separately.

| Separate profile | Target step at 20 ms | Target velocity bound | Acceleration setting |
| --- | ---: | ---: | ---: |
| `diagnostic_003` | 0.03 rad | 1.5 rad/s | 8 rad/s² |
| `formal_004` | 0.04 rad | 2.0 rad/s | 8 rad/s² |

The 0.03 intervention is not a physical motor-speed limit. The 0.04 profile is the common formal software bound. They require separate plans, admissions and checkpoints; their metrics must not be pooled.

`candidate_env.py` appends 36 noiseless, controller-known state values to each original 63-wide frame: target position minus nominal pose (18), then target velocity divided by the selected bound (18). Five 99-wide frames give **495 actor inputs**. The critic gets those 495 plus navigation-frame linear-velocity truth, giving **498**. Actor sensor noise remains on the original IMU/encoder fields only. Repeated reads within one step are identical; partial resets overwrite all five frames for reset robots. The controller performs a canonical physical reset before exposing its first state, avoiding imported zero coordinates outside soft joint limits.

Zero **action** decelerates the target velocity to zero and holds the reached target. Zero **body command** does not mask the actor, lock joints, teleport the body or disable feedback. A learned actor can therefore correct disturbances, but it can still jitter or drift: successful quiet standing is a measured requirement, not guaranteed by the action parameterization.

## Checked CPU evidence

Run from the repository root:

```sh
.venv/bin/python -m unittest discover -s tmp/omni_velocity_candidate_001 -p 'test_*.py' -v
.venv/bin/python tmp/omni_velocity_candidate_001/cpu_noise_study.py
```

The controller, actual adapter methods and checkpoint tests pass, including named joint permutation, saturation/braking, reversal, 4,000 float32 noise steps, inference-mode reset compatibility, partial resets, history, reserved identity rejection, immutable checkpoint files/sidecars, strict Torch state readback, and pre-reset motor capture. Independent SciPy yaw/roll fixtures verify that actual Isaac XYZW quaternions are explicitly converted to the existing quiet-review WXYZ format. The historical shared diagnostic helper's mislabeled quaternion and absolute-position target fields are corrected only in the new candidate snapshots.

[CPU noise results](cpu_noise_report.json) use 32 replicas, 10 seconds and the named C joint limits/40°–120° stance. Zero actions hold exactly in both profiles. With std 0.05, target-velocity RMS is 0.0686 rad/s at 0.03 and 0.0842 rad/s at 0.04; final target drift RMS is 0.0340 and 0.0462 rad respectively. These are target-space statistics, not physical gait results. This drift is why the next run calibrates the actual actor's exploration against motor/contact evidence instead of assuming a small std is safe.

## Frozen source integration

Overlay these **six new files** into `tools/` in a fresh archive of reviewed main. Do not replace existing main source, pinned runtime, production shims, archived snapshots or benchmarks:

- `velocity_action.py`
- `candidate_env.py`
- `candidate_runner.py`
- `candidate_diagnostics.py`
- `candidate_asset_audit.py`
- `train_velocity_candidate.py`

The entrypoint derives its C asset/configuration setup from the verified main `tools/train_length_study.py`; [BASE_SOURCE_SHA256.json](BASE_SOURCE_SHA256.json) records that provenance. Validation alone uses the existing USD inertia writer. Probe/train/evaluate use the candidate's read-only mass, COM, tensor, body/joint layout and contact-report audit, supporting a read-only admitted asset mount.

Use the **previously admitted full C plan/package**, not main's generic length grid stance. The preflight checks the exact serial C URDF SHA `e916b1ebbb2720c90f23974bbffa5da120b32d2e5409fe419fb1492b8556746c`, named femur 40°/tibia 120°/coxa 0°, plate height 0.13053251856352807 m and reset height 0.13653251856352808 m. It requires 32 validation robots, 1,000 standing steps, 1,024 training robots, 48 evaluation robots, physics dt 0.0025 and decimation 8.

The plan must declare `omni.architecture = "c_serial_omni_target_velocity_v1"`, `omni.velocity_candidate = {"profile": "formal_004", "max_acceleration_rad_s2": 8.0}`, observation width 99 per frame/495 actor/498 critic, and override noise 1/filter 0. Remove old `reference_controller` and `repair_training`. Any declared legacy slew must match the selected profile. The existing all-bearing command sampler and reward terms remain, with any explicit reward overrides recorded in the immutable plan. This first admission is not a reward comparison.

Code identity hashes executable Python in `tools/`, `isaaclab/`, `packages/` and the pinned study runtime, plus its JSON manifests. Source/plan/asset/stance identities bind admission, calibration, runner smoke and checkpoint. Checkpoints embed their contract and have SHA sidecars; old 315/318 schemas, another rate profile, modified sources, colliding identity keys, missing sidecars and existing output checkpoints fail closed. The copied reviewed inference-buffer compatibility helper changes only writable buffer storage for strict reload; it preserves tensors and optimizer state.

## Guarded Spark sequence

The host launcher owns locks, unrelated CUDA checks, forecasting pause/restoration, timeouts and immutable output directories. Common arguments below are illustrative container paths; the root launcher supplies its verified mounts.

```sh
python tools/train_velocity_candidate.py --package /study --variant f050_t060 --stance-index 0 --mode validate --output /outputs/flat --headless --device cuda:0
python tools/train_velocity_candidate.py --package /study --variant f050_t060 --stance-index 0 --mode probe --admission /outputs/flat/admission.json --output /outputs/probe --headless --device cuda:0
```

Probe runs 250 deterministic zero-mean steps and 500 stochastic std-0.05 steps at zero body commands. Every replica retains physical joint RMS/range, finite-difference rate, target motion, body/yaw drift, requested/applied torque, joint-specific saturation and contact/failure evidence. Nonfinite state or applied torque fails immediately. The zero-mean trial must pass the short quiet window; sampled exploration must pass motor/contact bounds, including no individual joint above 0.5% saturation. This does not substitute for the fixed 30-second quiet review.

Only after both calibration trials pass does probe perform 100 zero-action settling steps and **two actual stand-only RSL PPO updates**, capture each optimization step, save an immutable `policy/schema_smoke.pt`, reload it strictly, and require identical deterministic actions on fixed observations plus finite, passing motor evidence. PPO uses fresh Adam, learning rate 1e-4 with the existing adaptive schedule, entropy coefficient 0, 24-step rollouts and the existing 256/256/128 MLP. `initialization.json` and `runner_smoke.json` report actual initial/learned std, optimizer entries and load compatibility.

No walking run is launched automatically. After review, a separate scratch pilot may use:

```sh
python tools/train_velocity_candidate.py --package /study --variant f050_t060 --stance-index 0 --mode train --admission /outputs/flat/admission.json --calibration /outputs/probe/calibration.json --runner-smoke /outputs/probe/runner_smoke.json --iterations 50 --output /outputs/pilot --headless --device cuda:0
python tools/train_velocity_candidate.py --package /study --variant f050_t060 --stance-index 0 --mode evaluate --admission /outputs/flat/admission.json --checkpoint /outputs/pilot/policy/final.pt --output /outputs/post --headless --device cuda:0
```

Training rejects `--checkpoint` even from the stand smoke. It starts a separate zero-mean/std-0.05 scratch actor and writes `initial.pt` before all-bearing PPO, then `final.pt`. Evaluation runs 12 direction/arc cases with four replicas and pre-reset traces, plus the existing quiet-stand and stop-to-stand review. It audits all 495/498 observation fields/history and noiseless executable state. Run the same evaluate command on `initial.pt` into a fresh directory for a matched baseline. Promotion requires each direction/stand/stop evidence, not only a median; the full 154 cases, transitions, paths and visual comparison with Benchmark 1 still follow any promising pilot. Video/full qualification support is deliberately not claimed by this first candidate entrypoint.

The next decision is whether the physically admitted velocity actor learns useful motion while reducing target oscillation. If integration drift dominates, compare a directly regularized policy mapping (such as CAPS) or a support-aware reference/residual action under a fresh explicit contract. Neither is selected by these CPU results. Later terrain/perception must keep measured/proprioceptive controller state, body-twist commands and explicit sensor validity/age; simulator truth must not silently enter the actor.
