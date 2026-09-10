# Preserved direct PPO: bounded native comparison

This native002 packaging successor preserves the frozen native001 runtime.
The only builder addition copies the preserved BSD license referenced by
`caps_ppo.py` into `tools/inputs/RSL_LICENSE`. The native001 inventory remains
in `PARENT_NATIVE001_SHA256.json`; no frozen native001 file was edited.

This successor keeps the original 315/318 direct-position policy and formal
0.04 rad per 20 ms target-slew profile. The cold parent is the 589-file source
`4671eab012aaf79ce49769cfdc1667142237a4b4a45966fd0399a9caa900c51b`.
The original checkpoint is
`1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8`.
No old source, checkpoint, production package or GPU job was modified here.

The measured cold policy tracks useful commands but has poor quiet standing and
8.91–11.41% requested-torque saturation, with peaks 5.78–7.05 Nm. Applied torque
remains clipped to 1.6 Nm. This experiment tests whether more frequent stand/stop
experience and policy-mean smoothness regularization improve those deficiencies.
It does not assume that CAPS resolves the actuator mismatch.

## Fixed behavior and explicit allocations

`caps.py`, `caps_ppo.py` and `curriculum.py` are byte-identical to the reviewed
20-file training proposal (`4befdbce…af06e2`). Its actual RSL 5.0.1 CPU regression
already completed two updates in ordinary PPO, zero-CAPS and positive-CAPS
branches. Ordinary/zero actor and critic outputs were bit-exact; all three
checkpoints strictly reloaded. That learning regression was not repeated here.
The new checkpoint finalizer was tested by loading that existing real checkpoint.

One new source is reused for every phase. The new explicit allocation protocol
supersedes the earlier proposal's 128×256 setting:

| Allocation | Environments | Controls/update | Updates | Simulated time/replica |
|---|---:|---:|---:|---:|
| Positive-CAPS integration smoke | 32 | 24 | 2 | 0.96 s |
| Independently dispatched curriculum pilot | 1024 | 24 | 50 | 24 s |
| Independently dispatched curriculum + CAPS pilot | 1024 | 24 | 50 | 24 s |

The pilot batch and experience budget match previously executed direct-PPO
repair runs. Native 20 s episode timeouts and randomized initial episode ages
remain; 24 s is not a promise of one uninterrupted trajectory. The matched cold
stop evaluations independently exercise complete command/quiet windows.
There is no automatic long continuation or warm start from the smoke result.
Both pilot branches start afresh from the same original actor, critic and
normalizers, with std 0.10, new Adam moments, learning rate 5e-5 and entropy 0.
The original adaptive learning-rate schedule remains.

One quarter of rows receive zero targets. Other rows alternate 8 s movement and
8 s zero targets, with different phase offsets and balanced random body-frame
bearings, both yaw signs, and combined translation/yaw. Native command slew is
unchanged. The parent's `evaluation=True` flag disables only random command
sampling in this wrapper. Existing reward, observation, action, reset and
termination methods otherwise run normally. The exact reward callback was
previously tested to score the current observed command before advancing it.

The pilot branch difference is only CAPS weights: 0/0 versus 0.1/0.1 temporal and
spatial. Both penalize deterministic actor means, not Gaussian action samples.
Temporal pairs exclude resets and command changes. Spatial neighbors preserve
all commands and previous-action entries. Neighbor sigma corresponds to
0.015 rad/s gyro (observation scale 0.25), 0.01 projected gravity, 0.005 rad joint
position, and 0.05 rad/s reported joint velocity (observation scale 0.05), in each
of five history frames. The independent noise stream is clipped at 3 sigma.
Synthetic neighbors do not update normalizers or consume the action RNG.
They are local observation perturbations, not asserted physical trajectories.

## Physical comparison scope

Exact old C asset, named stance, masses, PD 30/0.6, 16/4 TGS iterations,
external-forces default false, 2.5 ms × 8, action scale 0.5, noise 1/filter off,
reward weights, native motor demands/terminations and standing gate are retained.
This is not source009's force/iteration profile. Only environment count changes
at the declared allocation site. The historical 0.03 diagnostic remains separate.
In particular, native direct training retains its original sustained computed
torque termination; it does not silently adopt the reference branch's per-step
requested-torque gate. Applied 1.6 Nm cap is unchanged and checked on every
captured control endpoint. Requested saturation remains raw failure evidence.

Training adds detached pre-reset per-row command, reward, target-motion, raw SDK
velocity, position, contact and torque summaries at 50 Hz; full joint values for
eight trace rows and every terminal/timeout row are retained. This adds memory
and timing overhead that the actual smoke must measure. It is not a 400 Hz
torque or sensor-fidelity qualification. A nonfinite observation/audit, applied
cap miss or wrong checkpoint aborts the bounded phase and preserves evidence.
Ordinary finite native episode failures remain learnable resets, not acceptance.
Optimizer updates and loss/timing values are explicitly counted. Saved decision
checkpoints label 2 or 10/25/50 completed updates; source/model hashes and actual
strict actor/critic/normalizer/Adam/deterministic-action reload bind the final.
Partial native model files remain unqualified if the phase fails.

## Matched evaluation and host contract

Constant evaluation is the exact existing 12-case, 48-replica, 12 s diagnostic.
The separate stop diagnostic uses the same 12 directions × four replicas:
4 s zero, 12 s movement, 16 s commanded stop. The last ten seconds are scored
contiguously against the unchanged quiet bounds: 10 mm planar excursion,
2 degrees heading, 0.03 rad/s maximum joint RMS, 0.02 rad position range,
0.002 rad target-step p95, 0.5% maximum per-joint requested saturation and
1.60001 Nm applied numerical tolerance. Any termination/timeout invalidates the
replica even if it settles after a reset. Full per-replica traces remain.
The new trace explicitly retains raw Isaac XYZW and derives WXYZ for the quiet
scorer; the old frozen constant trace's quaternion label is not rewritten.
Adjacent-angle rates are interval averages, separate from untrusted raw SDK
rates and never a substitution in the gates. Completion means data acquisition,
not that failed quiet/torque gates passed.

`direct_contract.py` and `direct_config.py` are standard-library-only and expose:

```python
verify_inputs(args)  # source, checkpoint, allocation, branch, optional smoke
runtime_arguments(phase, allocation, branch)
validate_result(directory, phase, identity, expected_checkpoint_sha256=None)
```

Smoke phases are standing → train → final_constant → final_stop. Each pilot is
standing → initial_constant → initial_stop → train → final_constant → final_stop.
A pilot requires the entire same-source smoke campaign to be completed, with
terminal integrity true and all four current accepted receipts exactly matching.
Its campaign, state and training-receipt hashes enter the pilot identity. Host
ownership/cleanup remains the separately reviewed exact cold-host supervisor;
only the host can authorize another allocation. Both input/output trees remain
immutable across completed phases. There is no implicit resume.

## Build and focused verification

Run the builder on a verified complete cold parent (no AppLauncher or GPU):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 build_source.py \
  --source /absolute/path/to/direct_omni_cold_source_001 \
  --output /absolute/path/to/fresh_direct_omni_train_source_001
```

It verifies every parent byte, copies that source once, changes the explicit
training protocol/entry glue, adds eight runtime modules plus the license, and emits the complete
new map and origin receipt. Existing entry functions and physical assignments
are AST-compared; only the declared environment-count assignment is added.
Root binds the generated source map into the separately owned successor host.

Focused local tests (including full 1600-control fake-API stop rollout, actual
pre-reset audit hook, old checkpoint reload, parser flags, allocation and terminal
smoke admission failures):

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=tmp/reference_residual_ppo_001/_deps:tmp/direct_omni_recovery_001/native_002 \
.venv/bin/python3 -m unittest discover \
  -s tmp/direct_omni_recovery_001/native_002 -p 'test_*.py' -v
```

These tests use explicit frozen local prior fixtures listed in `DEPENDENCIES.json`.
They do not claim a new actual motor, training or project completion result.
