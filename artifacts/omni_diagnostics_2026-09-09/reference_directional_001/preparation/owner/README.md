# Four-case directional reference discriminator001

This is a bounded actual-physics preparation for reverse, strafe, turning and combined translation/turning at the proven reference's existing low command limits. It does not train PPO or admit the full omnidirectional envelope. Root approved four independent cold cases after one fresh shared32-replica standing/quiet admission. The campaign stops at its first rejected case; any later cases remain untested.

| Case | Requested forward (m/s) | Requested left (m/s) | Requested yaw (rad/s) |
|---|---:|---:|---:|
| reverse | −0.005 | 0 | 0 |
| left_strafe | 0 | +0.005 | 0 |
| left_turn | 0 | 0 | +0.015 |
| forward_right_arc | +0.0035355339059327372 | −0.0035355339059327372 | −0.010 |

Every case has the original200-control/4s startup (2s bounded C2 transition from the randomized physical reset target to canonical C40/120, then2s hold),1,200 controls/24s motion and1,000 controls/20s zero-command stopping. Full length is2,400 controls/48s. The20s stop retains enough time for the generator's finite stop and the original measured≥10s quiet window after its additional2s settling. Every case is a new one-robot environment with seed0; the previous case's physical or generator state is never reused. Only its declared command differs.

## Source and state identity

The exact parent is reference009, map `04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e`. `SOURCE_BUILD.json` binds all930 source payloads: four added tools and changed lineage metadata; all925 other parent payloads are byte-identical. The full C study geometry/accurate properties,550 admitted asset payloads, pinned16-file runtime, named joints, RS05/PD settings, original random reset, TGS16/1 plus external-force timing, contact buffers and complete400Hz observer remain unchanged. There is no origin-state injection, pair-support generator or production-model modification.

`wave_reference.py` is exactly wave005 (`8bacf0c61f7fdc6d43d9f9ebdb5adc643b026495bb303ec3c18157d121c78893`). Its smoothing,2s swing,7mm vertical lift,.8 horizontal timing, preload preservation, qualified-liftoff/landing/stop state and target P/V/A checks are unchanged. The new contract binds the wave005 observation-owner freeze `22402b0079b0b5ac0acdd4e1d7e819fe206b20ee1883077f7de6eaca55808a63` and actual runtime-order schema `c2ab56b61c8ca56412090cd42cf109c90576bd5e78602a17eb29ada6d0559845`. This is compatibility of the recorded scalar wave state with that846-actor/849-critic schema; this runner does not emit a packed device actor observation or test PPO.

The inherited solver protocol name is retained as configuration lineage. `directional_protocol` independently declares these phases and exact command/time cases. Each directional identity includes the exact fresh standing identity and admission hash. An actual reference009 receipt or a failed-quiet receipt cannot authorize the new source.

## Measurements and limited criteria

Raw SDK quaternions are XYZW; telemetry also retains the correctly converted WXYZ channel for the unchanged quiet scorer. Body forward is−Y, left+X and positive yaw+Z. Each50Hz pre-reset row records the requested command, admitted target command, filtered reference command and actual measured navigation twist `[-body_vy,body_vx,body_gyro_z]`. The complete reference states preserve the governor/filter and their previous measured sample. Commands have exact named values and times; silent derating is rejected. Raw world link/COM pose and velocities,400Hz joint angles/rates/torques, counters and matching control endpoints remain separate evidence.

All existing physical/contact/torque/flight/quiet requirements remain: zero residual, finite state, no terminal/reset or forbidden contact, at least5 distal supports, six distinct legs with≥2mm actual flight and confirmed landing, sampled requested torque≤1.6Nm, exact executable-reference target and final measured quiet≥10s. Existing torque startup exclusions are retained; no hardware-startup admission is implied. As in009, the formal wave screen uses its original50Hz physical gates, with complete400Hz torque/kinematics evidence reported independently; a control-rate pass must not be described as a400Hz pass without inspecting that evidence.

For a nonzero requested translation, both mean measured body-navigation velocity along the command axis and independent actual position increments projected along the corresponding measured world command direction must reach at least half the requested distance/speed. At each measured body pose, `[left,−forward,0]` is rotated into world coordinates and projected onto the ground; adjacent directions define the interval projection. This supports arcs without treating their path as fixed-heading straight motion. The original full24s **world displacement versus integrated world-velocity discrepancy must still be≤5mm**; the fixed-initial-forward diagnostic is also preserved verbatim by the existing helper.

A nonzero yaw request additionally requires both measured body yaw-rate and actual unwrapped XY heading change to reach at least half the requested signed rotation. Pure turning requires no nonzero translation; it instead has an explicit proposed≤10mm planar excursion bound during motion. Heading-minus-body-gyro-integral difference is recorded as a separate diagnostic. On tilted bodies these are related rather than identical rotation quantities, so that difference is not treated as a new native velocity-fidelity gate. Untested pure right strafe/right turn and other bearings are not admitted by these four cases. These checks establish limited motion feasibility, not accurate general path tracking.

The known0.398–0.410rad standing joint-angle/rate integral discrepancy remains unresolved. The adapter does not replace SDK rates with finite differences or infer a solver fix. Actual motion requires both measured position/heading and rate evidence; no gate is selected merely because it is favorable.

## Verification and dispatch

`tests_002.log` records11 passing CPU regressions: signed reverse/strafe/turn/arc, wrong-way rejection, pure turn without translation, pure-turn drift rejection, unchanged5mm consistency gate, command/admitted/actual channel and timestamp binding, contact/torque/terminal/quiet requirements, exact cold-case modes/counts, inherited cleanup AST and fresh standing/wave identity. These synthetic metric fixtures are not physics evidence. `INTEGRATION_PREFLIGHT.json` verifies all930 source hashes and pinned16 runtime, checks all four case identities and rejects actual009 admission plus a failed-quiet synthetic receipt.

Focused command from repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run python -B -m unittest discover -s tmp/reference_directional_adapter_001 -p 'test_*.py'
```

Preparation scripts are one-shot and reject existing outputs; do not rerun them inside a frozen owner directory. Root owns source transfer, reviewed map verification, both GPU locks, workload/coordination checks, previous-job restoration verification and a fresh forecast pause. Suggested remote source/output names are `reference_directional_source_001` and `directional_reference_001`; no dispatch is claimed here.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B /home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_directional_source_001/tools/launch_directional_physics_spark.py --source /home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_directional_source_001 --output /home/orionh/HEXAPOD_runs/mock_length_study_20260909/directional_reference_001 --isaaclab /home/orionh/IsaacLab
```

The root guard must put source/tools on PYTHONPATH and bind the complete new source map. The approved host cleanup function is AST-identical to009: exact owned-container identity, unknown-inspection failure handled as cleanup uncertainty, both locks and coordination honored, unrelated workloads protected. Source and admitted study are mounted readonly. Each of five possible phases is bounded to600s with90s AppReady deadline,45s traceback and unbuffered output; root must set a finite outer bound allowing those phases plus cleanup, with its independent restoration fallback. No automatic retry, resume, worker signal or further job is started by this preparation.

Outputs retain standing admission and per-case `state.json`, `trace.npz`, reference states, startup target contract, resolved environment and authored solver readbacks,complete observer traces (standing8,001 samples; each complete directional case19,201), all logs and contact-overflow audits, job/container receipts, source/study integrity checks and final campaign status. Failure retains the last pre-reset sample and available partial traces; missing later cases remain untested. Root should append actual results and cleanup evidence to a separate publication wrapper without changing preparation or old raw bytes.
