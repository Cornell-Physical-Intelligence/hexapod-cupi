# Actual Isaac device bridge smoke, version 1

This is a short, bounded integration test prepared for root review and dispatch. CPU tests have passed; no CUDA or Isaac execution of this bridge has occurred. It does not admit PPO, complete Stage 2, establish a new quiet-stand admission, or validate native velocity fidelity.

## Exact inputs

- Frozen reference physics source009: 926 files, map `04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e`.
- Completed source009 campaign, fresh standing admission, wave state and complete 550-file read-only admitted package: exact receipt hashes in `smoke_contract.py`.
- Frozen observation005: 160 files, manifest `22402b0079b0b5ac0acdd4e1d7e819fe206b20ee1883077f7de6eaca55808a63`, including the exact wave005 tensor state machine and its scalar oracle.
- This adapter's complete freeze manifest. The outer host must pin its manifest hash; the entry point rechecks all adapter bytes before and after execution.
- Exact four installed sensor source modules, hashes and local source copies in `sensor_source_contract.json`. Installed code mismatch stops the test.

The pinned 16-file C-study runtime, source009 body geometry, named stance, masses, PD gains, motor limits, 16/1 TGS solver settings and external-forces setting remain unchanged. The source009 environment builder is called directly. No old actor or checkpoint is loaded. The scalar wave generator is replaced only after startup by the separately versioned tensor implementation, with zero requested twist and zero residual action throughout.

## Bounded physical and measurement sequence

The only permitted allocations are 1 or 32 environments and exactly 264 controls at 20 ms. Source009's inherited randomized physical reset is retained. Its exact C2 canonical-target startup runs for 2 s, followed by 2 s settling. All 264 controls retain the source009 400 Hz observer and its exact final-substep versus pre-reset angle/rate/torque/pose checks. After the 200-control startup, the next 64 controls use the measured settled state to initialize and run BatchWave005's zero-command hold.

The original done predicate executes once. A device snapshot is taken before any automatic reset, preserving those exact done flags. A malformed, nonfinite, stale or terminal row aborts before it can feed the next reference update. Post-startup hold additionally requires all six distal contacts, no nonfoot contact and requested/applied torque at or below 1.6 N·m at every recorded 400 Hz substep. These short checks grant no 10 s quiet admission.

Sensor freshness is read from actual Warp `_timestamp`, `_timestamp_last_update` and `_is_outdated` arrays after ordinary `.data` lazy update, using `wp.to_torch`. All 14 sensors must advance by exactly eight sequential float32 additions of 2.5 ms. Age is the measured cache-clock difference; it is never supplied as an assumed zero. An unknown or stale row latches invalid. This smoke aborts any physical reset; selected sensor-clock reset integration for future PPO is not implemented or claimed.

The new actor/critic packets have explicit widths **846/849**, with complete wave005, landing/unloading, residual and history state. Seed time is 4.0 s with step index zero; subsequent packets pair the executed reference/controller state with the next physical sample. Five-frame history has explicit validity. Raw SDK joint velocity and the separate 20 ms position-difference average, interval duration and validity are preserved. Native rate consistency remains unqualified. New observation values are instrumented simulator state; inherited old observation noise is not injected into this new packet and this is not yet a training/noise contract.

At completion, the device telemetry is independently compared with the frozen NumPy source009 reader at the same final simulator state. Unknown contact coordinates remain raw NaNs with explicit masks. Timing includes synchronized reference/target setting, the physical step and nested device snapshot, and observation encoding/history. The physical-step timing still includes legacy CPU telemetry copies, source009's full 400 Hz observer, and sensor-clock evidence copies. It is a measured CUDA hold cost, **not a prediction of PPO throughput or dynamic walking throughput**.

## CPU evidence

Ten tests pass. They cover exact sensor-clock recurrence and duplicate reads, missing updates/stale caches/reset invalidation, dtype and finite checks, original pre-reset done ordering/restoration, immutable input corruption, allocation limits, installed source bindings, and the absence of actor/body-write calls.

A self-contained 65-boundary slice of actual source009 standing controls 200–264 tests the exact reference → executable residual → next measured packet → history order at both 1 and 32 rows. The original raw trace hash and exact array slice are recorded in `inputs/PROVENANCE.json`. This replay retains real measured input values but uses explicitly synthetic freshness fields; it does not claim recorded freshness or new physical simulation.

## Invocation and host requirements

Run inside the existing Isaac container, mounting this adapter, the observation bundle, source009 and the admitted study package read-only. Output must be fresh and outside every input.

```sh
/workspace/isaaclab/_isaac_sim/python.sh /adapter/run_device_smoke.py \
  --source-root /workspace/hexapod \
  --run /qualified \
  --observation-bundle /observation \
  --package /study \
  --output /outputs/replicas_1 \
  --num-envs 1 --controls 264 --headless --device cuda:0
```

Use `--num-envs 32` only in a separate bounded allocation, with a new output. `--preflight-only` validates all bytes and identities without AppLauncher or CUDA. Root must review and supply the established two job locks, unrelated-CUDA preflight, exact container cleanup, 90 s AppReady/600 s phase limits, contact-log overflow rejection, forecasting pause/restoration and pinned adapter hash. This directory has no GPU host launcher and dispatches nothing automatically.

Expected completed outputs include `state.json`, `environment.yaml`, `solver_comparison.json`, `sensor_clocks.npz`, `physics_substeps.npz`, `physics_control_integrals.npz`, `physics_substep_review.json`, `last_device_sample.npz`, `last_reference_state.npz`, `final_observation.npz`, `observation_schema.json` and `timings.json`. Failure preserves available pre-reset snapshots, clocks, reference state and substeps, marks failure, and rechecks input integrity. Completion must require `bridge_passed=true`, 264 controls, exact 846/849 shapes, source integrity and pinned runtime identity; no automatic successor training.
