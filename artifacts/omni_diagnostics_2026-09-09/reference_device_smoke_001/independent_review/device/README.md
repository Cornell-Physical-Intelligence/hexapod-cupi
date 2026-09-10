# Independent review of the bounded device bridge001

**No concrete code or CPU-contract blocker was found for the separately guarded short device smoke.** This review binds bridge freeze `be4870a8f7ca0857ba50ffd3a0c92ac5925b9a816767d66d826af7b59957aa0e`. All 18 files match, and all **10 CPU tests pass** independently. Read-only preflight passes for both 1 and 32 replicas against exact source009, observation005, completed admission receipts, and the full 550-file asset package. Frozen bundles and main were not edited; no GPU job ran during review.

The scope is 264 controls: 200 controls of the unchanged canonical C2 startup/settle, followed by 64 zero-command hold controls. It is not a full quiet-standing, walking, PPO, or velocity-fidelity admission. The root-owned host/guard and actual installed CUDA execution remain separate checks.

## Sensor timestamps and measurement provenance

The four copied installed sensor modules match their declared SHA values. Their Warp kernel advances float32 `_timestamp` by physical dt, and successful lazy buffer update sets `_timestamp_last_update` to that timestamp and clears `_is_outdated`. `ContactFreshness` reads those device arrays through `warp.to_torch` after ordinary `.data` access. It does not call sensor reset/update or step simulation. All 14 sensor clocks must advance by the exact eight repeated float32 additions of 2.5 ms; current/last-update equality, finite nonnegative values, and a clear outdated flag are required. Missing increments, stale caches, clock reset, wrong dtype/period, and nonfinite values reject and latch affected rows. Zero contact age is an observed cache difference, not a caller-filled timestamp.

This proves consistency with the inspected installed-source contract and CPU fixtures, not execution of the actual Warp-to-Torch bridge on CUDA. Actual cadence/readback and property availability are precisely what the proposed smoke must measure.

Raw SDK joint velocities, raw angle samples, explicit XYZW poses, and COM versus root-link velocities retain separate meanings. The 846/849 encoder exposes the raw SDK joint-rate channel and a separately valid 20 ms position-interval rate. The final artifact saves both; no rate is substituted into the physical quiet metrics or simulator state. Missing contact points keep a raw NaN/validity pair and are not replaced by toe anchors.

## Pre-reset sequence, targets, and observer

`capture_before_reset` calls the original done predicate once, passes clones of those exact device flags to the snapshot reader, and restores the original method even if capture raises. `require_one` checks one new capture and equality with returned termination/truncation flags. The main loop stops on an invalid measured/freshness row or terminal result before that sample can feed another reference step. This is not post-reset continuation.

The startup preserves the exact emitted randomized initial target, moves through the unchanged canonical quintic, and checks the named nominal target against the actual articulation default. After 200 controls, the tensor reference resets from the actual settled packet; history seeds at 4.0 s with step index zero. Each following reference/target emission precedes the corresponding physical step and matching observation index. Raw residual actions and requested body twist remain zero. Source, geometry, named order, reference P/V/A, and full target guards remain bound to the frozen source009 and residual002 code.

The original source009 `PhysicsSubstepRecorder` remains unchanged (SHA `8d52e0f56e3671a18c194b3f5d623cd95ca9187c5d58289001a74868c8d0c091`). It observes all eight real scene updates, checks counters/timestamps, captures raw joint angle/rate and requested/applied torque, and requires equality with the control endpoint. The post-startup smoke checks every substep's requested/applied torque against 1.6 N·m and all six distal contacts with no forbidden contacts. Partial state, clock, and substep evidence is preserved on failure; failed export or changed source cannot produce a completed state.

The existing environment still contains legacy CPU telemetry copies, and the smoke adds synchronized timing/readback. Its measurements cannot be presented as PPO throughput or a fully optimized device rollout. The final NumPy-oracle comparison and schema check provide additional actual-run evidence if the smoke completes.

## Reproduce the CPU review

- [tests.log](tests.log): ten independent passing tests, including actual009 standing-slice reference/residual/history replay at 1 and 32 replicas.
- [verification.json](verification.json): all source/API identities and both preflight results.
- [verify_review.py](verify_review.py): repeatable read-only source and preflight inspection.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s tmp/reference_device_smoke_001 -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  tmp/reference_device_smoke_independent_review_001/verify_review.py
```

The previously frozen tensor/observation independent receipt remains unchanged at `tmp/reference_wave005_observation_independent_review_001` (freeze `d01aaa1ee926207f62ed4c32ed0ac0510a8ef08c2842b97705e4d6a9a4ae54d8`). This receipt adds only the bounded device-bridge source review, not GPU or launch approval.
