# Exact reference003 → reference004 lift and measurement delta

This is a frozen CPU-reviewed preparation. It makes no new physics or completion claim. The full fresh 32 × 1000 standing admission remains mandatory before the 1 × 2400 zero-residual wave. The simulator, physical asset, startup, controller, host and all acceptance gates remain unchanged.

Parent 924-file source manifest: `7c75f0372abcea9eace3a280a2c204164179b8c433fc9d6280f60dd189737e24`. Target 925-file source manifest: `a433e529d29d5360c828b406a3dfd769e078d69fc9deaf03f4fe110eb6fa7a63`. Changed files are `tools/wave_reference.py`, `tools/run_reference_physics.py` and `source_origin.json`; `tools/physics_substeps.py` is added. Every other parent byte is preserved. The target map and lineage are under `source_identity/`; no full parent source is duplicated.

The reference lift changes from 5 to 7 mm after actual003's RF foot lifted only 1.445 mm relative to its immediate preflight sample and failed the unchanged measured 2 mm gate. The complete owner freeze under `wave003/` preserves 23 passing synthetic command cases, 16 passing tests, the actual RF failure decomposition and the rejected 8 mm alternative. Those tests establish target feasibility, not actual clearance, torque or walking success. The previous C2 landing's bounded target overshoot and return remain part of the controller; there is no new landing correction or relaxed contact requirement.

Actual003 also has an unresolved discrepancy: 36.402 mm forward link displacement versus 25.879 mm integrated reported link velocity over its partial 8.34-second moving window; the full-vector difference is 10.541 mm. The old 50 Hz metric and its 5 mm gate remain unchanged. New 400 Hz data is diagnostic and cannot silently replace that gate. Installed PhysX getter/kernel readback confirms the intended link-position/link-velocity pairing, so no frame conversion or metric correction has been justified yet.

The observer reads each of the eight actual 2.5 ms post-physics `scene.update` calls, along with initial state, link/COM position and velocity, raw XYZW pose, joint velocity and computed/applied torque. It checks counters, timestamps and equality of the eighth sample to the unchanged control sample, then restores the original method. Backend-internal decimation, missing/extra samples and nonfinite state fail closed, retaining available raw evidence. Per-control substep integrals and torque extrema are reported separately. There is no physical-state write, extra simulation step, controller action change or acceptance substitution.

`integration_review/` retains the exact owner freeze, source preflight, installed SDK excerpts/hashes, observer source and six passing focused tests. The independent PPO agent also ran all six and found no remaining first-run blocker; its immutable review receipt is kept separately when available. The tests cover substep count/order/timing, endpoint identity, method restoration, backend rejection, intentionally aliased velocity, a hidden torque spike and initial/later nonfinite evidence retention. They do not diagnose the actual discrepancy without runtime data.

Verify the virtual exact source without materializing it:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 reconstruct_source.py --parent-source /path/to/reference_source003
```

Add `--output /fresh/reference_source004` to materialize a new immutable source when needed. The recipe verifies all parent files/no extras, the exact permitted overlays and every final byte. Obtain the parent through the published reference003 reconstruction chain. The observer tests can then be replayed with NumPy installed:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/fresh/reference_source004/tools python3 -B -m unittest discover -s integration_review -p 'test_physics_substeps.py' -v
```

This bundle does not launch a job, pause forecasts or claim compute ownership. Root supplies the separate guarded dispatch, raw result, source/asset audit and pause030 restoration receipt. Terminal evidence must be added in a new wrapper; this preparation remains immutable.
