# Canonical Phase B preparation — CPU proposal only

Read `PROPOSAL.md` for the decision and `PROPOSED_CONTRACT.json` for exact proposed values. This bundle contains CPU geometry, statics and numerical-servo studies, the actual tools-disabled Fable5.1 MAX final response, its independent corrections, and exact source/input receipts. It has no GPU launcher and grants no physical, policy or hardware admission.

Native003 has now completed the exact-model import and eight zero-G steps. `PHASE_A_INPUT.json` binds the actual receipt while retaining both no-admission flags. Its CUDA-free-memory observation does not identify native SDF allocation; measure during-run host RSS/available memory and CUDA free memory before scaling 1→32. No geometry simplification is proposed.

The useful CPU result is a neutral first stance candidate, requiring 0.581644 N·m in an ideal vertical-force minimax calculation. The candidate servo is Kp12 with name-bound inertia-derived damping, capped at1.6 N·m and a declared speed envelope. Neither result is a native standing pass. Exact authored tibia geometry has a distal +X toe; the historical +Y sphere classifier is incompatible.

## Reproduce from the repository root

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python3 -m unittest discover -s tmp/updated_native_phase_b_design_001 -p 'test_*.py'
python3 -S tmp/updated_native_phase_b_design_001/verify_bundle.py
```

Nine tests pass. They independently check mass/energy/gravity derivatives, static world force/moment equilibrium, supported joint torque by virtual work, numerical free-response decay, and separate demand/ceiling/applied/power behavior. They do not run Isaac. `tests_final.log` is the final test receipt; earlier `tests.log` is retained. `servo_candidate.py`, `stance_probe.py` and `inertia_probe.py` refuse to overwrite their reports.

Geometry is referenced by the exact nine-file map in `INPUTS.json` and is not duplicated here. The proprietary installed native Tensor API is referenced by path/hash and interpreted signatures only. `provisional_actuator.py` is an independently written CPU proposal, not the historical four-bar actuator implementation.

Root must review/adopt a new runtime source before native effort tests. Proposed publication changes are in `PUBLICATION_PROPOSAL.md`; no tracked files or GPU state were changed by this preparation.
