# Cold baseline terminal audit

This small read-only utility is bound to cold001, unit `hexapod-direct-omni-cold-001-20260910.service`, invocation `ae3b95c756db4aa4b639ee4230e8c958`, and forecast pause 053. The expected dispatch is recorded; independent proof of the unit, source, assets, container absence and restoration belongs to the root audit. No GPU, allocation, signal, or remote write is performed here.

`remote_inventory.py` requires the exact cold source/checkpoint/host identity, a terminal producer campaign and finished jobs with cleanup receipts before hashing every raw output. `fetch.py` preserves those receipts and copies each missing file once. Existing files must match; resumed corrupt files and unexpected partial files fail closed. A second remote inventory must equal the first. There is a 512 MiB transfer cap and a 256 MiB free-disk reserve. No source trees or extra raw trace copies are made.

`analyze.py` keeps the frozen historical 0.03 rad/20 ms diagnostic and fresh 0.04 diagnostic separate. It binds the same original checkpoint, all 12 scenario names/commands, four replicas per scenario and the complete 600-control trace. It compares the other declared overrides/rewards without claiming a causal training improvement. Every original per-joint/raw reward statistic remains in the original diagnostic JSON; the compact report retains all scenario windows and failure counts.

The saved trace contains only one of four replicas per scenario. Its interval-angle rates and endpoint SDK rates are different channels; reset-crossing intervals are excluded explicitly. The legacy field `quaternion_world_wxyz` stores raw SDK XYZW on this build. The analyzer neither uses it to transform measurements nor silently changes its label. This 50 Hz trace cannot establish substep torque, hardware fidelity or oscillation above 25 Hz. The 12 constant-command cases contain no moving-to-stop transition.

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tmp/direct_omni_cold_audit_001/fetch.py --output tmp/direct_omni_cold_results_001
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tmp/direct_omni_cold_audit_001/analyze.py --run tmp/direct_omni_cold_results_001/run --historical tmp/direct_omni_recovery_001/baseline/inputs/old_diagnostics.json --output tmp/direct_omni_cold_results_001/analysis_002
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tmp/direct_omni_cold_audit_001 -p 'test_*.py'
```

The analysis output must be fresh. The documented actual result paths already exist and must not be overwritten. Historical comparator SHA: `f0c5c9feba99c9785bc47b69724c501f3d87b4782b3466dd98616b0382fa9ee6`; it is a read-only reference to the separately frozen baseline contract inputs.

The seven tests include the actual complete trace and profile comparison, shifted timestamps, wrong/missing scenarios, nonfinite metrics, unsafe paths, live campaigns and wrong checkpoint identity. This is diagnostic evidence preparation, not Stage 2 qualification or training admission.
