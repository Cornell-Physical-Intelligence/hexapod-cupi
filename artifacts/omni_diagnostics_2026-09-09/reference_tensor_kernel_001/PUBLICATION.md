# Batched reference geometry: verified CPU boundary

Root independently replayed all 12 tests and verified the 21-file frozen bundle. The [original methods, limitations and timing report](README.md) remain unchanged. Named batched FK/IK/center-of-mass calculations and a partial geometric observation block match their frozen scalar oracles, including invalid masks, joint order, per-replica resets and failure latches.

Local CPU geometry timing at 128 replicas improved from 85.81 to 1.97 ms for the measured operations; this excludes the contact state machine, full observation/history, transfers, physics and PPO. The 233-value partial state block cannot replace the complete 740-value actor input. Float64 is required, and CUDA/Spark throughput is not measured. Only the explicit wave002/wave003 source bindings are admitted by this CPU module; the later horizontal timing needs a successor binding.

This is no walking or policy admission. It addresses the [measured CPU bottleneck](../reference_cpu_preparation_001/profiling/README.md) before large training allocation. The next boundary is the complete measured-contact state machine with scalar parity and asynchronous reset/contact tests.

Replay the self-contained tests without rewriting the original timing report:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run python -B -m unittest discover \
  -s artifacts/omni_diagnostics_2026-09-09/reference_tensor_kernel_001 \
  -p 'test_*.py'
```

The original README retains staging-path examples from preparation; use the publication command above. No production or pinned runtime file was changed.
