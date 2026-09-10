Lazy wave005 prediction candidate

This CPU-only successor skips endpoint prediction on controls with no active launching row. Every dynamic input check and state update remains present. If any row launches, the complete original batch prediction runs, preserving all 198 float32 position-rounding steps and both polynomial clocks. The original source, observation bundle, current PPO002 campaign, and physical gates are unchanged.

`lazy_prediction.py` verifies the complete frozen 86-file wave005 dependency inventory and exact step source, then compiles a single explicit replacement into a separate subclass. It does not modify the parent file, class, or module. `step.patch` shows the complete method difference. The subclass reports a new implementation identity with the parent and successor hashes. The frozen observation builder will not silently accept this identity; later adoption needs an explicitly reviewed new consumer binding. No old checkpoint compatibility or policy admission is granted.

Four focused tests passed. They compare every executable state tensor exactly, including 2,200 controls of actual009 contact/stop evidence, the unqualified RR contact return, all 11 landings, asynchronous moving/quiet rows, partial resets, reversed runtime joint order, and invalid-row failure latching. The actual009 replay changes prediction calls from 2,200 to 11 while preserving the final quiet hold. The synthetic mixed test covers landing, unloading, stop and asynchronous reset states. These are recorded-input or synthetic CPU tests, not new physical rollouts.

The complete step-call timing includes validation, geometry, state updates and output copying. Three measured trials follow one warm-up, with parent/lazy order alternating. Input construction and state assertions are outside the timing. CPU execution uses float64, one Torch thread and inference mode.

| Replicas | Recorded workload | Parent ms/control | Lazy ms/control | CPU ratio |
|---:|---|---:|---:|---:|
| 1 | actual_quiet | 2.829 | 2.006 | 1.41× |
| 1 | actual_asynchronous | 2.992 | 2.092 | 1.43× |
| 32 | actual_quiet | 3.784 | 2.577 | 1.47× |
| 32 | actual_asynchronous | 3.773 | 2.782 | 1.36× |
| 128 | actual_quiet | 6.197 | 3.959 | 1.57× |
| 128 | actual_asynchronous | 6.154 | 4.426 | 1.39× |

The actual asynchronous workload uses distinct recorded009 phase offsets; the 128-row case repeats the 32 offsets. Only 14 of 80 controls contain a launch in those batches. At one replica the moving window contains one launch. The quiet windows contain none. These cases do not establish a general throughput multiplier. CUDA `launch.any()` introduces a synchronization point, and no Spark/CUDA/simulator/observation/RSL speedup has been measured for this successor. Keep the old live campaign immutable; compare an explicitly bound device smoke before adopting this change.

Reproduce from the repository root with the existing NumPy/Torch CPU environment:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tmp/reference_lazy_prediction_001 -p test_lazy_prediction.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tmp/reference_lazy_prediction_001/benchmark.py
```

The tests require the exact parent at `tmp/reference_tensor_wave_005_001`. On a clean checkout it can be copied from the published `artifacts/omni_diagnostics_2026-09-09/reference_policy_observation_005_001/owner/reference` tree into that unused path. The loader rejects any changed payload, symbolic substitution, or wrong already-imported geometry runtime. Reproduce into a new output copy rather than rewriting this frozen bundle's log/report files. The parent manifest is 5f46b6f99c172bb037004e758f4f3715d41da42725abd10bfd313ee2b84fe286; the actual009 input trace is ab7640bce1d98ce4d175882537ef993c68a620987adff88b7c919dc1a0c6c0e8.
