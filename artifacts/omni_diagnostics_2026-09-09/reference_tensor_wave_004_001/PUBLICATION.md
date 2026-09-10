# Reviewed batched wave004 reference prototype

Root independently verified all 43 owner-frozen payloads and passed all ten tests. The separate seven-file review receipt verifies the same source, preserves the actual004 failure and checks 24 mixed-precision predictor cases. Both original freezes remain unchanged.

From this directory, run `PYTHONDONTWRITEBYTECODE=1 uv run python -B -m unittest discover -s . -p test_batch_wave.py` using the workspace dependencies. Original staging-path examples remain in the frozen README. The first predictor comment is stale; the code and README explicitly preserve the fixed 198-step rounded recurrence.

Local CPU timings show 91.57 to 7.62 ms at 128 replicas. CUDA throughput, input packing, complete typed observation/history, residual execution and physical behavior remain unqualified. The 332-slot controller state is not an actor width; neither the old 233-value partial block nor the 740-value observation schema can represent this successor without a new contract. No PPO or checkpoint admission is granted.
