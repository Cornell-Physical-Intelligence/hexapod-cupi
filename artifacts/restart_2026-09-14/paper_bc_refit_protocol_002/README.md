# BC refit protocol 002: relocated helper

This protocol binds the maintained helper `experiments/paper_walk/refit_bc.py`
(SHA-256 `f9cb7250f5c81c4452cb8554529fe2a36ca68cb58124b86beac757e90049b439`) to the unchanged recipe, inputs, onset rows, weights and RNG contract
of [protocol 001](../paper_bc_refit_preparation_001/PROTOCOL.json). The helper body is
byte-identical from its first `import` statement; only the module docstring changed.
Protocol 001 and its helper remain frozen.

Read-only preflight:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run python -B experiments/paper_walk/refit_bc.py --repo-root . --protocol artifacts/restart_2026-09-14/paper_bc_refit_protocol_002/PROTOCOL.json
```

Execution writes `uniform/` and `onset_weight20/` candidates plus a paired `RESULT.json`
into a fresh output directory. A fitted candidate is a CPU fitting result; native
evaluation and every Stage 2 gate remain separate.
