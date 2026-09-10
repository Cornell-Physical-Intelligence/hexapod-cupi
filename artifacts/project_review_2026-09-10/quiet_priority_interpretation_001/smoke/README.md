# Independent quiet-objective review

See [REPORT.md](REPORT.md) for the concise actual-smoke findings and [calculations_003.json](calculations_003.json) for every retained KL/LR row and exact input references. Earlier calculations are preserved before the dot-product and weighted-component extensions. The scope is interpretation of two actual updates, with focused CPU checks. It grants no policy or physical qualification.

The Fable partner uses `claude-fable-5-1 --effort max`, supplied raw evidence, no tools, no saved session and final JSON output only. Its launch, final response and verification disposition are preserved; no internal thinking stream is captured. Independent calculations remain authoritative when a partner claim is not supported by source or raw data.

Reproduce the calculation into a **fresh** output path:

```sh
python3 tmp/direct_quiet_smoke003_independent_review_001/analyze.py --repo /absolute/path/to/HEXAPOD --output /fresh/path/calculations.json
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tmp/direct_quiet_smoke003_independent_review_001 -p test_review.py -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tmp/reference_residual_ppo_001/_deps .venv/bin/python -B -m unittest discover -s tmp/direct_omni_recovery_001/native_004 -p test_quiet_priority.py -v
```

This directory is tmp-only preparation. Root owns any publication and adoption. Under `docs/PROJECT_SITE.md`, eventual publication requires an append-only site update identifying this bounded evidence and its unresolved quiet result, and a registry change only if the presented implementation/evidence changes. No existing tracked files, frozen inputs, source, gates, checkpoint or GPU state were modified.
