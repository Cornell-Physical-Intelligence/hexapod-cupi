# Canonical moving PPO composition: CPU proposal

This isolated package implements the command/history, eight-substep reward and material-contact packet seams needed after frozen canonical PPO003's zero-command smoke. It does not register a task, run PPO or native physics, change gates, or admit walking. Read [DESIGN.md](DESIGN.md) for exact source paths, missing native accessor, explicit unresolved moving rule, bounded proposed next job and recording costs.

The five files in `oracles/` are byte-exact selected copies from PPO003 and objective001, bound by `INPUTS.json`; they are not complete copies of either parent. Frozen source005 is unchanged. No raw simulation data, proprietary SDK source, actor checkpoint or internal Claude event stream is copied. Fable input is final-response-only, with exact model/effort/session and prompt/input hashes; its claims require independent disposition.

Run from any location with Python and NumPy:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s /absolute/path/to/this/bundle -p 'test_*.py' -v
python3 -S /absolute/path/to/this/bundle/verify_bundle.py
```

`tests_initial.log` preserves the initial17 passing checks; `tests_002.log` adds two clock/frame tests,19 passing. `tests_003.log` records22 passing after the three independent review counterexamples. Native availability/performance is not exercised. `CommandHistory.state()` is an owned CPU object snapshot for exact continuation testing, not a new production checkpoint format. All numeric contact-count fixtures are explicitly unadopted examples.

For eventual publication, root must follow `docs/PROJECT_SITE.md`: a new bounded update record, relevant Markdown/status when state changes, validated site build and integrated source lineage if runtime code is adopted. This package modifies no tracked files, shared guides, GPU state or frozen parent.
