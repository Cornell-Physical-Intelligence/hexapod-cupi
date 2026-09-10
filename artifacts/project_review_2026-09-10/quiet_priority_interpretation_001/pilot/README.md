# Independent pilot training interpretation

Read [REPORT.md](REPORT.md) for actual 50-update training diagnostics and the later [physical decision note](PHYSICAL_DECISION.md) for raw quiet-result comparison and a bounded experiment recommendation. The physical note preserves reset-contaminated maxima and unchanged failures. Root owns the complete terminal audit, next allocation and publication.

Reproduce using a fresh output:

```sh
python3 tmp/direct_quiet_pilot50_independent_review_001/analyze.py --repo /absolute/path/to/HEXAPOD --output /fresh/path/calculations.json
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tmp/direct_quiet_pilot50_independent_review_001 -p test_review.py -v
```

The original helper adaptation initially unpacked all 50 update losses as a two-item sequence and failed before writing an output. That local analysis error was corrected to select the first and last updates explicitly; no input or producer source changed. The successful calculation then replayed all 1,000 rows.

Fable 5.1 was requested with `--effort max`, tools disabled and no saved session. The artifact preserves its final response/provenance and the independent disposition, without internal thinking streams.

Under `docs/PROJECT_SITE.md`, eventual publication must append a site update for this bounded evidence, preserve the incomplete Stage 2 milestone, and update the registry only where the presented evidence changes. This tmp-only review makes no tracked or remote edits and does not trigger a GPU job.
