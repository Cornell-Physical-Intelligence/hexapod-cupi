# Host005: Python 3.12 compilation-context repair

This bounded host-only successor fixes the rejection observed before the first extended campaign created its output. The frozen host004, source004/native005, smoke004 and failed extended001 evidence remain unchanged. A successful retry is not claimed here.

The actual Spark Python 3.12.3 probe reproduces host004’s rejection: imported `run_owned` differs from the isolated-AST recompilation in `co_code` and `co_linetable`. The first instruction differences concern `LOAD_GLOBAL`/`LOAD_ATTR` call setup for imported module names (`uuid`, `time`, `os`, `fcntl`). Compiling the complete exact source module instead produces full `CodeType` equality. This is a compiler-context mismatch in the verification method, not evidence of a changed supervisor.

Only `deadline_adapter.py` changes among runtime files. It retains the full verified source text and compiles that module without executing any of its statements, with `dont_inherit=True`. The entire extracted code object must still equal the loaded function; original module globals, defaults, closure, exact source hash and all three counted substitutions remain mandatory. Altered function code in the correct globals is still rejected. The adapted function text and cleanup AST are unchanged from host004. Extended training remains 1800 s, evaluation 600 s, AppReady 90 s; smoke/pilot use the original unadapted supervisor. The launcher, metadata entry, original checkpoint, source599/native28 identities, six phase folders, smoke admission, policy/physics/gates and 500-update contract are byte-identical to host004.

**Validation:** all 50 CPU tests pass, including the inherited fake-clock/process/cleanup tests and three new compilation-context/altered-code/runtime-delta tests. `diagnosis/spark_cpu_probe_result.json` additionally records the actual Spark 3.12.3 `load_supervisor` integration: old failure reproduced, new extended install passed, altered code rejected and smoke/pilot code unchanged. The exact unchanged host launcher was imported from disk and the new adapter supplied in memory. A Python audit hook rejected writable opens, subprocess execution and filesystem mutations; `run_owned` was never invoked. Root will repeat loading after transferring the final bundle before dispatch.

The first local new-test attempt incorrectly counted `ast.parse`’s compile call as the full bytecode compile. Its failed log is retained; only the test’s selection predicate was corrected. Runtime code did not change between that attempt and passing checks.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tmp/direct_omni_train_host_005 -p 'test_*.py'
```

`historical_host004`, `historical_host003`, `parent_evidence` and the source-build receipts are immutable copied history. Current identities are in FINAL_BINDINGS.json and CPU_READINESS_FINAL.json. No remote file, GPU job, tracked repository file or frozen original was changed. Eventual publication is owned by root and must follow docs/PROJECT_SITE.md: bounded central update record, STATUS/relevant Markdown and site validation/build.
