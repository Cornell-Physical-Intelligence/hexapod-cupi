# hexapod_eval

This package owns the evaluation side of the project: `gates.py`, the
machine-readable single source of truth for every acceptance threshold, screen
parameter, and RS05 safety limit in this repository, transcribed verbatim from
the grader or config constant that actually enforces each number and bound to it
by `isaaclab/tests/test_eval_gates_contract.py`; and `dispatch.py` plus
`__main__.py`, the one documented entry point
(`python -m hexapod_eval <action> --stage <stage> [args ...]`, or `--list` for
the mapping table) that execs an existing evaluation script under the current
interpreter with the remaining argv forwarded untouched. Don'ts: never change a
threshold value here to admit a candidate — a threshold changes only by an
explicit user decision recorded in `STATUS.md`, and then it changes in the
enforcing grader first and in `gates.py` second; never import from
`hexapod_train` (or from `hexapod_env`, torch, or Isaac Lab) — this package
stays stdlib-only so it runs under a bare system interpreter; never move,
rename, wrap, or alter the CLI of the scripts at `isaaclab/` top level, because
the hardened launchers reference them by path and the release manifest pins
their hashes — the dispatcher maps to them and adds nothing; and never invent,
round, or normalize a number that cannot be found verbatim in a cited source
file, since every value here is expected to survive an AST re-extraction on
every test run.
