# hexapod_train

This package owns the operator side of a training attempt: run composition
(`configio.py`, `compose.py`, `labels.py` turn a named experiment config plus an
intervention delta into the exact argv of an existing hardened launcher), the
run contract (`contract.py` — parent pin, gate report, milestone events, attempt
outcome), the resource gates (`gates.py`), and the attempt-aware startup
supervisor (`supervisor.py`) specified in `docs/OPERATIONS.md` §6, exposed
through `ops/hexctl` (`cli.py`). Every decision — argv composition, label and
attempt allocation, gate predicates, the milestone state machine, terminal
classification, and the retry decision — is a pure function or a class with
injected clock, filesystem, and process interfaces; everything that touches the
real machine lives behind the single `SystemInterface` in `system.py`. Don'ts:
never bypass or reimplement a launcher's gates — the bash keeps doing the
flock, the Docker and service checks, the atomic artifacts, and the exact
container-ID cleanup, and this package only wraps and supervises it; never
auto-delete caches, Docker volumes, or containers beyond the launcher's own
exact-ID cleanup; never allow more than one retry of an attempt, and only for a
proven pre-AppReady stall; labels are immutable and are never reused or
rewritten, including failed ones; and never import torch, Isaac Lab, or
`hexapod_env` from here — dependencies point outward from this package, not
into it.
