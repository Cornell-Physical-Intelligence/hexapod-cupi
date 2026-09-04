# Handoff

The single monolithic handoff has been split by lifespan. Use these instead:

- [`STATUS.md`](STATUS.md) — current state: best checkpoint, current target,
  open contradictions, next actions. Rewritten in place, never appended.
- [`docs/TRAINING.md`](docs/TRAINING.md) — durable task and training design:
  stack, observation/action/timing contract, coordinate contract, curriculum,
  rewards, acceptance gates, sim-to-real gaps.
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — Spark runbook: environment,
  audit sequence, GPU lock protocol, launchers, formal screens, and the
  startup-supervisor specification.
- [`docs/incidents/`](docs/incidents/) — frozen forensic records of specific
  failures.
- [`docs/archive/HANDOFF-2026-08-26.md`](docs/archive/HANDOFF-2026-08-26.md) —
  the previous handoff, verbatim and frozen.

Do not restore a single combined handoff file. Put current state in
`STATUS.md`, durable design in `docs/`, and evidence in `artifacts/`.
