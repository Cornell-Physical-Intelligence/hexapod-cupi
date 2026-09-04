# Onboarding

Read in this order, then pick a workstream from `docs/ROADMAP.md` §4.

1. `README.md` — the map.
2. `CLAUDE.md` — the invariants. They bind people as much as agents.
3. `docs/ARCHITECTURE.md` — the three layers and the contracts between them.
4. `docs/ROADMAP.md` — milestones, gates, who owns what.
5. `STATUS.md` — what is true right now.
6. The `CLAUDE.md` inside each `packages/hexapod_*/` you touch.

## Local setup (any laptop, no simulator)

```sh
git clone https://github.com/Cornell-Physical-Intelligence/hexapod-cupi.git
cd hexapod-cupi
python3 -m venv .venv && . .venv/bin/activate
pip install torch numpy pyyaml opencv-python-headless
python3 -m unittest discover -s isaaclab/tests
```

Without the four dependencies the torch-dependent test modules surface as
import errors and the rest still runs; read the count off the run. Isaac Sim
is never required locally.

Viewer: `cd viewer && npm install && npm run dev`.

## Spark access

Training and Isaac Sim evaluation run only on the DGX Spark over Tailscale.
Ask the project lead for a Tailscale invite and a Spark account. Before your
first launch, read `docs/OPERATIONS.md` §2-§4 in full and run the read-only
audit in §2. Never launch an attempt that is not in the shared queue
(`docs/ROADMAP.md` §7).

## Working rules

- Branch per task; PR to `main`; CI must be green; one reviewer from another
  workstream for anything that touches a contract.
- New numbers come with an artifact. A result without a checkpoint hash,
  evaluation payload, and screen config is a note, not a result.
- Failed attempts keep their label and their directory. Nothing under
  `artifacts/` or `docs/incidents/` is rewritten.
- Gates are written before the work and are never edited to admit a result.
- Decisions that change a contract or a plan get an ADR in `docs/decisions/`.
