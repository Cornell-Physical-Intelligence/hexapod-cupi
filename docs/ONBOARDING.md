# Onboarding

Read in this order, then pick a workstream from `docs/ROADMAP.md` §4.

1. `dar.md` — the mission: survey a bounded area drawn on the fly, steadily.
   Everything else exists to deliver that.
2. `README.md` — the map.
3. `CLAUDE.md` — the invariants. They bind people as much as agents.
4. `docs/ARCHITECTURE.md` — the three layers and the contracts between them.
5. `docs/ROADMAP.md` — milestones, gates, who owns what.
6. `STATUS.md` — what is true right now.
7. `AGENTS.md` — the robot model and the Isaac Sim runbook, if you touch either.
8. The `CLAUDE.md` inside each `packages/hexapod_*/` you touch.

## Local setup (any laptop, no simulator)

The repository is a [uv](https://docs.astral.sh/uv/) workspace: one lockfile,
six packages installed editable, test dependencies pinned for everyone.

```sh
git clone https://github.com/Cornell-Physical-Intelligence/hexapod-cupi.git
cd hexapod-cupi
uv sync                                   # creates .venv from uv.lock
uv run python -m unittest discover -s isaaclab/tests
```

`uv sync` installs the interpreter in `.python-version` if it is missing. Add
a dependency with `uv add <name>` (a test-only one with `uv add --dev <name>`)
and commit the updated `pyproject.toml` and `uv.lock` together; never edit the
lockfile by hand. Running the suite with a bare `python3` still works: the
torch- and numpy-dependent modules then surface as import errors and the rest
runs, which is the expected result on a machine without the dependencies. Read
the test count off the run. Isaac Sim is never required locally, and uv never
touches the Spark container (`docs/OPERATIONS.md` §2).

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
