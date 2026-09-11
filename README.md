# Hexapod

Survey an operator-drawn region, stopping with a steady deck to collect measurements
for a 3D terrain map in local site coordinates.

- [Architecture](ARCHITECTURE.md): agreed requirements, boundaries and team workflow.
- [Progress](https://cornell-physical-intelligence.github.io/hexapod-cupi/#roadmap): the four roadmap markers, evidence and open definitions.
- [STATUS](STATUS.md): generated text view of the same progress.
- [Contributor guide](CLAUDE.md): source locations, invariants and commands; `AGENTS.md` points here.
- [Reference documents](docs/README.md): model/gate references, operations and archived history.

The approved robot is selected by [robot/active_model.json](robot/active_model.json).
Historical walking results have their own model and qualification limits.

```sh
uv sync --locked
uv run python -m unittest discover -s isaaclab/tests
uv run python -m unittest discover -s robot/tests
python3 tools/source_inventory.py list --classification current
python3 tools/project_site.py check
python3 tools/project_site.py build
```

Start with your issue's architecture section and input fixtures. Evidence and
historical source copies remain available under `artifacts/`; they are excluded
from default source search. The CAD viewer is under `viewer/`.
