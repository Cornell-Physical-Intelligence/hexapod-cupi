# Hexapod

You develop a robot that surveys an operator-drawn region and stops to collect
measurements for a terrain map. Start with [ARCHITECTURE](ARCHITECTURE.md) for
requirements and [STATUS](STATUS.md) for measured progress.

| Directory | Maintained scope |
| --- | --- |
| `robot/` | Approved model, portable simulation inputs and CAD inspection. |
| `locomotion/` | Native simulator, current PPO, evaluation and guarded launch; optional optimizer in `priors/`. |
| `contracts/` | Velocity commands, planar poses and existing release identity fields. |
| `navigation/` | Example waypoint follower; planning and localization remain pending. |
| `mission/` | Sensor-pattern and transport prototypes; survey recording/export remain pending. |
| `viewer/` | Interactive inspection of the approved robot. |
| `docs/`, `site/` | Maintained procedures and published progress. |

Read [TRAINING](docs/TRAINING.md) for your proposed paper-reproduction order and
its prerequisite review. The foundation preserves the current algorithm.

```sh
uv sync --locked
uv run python -m locomotion.inputs check
uv run python -m unittest discover -s locomotion/tests
uv run python -m unittest discover -s tests
python3 tools/project_site.py check
```

[CLAUDE](CLAUDE.md) owns contributor instructions. [CONTRIBUTING](CONTRIBUTING.md)
lists the required checks. Run the viewer with `npm ci` and `npm run dev` from
`viewer/`. Use the existing [Pages site](https://cornell-physical-intelligence.github.io/hexapod-cupi/)
for videos and the append-only update history.

The [archive index](configs/archive.json) pins retired results and source in Git.
Use [source release procedures](docs/PIPELINE_LINEAGES.md) to restore a named path.
A shallow clone supports normal development; full Git history retains its size.
Keep new run payloads outside the checkout and publish selected verified files.
