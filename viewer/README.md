# Viewer

The viewer has two pages. **Robot** is the joint inspector for the active robot model. **Runs** replays
evaluation runs from native Spark attempts and from the surrogate side by side on one clock.

```sh
npm ci
npm run dev
```

Open the address Vite prints. Add `#runs` for the Runs page.

## Runs

Both evaluators write `evaluation/batch_*/report.json` and `control_trace.npz`. `run_view.py` turns one
evaluation into a run file:

```sh
uv run python viewer/run_view.py /path/to/eval_attempt --output attempt.json
uv run python viewer/run_view.py /path/to/surrogate_output --output surrogate.json
```

The folder can be a native attempt (`<attempt>/run/standing/evaluation`), a surrogate output folder
(`<output>/evaluation`) or the evaluation folder itself. Copy native attempts from Spark first; only batches
with a control trace are drawn, and the file lists the others.

On the Runs page, open one or more run files or drop them on the page. Pick the cases to show from the
table; each case shows:

- the robot from the top, in 3D or from the side, with its legs drawn from the robot model's joint frames;
- the body path in grey, and toes filled while on the floor;
- the ghost, as a dashed orange line and dot. It starts at the reset pose (the origin, heading zero), follows
  the command exactly once per control and restarts at the body when the command changes, as the destination
  experiments train it;
- the commanded direction or turn, the heading against the commanded heading, and the distance to the ghost;
- the evaluator's own pass or fail and failed checks.

A server that serves the built viewer and the run files can open them directly: `?run=attempt.json#runs`,
with `run` repeated for more files.

`npm test` reads `test/fixtures/sample_run.json`. `tests/test_run_view.py` checks that `run_view.py` writes
the same file from its synthetic attempt.
