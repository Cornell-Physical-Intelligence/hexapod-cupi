# Contributing

Use [README](README.md) to find the affected package and
[ARCHITECTURE](ARCHITECTURE.md) for its requirements. The team also maintains
[onboarding standards](https://wiki.cornellphysicalintelligence.com/#/page/software-onboarding-repo-standards)
behind Cornell sign-in. Agent constraints live in [CLAUDE](CLAUDE.md).

## Environment and review

Use `uv sync --locked` and run Python through `uv run`; do not create separate
pip, conda or venv environments. Add dependencies with `uv add` and commit
`pyproject.toml` with `uv.lock`. You can run CPU tests and the viewer on your
laptop. Read [OPERATIONS](docs/OPERATIONS.md) before using Spark for Isaac.

Name branches `[netid]/[task]`. Submit changes through a PR to `main`, review
the diff and merge after green CI. Use forward commits to preserve published
history. Link the issue when one applies and explain the changed behavior,
validation and remaining limits in the [PR template](.github/pull_request_template.md).
The author must understand submitted code, including agent output.

## Results and publication

Keep secrets and full run payloads outside the checkout. Preserve failed attempts
and immutable checkpoint/manifest bytes. Give new results fresh identities;
bind measurements and videos to the source, model, inputs and checkpoint hash.
Use the recorded 0.040 rad / 20 ms limiter for formal motion comparisons.
Keep measurements with other limiters in separate comparisons.

Publish selected media and small reports under `site/assets/`. Follow
[PROJECT_SITE](docs/PROJECT_SITE.md) for each push: add an append-only update,
change the registry when facts change, and verify the full diff
and rendered site. Follow [PIPELINE_LINEAGES](docs/PIPELINE_LINEAGES.md) when
covered source changes need a new manifest. Register new source commands in
`configs/source_inventory.json`.

## Required checks

Run focused tests while editing, then these checks before publication:

```sh
uv sync --locked
uv run python -m unittest discover -s contracts/tests
uv run python -m unittest discover -s navigation/tests
uv run python -m unittest discover -s mission/tests
uv run python -m unittest discover -s robot/tests
uv run python -m unittest discover -s locomotion/tests
uv run python -m unittest discover -s locomotion/priors/tests
uv run python -m unittest discover -s tests
python3 tools/source_inventory.py check
python3 tools/archive.py check
uv run python -m locomotion.inputs check
python3 tools/check_pipeline_lineages.py current
python3 tools/project_site.py check --base <integration-base-SHA>
python3 tools/project_site.py build
node site/test_render.mjs
```

Use the actual integration-base commit for the site check. After asset or viewer
changes, run `npm ci` and `npm run build` from `viewer/` and check the joint
controls in the rendered viewer. Historical Git-object verification is a
separate archive maintenance check. Preserve existing tests and gate values.
