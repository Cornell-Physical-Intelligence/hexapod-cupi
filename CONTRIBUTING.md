# Contributing

Use [README](README.md) to find the affected package and
[ARCHITECTURE](ARCHITECTURE.md) for its requirements. The team also maintains
[onboarding standards](https://wiki.cornellphysicalintelligence.com/#/page/software-onboarding-repo-standards)
behind Cornell sign-in. Agent constraints live in [AGENTS](AGENTS.md).
If available, use the graphify skill to become acquainted with the codebase/update.
## Environment and review

Use `uv sync --locked` and run Python through `uv run`; do not create separate
pip, conda or venv environments. Add dependencies with `uv add` and commit
`pyproject.toml` with `uv.lock`. You can run CPU tests and the viewer on your
laptop. Follow [SPARK](SPARK.md) for personal worktrees and frozen runs on Spark;
read [OPERATIONS](docs/OPERATIONS.md) before using Isaac.

Name branches `[netid]/[task]`. Submit changes through a PR to `main`, review
the diff and merge after green CI. Use forward commits to preserve published
history. Link the issue when one applies and explain the changed behavior,
validation and remaining limits in the [PR template](.github/pull_request_template.md).
The author must understand submitted code, including agent output.

GitHub requires passing `unit` and `poster` checks from GitHub Actions before
merging to `main`. Branches must be up to date. These requirements also apply
to repository administrators.

## Python style

Use the Ruff version pinned in `pyproject.toml` and `uv.lock`. Ruff formats
maintained Python with four-space indentation, double quotes and a 100-column
wrapping target. It sorts imports and checks syntax, undefined names, unused
imports and variables, and ambiguous variable names. `.editorconfig` sets shared
editor defaults.

Before committing, run:

```sh
uv run ruff check . --fix
uv run ruff format .
```

Review fixes and run the affected tests. CI runs both checks without modifying
files and fails on violations. Do not use unsafe fixes without reviewing their
behavior. Keep a rule suppression local and explain the required exception.

Ruff excludes published evidence, frozen test fixtures, the retained original
`src/locomotion/hexapod/` experiment and `src/locomotion/evaluation/scoring.py`.
Tests bind the retained experiment to its pinned source and require the scorer
to remain byte-identical to the archived oracle. Formatting other native runtime files
changes their source hashes and requires fresh standing admission and checkpoint
compatibility evidence before those files can run an admitted checkpoint.

## Results and publication

Keep secrets and full run payloads outside the checkout. You commit an admitted
training dataset under 20 MB with its manifest and review record; its raw native
runs stay outside the checkout. Preserve failed attempts
and immutable checkpoint/manifest bytes. Give new results fresh identities;
bind measurements and videos to the source, model, inputs and checkpoint hash.
Use the recorded 0.040 rad / 20 ms limiter for formal motion comparisons.
Keep measurements with other limiters in separate comparisons.

Publish selected media and small reports under `site/assets/`. Follow
[PROJECT_SITE](docs/PROJECT_SITE.md) for each push: add an append-only update,
change the registry when facts change, and verify the full diff
and rendered site. CI generates the source manifest when a change merges; see
[PIPELINE_LINEAGES](docs/PIPELINE_LINEAGES.md). Register new source commands in
`configs/source_inventory.json`.

## Required checks

Run focused tests while editing, then these checks before publication:

```sh
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s src/contracts/tests
uv run python -m unittest discover -s src/navigation/tests
uv run python -m unittest discover -s src/mission/tests
uv run python -m unittest discover -s robot/tests
uv run python -m unittest discover -s src/locomotion/tests
uv run python -m unittest discover -s src/locomotion/amp/tests
uv run python -m unittest discover -s src/locomotion/surrogate/tests
uv run python -m unittest discover -s tests
python3 tools/source_inventory.py check
python3 tools/archive.py check
uv run python -m locomotion.spark.inputs check
uv run python -m locomotion.amp.dataset check
python3 tools/check_pipeline_lineages.py current
python3 tools/project_site.py check --base <integration-base-SHA>
python3 tools/project_site.py build
node site/test_render.mjs
node site/test_code_map.mjs
```

Use the actual integration-base commit for the site check. After asset or viewer
changes, run `npm ci` and `npm run build` from `viewer/` and check the joint
controls in the rendered viewer. Historical Git-object verification is a
separate archive maintenance check. Preserve existing tests and gate values.
