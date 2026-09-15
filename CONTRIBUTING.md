# Contributing

The team standards live on the wiki page
[Software Onboarding - Repo Standards](https://wiki.cornellphysicalintelligence.com/#/page/software-onboarding-repo-standards)
(cornell.edu sign-in). Read it, then `README.md` and `CLAUDE.md`, before your first
pull request. This file repeats the rules that shape every change here.

## Environment

- `uv` manages every Python environment. Run `uv sync --locked` once and every
  script through `uv run`. Do not use `pip`, `conda` or `python -m venv`.
- Add a dependency with `uv add <package>` and commit `pyproject.toml` and
  `uv.lock` together.
- Isaac Sim is never required on your laptop. The test suites and the viewer run
  locally; training runs on the shared DGX Spark under
  [docs/OPERATIONS.md](docs/OPERATIONS.md).

## Branches and pull requests

- Name branches `[netid]/[task]`, for example `jd632/height-scan-validator`.
- Every change reaches `main` through a pull request. CI must be green before
  merge. Use forward commits and never rewrite published history.
- Fill in [the pull request template](.github/pull_request_template.md): work
  packet, verification evidence, reproduction procedure, roadmap marker and the
  `site/updates/` record. The author must explain every submitted line,
  including agent output.
- Do not commit secrets, `.env` contents, credentials or Spark paths with
  credentials. Checkpoints and other files under `artifacts/` belong in Git LFS;
  this repository has not adopted LFS yet, so keep new artifact payloads small
  and hash-bound until the leads migrate it.

## Every change

- Follow [docs/PROJECT_SITE.md](docs/PROJECT_SITE.md): update `site/project.json`
  when displayed progress changes, regenerate `STATUS.md`, add an append-only
  `site/updates/` record, and run the check, build and renderer commands.
- Covered source changes need a new release manifest and CI selection; see
  [docs/PIPELINE_LINEAGES.md](docs/PIPELINE_LINEAGES.md).
- Add new tools and prototypes to `configs/source_inventory.json`.

## Results

- A trained policy is a specific checkpoint file. Report every checkpoint with
  its SHA-256, the evaluation output and the exact test settings. A number
  without those is a note, not a result.
- Every formal measurement uses the 0.040 rad per 20 ms playback limiter.
  Numbers taken at different limiters never share a table.
- Frozen artifacts, published manifests, failed attempts and gates are immutable.
  New results get new identities.
