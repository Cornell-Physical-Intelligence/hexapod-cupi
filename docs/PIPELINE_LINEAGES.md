# Pipeline source lineages

The Stage2 mock and the physical four-bar task are separate releases. Source
integrity checks preserve both. They do not admit training or validate robot
dynamics; the physical validation reports and checkpoint contract remain
separate requirements.

`isaaclab/deploy/stage2_pipeline.sha256` remains byte-for-byte frozen, with
SHA256 `19fc816cf9c53a79be8e14831daa58a12eba3f7f07c5fca80d03f2dc947ccda1`.
Its 112 entries are verified from the immutable Git source commit
`81d7c6f2a43c7de99f32cd6bb1b7efb0f54874df`. The checker creates an isolated
Git archive, hashes every listed file from that archive, and checks the archived
manifest itself against the unchanged local copy. It neither checks out nor
executes archived source.

```sh
uv run python tools/check_pipeline_lineages.py historical
```

The current package configuration has two explicit, separately versioned
changes: `packages/hexapod_core/pyproject.toml` includes the RS05 v2 JSON data,
and `packages/hexapod_env/pyproject.toml` includes the actuator and task
subpackages. Requiring these current files to have their historical hashes
would remove needed packaging. The historical source and its old package
configuration remain recoverable at the pinned commit.

The original four-bar manifest `isaaclab/deploy/mkii_fourbar_v1_pipeline.sha256` remains preserved with source commit `3dc3fa0c9d0f9d05dd625c00f599ca456e94779c`. It describes the first 32/4-iteration implementation, whose driven solver test subsequently failed. Its bytes are included in the new release manifest.

`isaaclab/deploy/mkii_fourbar_v1_tgs_pipeline.sha256` covers **all 112 historical
paths** at their current hashes, every source/configuration/asset path in the
four-bar runtime identity, and the current launcher, CI workflow, lineage
checker, lineage regression tests, this document and workspace dependency
files. Verification requires the complete expected path set: missing entries,
extra entries and changed hashes all fail. Changes to any historical path
outside the two named packaging files fail, even if someone generates a new
manifest from that change.

```sh
uv run python tools/check_pipeline_lineages.py current
```

CI fetches full history and runs both checks independently. A raw
`sha256sum -c isaaclab/deploy/stage2_pipeline.sha256` remains appropriate when
reproducing the **matching historical checkout**. It is not the current
four-bar release check.

To create the new manifest after the source and asset files are finalized:

```sh
uv run python tools/check_pipeline_lineages.py generate
uv run python tools/check_pipeline_lineages.py historical
uv run python tools/check_pipeline_lineages.py current
```

Generation first verifies the historical lineage and refuses to overwrite an
existing destination. Future published releases need a new manifest filename
passed with `--manifest`, and a reviewed CI pointer update. Keep prior manifest
bytes and their source commits available. Do not regenerate a published
manifest to hide an unexpected difference.

Each owned Spark run also stores `source.SHA256SUMS` and `source.tar.gz` with
their digests in `supervisor.json`. That preserves the actual isolated run
snapshot, including uncommitted files, even if the staging directory is later
updated. A release manifest and a per-run archive serve different purposes;
neither replaces the live physical acceptance evidence.
