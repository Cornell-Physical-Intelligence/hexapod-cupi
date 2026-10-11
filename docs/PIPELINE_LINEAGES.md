# Source release and archive verification

You verify current code without fetching retired experiments:

```sh
python3 tools/check_pipeline_lineages.py current
python3 tools/archive.py check
```

`current` hashes maintained source and tests, with the selected model
identities, and reports the digest of their manifest. Add `--manifest <path>`
to compare the tree, or a copy without Git metadata such as the Spark mirror
(`--root`), with a named manifest.
The fixture manifest and model-input manifest pin required immutable inputs.
Source identity supplies no native admission or walking acceptance.

## Restore historical work

`configs/archive.json` pins the complete pre-consolidation tree at
`5e65918020dc9ef2d72da8dad0a346016b98728d`. Published reference mappings retain
older commits when their files preceded that tree. Existing update records and
result bytes retain their original identities. Git history has not been rewritten.

```sh
git fetch --unshallow
git fetch origin refs/pull/33/head:refs/archive/pr33
python3 tools/archive.py check --git-objects
python3 tools/archive.py restore artifacts --destination ../hexapod-restored
python3 tools/check_pipeline_lineages.py historical
```

Use `git fetch --unshallow` for a shallow clone. Retired tripod evidence from PR #33 pins
commit `05f706eb`; the second fetch keeps it under a local ref that `git gc` cannot prune. Choose a fresh restore destination;
restore accepts a single file or subtree and refuses to overwrite a directory.
The historical check uses the unchanged Stage 2 manifest at its original commit.
Run historical tests from a restored checkout of their source revision.

## Publish current code

A pull request adds no manifest. For each revision, CI fetches the archived
Stage 2 commit and runs `generate`, which verifies the historical lineage and
writes the manifest. For each push to `main`, CI keeps that manifest as the
`source-manifest-<revision>` workflow artifact for the repository's artifact
retention period. After that period, regenerate the manifest from Git:

```sh
git checkout <revision>
python3 tools/check_pipeline_lineages.py generate --manifest <fresh path>
```

Generation rejects overwriting a published manifest. The manifests under
`configs/releases/` stay unchanged as records of earlier releases. Add an
append-only site update and validate the complete diff before pushing. Verify
exact-revision CI and the deployed Pages revision.
