# Source release and archive verification

You verify current code without fetching retired experiments:

```sh
python3 tools/check_pipeline_lineages.py current
python3 tools/archive.py check
```

The current manifest is `configs/releases/tripod_target_bounds_20260922_v1.sha256`.
It covers maintained source and tests, with the selected model identities.
The fixture manifest and model-input manifest pin required immutable inputs.
Source identity supplies no native admission or walking acceptance.

## Restore historical work

`configs/archive.json` pins the complete pre-consolidation tree at
`5e65918020dc9ef2d72da8dad0a346016b98728d`. Published reference mappings retain
older commits when their files preceded that tree. Existing update records and
result bytes retain their original identities. Git history has not been rewritten.

```sh
git fetch --unshallow
python3 tools/archive.py check --git-objects
python3 tools/archive.py restore artifacts --destination ../hexapod-restored
python3 tools/check_pipeline_lineages.py historical
```

Use `git fetch --unshallow` for a shallow clone. Choose a fresh restore destination;
restore accepts a single file or subtree and refuses to overwrite a directory.
The historical check uses the unchanged Stage 2 manifest at its original commit.
Run historical tests from a restored checkout of their source revision.

## Publish current code

Add a fresh manifest under `configs/releases/`, update the checker and CI, then
run `python3 tools/check_pipeline_lineages.py generate --manifest` with that path.
Generation rejects overwriting a published manifest. Add an append-only site
update and validate the complete diff before pushing. Preserve earlier manifests
in their Git revisions. Verify exact-revision CI and the deployed Pages revision.
