# Progress site

The [progress site](https://cornell-physical-intelligence.github.io/hexapod-cupi/)
shows where the project stands. `python3 tools/project_site.py build` builds it
from `site/project.json`. `python3 tools/project_site.py status` writes the latest
result, roadmap and accomplishments to an untracked `STATUS.md`, and each Pages
build publishes that file.

## Presentation standard

The site, the README and the other pages we publish follow this standard.

- Lead with the visual result. Give each video or image one short caption in
  plain words for a reader outside the team.
- Write as the team describing its work ("We trained…"), in current terms.
- Keep the page to five sections: latest progress, roadmap, system
  connections, the robot and the terms used on the page.
- Leave out small grey explanation text, evidence lists, status cards, test
  result tables, earlier work and change history. Evidence stays in
  `site/assets/` and in the update records.
- Add each technical word the page uses to the terms list.

`site/test_render.mjs` fails when a removed section returns or when the page
shows more than 650 words.

## Developer code map

You can inspect the separate [locomotion code map](https://cornell-physical-intelligence.github.io/hexapod-cupi/code/).
The Pages build generates its HTML and JSON from Python syntax and the source inventory.
You can search definitions and open source lines without importing the simulator.
The map uses the progress site's `poster.css`, header and type. It draws each package
as a square sized by its lines, with arrows for imports between packages. A selected
package shows its files as tiles sized by lines, and a selected file shows its classes
and functions as blocks along one bar. Search marks matches in red.
You can download the JSON index for hashes and import references. The index records
the build revision and excludes frozen fixtures. Keep generated map files out of Git.

## Update the page

Change `site/project.json` when a result changes:

1. `latest`: the date, a one-line title, at most three sentences, the headline
   number and up to three videos. The check compares the number with each JSON
   field in `metric.sources`.
2. `accomplishments`: one line per result, newest first, with its evidence path.
3. `roadmap`: move `now` when a step finishes; completed steps come first. A
   step becomes `done` only with `accepted_by` (the person who accepted it) and
   a `source` path.
4. `nodes`: a part's one sentence and its state (`working`, `in_progress`,
   `not_started`).

Register each new video in `media` with its SHA-256, caption, controller
identity, evidence path and a first-frame poster. Earlier results and their
evidence links stay in the
[registry of 13 October 2026](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/9d819c68f7311477fc5d1fd61b00026a146656a0/site/project.json).

## Every repository change

Every push includes a matching Pages update. The program lead set this rule on
10 September 2026 and restated it on 14 September 2026. Add a new `site/updates/YYYYMMDDTHHMMSS_name.json` record with:

- `id` matching its filename and an ISO-8601 UTC `date`;
- a plain-English `title`, `summary`, affected `areas` and the `next` action;
- `changes`: exact repository-relative paths or bounded globs covering the diff;
- `evidence`: existing repository paths or public HTTPS references;
- boolean `no_project_impact`, and a specific `reason` when it is true.

Records are append-only. A later record may declare
`evidence_reference_corrections`. Each entry pins the older record by ID and
hash, and it names the original reference with its replacement path and hash. Evidence paths must match
on-disk capitalization so macOS checks agree with Linux CI.

For moved or edited source files, add the original path and its exact Git commit
and hash to `source_archives` in the source inventory. The site resolves retired
paths to that revision. A correction that pins earlier source bytes uses the
matching archived revision. The validator checks the Git blob hash; the published
record keeps its original bytes.

```sh
python3 tools/project_site.py status
python3 tools/project_site.py check --base <integration-base-SHA>
python3 tools/project_site.py build
node site/test_render.mjs
node site/test_code_map.mjs
```

`.github/workflows/pages.yml` runs these checks on each push and pull request
and deploys `main`. Only the media listed in the registry enter `site/dist`.
After a merge, confirm that the deployed `version.json` names the merged
revision. CI generates each merged revision's source manifest, so a pull request
adds none; see [PIPELINE_LINEAGES.md](PIPELINE_LINEAGES.md).
