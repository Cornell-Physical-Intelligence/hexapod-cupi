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
- Show structure as a picture: a diagram, tiles or a chart in place of a list
  or a paragraph.
- Use black and white. Red (`#b31b1b`) is the one accent, for the logo,
  section titles and the current or selected item. `site/poster.css` holds
  these colours for each page.
- Put text and pictures straight on the white page. Leave out cards: grey
  boxes, coloured side tabs or bars, pill badges and drop shadows.

`site/test_render.mjs` fails when a removed section returns or when the page
shows more than 650 words.

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

```sh
python3 tools/project_site.py status
python3 tools/project_site.py check --base <integration-base-SHA>
python3 tools/project_site.py build
node site/test_render.mjs
```

`.github/workflows/pages.yml` runs these checks on each push and pull request
and deploys `main`. Only the media listed in the registry enter `site/dist`.
After a merge, confirm that the deployed `version.json` names the merged
revision. CI generates each merged revision's source manifest, so a pull request
adds none; see [PIPELINE_LINEAGES.md](PIPELINE_LINEAGES.md).
