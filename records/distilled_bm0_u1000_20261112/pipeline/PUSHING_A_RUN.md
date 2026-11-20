# Running the bm0 chain on another computer and pushing the result

The whole chain from the v70 odometry policy to bm0 runs on the CPU MuJoCo surrogate. There is no
GPU or Spark step, so "queueing" it just means starting it in the background on a computer that can
stay awake for about 2 hours.

```
od1 u1200 (records/curriculum_20261109/od1, the v70 odometry policy)
  1. predictor B   od1's inputs on the fitted sim, seeds 20261011-14 -> merged -> fit (120 epochs, 5 folds)
  2. pfit1         2000 PPO updates from od1, predictor 2 controls ahead, anchor 0.006, seed 20261110
  3. bm0           1000 DAgger updates from "B + pfit1", nylon + 0.75 kg body, seed 20261110
  4. judge         bm0 u500 and u1000 alone, all five fitted axes, test seeds 20261110 and 20261111
  5. record        chain_record.txt: commit, every hash, the four scores
```

## 1. Set up (once per computer)

```bash
git clone --branch jg2399/distilled-bm0 https://github.com/S0ulSyte/hexapod-cupi.git
cd hexapod-cupi
git checkout 41d3e2432f4a70d14a7b637655caa11f447ec9a3   # the pre-merge code the chain runs on
uv sync --frozen
```

Needs `git` and [uv](https://docs.astral.sh/uv/). `uv sync` builds `.venv` from `uv.lock`.
After the merge with main, `run_chain.sh` refuses any source tree but that commit's, because main
replaced the 96-case scorer its judge stage reads.

## 2. Check the wiring (about 15 minutes)

```bash
bash records/distilled_bm0_u1000_20261112/pipeline/run_chain.sh ~/bm0_smoke --smoke
```

This runs every stage at toy size: two collections, a 1-epoch predictor, 2 PPO updates,
2 distillation updates and the stand cases. It should end with `CHAIN DONE`. Its numbers mean
nothing. Delete `~/bm0_smoke` afterwards.

## 3. Start the real run

```bash
nohup caffeinate -i bash records/distilled_bm0_u1000_20261112/pipeline/run_chain.sh ~/bm0_chain > ~/bm0_chain.log 2>&1 &
```

On Linux, drop `caffeinate -i`. Watch progress with `tail -f ~/bm0_chain/chain.log`.

- To start later in the chain, add `--from pfit1` (reuse the bundled predictor B), `--from distill`
  (reuse B and the bundled pfit1 u2000), or `--from judge` (re-judge an existing `OUT/bm0`).
- Thread counts: `PFIT_THREADS` (default 6) and `DISTILL_THREADS` (default 2).
- The script checks the hash of od1 and of every script before it starts. It refuses to overwrite
  any output, so give every run a new folder.

Recorded results to compare against (2026-11-10 to 12, on the original laptop):

| stage | recorded output | sha256[:16] |
|---|---|---|
| predictor B | `lineage/predictor_B/predictor.pt` | 1dff0e80c45da086 |
| pfit1 u2000 | `lineage/pfit1_u2000/checkpoint_update002000.pt` | e2ec2f0c433d21d8 |
| bm0 u1000 | `checkpoint_update001000.pt` | d84cb9285a1312da |
| bm0 scores (passes of 96, s20261110 / s20261111) | u500 20, 17; u1000 19, 20 | |

A rerun on different hardware or library builds will not give byte-identical weights. Floating-point
order and thread scheduling differ. Compare the scores: the seed-to-seed spread of this recipe is
about 16 to 21 passes.

## 4. Put the run in the repo

```bash
bash records/distilled_bm0_u1000_20261112/pipeline/record_run.sh ~/bm0_chain bm0_chain_YYYYMMDD
```

This copies into `records/bm0_chain_YYYYMMDD/`:

- `chain_record.txt` and `chain.log`
- `predictor/`: `predictor.pt`, `collect.json` and `report.*`, plus the per-seed `collect.json` and logs
- `pfit1/`: the last checkpoint and its `.json`, `predictor_in_loop.json`, `ppo_config.json`,
  `state.json`, `force_metrics.json` and the train log
- `bm0/`: each judged checkpoint and its `.json`, `body_mass.json`, `distillation.json`, the configs
  and the train log
- `judges/`: the four judge logs

It leaves out the files too big for git: `inputs.npz` (16 to 66 MB), `metrics.jsonl` (50 to 100 MB)
and the judge folders (about 1.8 GB each). Keep those on the computer if you want them. Every file
it copies is staged with `git add -f`, because the repo's `.gitignore` excludes `*.pt` and `*.log`.

Then write a short `records/bm0_chain_YYYYMMDD/README.md`, commit and push:

```bash
git commit -m "Record the bm0 chain run YYYYMMDD"
```

```bash
git push fork jg2399/distilled-bm0
```

The README should say: what differs from the recorded bm0 (computer, any recipe change), the four
scores next to the recorded ones, and the sha256 lines from `chain_record.txt`. Copy the format of
this folder's `README.md`.

On a fresh clone the remote is called `origin` and points at your fork, so the push is
`git push origin jg2399/distilled-bm0`. Check with `git remote -v` first.

## Rules that still apply

- Push to your fork only (`S0ulSyte/hexapod-cupi`), on a `jg2399/*` branch. Never push to
  `Cornell-Physical-Intelligence/hexapod-cupi` or to `main`, and open a PR only when you mean to.
- bm0 is a surrogate diagnostic. Admitting it to the lineage needs the lead's (palerdr's) OK.
- GitHub rejects files over 100 MB and warns above 50 MB. `record_run.sh` keeps every copied file
  far below that (checkpoints are about 4 MB).
- Add a decision-log entry for the run, in the same style as the bmass1 entries in
  `hexapod-surrogate/runs/curriculum/decisions.md`. Put it in the run's README if you are not on
  the original laptop.
