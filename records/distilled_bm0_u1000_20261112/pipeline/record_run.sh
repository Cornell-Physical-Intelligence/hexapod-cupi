#!/usr/bin/env bash
# Copy a finished run_chain.sh output into records/NAME/ and stage it with git, ready to commit.
# Copies what the repo keeps for a run: checkpoints with their declarations, the small JSON records,
# the logs and the chain record. Leaves out what is too big for git: the collected inputs
# (inputs.npz, 16-66 MB), metrics.jsonl (50-100 MB) and the judge folders (about 1.8 GB each;
# their logs carry every case's result).
# Usage, from inside the hexapod-cupi checkout:
#   bash records/distilled_bm0_u1000_20261112/pipeline/record_run.sh OUT NAME
#     OUT   the run_chain.sh output folder (must contain chain_record.txt: a finished chain)
#     NAME  the new records folder, e.g. bm0_chain_20261120 (refused if it exists)
# It stages the files and prints the commit and push commands; it does not commit or push.
set -euo pipefail
[[ $# -eq 2 ]] || { sed -n '2,12p' "$0"; exit 2; }
OUT=$(cd "$1" && pwd); NAME=$2
BUNDLE=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(git -C "$BUNDLE" rev-parse --show-toplevel)
DEST=$REPO/records/$NAME
[[ -f $OUT/chain_record.txt ]] || { echo "$OUT has no chain_record.txt: the chain did not finish"; exit 1; }
grep -q "smoke: 1" "$OUT/chain_record.txt" && echo "WARNING: this is a --smoke run; its numbers mean nothing"
[[ ! -e $DEST ]] || { echo "$DEST exists; pick another NAME"; exit 1; }

copy() {  # SRC DEST_SUBDIR (skips a missing file)
  [[ -e $1 ]] || return 0
  mkdir -p "$DEST/$2"; cp "$1" "$DEST/$2/"
}
copy "$OUT/chain_record.txt" .
copy "$OUT/chain.log" .

# 1. predictor (only when this run fitted its own)
if [[ -d $OUT/predictor ]]; then
  for f in predictor.pt collect.json report.json report.txt; do copy "$OUT/predictor/merged/$f" predictor; done
  for f in "$OUT"/predictor/*.log; do copy "$f" predictor; done
  for d in "$OUT"/predictor/s*/; do
    [[ -f $d/collect.json ]] && { mkdir -p "$DEST/predictor/$(basename "$d")"; cp "$d/collect.json" "$DEST/predictor/$(basename "$d")/"; }
  done
fi

# 2. pfit1 (only when this run trained its own): the last checkpoint and the records beside it
if [[ -d $OUT/pfit1 ]]; then
  last=$(ls "$OUT"/pfit1/checkpoint_update*.pt | sort | tail -1)
  copy "$last" pfit1; copy "${last%.pt}.json" pfit1
  for f in predictor_in_loop.json ppo_config.json state.json force_metrics.json; do copy "$OUT/pfit1/$f" pfit1; done
  copy "$OUT/pfit1_train.log" pfit1
fi

# 3. bm0: every judged checkpoint and the distillation records
for pt in "$OUT"/bm0/checkpoint_update*.pt; do
  u=$((10#$(basename "$pt" .pt | sed 's/checkpoint_update//')))
  if ls "$OUT"/judge_bm0_u${u}_s*.log >/dev/null 2>&1; then copy "$pt" bm0; copy "${pt%.pt}.json" bm0; fi
done
for f in body_mass.json distillation.json ppo_config.json state.json force_metrics.json; do copy "$OUT/bm0/$f" bm0; done
copy "$OUT/bm0_train.log" bm0

# 4. judges: the logs (one line per case, then the FITTED pass line)
for f in "$OUT"/judge_*.log; do copy "$f" judges; done

cd "$REPO"
git add -f "records/$NAME"
echo
echo "Staged records/$NAME ($(du -sh "$DEST" | cut -f1)):"
git status --short -- "records/$NAME"
echo
echo "Scores:"; grep -A8 '^judge' "$DEST/chain_record.txt" | tail -n +2
cat <<EOF

Next (see PUSHING_A_RUN.md):
  1. add a README.md in records/$NAME (what changed vs the recorded bm0, the scores, the hashes)
  2. git commit -m "Record the bm0 chain run $NAME"
  3. git push fork $(git branch --show-current)     # your fork only; never origin or main
EOF
