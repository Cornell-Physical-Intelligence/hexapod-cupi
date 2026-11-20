#!/usr/bin/env bash
# The bm0 chain, end to end, on the CPU MuJoCo surrogate:
#   v70 odometry policy od1 u1200
#   -> 1. predictor B: od1's inputs collected on the fitted simulator (4 seeds), merged, then fitted
#   -> 2. pfit1: 2000 PPO updates from od1 with predictor B in front, all five fitted axes
#   -> 3. bm0: 1000 DAgger updates distilling "B + pfit1 u2000" into one actor, nylon + 0.75 kg body
#   -> 4. judge: u500 and u1000 alone, all five fitted axes, nylon body, test seeds 20261110/20261111
#   -> 5. record: chain_record.txt (commit, hashes, scores) for record_run.sh
# The recipe is the one recorded for predictor2_20261110/merged4, pfit1 and bm0 (bmass1) on 2026-11-10..12.
#
# Usage, from anywhere inside a hexapod-cupi checkout of this branch (after `uv sync`):
#   bash records/distilled_bm0_u1000_20261112/pipeline/run_chain.sh OUT [--from STAGE] [--smoke]
#     OUT       a new folder for every output (refused if a stage's output already exists)
#     --from    predictor (default) | pfit1 | distill | judge. A later start uses the bundled
#               lineage/ files for the earlier stages (predictor B and pfit1 u2000 as recorded).
#     --smoke   wiring check in minutes: 1-epoch predictor, 2 PPO and 2 distill updates, stand cases.
#               Its numbers mean nothing.
# Environment: PFIT_THREADS (default 6), DISTILL_THREADS (default 2), SEEDS_PARALLEL=0 runs the four
# collections one after another (the default runs them at the same time).
# To leave it running unattended (macOS):  nohup caffeinate -i bash .../run_chain.sh OUT > OUT.log 2>&1 &
# Linux:                                    nohup bash .../run_chain.sh OUT > OUT.log 2>&1 &
# Recorded wall times on the original laptop (Apple silicon, CPU only): the four collections 11 min
# (at the same time), predictor fit 31 min, pfit1 38 min (6 threads), bm0 33 min (2 threads); the
# judge round that scored bm0 ran 18 judges in 45 min (this chain runs 4). About 2 h in all.
set -euo pipefail

usage() { sed -n '2,24p' "$0"; exit 2; }
[[ $# -ge 1 ]] || usage
OUT_ARG=$1; shift
FROM=predictor; SMOKE=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --from) FROM=${2:?}; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    *) usage ;;
  esac
done
case $FROM in predictor|pfit1|distill|judge) ;; *) echo "unknown stage $FROM"; exit 2 ;; esac

BUNDLE=$(cd "$(dirname "$0")/.." && pwd)
S=$BUNDLE/scripts
REPO=$(git -C "$BUNDLE" rev-parse --show-toplevel)
mkdir -p "$OUT_ARG"; OUT=$(cd "$OUT_ARG" && pwd)
LOG=$OUT/chain.log
OD1=$REPO/records/curriculum_20261109/od1/checkpoint_update001200.pt
cd "$REPO"

if command -v sha256sum >/dev/null; then sha() { sha256sum "$1" | cut -d' ' -f1; }
else sha() { shasum -a 256 "$1" | cut -d' ' -f1; }; fi
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
die() { say "STOPPED: $*"; exit 1; }
fresh() { [[ ! -e $1 ]] || die "$1 exists; not overwriting (use a new OUT)"; }

# ---- inputs are the recorded ones, or stop
expect() {  # FILE SHA256_PREFIX
  local got; got=$(sha "$1")
  [[ $got == "$2"* ]] || die "$1 is $got, expected $2…: the bundle or the start checkpoint changed"
}
expect "$OD1" 3bdfb00559840787
expect "$S/delay_predictor.py" 55d8c8151c21221e
expect "$S/predictor_train.py" f7c892af13edef8d
expect "$S/distill_train.py" fd9b35e3497c4d15
expect "$S/body_mass.py" 074b57d6a0bfe922
expect "$S/body_center_parts.json" b42fe28f86a00265
expect "$S/fitted_eval.py" 3c6f4b0314a54d41
expect "$BUNDLE/full96_suite.py" a4f5949ee1c039ab
[[ -n $(command -v uv) ]] || die "uv not found (install uv, then run 'uv sync' in $REPO)"

say "chain start: from $FROM, smoke $SMOKE, repo $(git rev-parse HEAD), out $OUT"
[[ -z $(git status --porcelain -- src) ]] || say "NOTE: src/ has uncommitted changes; the record names HEAD only"

# Recipe shared by pfit1 and bm0 (fit1_v70.sh / pfit1.sh / bmass1.sh)
TRAIN_ARGS=(
  --task "locomotion.omniwalk.destination_task:experiment_task(destination_v70)"
  --wrapper locomotion.rl.destination_wrapper:DestinationVecEnv
  --initialize-rate checkpoint
  --action-mean tanh --observation-normalization none --observation-scaling fixed
  --command-segments bootstrap --learning-rate-max 1e-4 --action-std 0.05 --action-std-final 0.05
  --gait-clock 60 --action-smoothing mean2 --velocity-noise 0.5 --episode-seconds 32.0
  --contact "touchdown_overshoot=False"
  --randomize "action_delay_controls=(1, 2)"
  --randomize "kd_scale=(0.13, 0.37)"
  --randomize "friction_scale=(0.45, 0.6)"
  --randomize "joint_friction_nm=((0.0335, 0.0622), (0.1387, 0.2576), (0.018, 0.0335))"
  --randomize "armature_kg_m2=((0.0, 0.0001), (0.0, 0.00058), (0.0, 0.00049))"
  --ppo-config-overrides '{"num_steps_per_env": 36, "algorithm": {"num_mini_batches": 6}, "environment_wrapper": {"destination_input": "terrain_v2_odom", "gait_clock_scaling": "speed_v1"}}'
)
if [[ $SMOKE == 1 ]]; then
  PFIT_UPDATES=2; DISTILL_UPDATES=2; EPOCHS=1; FOLDS=2; MATCH=(--match stand); JUDGE_AT=(2); LAST=2
else
  PFIT_UPDATES=2000; DISTILL_UPDATES=1000; EPOCHS=120; FOLDS=5; MATCH=(); JUDGE_AT=(500 1000); LAST=1000
fi
ckpt() { printf '%s/checkpoint_update%06d.pt' "$1" "$2"; }

# ---- 1. predictor B
PRED=$BUNDLE/lineage/predictor_B/predictor.pt
if [[ $FROM == predictor ]]; then
  P=$OUT/predictor; fresh "$P"; mkdir -p "$P"
  COLLECT_SEEDS=(20261011 20261012 20261013 20261014)
  [[ $SMOKE == 1 ]] && COLLECT_SEEDS=(20261011 20261012)
  say "1. predictor: collecting od1 u1200 inputs on the fitted simulator, seeds ${COLLECT_SEEDS[*]}"
  collect() {
    uv run python "$S/delay_predictor.py" collect "$P/s$1" "od1_u1200=$OD1" --seed "$1" \
      --axes kd,toe_friction,joint_friction,armature --motor-check drawn > "$P/collect_s$1.log" 2>&1
  }
  pids=()
  for seed in "${COLLECT_SEEDS[@]}"; do
    if [[ ${SEEDS_PARALLEL:-1} == 1 ]]; then collect "$seed" & pids+=($!); else collect "$seed" || die "collect s$seed failed (see $P/collect_s$seed.log)"; fi
  done
  for pid in ${pids[@]+"${pids[@]}"}; do wait "$pid" || die "a collection failed (see $P/collect_s*.log)"; done
  PARTS=(); for seed in "${COLLECT_SEEDS[@]}"; do PARTS+=("$P/s$seed"); done
  uv run python "$S/delay_predictor.py" merge "$P/merged" "${PARTS[@]}" > "$P/merge.log" 2>&1 || die "merge failed"
  say "1. predictor: fitting ($EPOCHS epochs, $FOLDS held-out folds)"
  uv run python "$S/delay_predictor.py" train "$P/merged" --epochs "$EPOCHS" --folds "$FOLDS" > "$P/train.log" 2>&1 \
    || die "predictor fit failed (see $P/train.log)"
  PRED=$P/merged/predictor.pt
  [[ -f $PRED ]] || die "no $PRED"
  say "1. predictor done: $(sha "$PRED") (recorded B: 1dff0e80c45da086…)"
else
  expect "$PRED" 1dff0e80c45da086
  say "1. predictor: using the bundled predictor B $(sha "$PRED")"
fi

# ---- 2. pfit1: PPO from od1 with the predictor in front
TEACHER=$BUNDLE/lineage/pfit1_u2000/checkpoint_update002000.pt
if [[ $FROM == predictor || $FROM == pfit1 ]]; then
  fresh "$OUT/pfit1"
  say "2. pfit1: $PFIT_UPDATES PPO updates from od1 u1200, predictor 2 controls ahead, anchor 0.006, seed 20261110"
  uv run python "$S/predictor_train.py" "$PRED" 2 --output "$OUT/pfit1" --initialize-from "$OD1" \
    "${TRAIN_ARGS[@]}" --updates "$PFIT_UPDATES" --seed 20261110 --threads "${PFIT_THREADS:-6}" --anchor-weight 0.006 \
    > "$OUT/pfit1_train.log" 2>&1 || die "pfit1 training failed (see $OUT/pfit1_train.log)"
  TEACHER=$(ckpt "$OUT/pfit1" "$PFIT_UPDATES")
  [[ -f $TEACHER ]] || die "no $TEACHER"
  say "2. pfit1 done: $(sha "$TEACHER") (recorded u2000: e2ec2f0c433d21d8…)"
else
  expect "$TEACHER" e2ec2f0c433d21d8
  say "2. pfit1: using the bundled pfit1 u2000 $(sha "$TEACHER")"
fi

# ---- 3. bm0: distill teacher + predictor into one actor on the nylon + 0.75 kg body
STUDENT_DIR=$OUT/bm0
if [[ $FROM != judge ]]; then
  fresh "$STUDENT_DIR"
  say "3. bm0: $DISTILL_UPDATES DAgger updates, teacher share 0.5 -> 0 over 200, nylon + 0.75 kg body, seed 20261110"
  uv run python "$S/body_mass.py" run "$S/distill_train.py" "$PRED" 2 --beta0 0.5 --beta-updates 200 \
    --output "$STUDENT_DIR" --initialize-from "$TEACHER" \
    "${TRAIN_ARGS[@]}" --updates "$DISTILL_UPDATES" --seed 20261110 --threads "${DISTILL_THREADS:-2}" \
    > "$OUT/bm0_train.log" 2>&1 || die "bm0 distillation failed (see $OUT/bm0_train.log)"
  [[ -f $(ckpt "$STUDENT_DIR" "$DISTILL_UPDATES") ]] || die "no final bm0 checkpoint"
  say "3. bm0 done: $(sha "$(ckpt "$STUDENT_DIR" "$DISTILL_UPDATES")") (recorded u1000: d84cb9285a1312da…)"
  grep -h "DISTILL update" "$OUT/bm0_train.log" | tail -1 | tee -a "$LOG" || true
else
  [[ -f $(ckpt "$STUDENT_DIR" "$LAST") ]] || die "--from judge needs $(ckpt "$STUDENT_DIR" "$LAST")"
fi

# ---- 4. judge the student alone under all five fitted axes, nylon body
say "4. judge: bm0 at u${JUDGE_AT[*]}, all five fitted axes, motor check drawn, test seeds 20261110 20261111"
pids=()
for u in "${JUDGE_AT[@]}"; do
  for seed in 20261110 20261111; do
    root=$OUT/judge_bm0_u${u}_s$seed; fresh "$root"
    uv run python "$S/body_mass.py" run "$S/fitted_eval.py" "$root" "bm0_u$u=$(ckpt "$STUDENT_DIR" "$u")" \
      --motor-check drawn --seed "$seed" ${MATCH[@]+"${MATCH[@]}"} > "$root.log" 2>&1 & pids+=($!)
  done
done
for pid in ${pids[@]+"${pids[@]}"}; do wait "$pid" || die "a judge run failed (see $OUT/judge_*.log)"; done

# ---- 5. record
REC=$OUT/chain_record.txt
{
  echo "bm0 chain record"
  echo "date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "host: $(uname -srm)"
  echo "repo commit: $(git rev-parse HEAD)"
  echo "from: $FROM  smoke: $SMOKE"
  echo "sha256:"
  echo "  od1 u1200        $(sha "$OD1")"
  echo "  predictor        $(sha "$PRED")  $PRED"
  echo "  teacher (pfit1)  $(sha "$TEACHER")  $TEACHER"
  for u in "${JUDGE_AT[@]}"; do echo "  bm0 u$u          $(sha "$(ckpt "$STUDENT_DIR" "$u")")"; done
  for f in "$S"/*.py "$S"/*.json "$S"/*.sh "$BUNDLE/full96_suite.py" "$0"; do echo "  $(basename "$f")  $(sha "$f")"; done
  echo "judge (passes; recorded bm0: u500 20, 17; u1000 19, 20):"
  for u in "${JUDGE_AT[@]}"; do for seed in 20261110 20261111; do
    echo "  u$u s$seed: $(grep -o 'FITTED bm0_u[0-9]*: pass [0-9]*/[0-9]*' "$OUT/judge_bm0_u${u}_s$seed.log" | tail -1)"
  done; done
} > "$REC"
cat "$REC" | tee -a "$LOG" >/dev/null
say "CHAIN DONE. Record: $REC. To put it in the repo: bash $BUNDLE/pipeline/record_run.sh $OUT NAME"
