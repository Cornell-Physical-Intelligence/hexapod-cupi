#!/bin/zsh
# bmass1: distill1's recipe (DAgger from "predictor B 2 ahead + pfit1 u2000", student starts at pfit1 u2000,
# BETA 0.5 -> 0 over 200 updates, no anchor, all five fitted axes) with ONE change: the surrogate body has
# its hexagonal centre piece in nylon (1140 kg/m^3) plus 0.75 kg (body_mass.py; robot 7.466 -> 7.280 kg).
# Arm "bm", seeds 20261110-12. Control: distill1 / dc1 / dc2 (same recipe and seeds, old body).
# Usage: zsh bmass1.sh UPDATES [SUFFIX]. Then u500 and the last checkpoint of every bm run, and the
# controls' u1000, under all five fitted axes with the new body (motor check drawn, seeds 20261110/11).
set -o pipefail
UPDATES=${1:?usage: bmass1.sh UPDATES [SUFFIX]}
SUFFIX=${2:-}
CUPI=/Users/new/Desktop/cupi/hexapod-cupi
C=/Users/new/Desktop/cupi/hexapod-surrogate/runs/curriculum
S=$C/session_scripts_20261110
PRED=$C/predictor2_20261110/merged4/predictor.pt
TEACHER=$C/pfit1/checkpoint_update002000.pt
THREADS=${THREADS:-2}
cd $CUPI || exit 1
train_arm() {  # NAME SEED
  if [[ -e $C/$1 || -e $C/${1}_train.log ]]; then echo "$1 exists; not overwriting"; return 1; fi
  uv run python $S/body_mass.py run $S/distill_train.py $PRED 2 --beta0 0.5 --beta-updates 200 \
    --output $C/$1 \
    --task "locomotion.omniwalk.destination_task:experiment_task(destination_v70)" \
    --wrapper locomotion.rl.destination_wrapper:DestinationVecEnv \
    --initialize-from $TEACHER --initialize-rate checkpoint \
    --action-mean tanh --observation-normalization none --observation-scaling fixed \
    --command-segments bootstrap --learning-rate-max 1e-4 --action-std 0.05 --action-std-final 0.05 \
    --gait-clock 60 --action-smoothing mean2 --velocity-noise 0.5 --episode-seconds 32.0 \
    --updates $UPDATES --seed $2 --threads $THREADS \
    --contact "touchdown_overshoot=False" \
    --randomize "action_delay_controls=(1, 2)" \
    --randomize "kd_scale=(0.13, 0.37)" \
    --randomize "friction_scale=(0.45, 0.6)" \
    --randomize "joint_friction_nm=((0.0335, 0.0622), (0.1387, 0.2576), (0.018, 0.0335))" \
    --randomize "armature_kg_m2=((0.0, 0.0001), (0.0, 0.00058), (0.0, 0.00049))" \
    --ppo-config-overrides '{"num_steps_per_env": 36, "algorithm": {"num_mini_batches": 6}, "environment_wrapper": {"destination_input": "terrain_v2_odom", "gait_clock_scaling": "speed_v1"}}' \
    > $C/${1}_train.log 2>&1
}
judge() {  # NAME CHECKPOINT SEED
  root=$C/fitted96_all5_drawnkd_nylon750_${1}_s$3
  [[ -e $root || -e $root.log ]] && { echo "$root exists; not overwriting"; return 1; }
  uv run python $S/body_mass.py run $S/fitted_eval.py $root $1=$2 --motor-check drawn --seed $3 $MATCH > $root.log 2>&1
}
SEEDS=(20261110 20261111 20261112)
[[ $UPDATES -lt 100 ]] && SEEDS=(20261110)
echo "=== bmass1$SUFFIX training $(date '+%Y-%m-%d %H:%M:%S') ==="
ARMS=()
for seed in $SEEDS; do n=${seed: -1}; train_arm bm$n$SUFFIX $seed & ARMS+=(bm$n$SUFFIX); done
wait
K=$(printf %06d $UPDATES)
for A in $ARMS; do
  echo "$A: $(grep -E 'DISTILL update' $C/${A}_train.log | tail -1)"
  grep -h "BODY MASS" $C/${A}_train.log | cut -c1-400
  [[ -f $C/$A/checkpoint_update$K.pt ]] || { echo "BMASS1 STOPPED: $A has no checkpoint $K"; tail -8 $C/${A}_train.log; exit 1; }
done
echo "=== trained $(date '+%Y-%m-%d %H:%M:%S') ==="
MATCH=()
CHECKS=($UPDATES)
[[ $UPDATES -lt 100 ]] && MATCH=(--match translate_0.10_0deg stand)
[[ $UPDATES -ge 1000 ]] && CHECKS=(500 $UPDATES)
JOBS=()
for U in $CHECKS; do KU=$(printf %06d $U); for A in $ARMS; do JOBS+=("${A}_u$U $C/$A/checkpoint_update$KU.pt"); done; done
if [[ $UPDATES -ge 1000 ]]; then
  JOBS+=("distill1_u1000 $C/distill1/checkpoint_update001000.pt" "dc1_u1000 $C/dc1/checkpoint_update001000.pt" "dc2_u1000 $C/dc2/checkpoint_update001000.pt")
fi
for J in $JOBS; do
  for seed in 20261110 20261111; do
    judge ${=J} $seed &
    (( ${#jobstates} >= 6 )) && wait
  done
done
wait
grep -h "Traceback\|Error" $C/fitted96_all5_drawnkd_nylon750_*_s*.log | head -5
grep -H -o "^FITTED [a-z0-9_]*: pass [0-9]*/[0-9]*" $C/fitted96_all5_drawnkd_nylon750_*_s*.log | sed 's#.*/fitted96_all5_drawnkd_nylon750_##'
echo "BMASS1 DONE $(date '+%Y-%m-%d %H:%M:%S')"
