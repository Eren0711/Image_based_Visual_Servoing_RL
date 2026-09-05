#!/bin/zsh
# Train a specified list of seeds for one λ. Usage:
#   zsh scripts/legacy/training/run_hardnet_d_seeds.sh <lambda> <seed1> [seed2] [seed3] ...
set -e
REPO_ROOT=${0:A:h}/../../..
cd "$REPO_ROOT"
PYTHON_BIN=${PYTHON:-python3}
LAM=$1; shift
TAG=$(echo $LAM | tr -d '.')
WARM=logs/stages/stage4b_dr_finetune/models/ibvs_ppo_best
for SEED in "$@"; do
  echo "================ D lambda=$LAM seed=$SEED  $(date) ================"
  "$PYTHON_BIN" train.py \
    --config configs/legacy/stage3_stage4.yaml \
    --stage stage4a_hardnet_d_lam${TAG}_seed${SEED} \
    --timesteps 3000000 --n-envs 16 \
    --noise-delay --dkf --hardnet --stage4b --lr-decay \
    --feasibility-coef $LAM \
    --seed $SEED --resume $WARM
done
echo "================ LAMBDA $LAM SEEDS [$@] DONE  $(date) ================"
