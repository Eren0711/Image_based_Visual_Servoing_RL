#!/bin/zsh
# Train a specified list of seeds for one λ. Usage:
#   zsh run_hardnet_d_seeds.sh <lambda> <seed1> [seed2] [seed3] ...
set -e
LAM=$1; shift
TAG=$(echo $LAM | tr -d '.')
PY=/opt/homebrew/Caskroom/miniconda/base/envs/ibvs_rl/bin/python
WARM=logs/stages/stage4b_dr_finetune/models/ibvs_ppo_best
for SEED in "$@"; do
  echo "================ D lambda=$LAM seed=$SEED  $(date) ================"
  $PY train.py \
    --stage stage4a_hardnet_d_lam${TAG}_seed${SEED} \
    --timesteps 3000000 --n-envs 16 \
    --noise-delay --dkf --hardnet --stage4b --lr-decay \
    --feasibility-coef $LAM \
    --seed $SEED --resume $WARM
done
echo "================ LAMBDA $LAM SEEDS [$@] DONE  $(date) ================"
