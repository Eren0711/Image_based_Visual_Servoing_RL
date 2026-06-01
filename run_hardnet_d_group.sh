#!/bin/zsh
# Train ONE λ-group: 3 seeds for the λ passed as $1. Used for the gated
# λ-sweep (one group at a time, pause for confirmation between groups).
# Usage: zsh run_hardnet_d_group.sh <lambda>
set -e
LAM=$1
TAG=$(echo $LAM | tr -d '.')
PY=/opt/homebrew/Caskroom/miniconda/base/envs/ibvs_rl/bin/python
WARM=logs/stages/stage4b_dr_finetune/models/ibvs_ppo_best
for SEED in 42 123 7; do
  echo "================ D lambda=$LAM seed=$SEED  $(date) ================"
  $PY train.py \
    --stage stage4a_hardnet_d_lam${TAG}_seed${SEED} \
    --timesteps 3000000 --n-envs 16 \
    --noise-delay --dkf --hardnet --stage4b --lr-decay \
    --feasibility-coef $LAM \
    --seed $SEED --resume $WARM
done
echo "================ LAMBDA $LAM GROUP DONE  $(date) ================"
