#!/bin/zsh
# Intervention D: auxiliary feasibility loss (λ=0.05). C and B OFF (they hurt
# in the C+B study). Exact replica of the baseline HardNet recipe otherwise:
# 3M steps, warm-start from 4b 1M, full DR from step 0, HOCBF context,
# linear LR decay. 3 seeds.
set -e
REPO_ROOT=${0:A:h}/../../..
cd "$REPO_ROOT"
PYTHON_BIN=${PYTHON:-python3}
WARM=logs/stages/stage4b_dr_finetune/models/ibvs_ppo_best
for SEED in 42 123 7; do
  echo "================ HardNet D seed=$SEED  $(date) ================"
  "$PYTHON_BIN" train.py \
    --config configs/legacy/stage3_stage4.yaml \
    --stage stage4a_hardnet_d_seed${SEED} \
    --timesteps 3000000 --n-envs 16 \
    --noise-delay --dkf --hardnet --stage4b --lr-decay \
    --feasibility-coef 0.05 \
    --seed $SEED --resume $WARM
done
echo "================ D ALL SEEDS DONE  $(date) ================"
