#!/bin/zsh
# Intervention C+B: entropy/std fix + curriculum DR. 3 seeds, exact replica
# of the baseline HardNet config EXCEPT: --curriculum (2M anneal),
# --max-log-std 0.0 (std cap), --ent-coef 0.001.
set -e
REPO_ROOT=${0:A:h}/../../..
cd "$REPO_ROOT"
PYTHON_BIN=${PYTHON:-python3}
WARM=logs/stages/stage4b_dr_finetune/models/ibvs_ppo_best
for SEED in 42 123 7; do
  echo "================ HardNet C+B seed=$SEED  $(date) ================"
  "$PYTHON_BIN" train.py \
    --config configs/legacy/stage3_stage4.yaml \
    --stage stage4a_hardnet_cb_seed${SEED} \
    --timesteps 3000000 --n-envs 16 \
    --noise-delay --dkf --hardnet --stage4b --lr-decay \
    --curriculum --curriculum-steps 2000000 \
    --max-log-std 0.0 --ent-coef 0.001 \
    --seed $SEED --resume $WARM
done
echo "================ C+B ALL SEEDS DONE  $(date) ================"
