#!/bin/zsh
# Multi-seed HardNet variance study (Plan A).
# Replicates the original stage4a_hardnet_finetune config exactly, varying
# only the master seed. Runs sequentially so each gets full CPU.
set -e
REPO_ROOT=${0:A:h}/../../..
cd "$REPO_ROOT"
PYTHON_BIN=${PYTHON:-python3}
WARM=logs/stages/stage4b_dr_finetune/models/ibvs_ppo_best

for SEED in 42 123 7; do
  echo "================ HardNet seed=$SEED  $(date) ================"
  "$PYTHON_BIN" train.py \
    --config configs/legacy/stage3_stage4.yaml \
    --stage stage4a_hardnet_seed${SEED} \
    --timesteps 3000000 --n-envs 16 \
    --noise-delay --dkf --hardnet --stage4b --lr-decay \
    --seed $SEED \
    --resume $WARM
done
echo "================ ALL SEEDS DONE  $(date) ================"
