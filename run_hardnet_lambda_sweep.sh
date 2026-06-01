#!/bin/zsh
# Intervention D λ-sweep: feasibility_coef ∈ {0.02, 0.1, 0.2} × seeds {42,123,7}.
# λ=0.05 already done (stage4a_hardnet_d_seed*). This fills in the ablation
# to locate the optimum and the lazy-policy boundary. Same recipe as D:
# 3M steps, warm-start from 4b 1M, full DR, HOCBF context, linear LR decay.
# Ordered λ-outer so each λ's 3 seeds complete together (usable partial results).
set -e
PY=/opt/homebrew/Caskroom/miniconda/base/envs/ibvs_rl/bin/python
WARM=logs/stages/stage4b_dr_finetune/models/ibvs_ppo_best
for LAM in 0.02 0.1 0.2; do
  TAG=$(echo $LAM | tr -d '.')   # 0.02→002, 0.1→01, 0.2→02
  for SEED in 42 123 7; do
    echo "================ D lambda=$LAM seed=$SEED  $(date) ================"
    $PY train.py \
      --stage stage4a_hardnet_d_lam${TAG}_seed${SEED} \
      --timesteps 3000000 --n-envs 16 \
      --noise-delay --dkf --hardnet --stage4b --lr-decay \
      --feasibility-coef $LAM \
      --seed $SEED --resume $WARM
  done
done
echo "================ LAMBDA SWEEP ALL DONE  $(date) ================"
