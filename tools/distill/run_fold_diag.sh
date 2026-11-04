#!/usr/bin/env bash
# Diagnostic (unofficial): does unlabeled in-domain data close the student's gap? Fine-tune the pool-distilled
# B/16 for 2 epochs with the dev images outside fold K added (label-free), and a control with the pool alone.
# Only fold K's accuracy of the first run is clean; the screen's other folds saw their own images.
#   setsid nohup flock /tmp/626-gpu.lock tools/distill/run_fold_diag.sh K > LOG 2>&1 < /dev/null &
set -euo pipefail
K=$1
R=/home/carl/626-runs/distill
M=/home/carl/Projects/626-Project/data/v2/manifest.json
cd "$R/repo"
export HF_HOME=/home/carl/models_cache/hf PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python tools/distill/fold_pool.py --manifest "$M" --fold "$K" --out "$R/foldpool_$K"
.venv/bin/python tools/distill/teacher.py --pool "$R/foldpool_$K" --out "$R/teacher_fold$K" --views-component 8 \
  --workers 10
ft=(--student siglip2_b16_256 --pool "$R/pool" --teacher "$R/teacher_v8g2" --text-weights "$R/v2meta/text_weights.npy"
    --init "$R/runs/siglip2_b16_256/final.pt" --component-repeat 1 --epochs 2 --batch 48 --lr 5e-5 --head-lr 2e-4
    --warmup-steps 200 --layer-decay 0.8 --drop-path 0.1 --ema 0.9995 --workers 10)
.venv/bin/python tools/distill/train.py "${ft[@]}" --out "$R/runs/siglip2_b16_256_ft_fold$K" \
  --extra-pool "$R/foldpool_$K" --extra-teacher "$R/teacher_fold$K" --extra-repeat 4
.venv/bin/python tools/distill/train.py "${ft[@]}" --out "$R/runs/siglip2_b16_256_ft_poolonly"
.venv/bin/python tools/distill/export.py screen --run "$R/runs/siglip2_b16_256_ft_fold$K" \
  "$R/runs/siglip2_b16_256_ft_poolonly" --manifest "$M" --head "$R/v2meta/head.npz" --cache "$R/cache/features" \
  --report "$R/screen_fold_diag_$K.json"
