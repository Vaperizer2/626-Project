#!/usr/bin/env bash
# The exact distillation runs, in order, on carlpc2 (RTX 4060 Ti 8 GB) under the shared GPU lock.
# Batches are the largest that fit in 8 GB with bf16 autocast (B/16@256: 5.7 GB at 48; FastViT-MCi0: 5.9 GB at 64).
#   setsid nohup flock /tmp/626-gpu.lock tools/distill/run_carlpc2.sh STUDENT... > LOG 2>&1 < /dev/null &
set -euo pipefail
R=/home/carl/626-runs/distill
cd "$R/repo"
export HF_HOME=/home/carl/models_cache/hf PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
common=(--pool "$R/pool" --teacher "$R/teacher_v8g2" --text-weights "$R/v2meta/text_weights.npy" --workers 10)
for s in "$@"; do
  case "$s" in
    siglip2_b16_256|siglip2_b16_224)
      .venv/bin/python tools/distill/train.py --student "$s" --out "$R/runs/$s" "${common[@]}" \
        --epochs 8 --batch 48 --lr 1e-4 --head-lr 1e-3 --layer-decay 0.8 --drop-path 0.1 --ema 0.9995 ;;
    mobileclip2_s0)
      .venv/bin/python tools/distill/train.py --student "$s" --out "$R/runs/$s" "${common[@]}" \
        --epochs 8 --batch 64 --lr 2e-4 --head-lr 1e-3 --layer-decay 0.9 --drop-path 0.05 --ema 0.9995 ;;
    *) echo "unknown student $s" >&2; exit 2 ;;
  esac
done
.venv/bin/python tools/distill/export.py screen --run "${@/#/$R/runs/}" \
  --manifest /home/carl/Projects/626-Project/data/v2/manifest.json --head "$R/v2meta/head.npz" \
  --cache "$R/cache/features" --report "$R/screen_$(IFS=_; echo "$*").json"
