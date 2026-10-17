# Learning path — from hand-crafted features to vision-language models

An ordered route through the docs and the code. Each stop has something to **read**, something to **run**, and a
**question** to answer from the results. Run everything from the repo root after `uv sync`. Results accumulate in
`results/runs.jsonl`; this one-liner summarises them at any point:

```bash
uv run python -c "import pandas as pd; d=pd.read_json('results/runs.jsonl', lines=True); print(d.groupby(['method','name'])[['accuracy','macro_f1']].agg(['mean','std','count']))"
```

Fold 0 alone is enough for exploring; use `--fold 0 1 2 3 4` (or `--folds …`) for anything you will report.

## Stop 0 — the ground rules (data and evaluation)

* Read: [`concepts/evaluation.md`](concepts/evaluation.md), [`concepts/data_quality.md`](concepts/data_quality.md),
  `src/compclass/data.py` (manifest validation, `get_fold`, the sealed `get_test`), `src/compclass/config.py`
  (`base:` inheritance, `--set` overrides).
* Run: `uv run pytest -q` (CPU smoke tests on a synthetic 3-class manifest) and
  `uv run python -c "from compclass.models import describe; print(describe())"` (every family's docstring).
* Question: why does `load_manifest` refuse a manifest whose fingerprint does not match its rows?

## Stop 1 — classical computer vision

* Read: [`models/classical.md`](models/classical.md), `src/compclass/classical.py`.
* Run: `uv run python -m compclass.classical`
* Question: which classes does HOG + colour handle best, and why those? (Look at `per_class` in the run log.)

## Stop 2 — a CNN from first principles

* Read: [`models/scratch_cnn.md`](models/scratch_cnn.md), `src/compclass/models/scratch_cnn.py`, then the training loop
  in `src/compclass/train.py` (`run`, `make_optimizer`, `warmup_cosine`).
* Run: `uv run python -m compclass.train --config configs/scratch_cnn.yaml --fold 0`
* Question: does the learned CNN beat the classical baseline? How do the training-loss and validation-F1 curves in the
  `history` field differ?

## Stop 3 — ImageNet-pretrained CNNs and transfer learning

* Read: [`concepts/transfer_learning.md`](concepts/transfer_learning.md), then [`models/resnet.md`](models/resnet.md) →
  [`models/mobilenet.md`](models/mobilenet.md) → [`models/efficientnet.md`](models/efficientnet.md) →
  [`models/convnext.md`](models/convnext.md); `src/compclass/models/registry.py` (`_timm`, `PreprocessSpec`).
* Run:
  ```bash
  uv run python -m compclass.train --config configs/resnet50.yaml --fold 0
  uv run python -m compclass.train --config configs/mobilenetv3_large.yaml --fold 0
  uv run python -m compclass.train --config configs/efficientnet_v2_s.yaml --fold 0
  uv run python -m compclass.train --config configs/convnext_tiny.yaml --fold 0
  ```
* Question: how large is the jump from Stop 2 to ResNet-50? Plot macro-F1 against `params` and `train_img_per_s` from
  the run log — which model would you deploy on a CPU?

## Stop 4 — Vision Transformers

* Read: [`models/vit.md`](models/vit.md) → [`models/swin.md`](models/swin.md).
* Run:
  ```bash
  uv run python -m compclass.train --config configs/vit_b16.yaml --fold 0
  uv run python -m compclass.train --config configs/swin_t.yaml --fold 0
  uv run python -m compclass.train --config configs/vit_b16.yaml --fold 0 --set train.layer_decay=null
  ```
* Question: Swin-T and ConvNeXt-T have almost the same size and pretraining data. What does their comparison say
  about attention vs convolution? What does removing layer-wise LR decay do to ViT-B?

## Stop 5 — self-supervised features

* Read: [`models/dinov2.md`](models/dinov2.md), `src/compclass/features.py`.
* Run:
  ```bash
  uv run python -m compclass.features --model dinov2_vitb14
  uv run python -m compclass.train --config configs/dinov2_vitb14.yaml --fold 0
  ```
* Question: how close is a *linear probe* on frozen DINOv2 features to the best fine-tuned model from Stops 3–4, at what
  fraction of the compute?

## Stop 6 — vision-language models

* Read: [`models/clip.md`](models/clip.md) → [`models/siglip.md`](models/siglip.md), `src/compclass/zeroshot.py`.
* Run:
  ```bash
  uv run python -m compclass.zeroshot --model clip_vitb16
  uv run python -m compclass.zeroshot --model siglip2_b16
  uv run python -m compclass.features --model siglip2_b16
  uv run python -m compclass.train --config configs/clip_vitb16.yaml --fold 0
  ```
* Question: zero-shot uses *no* training images. Which classes does it get wrong, and are those the classes with the
  most `ood_suspect` flags?

## Stop 7 — from checkpoint to deployment

* Read: `src/compclass/evaluate.py`, `src/compclass/export.py`.
* Run:
  ```bash
  uv run python -m compclass.evaluate --checkpoint runs/<name>-<hash>/fold0/best.pt
  uv run python -m compclass.export --checkpoint runs/<name>-<hash>/fold0/best.pt --out exports/<name>
  ```
* Question: `preprocess.json` lists resize, crop, mean and std. What breaks if the OpenCV demo uses ImageNet mean/std
  for a SigLIP model?

## Stop 8 — the comparison you will write up

Run the 5-fold versions of the stops you care about, then compare families on mean ± std macro-F1, parameters,
training throughput and inference latency, with paired tests for close calls ([`concepts/evaluation.md`](concepts/evaluation.md) §5).
Only at the very end: one sealed-test evaluation of the frozen final model.

## Results on data v2

> **Placeholder — no v2 results exist yet.** See `results/benchmark.csv` (generated from `results/runs.jsonl`) once the benchmark has run. Report mean ± std over the 5 dev folds; never quote the sealed test here.

| method | key | CV macro-F1 | CV accuracy | notes |
|---|---|---:|---:|---|
| — | — | *pending* | *pending* | |


## Map of the docs

| doc | topic |
|---|---|
| [`models/classical.md`](models/classical.md) | HOG + HSV histogram + RBF-SVM |
| [`models/scratch_cnn.md`](models/scratch_cnn.md) | VGG-style CNN from random init |
| [`models/resnet.md`](models/resnet.md) | residual CNNs (ResNet-50, RSB weights) |
| [`models/mobilenet.md`](models/mobilenet.md) | mobile CNN (MobileNet-V3-Large) |
| [`models/efficientnet.md`](models/efficientnet.md) | NAS + compound scaling (EfficientNet-V2-S) |
| [`models/convnext.md`](models/convnext.md) | modernised CNN (ConvNeXt-T) |
| [`models/vit.md`](models/vit.md) | Vision Transformer (ViT-B/16) |
| [`models/swin.md`](models/swin.md) | hierarchical shifted-window transformer (Swin-T) |
| [`models/dinov2.md`](models/dinov2.md) | self-supervised ViT (DINOv2 S/14, B/14) |
| [`models/clip.md`](models/clip.md) | contrastive image–text (CLIP B/16, L/14) |
| [`models/siglip.md`](models/siglip.md) | sigmoid image–text (SigLIP, SigLIP 2 B/16) |
| [`concepts/transfer_learning.md`](concepts/transfer_learning.md) | regimes, forgetting, LLRD, every training knob |
| [`concepts/evaluation.md`](concepts/evaluation.md) | grouped CV, sealed test, leakage, macro-F1, confidence intervals |
| [`concepts/data_quality.md`](concepts/data_quality.md) | label noise, out-of-scope images, confident learning |
