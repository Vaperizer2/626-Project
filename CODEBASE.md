# Codebase map

```
src/compclass/
  config.py          YAML loading: `base:` inheritance, `--set a.b=v` overrides, config hash
  data.py            v2 manifest loader + validation (schema, cluster/fold integrity, fingerprint),
                     get_fold / get_dev / get_test(confirm=True), ImageDataset, transforms, Mixup/CutMix,
                     make_synthetic_manifest (tests and smoke runs)
  models/registry.py ONE registry: key -> builder -> (nn.Module, PreprocessSpec); a docstring per family
  models/scratch_cnn.py  the from-first-principles CNN
  train.py           fine-tuning loop + CLI (AdamW, warm-up+cosine, layer-wise LR decay, frozen-head warm-up,
                     label smoothing / mixup, AMP fp16|bf16, grad accumulation, early stopping on macro-F1)
  evaluate.py        metrics (acc, macro-F1, per-class P/R/F1, confusion), predict(), log_run -> results/runs.jsonl,
                     CLI for re-evaluating checkpoints (and the one sealed-test look)
  features.py        frozen-feature extraction (cached .npy) + logistic-regression linear probe (C chosen inside
                     the training fold by GroupKFold)
  zeroshot.py        CLIP / SigLIP zero-shot with prompt templates (open_clip text towers)
  classical.py       HOG + HSV colour histogram + RBF-SVM baseline
  export.py          ONNX export + labels.json + preprocess.json + onnxruntime parity check
configs/             base.yaml (every knob, commented) + one file per family
tests/test_smoke.py  CPU smoke tests on a synthetic 3-class manifest (~10 s)
scripts/gpu_witness.py  per-model download/build/fp16 train-step benchmark + end-to-end CLI witness on the GPU PC
legacy/              pre-v2 notebooks and docs (numbers invalid: duplicate leakage)
```

## Model registry

| key | family | timm weights | input | zero-shot |
|---|---|---|---:|---|
| scratch_cnn | Scratch CNN | — (random init) | 160 | |
| resnet50 | ResNet | resnet50.a1_in1k | 224 | |
| efficientnet_v2_s | EfficientNet-V2 | tf_efficientnetv2_s.in21k_ft_in1k | 300 | |
| mobilenetv3_large | MobileNet-V3 | mobilenetv3_large_100.ra_in1k | 224 | |
| convnext_tiny | ConvNeXt | convnext_tiny.fb_in22k_ft_in1k | 224 | |
| vit_b16 | ViT | vit_base_patch16_224.augreg_in21k_ft_in1k | 224 | |
| swin_t | Swin | swin_tiny_patch4_window7_224.ms_in22k_ft_in1k | 224 | |
| dinov2_vits14 / dinov2_vitb14 | DINOv2 | vit_{small,base}_patch14_dinov2.lvd142m | 224 (native 518) | |
| clip_vitb16 / clip_vitl14 | CLIP | vit_{base_patch16,large_patch14}_clip_quickgelu_224.openai | 224 | ViT-B-16 / ViT-L-14 openai |
| siglip_b16 / siglip2_b16 | SigLIP / SigLIP 2 | vit_base_patch16_siglip_224.{webli,v2_webli} | 224 | ViT-B-16-SigLIP(2) webli |

Add a family by writing one decorated builder in `registry.py` (with an explainer docstring) and one YAML file.

## How a run flows

1. `load_config(configs/<x>.yaml, --set ...)` -> dict; `config_hash` identifies the recipe.
2. `load_manifest` validates the manifest and its fingerprint; `get_fold(m, k)` gives dev-only train/val rows.
3. `build_model(key, n_classes)` returns the network and its `PreprocessSpec`; `build_transforms(spec, aug, train)`
   derives train/eval pipelines from it (so preprocessing always matches the weights).
4. `train.run` trains, keeps `runs/<name>-<hash>/fold<k>/best.pt` (+ `val_preds.npz` for OOF analysis/ensembles)
   and appends a record to `results/runs.jsonl`.
5. `export.py` turns a checkpoint into `model.onnx` + `labels.json` + `preprocess.json` for the OpenCV demo.

## Commands

```bash
uv run pytest -q
uv run python -m compclass.train --config configs/vit_b16.yaml --fold 0 1 2 3 4 [--set train.lr=5e-5 ...]
uv run python -m compclass.features --model dinov2_vitb14 [--folds 0 1 2 3 4]
uv run python -m compclass.zeroshot --model clip_vitl14
uv run python -m compclass.classical
uv run python -m compclass.evaluate --checkpoint runs/<run>/fold0/best.pt [--sealed-test]
uv run python -m compclass.export --checkpoint runs/<run>/fold0/best.pt --out exports/<name> [--dynamic-batch]
uv run python scripts/gpu_witness.py --out results/gpu_witness.json
```

Reading results: `pandas.read_json("results/runs.jsonl", lines=True)` then group by `name`/`method` over `fold`.

## Gotchas

* PyYAML reads `1e-3` as a string; write `1.0e-3` in YAML (CLI `--set` values are coerced for you).
* The lockfile pins CUDA 12.8 wheels on Linux (torch 2.11.0+cu128 — the newest on that index); other platforms
  resolve plain PyPI torch.
* DINOv2 runs at 224 px (position embeddings interpolated from 518) to keep training affordable on 8 GB.
* Layer-wise LR decay applies to timm models only (they expose `group_matcher`); for ConvNeXt the fine-grained
  grouping has ~20 layers, so 0.8 already gives the stem a ~70x smaller LR than the head.
