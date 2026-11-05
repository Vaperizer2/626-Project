# Codebase map

For setup, model downloads, running the demo and release maintenance, use [the runbook](docs/RUNBOOK.md).
For completed work and remaining limitations, use [STATUS.md](STATUS.md).

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
  export.py          checkpoint ONNX export + labels.json + preprocess.json + parity check
  final.py           fit/calibrate fused SigLIP head, export, held-out evaluation
  demo.py            Python file/folder/URL/webcam inference through onnxruntime
Cargo.toml           Rust workspace: crates/inference and apps/gpui-demo (build from here; binaries in target/)
crates/inference/    image models without a UI: manifests and registry, inference service, onnxruntime with
                     execution-provider fallback and TensorRT engine cache, weight integrity (no GPUI dependency)
apps/gpui-demo/      Rust GPUI Kit 0.7.3 desktop app: Classify, Demo and Benchmark tabs over models
                     A/B/C/Deployed/Fast, headless classify/bench/models, portable Linux/NixOS launchers
scripts/new_app.sh   a new desktop AI app from apps/gpui-demo without the classifier's parts (docs/TEMPLATE.md)
exports/models.json  the app's model index; each export folder has a model.json (tools/gen_manifests.py writes them)
notebooks/           ITRI626_four_models.ipynb (executed A/B/C/D comparison), ITRI626_training.ipynb (one section per
                     model: A, B, C, Deployed, Fast)
tools/               dataset builders, export scripts, tools/speed/ (the frozen speed harness) and the speed tracks
                     behind Fast (speed2-4, compose, quant, prune, tokens, trtbugs, distill), tools/hardcases/
configs/             base.yaml (every knob, commented) + one file per family
tests/test_smoke.py  CPU smoke tests on a synthetic 3-class manifest (~10 s)
scripts/gpu_witness.py  per-model download/build/fp16 train-step benchmark + end-to-end CLI witness (last run on
                     the GPU PC, which is gone: results/gpu_witness_carlpc2.json)
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

## Hardware witness

`results/gpu_witness_carlpc2.json` (RTX 4060 Ti 8 GB, torch 2.11.0+cu128, 2026-10-17): all 13 registry models download,
build and run fp16-AMP train steps (forward + backward + AdamW) at batch 16 and native input size. Peak VRAM ranges
from 0.1 GB (scratch_cnn) to 2.7 GB (dinov2_vitb14) for the base-size models; clip_vitl14 OOMs at batch 16 and needs
gradient checkpointing (5.8 GB, ~32 img/s). The witness also runs 1-epoch synthetic fine-tunes through the CLI, ONNX
export + onnxruntime parity, all four zero-shot models, the linear probe and the classical baseline.

## Gotchas

* PyYAML reads `1e-3` as a string; write `1.0e-3` in YAML (CLI `--set` values are coerced for you).
* The lockfile pins CUDA 12.8 wheels on Linux (torch 2.11.0+cu128 — the newest on that index); other platforms
  resolve plain PyPI torch.
* DINOv2 runs at 224 px (position embeddings interpolated from 518) to keep training affordable on 8 GB.
* `--features turbojpeg` (the app README's recommended build, and the binary that `apps/gpui-demo/scripts/golden.py`
  checks) links the system libjpeg-turbo, and `turbojpeg-sys` 1.2.0 asks pkg-config for version 3.0 or newer. Ubuntu
  24.04 ships 2.1.5, so the build fails there. Install libjpeg-turbo 3.x, or build `--features turbojpeg-vendored`
  (needs cmake and nasm). The portable bundle uses `turbojpeg-vendored` and is not affected.
* `[profile.dev.package."*"]` in the root `Cargo.toml` leaves out workspace members, so `inference` has its own
  `opt-level = 2` line. Without it the GUI tests decode the 68-picture demo stack too slowly and time out.
* Layer-wise LR decay applies to timm models only (they expose `group_matcher`); for ConvNeXt the fine-grained
  grouping has ~20 layers, so 0.8 already gives the stem a ~70x smaller LR than the head.
