# compclass — electronic-component image classification across model families

One importable package (`src/compclass`) trains, probes and evaluates every model family the project compares,
from a CNN written from scratch to CLIP / SigLIP / DINOv2 foundation models. All families share one leakage-free
dataset (`data/v2/manifest.json`), cluster-grouped 5-fold cross-validation and a sealed test split. The
repository has three parts:

1. A **comparative study** of 11 model families under three adaptation regimes (zero-shot, linear probe, fine-tune) plus a classical baseline, on 20-class component recognition:
   [`docs/report/REPORT.md`](docs/report/REPORT.md).
2. A **deployable classifier** with calibrated abstention, exported to ONNX, and an OpenCV demo for files, folders,
   URLs and webcams: [`docs/DEMO.md`](docs/DEMO.md).
3. A **learning resource** on model families, transfer learning and honest evaluation:
   [`docs/LEARNING_PATH.md`](docs/LEARNING_PATH.md) and the interactive, offline explorer
   [`docs/explorer/index.html`](docs/explorer/index.html) (open in a browser: walkthrough, benchmark explorer,
   abstention slider, and an in-depth page per model family).

Also: the IEEE-format paper [`docs/paper/main.pdf`](docs/paper/main.pdf) (source `docs/paper/main.tex`), the
GPUI desktop demo [`apps/gpui-demo/`](apps/gpui-demo/README.md) (Rust, onnxruntime), and the evaluation on the
original Kaggle download [`results/kaggle_original/`](results/kaggle_original/summary.json)
(`scripts/eval_original_kaggle.py`): 83.0 % on the 493 never-seen files; 87.7 % vs the old notebooks' 71.3 % when the
old 70/15/15 protocol is replayed on the same legacy-test files.

## Headline results

Development numbers are grouped 5-fold CV on 2,512 images (mean ± SD over folds). Sealed-test numbers come from
one look at 418 held-out images. Sources: `results/benchmark.csv`, `results/final/final_test_v2.json`.

| Model | Regime | CV macro-F1 | Notes |
|---|---|---:|---|
| SigLIP So400m/14@384 | frozen linear probe | **0.824 ± 0.012** | best family; basis of the final model |
| SigLIP So400m/14@384 | partial fine-tune (folds 0–2) | 0.812 ± 0.023 | 76M trainable params, 17.5 min/fold, no gain over the probe |
| SigLIP So400m/14@384 | zero-shot (no training) | 0.776 ± 0.018 | beats every non-SigLIP network |
| SigLIP 2 B/16 / CLIP ViT-B/16 | full fine-tune | 0.733 / 0.702 | |
| ImageNet CNNs & transformers | full fine-tune | 0.548–0.654 | ViT-B/16 best, MobileNetV3-L worst |
| Scratch CNN | trained from random init | 0.355 ± 0.015 | |
| HOG + colour histogram + SVM | classical | 0.206 ± 0.017 | |

**Final model** (`exports/final/`): frozen SigLIP So400m image tower + logistic-regression probe fused with the
model's own zero-shot text head, temperature-calibrated (T = 0.898), abstaining below confidence 0.5.
**Sealed test:** accuracy 0.837 (95 % CI 0.799–0.870), macro-F1 0.834, in-scope accuracy 0.869, ECE 0.034. When it
answers (87 % of images), it is right 90.9 % of the time.

Caveats, all detailed in the report:

* The source data is a noisy web scrape. An automated CLIP screen flags 27.8 % of images as out of scope
  (e.g. Nintendo Switch consoles labelled *switch*). These flags were not human-verified.
* With 418 test images the accuracy SE is about ±1.8 points. The zero-shot fusion is a tie with the plain probe (OOF
  0.826 vs 0.825; test 0.837 vs 0.845), and the LR-schedule fix, flip TTA and data cleaning are null results.
* "Frozen beats fine-tuned" holds for SigLIP only. For ImageNet, DINOv2 and CLIP-B backbones, fine-tuning
  gains 3.6–6.5 points.
* The 1.6 GB ONNX graph runs exactly in onnxruntime (max |Δp| 6e-6, ~0.9 s/image on CPU), but OpenCV's DNN importer
  segfaults on it, so the demo uses onnxruntime for inference.
* On 407 fresh Wikimedia Commons photos (`data/web_holdout_v1/`, evaluated once) the final model scores
  **0.821 accuracy / 0.816 macro-F1**; here the zero-shot head helps (+4.2 points over the probe alone).
* The old notebooks' 0.71–0.76 accuracies (`legacy/`) were inflated by duplicate leakage (+5.5 to +13.6 points in a
  replay; report §3.3).

## Quickstart (uv)

```bash
uv sync                      # .venv from pyproject.toml + uv.lock (CUDA 12.8 torch wheels on Linux)
uv run pytest -q             # ~10 s CPU smoke tests on a synthetic manifest
uv run python -c "from compclass.models import describe; print(describe())"   # family explainers
export HF_HOME=~/models_cache/hf TORCH_HOME=~/models_cache/torch               # persistent weight caches

# classify images with the exported final model (see docs/DEMO.md for all options)
uv run python -m compclass.demo --source path/to/image_or_folder_or_url --out annotated/
uv run python -m compclass.demo --source 0                                       # webcam

# rebuild every report figure and table from the committed results (CPU, seconds)
uv run python scripts/make_report_figures.py
```

`exports/final/model.onnx(.data)` is not committed (1.6 GB). Recreate it with `compclass.final export` (below).

## Repository map

```
src/compclass/        package: data (manifest, folds, transforms), models/registry.py (one builder per family),
                      train, features (frozen probe), zeroshot, classical, evaluate, export, final, demo
configs/              base.yaml (every knob, commented) + one YAML per family / ablation
tools/                build_dataset_v2.py (dataset v2 + leakage report), build_web_holdout.py
scripts/              make_report_figures.py (report figures/tables), gpu_witness.py (hardware check)
data/v2/              manifest.json, leakage_report.json, review/ (duplicate, conflict and OOD review sheets)
results/              phase1/ (probes, zero-shot, HOG, label audit, cleaning), phase1b/ (TTA probes),
                      phase2/ (fine-tunes: runs_fixed.jsonl main, raw_oldschedule/ ablation, per-fold preds),
                      benchmark.csv, leakage_simulation.csv, schedule_ablation_folds012.csv, final/ (sealed test)
exports/final/        head.npz, labels.json, preprocess.json, fit_report.json, export_report.json
docs/report/          REPORT.md + figures/ + tables/ (CSV and booktabs .tex)
docs/models/, docs/concepts/, docs/LEARNING_PATH.md   per-family explainers, concept notes, reading order
docs/DEMO.md          OpenCV demo guide
legacy/               pre-v2 notebooks and docs (numbers invalid: duplicate leakage)
```

Further reading: [`CODEBASE.md`](CODEBASE.md) (code map, commands, gotchas), [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)
(dated decisions and results), [`STATUS.md`](STATUS.md) (resume checkpoint).

## Reproducing each phase

GPU phases ran on one RTX 4060 Ti (8 GB). Each command appends one JSON line per fold to its `--results` file (or to
the config's `output.results`).

```bash
# Phase 0 — dataset v2 (expects the Kaggle folders under data/raw/electronic-component-recognizer/)
uv run --no-config tools/build_dataset_v2.py            # writes data/v2/{manifest.json,leakage_report.json,review/}

# Phase 1 — frozen probes, zero-shot, classical baseline (5 folds each)
uv run python -m compclass.features --model siglip_so400m_384 --results results/phase1/raw/runs.jsonl
uv run python -m compclass.zeroshot --model siglip_so400m_384 --results results/phase1/raw/runs.jsonl
uv run python -m compclass.classical --results results/phase1/raw/runs.jsonl
#   other keys: siglip2_so400m_384 siglip_b16 siglip2_b16 clip_vitl14_336 clip_vitl14 clip_vitb16
#               dinov2_vitl14 dinov2_vitb14 dinov2_vits14 convnext_tiny resnet50 (probe only for the last two)
uv run python -m compclass.features --model siglip_so400m_384 --tta --results results/phase1b/runs.jsonl  # flip TTA

# Phase 2 — fine-tune benchmark (one YAML per family; main table = fixed schedule, all 5 folds)
uv run python -m compclass.train --config configs/convnext_tiny.yaml --fold 0 1 2 3 4
uv run python -m compclass.train --config configs/siglip_so400m_384_partial_v2.yaml --fold 0 1 2

# Phase 3 — final model (fit/calibrate on dev OOF, export ONNX + parity, ONE sealed-test look)
uv run python -m compclass.final fit
uv run python -m compclass.final export
uv run python -m compclass.final test --confirm         # already spent: results/final/final_test_v2.json
uv run python -m compclass.final test --manifest data/web_holdout_v1/manifest.json --confirm   # web holdout (done once)

# Phase 4 — demo and report
uv run python -m compclass.demo --source <file|folder|url|webcam-index>
uv run python scripts/make_report_figures.py
```

The summary tables `results/benchmark.csv`, `results/leakage_simulation.csv`, `results/schedule_ablation_folds012.csv`
and `results/phase1/cleaning_ablation_siglip_so400m.csv` were aggregated by analysis code that is not in the
repository. `scripts/make_report_figures.py` re-derives `benchmark.csv` and the schedule ablation from the per-fold
records and asserts that they agree.

## Syncing the laptop checkout to the GPU PC

The GPU PC (`carlpc2`) keeps its copy at `~/Projects/626-Project`. Push code with rsync over Tailscale SSH (port
2222). The virtualenv and raw data are excluded, because each machine builds its own `.venv` with `uv sync` and only
`data/v2` is needed for training:

```bash
rsync -avz --delete -e "ssh -p 2222" \
  --exclude .venv/ --exclude .git/ --exclude data/raw/ --exclude runs/ --exclude runs_v2/ --exclude results/cache/ \
  --exclude '__pycache__/' --exclude '*.onnx' --exclude '*.onnx.data' \
  ~/Work/Prototypes/626-Project/ carl@carlpc2.tail498db1.ts.net:~/Projects/626-Project/
ssh -p 2222 carl@carlpc2.tail498db1.ts.net 'cd ~/Projects/626-Project && uv sync --python 3.12'
```

`--delete` makes the PC mirror the laptop. `runs/`, `runs_v2/` and `results/cache/` are excluded, so trained
checkpoints on the PC are never deleted. To bring results back, reverse the direction for the specific `results/…`
file you need.

## Evaluation contract (short)

Model choices use only the `dev` split through `compclass.data.get_fold(manifest, k)`. Near-duplicate clusters never
straddle folds. The `test` split is opened once, for the final model, via `get_test(manifest, confirm=True)` /
`compclass.final test --confirm`. That look has been spent (2026-10-18). Details in
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) and report §3.
