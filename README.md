# compclass — electronic-component image classification across model families

**Share / run:** [private GitHub repository](https://github.com/Vaperizer2/626-Project) ·
[download models and Linux demo](https://github.com/Vaperizer2/626-Project/releases/tag/share-2026-11-04) ·
[runbook](docs/RUNBOOK.md) · [current status](STATUS.md) · [agent handoff](AGENTS.md).
That release has all five models (A, B, C, Deployed and Fast) and the 68-picture demo stack. Cloning the source
alone does not download the ONNX weights.
Colleagues need repository access (Settings → Collaborators); send the release link after they accept the invitation.

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

**ITRI626 course submission** (three models on one fixed split, report, notebook, split files): see
[ITRI626 submission](#itri626-submission) at the end of this file.

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

`exports/final/model.onnx` and `model.onnx.data` are release assets inside the Linux bundle, not Git objects.
Follow [the runbook](docs/RUNBOOK.md) to download the models into the source checkout or run the bundled demo.
The shared SigLIP weights alone are 1,714,593,792 bytes (1.60 GiB). Exporting again requires the training inputs/caches;
use the release for inference without rebuilding the dataset.

## Repository map

```
src/compclass/        package: data (manifest, folds, transforms), models/registry.py (one builder per family),
                      train, features (frozen probe), zeroshot, classical, evaluate, export, final, demo
configs/              base.yaml (every knob, commented) + one YAML per family / ablation
tools/                build_dataset_v2.py (dataset v2 + leakage report), build_web_holdout.py, export_splits.py,
                      export_itri626_models.py, gen_manifests.py (the app's model.json files); speed/ (the frozen
                      speed harness), speed2/ to speed4/, compose/, quant/, prune/, tokens/, trtbugs/, distill/
                      (the speed work; Fast comes from compose/ and quant/), hardcases/ (the hard-case study)
scripts/              make_report_figures.py (report figures/tables), gpu_witness.py (hardware check),
                      itri626_experiments.py (models A, B, C), new_app.sh (a new app from the template)
notebooks/            ITRI626_three_models.ipynb (executed A/B/C comparison), ITRI626_training.ipynb (training)
crates/inference/     Rust: the models without a UI (manifests, inference service, onnxruntime)
apps/gpui-demo/       Rust: the GPUI desktop app over crates/inference (README.md there)
data/v2/              manifest.json, leakage_report.json, review/*.csv (duplicate, conflict and OOD audit records)
results/              phase1/ (probes, zero-shot, HOG, label audit, cleaning), phase1b/ (TTA probes),
                      phase2/ (fine-tunes: runs_fixed.jsonl main, raw_oldschedule/ ablation, per-fold preds),
                      benchmark.csv, leakage_simulation.csv, schedule_ablation_folds012.csv, final/ (sealed test)
exports/              models.json (the app's model index); final/ (head.npz, labels.json, preprocess.json,
                      model.json, fit_report.json, export_report.json), fast/, itri626/ (A and B)
docs/report/          REPORT.md + figures/ + tables/ (CSV and booktabs .tex)
docs/models/, docs/concepts/, docs/LEARNING_PATH.md   per-family explainers, concept notes, reading order
docs/DEMO.md          OpenCV demo guide
legacy/               pre-v2 notebooks and docs (numbers invalid: duplicate leakage)
```

Further reading: [`CODEBASE.md`](CODEBASE.md) (code map, commands, gotchas), [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)
(dated decisions and results), [`STATUS.md`](STATUS.md) (resume checkpoint).

## Reproducing each phase

These are historical reproduction commands, not onboarding steps. Both held-out evaluations have already been
used; consult the committed results instead of using those sets for further model selection.

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

## Where training runs

Since 2026-11-03 GPU work runs on the laptop (RTX 3060 Laptop, 6 GB), the only GPU. Heavy CPU work, such as Rust
release builds and test suites, runs on Boat.dev CPU sandboxes (`boat` CLI). The GPU PC used before (`carlpc2`) is no
longer available; its sync commands are in Git history. Its speed3 outputs were copied to the laptop, but speed3's
fp32 base graphs were lost with it, and so were the inputs to rebuild them (the distillation pool images, the
teacher feature cache and the MLP activation statistics). The pool's Caltech-256 and COCO val2017 sources must be
downloaded again; the rest of the chain is in `tools/distill/`, `tools/prune/recover.py`, `tools/prune/build.py` and
`tools/speed2/remote_build.sh`, with GPU PC paths to change. Training that needs more than 6 GB of GPU memory does
not fit the laptop; speed3's full-rank QAT run (Q5) peaked at 6.7 GB. The remote subcommands of
`tools/speed3/run.sh` no longer work.

## Evaluation contract (short)

Model choices use only the `dev` split through `compclass.data.get_fold(manifest, k)`. Near-duplicate clusters never
straddle folds. The `test` split is opened once, for the final model, via `get_test(manifest, confirm=True)` /
`compclass.final test --confirm`. That look has been spent (2026-10-18). Details in
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) and report §3.

## ITRI626 submission

Module ITRI626 *Deep learning for image classification* (Prof. Absalom Ezugwu). Three model families trained on
**one** fixed train/validation/test split and evaluated on the **same** 418-image test set:

| | Model | Family | Transfer learning | Test acc. | Test macro-F1 |
|---|---|---|---|---:|---:|
| A | custom VGG-style CNN (1.19 M params) | plain CNN | none (random init) | 0.354 | 0.336 |
| B | ConvNeXt-Tiny, fully fine-tuned | modernised CNN | ImageNet-22k → 1k weights, fine-tuned | 0.636 | 0.628 |
| C | SigLIP So400m/14 @ 384 px, frozen + `nn.Linear(1152, 20)` | Vision Transformer | frozen features, linear head trained | 0.823 | 0.817 |

Source of these numbers: `results/itri626/metrics.json`.

| Deliverable | Location |
|---|---|
| Report (PDF, LaTeX source) | [`docs/itri626_report/main.pdf`](docs/itri626_report/main.pdf), `docs/itri626_report/main.tex` |
| Rubric compliance checklist | [`docs/itri626_report/RUBRIC_CHECKLIST.md`](docs/itri626_report/RUBRIC_CHECKLIST.md) (also report Appendix E) |
| Executed notebook (outputs, figures, metric tables) | [`notebooks/ITRI626_three_models.ipynb`](notebooks/ITRI626_three_models.ipynb) (+ `notebooks/itri626_helpers.py`) |
| Training notebook (one section per model: A, B, C, Deployed, Fast) | [`notebooks/ITRI626_training.ipynb`](notebooks/ITRI626_training.ipynb) |
| Experiment script | `scripts/itri626_experiments.py` → `results/itri626/` (metrics, per-class, histories, predictions, figures) |
| Data preparation | `tools/build_dataset_v2.py` (cleaning, duplicate clustering, grouped split), `tools/export_splits.py` |
| Split files | `data/v2/splits/{train,val,test}.csv`, `split_summary.csv` (per-class counts), `split_info.json` |
| Presentation slides | to be prepared by the group (not in this repository) |

### Dataset source and licence

* *Electronic Component Recognizer* by **mdfaisalahmed025** on Kaggle:
  <https://www.kaggle.com/datasets/mdfaisalahmed025/electronic-component-recognizer>. Licence: **MIT** (use, copy,
  modify, redistribute with the copyright notice; no warranty).
* The author collected the images with DuckDuckGo image search, one query per class; the class labels are the search
  terms. The MIT licence covers the collection; the copyright of the individual web images stays with their owners,
  so the images are used here for non-commercial, educational purposes only and are not tracked in Git.
* The Kaggle card states 3,661 images; the download used here contained **3,530** files. After exact de-duplication
  and removal of 13 conflicting-label groups, **2,930** images remain (`data/v2/leakage_report.json`).
* The supplementary web holdout (`data/web_holdout_v1/`) consists of Wikimedia Commons photographs under PD/CC
  licences; author, licence and source of every image are in `data/web_holdout_v1/ATTRIBUTION.md`.

### Getting the data

`data/v2/manifest.json`, `data/v2/leakage_report.json` and `data/v2/splits/` are tracked. The cleaned images
(`data/v2/images/`, 2,930 JPEG files, also packed as `data/v2/images.tar`) are git-ignored. If you have them (submission
archive or an existing checkout), nothing needs to be downloaded: every step below except the dataset rebuild uses
only `data/v2/`.

To rebuild `data/v2/` from scratch, download the Kaggle dataset (web page "Download", or
`kaggle datasets download -d mdfaisalahmed025/electronic-component-recognizer` with a Kaggle API token) and unpack it
so that the images sit at `data/raw/electronic-component-recognizer/<class>/<class>/<image files>`, the layout the
build script expects (it reports any other layout as a warning; `data/raw/` is not in the repository). Then run `tools/build_dataset_v2.py` (below). It never
modifies `data/raw/`, and it refuses to write a manifest whose split fingerprint differs from `431315e87d2612cb`
unless `--overwrite` is given.

### Setup (uv)

```bash
uv sync                  # Python 3.12 environment from pyproject.toml + uv.lock (CUDA 12.8 torch wheels on Linux)
uv sync --group dev      # + pytest, ipykernel, jupyterlab (needed to execute the notebook)
uv run pytest -q         # optional smoke tests (CPU, seconds)
```

### Reproducing the ITRI626 results

```bash
# 0. (only without data/v2/images/) rebuild dataset v2 from the Kaggle download
uv run --no-config tools/build_dataset_v2.py

# 1. verify the fixed split files against the manifest (no re-randomisation; seed 42, fingerprint 431315e87d2612cb)
uv run python tools/export_splits.py --check

# 2. train A and B, extract frozen SigLIP features and train head C, score the test set once, profile the models
uv run python scripts/itri626_experiments.py                        # GPU recommended, ~15-25 min on an 8 GB GPU
uv run python scripts/itri626_experiments.py --steps head reference collect   # CPU-only parts (cached features)

# 3. execute the notebook (RETRAIN = False in its first code cell loads results/itri626/; True re-runs step 2 first)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/ITRI626_three_models.ipynb

# 4. regenerate the report's tables and number macros from the result files, then build the PDF
uv run --no-project --with numpy --with scipy python docs/itri626_report/make_tables.py
cd docs/itri626_report && latexmk -pdf main.tex
```

Split: train = development folds 1–4 (2,010 images), validation = development fold 0 (502), test = the sealed test
set (418), from a cluster-grouped `StratifiedGroupKFold` (seed 42) over near-duplicate clusters, so no image or
re-encoded copy crosses a split boundary. Model selection (early stopping, head C's weight decay) used the validation
split only; the test split was scored once per model after all settings were frozen. Training seed 0 for all models;
GPU training of model A is not bit-for-bit reproducible (cuDNN), so re-runs can differ slightly.

**Hardware used:** one NVIDIA GeForce RTX 4060 Ti (7.7 GiB usable memory), AMD Ryzen 7 8700F (16 logical cores),
30.7 GiB RAM, Linux, PyTorch 2.11.0 + CUDA 12.8, timm 1.0.30 (`results/itri626/environment.json`). Head C trains on
the CPU from cached features in seconds.
