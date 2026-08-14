# Electronic component classification comparative

Self-contained Jupyter notebooks. `uv` is the reproducible path; each notebook also has a pip fallback for standalone sharing.

| Project file | Role |
|---|---|
| [`notebooks/01_primary_comparative.ipynb`](notebooks/01_primary_comparative.ipynb) | Paper baseline: HOG→SVM, SmallCNN, ResNet18 |
| [`notebooks/02_playground.ipynb`](notebooks/02_playground.ipynb) | Validation-first family/size table with one selected test evaluation |
| [`notebooks/03_optuna_tuning.ipynb`](notebooks/03_optuna_tuning.ipynb) | Validation-only tuning for the three paper models |
| [`notebooks/04_eli5_self_coding_models.ipynb`](notebooks/04_eli5_self_coding_models.ipynb) | Readable NumPy/PyTorch self-coding reference |
| [`notebooks/05_resnet_regularization_trial.py`](notebooks/05_resnet_regularization_trial.py) | Validation-only ResNet18 regularisation trial |
| [`notebooks/06_indexed_audit_review.py`](notebooks/06_indexed_audit_review.py) | Indexed high-loss/data-quality audit |
| [`notebooks/07_curated_matched_retrain.py`](notebooks/07_curated_matched_retrain.py) | Split-safe curated retraining experiment |
| [`notebooks/08_raw_test_curation_eval.py`](notebooks/08_raw_test_curation_eval.py) | Data-only raw-test curation audit |
| [`notebooks/09_model_family_trials.py`](notebooks/09_model_family_trials.py) | ResNet50/EfficientNet family comparison |
| [`notebooks/10_cnn_trials.py`](notebooks/10_cnn_trials.py) | Controlled scratch-CNN comparison; width-64 variant skipped at checkpoint |
| [`notebooks/11_model_error_audit.py`](notebooks/11_model_error_audit.py) | Evaluation-only per-class error and confusion audit |
| [`notebooks/12_model_family_benchmark.py`](notebooks/12_model_family_benchmark.py) | Resumable validation-first family/size sweep with per-epoch checkpoints |
| [`notebooks/13_ensemble_benchmark.py`](notebooks/13_ensemble_benchmark.py) | Validation-first TTA-logit ensemble comparison with optional held-out test |
| [`notebooks/14_model_family_walkthrough.ipynb`](notebooks/14_model_family_walkthrough.ipynb) | One explained execution cell per model, grouped by family and size, with final stats table |
| [`notebooks/15_family_hyperparam_tuning.py`](notebooks/15_family_hyperparam_tuning.py) | Optuna validation-only tuning for the top-three and one-per-family candidates |
| [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) | Chronological metrics, decisions, and invalidated-run notes |
| [`HANDOVER.md`](HANDOVER.md) | Resume point, checkpoint inventory, and next steps |
| [`DATASET_IMPROVEMENT_RESEARCH.md`](DATASET_IMPROVEMENT_RESEARCH.md) | Evidence-led dataset quality, leakage, and collection recommendations |
| [`OPENCV_RESISTANCE_RESEARCH.md`](OPENCV_RESISTANCE_RESEARCH.md) | Secondary resistance-detection design for the OpenCV demo |

---

## Setup: initialize the environment with `uv`

### 1. Install `uv` (once per machine)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: pacman / brew / pipx — see https://docs.astral.sh/uv/getting-started/installation/
uv --version
```

### 2. Create / sync the project env (recommended)

From the repo root:

```bash
cd /path/to/626-Project
uv sync --all-groups
```

This creates `.venv/` from `pyproject.toml` + `uv.lock` (torch, sklearn, jupyter, Kaggle, Rich, Optuna, …).

### 3. Register a Jupyter kernel (once)

```bash
uv run python -m ipykernel install --user --name=compclass --display-name="Python (compclass)"
```

### 4. Open a notebook

**Cursor / VS Code:** open the `.ipynb` → select kernel **Python (compclass)** or `.venv/bin/python`.

**Terminal:**

```bash
cd notebooks
uv run jupyter lab
```

Launching from `notebooks/` makes `./data/` and `./outputs/` shared by the notebooks and experiment scripts while keeping a copied notebook standalone.

The notebooks write `outputs/metrics/session_manifest.json` on startup. It records the dataset/split fingerprints, profile, selection metric, and held-out-test policy. If the kernel opens in another directory, set `COMPCLASS_WORK_DIR` to the directory containing `data/` and `outputs/`.

The session manifest also records the training variant (`robust-v2-tta`,
`strong-regularized`, or `accuracy-regularized`), seed, split sizes, and raw
dataset/split fingerprints. Validation selection is the default; notebooks do
not read held-out test labels until a selected row is explicitly evaluated.

Skip the notebook “Install” cell if you already ran `uv sync` — packages are already in `.venv`.

### Notebook-only (no repo checkout)

If someone only has the `.ipynb`:

```bash
mkdir component-lab && cd component-lab
uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install ipykernel notebook
uv run python -m ipykernel install --user --name=compclass --display-name="Python (compclass)"
```

Then open the notebook with that kernel and run its **Install** cell. It installs the complete notebook dependency list with `uv`, or with `python -m pip` if `uv` is unavailable.

---

## Data

- **Kaggle API:** `uv run kaggle auth login`, or `KAGGLE_API_TOKEN`, or `~/.kaggle/access_token`
- **Manual:** drop the [dataset](https://www.kaggle.com/datasets/mdfaisalahmed025/electronic-component-recognizer) `.zip` or ImageFolder under `data/raw/` (created next to the notebook cwd)

The downloaded dataset contains substantial web-scrape noise and some mislabeled or unrelated images. The notebooks now ignore hidden checkpoint files and use mild class balancing, label smoothing, and flip-based test-time averaging. Higher scores will require reviewing the irrelevant images rather than only increasing model size.

After changing the pipeline, force a fresh run from the notebooks/ directory:

~~~bash
COMPCLASS_FORCE_RETRAIN=1 COMPCLASS_PROFILE=paper uv run jupyter lab
~~~

For the resumable family/size sweep on a GPU host:

```bash
cd notebooks
../.venv/bin/python 12_model_family_benchmark.py
# optional: COMPCLASS_BENCHMARK_MODELS=resnet34,densenet121 ../.venv/bin/python 12_model_family_benchmark.py
```

The sweep skips completed matching checkpoints, saves `family_benchmark_*.progress.pt` after each validation pass, and writes `model_family_benchmark.json` after every completed trial. It selects on validation macro-F1; the held-out test is reported only for the final validation-selected model. The registered larger-model phase can be run without touching the test split until the final phase winner is known:

```bash
COMPCLASS_BENCHMARK_MODELS=resnet152,efficientnet_v2_s,convnext_small,swin_v2_t \
  ../.venv/bin/python 12_model_family_benchmark.py \
  --report model_family_benchmark_phase2.json --defer-test
```

After comparing the phase-2 validation winner with the earlier validation leader, run one final selected checkpoint with the default test policy. Use a separate `--report` filename when preserving multiple benchmark phases.

The registered largest-model phase uses the same deferred-test policy:

```bash
COMPCLASS_BENCHMARK_MODELS=convnext_base,efficientnet_v2_m,swin_b \
  ../.venv/bin/python 12_model_family_benchmark.py \
  --report model_family_benchmark_phase3.json --defer-test
```

The recipe phases use separate artifact prefixes so earlier checkpoints remain
reproducible:

```bash
../.venv/bin/python 12_model_family_benchmark.py \
  --models convnext_tiny,convnext_base \
  --recipe strong-regularized --artifact-prefix family_recipe_strong \
  --report model_family_benchmark_phase4_strong.json --defer-test
```

For a conservative data-quality experiment, only reviewed `train-*` entries
are removed; validation and test samples remain unchanged:

```bash
../.venv/bin/python 12_model_family_benchmark.py \
  --models convnext_base --recipe strong-regularized \
  --training-exclusions data/curation/reviewed_exclusions_v1.json \
  --artifact-prefix family_recipe_train_clean \
  --report model_family_benchmark_phase7_train_clean.json --defer-test
```

Compare pre-specified ensembles without touching test labels:

```bash
../.venv/bin/python 13_ensemble_benchmark.py \
  --checkpoints family_recipe_strong_convnext_base.pt,family_benchmark_swin_t.pt \
  --report ensemble_benchmark_phase5_base_swin.json
```

For a guided run with one cell per model, open
`notebooks/14_model_family_walkthrough.ipynb`. Run the setup cell, then the
family/model cells in order, and finish with the summary cell. It writes
`model_walkthrough_*_summary.csv` and `.json` under `outputs/metrics/`.
Set `COMPCLASS_PROFILE=smoke` for a quick pipeline check or set
`COMPCLASS_WALKTHROUGH_ARTIFACT_PREFIX` before opening the notebook to keep its
checkpoints separate.

The registered hyperparameter follow-up tunes the eight unique models formed
from the top three overall plus the best model from each family. It searches
learning rate, weight decay, label smoothing, class-weight power, and initial
backbone freezing with Optuna. The tuner loads train/validation only; the test
split is deliberately not constructed or evaluated:

```bash
../.venv/bin/python 15_family_hyperparam_tuning.py \
  --profile paper --trials 5 --epochs 15 --patience 4 \
  --batch-size 32 --num-workers 4 \
  --report family_hparam_tuning_v3.json \
  --artifact-prefix family_hparam_v3
```

Run this command from the compute host when reproducing the recorded search.

## Outputs

- `outputs/figures/` — confusion matrices (also embedded in the saved notebook)
- `outputs/weights/` — model weights
- `outputs/metrics/primary_metrics.json` — numbers for a later LaTeX write-up
- `outputs/metrics/cnn_trials.json` — completed scratch-CNN comparison and validation-selected test result
- `outputs/metrics/model_error_audit.json` / `.csv` — ResNet50 versus scratch-CNN per-sample and per-class audit
- `outputs/figures/model_error_audit_confusion.png` — normalized validation/test confusion matrices
- `outputs/metrics/session_manifest.json` — run contract and frozen split fingerprints
- `outputs/metrics/model_family_benchmark.json` — resumable family/size benchmark table
- `outputs/metrics/model_family_benchmark_phase2.json` — deferred-test larger-model phase (when run)
- `outputs/metrics/model_family_benchmark_phase3.json` — deferred-test largest-model phase (when run)
- `outputs/metrics/model_family_benchmark_phase4_strong.json` — strong-recipe validation phase
- `outputs/metrics/model_family_benchmark_final_test.json` — selected strong-recipe test evaluation
- `outputs/metrics/ensemble_benchmark_phase5*.json` — validation-only ensemble comparisons
- `outputs/metrics/model_family_benchmark_phase6_accuracy.json` — accuracy-focused recipe phase
- `outputs/metrics/model_family_benchmark_phase7_train_clean.json` — training-only reviewed-exclusion phase
- `outputs/metrics/model_walkthrough_*_summary.csv` / `.json` — guided 14-model walkthrough summary
- `outputs/metrics/family_hparam_tuning_v3.json` — validation-only Optuna results for eight unique candidates
- `outputs/metrics/family_hparam_tuning_v3.log` / `family_hparam_v3.db` — tuner log and resumable Optuna study
- `outputs/weights/family_benchmark_*.pt` — completed and per-epoch benchmark checkpoints
- `outputs/weights/family_recipe_*.pt` — recipe-phase checkpoints
- `outputs/weights/family_hparam_v3_*.pt` — best validation checkpoint for each tuned candidate
- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) — running history of all model runs and decisions
- [`HANDOVER.md`](HANDOVER.md) — current checkpoint and exact resume guidance
- `data/splits/split_manifest_raw.json` — frozen 70/15/15 raw split

## Adding packages later

Prefer `uv` for project changes:

```bash
uv add some-package              # project (updates pyproject + lock)
uv pip install --python .venv/bin/python some-package   # ad-hoc into venv
```
