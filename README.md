# compclass — electronic-component image classification across model families

One importable package (`src/compclass`) that trains, probes and evaluates every model family the project
compares — from a CNN written from scratch to CLIP/SigLIP/DINOv2 foundation models — on one leakage-free
dataset (`data/v2/manifest.json`), with grouped 5-fold cross-validation and a sealed test split.

* **Code map and how to run:** [`CODEBASE.md`](CODEBASE.md)
* **Results and decisions:** [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) (human) and `results/runs.jsonl` (one JSON line per run)
* **Old notebooks and reports:** [`legacy/`](legacy/README.md) — kept for reference; their 0.71–0.76 accuracies were
  inflated by train/test duplicate leakage.

## Setup (uv)

```bash
uv sync                      # creates .venv from pyproject.toml + uv.lock (CUDA 12.8 torch on Linux)
uv run pytest -q             # ~10 s CPU smoke tests on a synthetic manifest
uv run python -c "from compclass.models import describe; print(describe())"   # read the family explainers
```

Pretrained weights download on first use; point the caches somewhere persistent:

```bash
export HF_HOME=~/models_cache/hf TORCH_HOME=~/models_cache/torch
```

## Quick start

```bash
# fine-tune one family on CV fold 0 (or --fold 0 1 2 3 4)
uv run python -m compclass.train --config configs/convnext_tiny.yaml --fold 0
# frozen-feature linear probe, zero-shot, classical baseline (all five folds by default)
uv run python -m compclass.features  --model dinov2_vitb14
uv run python -m compclass.zeroshot  --model siglip2_b16
uv run python -m compclass.classical
# export the chosen checkpoint for the OpenCV demo
uv run python -m compclass.export --checkpoint runs/convnext_tiny-<hash>/fold0/best.pt --out exports/convnext_tiny
```

## Syncing the laptop checkout to the GPU PC

The GPU PC (`carlpc2`, RTX 4060 Ti 8 GB) keeps its copy at `~/Projects/626-Project`. Push code changes from the
laptop with rsync over Tailscale SSH (port 2222); the virtualenv and raw data are excluded because each machine
builds its own `.venv` with `uv sync` and only the converted `data/v2` is needed for training:

```bash
rsync -avz --delete -e "ssh -p 2222" \
  --exclude .venv/ --exclude .git/ --exclude data/raw/ --exclude runs/ --exclude results/cache/ \
  --exclude '__pycache__/' --exclude '*.onnx' \
  ~/Work/Prototypes/626-Project/ carl@carlpc2.tail498db1.ts.net:~/Projects/626-Project/
ssh -p 2222 carl@carlpc2.tail498db1.ts.net 'cd ~/Projects/626-Project && uv sync --python 3.12'
```

`--delete` makes the PC mirror the laptop, but `runs/` and `results/cache/` are excluded so trained checkpoints on
the PC are never deleted. To bring results back, reverse the direction for just `results/runs.jsonl` (or a run dir):

```bash
rsync -avz -e "ssh -p 2222" carl@carlpc2.tail498db1.ts.net:~/Projects/626-Project/results/runs.jsonl results/runs.pc.jsonl
```

## Evaluation contract (short)

Model choices use only the `dev` split through `compclass.data.get_fold(manifest, k)`; near-duplicate clusters
never straddle folds. The `test` split is opened once, for the final model, via `get_test(manifest, confirm=True)`
or `compclass.evaluate --sealed-test`. Details in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).
