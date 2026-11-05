# Current status — 2026-11-04

The comparative study, held-out evaluations, course submission and desktop demo are implemented.
Use [the runbook](docs/RUNBOOK.md) for setup and operation, [AGENTS.md](AGENTS.md) for the short history,
and [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for dated research decisions. Older checkpoints are in Git history.

## Completed deliverables

- Deduplicated v2 dataset: 2,930 images, 2,512 development / 418 test; fingerprint `431315e87d2612cb`.
  Manifest/splits are tracked; training images, feature caches and training checkpoints are not.
- Grouped-CV model comparison and schedule/leakage ablations: `results/benchmark.csv`, `results/phase1/`,
  `results/phase2/`, `docs/report/REPORT.md`.
- Final frozen SigLIP + fused heads: sealed-test accuracy 0.8373, macro-F1 0.8337; 87.1% coverage at 90.9%
  accepted accuracy. Wikimedia holdout (407 images): accuracy 0.8206, macro-F1 0.8159.
  Both evaluations were recorded on October 18; they are no longer fresh model-selection data.
- ONNX export plus Python demo using onnxruntime (OpenCV handles I/O, not this graph's inference).
  Reports record numerical parity; `model.onnx.data` is 1,714,593,792 bytes.
- ITRI626 report, executed notebook and A/B/C comparison: `docs/itri626_report/main.pdf`,
  `notebooks/ITRI626_four_models.ipynb`, `results/itri626/metrics.json`. The training notebook
  `notebooks/ITRI626_training.ipynb` has one section per model: A, B and C, plus Deployed and Fast beyond the brief.
- IEEE paper, learning notes and offline explorer: `docs/paper/`, `docs/LEARNING_PATH.md`, `docs/explorer/`.
- GPUI Kit 0.7.3 desktop demo: Classify/Demo/Benchmark, A/B/C/Deployed/Fast picker, Bench themes, headless CLI;
  portable x86_64 Linux packaging and NixOS launcher. Export parity evidence is in `results/demo/` and `exports/`.
  The demo stack has 68 Wikimedia photos in four sections: Hand-picked (20), Random (40, seeded), Hard cases (5) and
  Not a component / abstains (3). Deployed gets 53 of 68 (`results/demo/golden/classify_all.json`).
- A fast model for the app: the deployed SigLIP at 308 px, quantised to INT8 (weights and activations), runs 9.2 times
  faster than the deployed model on the RTX 3060 Laptop, with the same 5-fold dev accuracy (0.8268). It is the app's
  default where TensorRT is installed. The acceptable faster variants cost 0.1 to 0.5 points. The best, Q5 (MLP pruning
  plus QAT distillation, 11.2 times faster, −0.08 points), is acceptable but not shipped: it was picked from five
  variants on the same folds, and on 2026-11-03 the owner chose to keep the current Fast (`compose_int8_res308`).
  The scripts are in `tools/speed*/`; [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) has the results.
- TensorRT 10.16.1 miscompiles these graphs after two onnxruntime rewrites (batch above 1; token merging with INT8).
  A manifest's `session_config` turns them off; the shipped Fast model is batch 1 without token merging and needs
  neither. [docs/TEMPLATE.md](docs/TEMPLATE.md) has the setting.
- The desktop app is also a template for new AI apps: a GUI-free inference crate (`crates/inference`), tabs as
  modules, `scripts/new_app.sh` and [docs/TEMPLATE.md](docs/TEMPLATE.md).
- Hard-case study (`tools/hardcases/`, research only): with the recommended change, none of the five hard demo
  pictures turns correct. That change resizes the whole picture to 384 × 384 instead of taking a centre crop, and it
  raises dev CV from 0.8268 to 0.8428 (McNemar p 6e-4). It is not shipped; shipping it is an open owner decision.
  [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) has the numbers.
- Private source repository: https://github.com/Vaperizer2/626-Project. `main` holds PRs #1 to #6 (2026-11-04).
  Sharing release: https://github.com/Vaperizer2/626-Project/releases/tag/share-2026-11-04, built from `main`
  `2f8483b` with all five models and the 68-picture stack. The older share-2026-10-30 stays available.

## Remaining limitations / optional work

- The release desktop binary targets x86_64 Linux; Windows/macOS need the Python route or a separate native build.
- GUI behavior across colleagues' desktops, webcam inference and laptop GPU performance need testing on those devices.
  Headless/model checks do not establish that a graphical session works everywhere.
- Source cloning does not supply training images/caches/checkpoints. Use the release for inference; reproduce training
  only with the dataset and inputs described in README.
- Web-scraped label noise and LED/diode ambiguity remain. Abstention is confidence-based, not an OOD detector.
- The IEEE paper still has author/affiliation placeholders described in `docs/paper/README.md`.
- Optional research: reviewed relabeling, target-camera data or smaller-model distillation, using fresh evaluation data.
- The GPU PC (`carlpc2`) is no longer available. GPU work runs on the laptop (RTX 3060 Laptop, 6 GB). Heavy CPU work,
  such as Rust release builds and test suites, runs on Boat.dev CPU sandboxes (`boat` CLI).
- The GPU runtime (onnxruntime-gpu with CUDA and TensorRT, several GB) is not bundled. `run.sh` uses it from
  `~/.local/share/onnxruntime-gpu/1.31` when present, else the bundled CPU library.

## 2026-11-04 — scratch CNN improvement study folded into the course deliverables

Model A of the ITRI626 report and notebook is now the 200-epoch recipe (test accuracy 0.400, macro-F1 0.392; was
0.354 / 0.336), imported from the improvement study's fold-0 run without a second test look. The study, the distilled
variant and the one-look Wikimedia holdout results are documented in `docs/SCRATCH_CNN_IMPROVEMENT_REPORT.md`,
report subsection "Can the scratch CNN be improved?" and notebook section 8. Both holdouts are spent for both CNN
recipes. The PDF must be rebuilt (`latexmk -pdf main.tex`) after this change.
