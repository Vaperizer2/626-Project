# Current status — 2026-11-03

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
  `notebooks/ITRI626_three_models.ipynb`, `results/itri626/metrics.json`.
- IEEE paper, learning notes and offline explorer: `docs/paper/`, `docs/LEARNING_PATH.md`, `docs/explorer/`.
- GPUI Kit 0.7.3 desktop demo: Classify/Demo/Benchmark, A/B/C/Deployed picker, Bench themes, headless CLI;
  portable x86_64 Linux packaging and NixOS launcher. Export parity evidence is in `results/demo/` and `exports/`.
- A fast model for the app: the deployed SigLIP at 308 px, quantised to INT8 (weights and activations), runs 9.2
  times faster than the deployed model on the RTX 3060 Laptop, with the same 5-fold dev accuracy (0.8268). It is the
  app's default where TensorRT is installed. The acceptable faster variants cost 0.1 to 0.5 points. The best, Q5 (MLP pruning plus
  QAT distillation, 11.2 times faster, −0.08 points), is acceptable but not shipped yet: it was picked from five
  variants on the same folds. The scripts are in `tools/speed*/`; [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) has the
  results.
- TensorRT 10.16.1 miscompiles these graphs after two onnxruntime rewrites (batch above 1; token merging with INT8).
  A manifest's `session_config` turns them off; the shipped Fast model is batch 1 without token merging and needs
  neither. [docs/TEMPLATE.md](docs/TEMPLATE.md) has the setting.
- The desktop app is also a template for new AI apps: a GUI-free inference crate (`crates/inference`), tabs as
  modules, `scripts/new_app.sh` and [docs/TEMPLATE.md](docs/TEMPLATE.md).
- Private source repository: https://github.com/Vaperizer2/626-Project.
  Sharing release: https://github.com/Vaperizer2/626-Project/releases/tag/share-2026-10-30.

## Remaining limitations / optional work

- The release desktop binary targets x86_64 Linux; Windows/macOS need the Python route or a separate native build.
- GUI behavior across colleagues' desktops, webcam inference and laptop GPU performance need testing on those devices.
  Headless/model checks do not establish that a graphical session works everywhere.
- Source cloning does not supply training images/caches/checkpoints. Use the release for inference; reproduce training
  only with the dataset and inputs described in README.
- Web-scraped label noise and LED/diode ambiguity remain. Abstention is confidence-based, not an OOD detector.
- The IEEE paper still has author/affiliation placeholders described in `docs/paper/README.md`.
- Optional research: reviewed relabeling, target-camera data or smaller-model distillation, using fresh evaluation data.
- The GPU PC is no longer available. Training and the speed work run on the laptop.
- The GPU runtime (onnxruntime-gpu with CUDA and TensorRT, several GB) is not bundled. `run.sh` uses it from
  `~/.local/share/onnxruntime-gpu/1.31` when present, else the bundled CPU library.
