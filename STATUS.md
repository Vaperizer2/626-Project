# Current status — 2026-10-30

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
- Historical PC shell changes and old detached job IDs are not current operational prerequisites; remote PC state
  has not been rechecked during this sharing pass.

## 2026-11-04 — scratch CNN improvement study folded into the course deliverables

Model A of the ITRI626 report and notebook is now the 200-epoch recipe (test accuracy 0.400, macro-F1 0.392; was
0.354 / 0.336), imported from the improvement study's fold-0 run without a second test look. The study, the distilled
variant and the one-look Wikimedia holdout results are documented in `docs/SCRATCH_CNN_IMPROVEMENT_REPORT.md`,
report subsection "Can the scratch CNN be improved?" and notebook section 8. Both holdouts are spent for both CNN
recipes. The PDF must be rebuilt (`latexmk -pdf main.tex`) after this change.
