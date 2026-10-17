# Legacy material (pre-v2)

This folder holds the first iteration of the project, moved here with `git mv` so its history is intact:
`notebooks/` (numbered 01–16 notebooks/scripts: paper baselines, Optuna tuning, family sweeps, ensembles,
audits) and `docs/` (the old README, handover notes, experiment log, dataset/augmentation/OpenCV research and
agent audit reports). It is kept for reference only — nothing in `src/compclass` imports it. **Treat every
headline number in here as invalid:** the 0.71–0.76 validation/test accuracies were inflated by train/test
duplicate leakage (≈22–25 % of the old val/test rows were byte-identical copies of training images, and 12
duplicate groups carried conflicting labels). The first leakage-free estimate (5-fold, exact-hash-grouped CV)
was ≈63–65 % macro-F1. The current, comparable record is the root [`EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md).
