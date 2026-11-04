# Project handoff

## TL;DR: how we got here

This ITRI626 honours project compares models for 20 electronic-component classes. Early notebooks used splits
with duplicate leakage. The October 2026 rebuild introduced a deduplicated manifest, cluster-grouped development
folds and held-out evaluation; the old notebooks remain in `legacy/` for context. The broad comparison found a
frozen SigLIP So400m probe strongest. The deployed model fuses that probe with a zero-shot head, calibrates its
probabilities and abstains below 0.5. Its recorded sealed-test accuracy is 83.7%; the independent Wikimedia set
is 82.1%. Fusion did not improve the original sealed test, but helped on Wikimedia.

The course submission separately compares A (scratch CNN), B (fine-tuned ConvNeXt-Tiny) and C (frozen SigLIP +
linear head) on a fixed split. A Rust GPUI Kit demo switches among A/B/C/Deployed/Fast. C and Deployed share a
1.7 GB external ONNX weight file. Fast is Deployed at 308 px, quantised to INT8 (weights and activations) for
TensorRT. October 30 sharing work published the source privately under `Vaperizer2/626-Project` and prepared a
release containing the models and portable Linux app. On 2026-11-04 `main` took PRs #1 to #4: the Fast model, the GUI-free inference crate
(`crates/inference`), the new-app template, the 68-picture demo stack, a hard-case study and the training notebook.

## Working rules

- Read [docs/RUNBOOK.md](docs/RUNBOOK.md) for setup, model downloads, demo operation or release maintenance.
- Read [CODEBASE.md](CODEBASE.md) before changing package structure; [STATUS.md](STATUS.md) records current state.
- Use development folds for model selection. Both recorded holdouts are already spent; use committed results
  for documentation updates and synthetic/dev inputs for routine checks.
- Keep ONNX graphs, external weights and preprocessing metadata together. Store large binaries in Releases;
  Git tracks source, reports, manifests and metadata. Preserve photo attribution and bundled licences.
- Use `uv` with the committed lockfile for Python and `cargo --locked` for Rust. Run checks appropriate to changes;
  do not retrain models or rerun held-out experiments for documentation work.
- Treat `legacy/` and dated experiment-log entries as historical evidence, not current run instructions.
- The laptop (RTX 3060 Laptop, 6 GB) is the only GPU. Run heavy CPU work, such as Rust release builds and test
  suites, on Boat.dev CPU sandboxes (`boat` CLI). The GPU PC `carlpc2` is gone.

## Where to look

- Research claims: `docs/report/REPORT.md`, `EXPERIMENT_LOG.md`, `results/benchmark.csv`, `results/final/`.
- Course deliverables: `docs/itri626_report/`, `notebooks/ITRI626_three_models.ipynb`,
  `notebooks/ITRI626_training.ipynb`, `results/itri626/`.
- Inference contract: `docs/DEMO.md`, `exports/itri626/README.md`, `apps/gpui-demo/README.md`.
- GitHub: https://github.com/Vaperizer2/626-Project (private; colleagues require collaborator access).
