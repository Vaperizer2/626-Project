# Runbook

Updated 2026-10-30. Source: https://github.com/Vaperizer2/626-Project (private).
Release: https://github.com/Vaperizer2/626-Project/releases/tag/share-2026-10-30.

## Give a colleague access

In GitHub repository **Settings → Collaborators**, invite their GitHub username. After they accept, send the repo
link for source/reports or the release link for the models and runnable demo. A private-repo link alone grants no
access. The same account permissions protect the release assets. For a person without GitHub, the downloaded
archive can be sent through an agreed file-sharing service; keep its licences and photo credits intact.

The source repository is separate from the existing `Vaperizer2/AI-Vision-Models` repository. No transfer or account
change is needed to use this one. Check `gh auth status` before publishing future updates.

## Run the portable Linux demo

Requires x86_64 Linux, glibc 2.28+, roughly 4 GB free RAM and at least 4 GB disk for archive plus extraction.
It includes CPU onnxruntime, four model choices and 28 credited demo photos; no Python/Rust install is needed.
Download the archive and its checksum from the release page, or use GitHub CLI after `gh auth login`:

```bash
mkdir -p component-classifier-download
cd component-classifier-download
gh release download share-2026-10-30 --repo Vaperizer2/626-Project \
  --pattern 'component-classifier-linux-x86_64.tar.gz*'
sha256sum -c component-classifier-linux-x86_64.tar.gz.sha256
tar -xzf component-classifier-linux-x86_64.tar.gz
cd component-classifier-linux-x86_64
sha256sum -c SHA256SUMS
./run.sh classify --model all demo_stack/01_battery_holder.jpg
./run.sh --tab demo
```

On NixOS replace `./run.sh` with `./run-nixos.sh`; `./run-nixos.sh --doctor` diagnoses runtime discovery.
The first NixOS launch fetches libraries through Nix and caches paths under `~/.cache/component-classifier`.
Later inference is offline once those dependencies are present. A graphical desktop with suitable GPU drivers is
needed for the window; `classify` works headlessly. See the bundled `README.txt` for shortcuts and troubleshooting.
Try the Demo tab, then Benchmark to compare A/B/C/Deployed. These 28 photos are a presentation set, not a new
independent accuracy estimate. Allow several seconds for the initial SigLIP load.

## Download models into a source checkout

Authenticate as an invited collaborator. From a terminal:

```bash
gh repo clone Vaperizer2/626-Project
cd 626-Project
mkdir -p dist
gh release download share-2026-10-30 --repo Vaperizer2/626-Project --dir dist \
  --pattern 'component-classifier-linux-x86_64.tar.gz*'
(cd dist && sha256sum -c component-classifier-linux-x86_64.tar.gz.sha256)
tar -xzf dist/component-classifier-linux-x86_64.tar.gz --strip-components=1 \
  component-classifier-linux-x86_64/exports
```

This restores `exports/final/` and `exports/itri626/{A_scratch_cnn,B_convnext_tiny}/` in the layout used by both apps.

| Asset | Bytes | Purpose |
|---|---:|---|
| `exports/final/model.onnx.data` | 1,714,593,792 | shared SigLIP weights; keep beside both graphs |
| `exports/final/model.onnx` | 2,036,686 | deployed fused/calibrated classifier |
| `exports/final/model_multi.onnx` | 2,129,342 | C + deployed outputs from one backbone pass |
| `exports/itri626/A_scratch_cnn/model.onnx` | 4,803,517 | scratch CNN |
| `exports/itri626/B_convnext_tiny/model.onnx` | 112,081,040 | fine-tuned ConvNeXt-Tiny |

The bundle contains matching labels/preprocessing metadata. Keep it with the weights. Model C has a slightly
different ONNX test count from the report (343 vs 344/418), documented in `exports/itri626/README.md`.

## Python inference and development

For inference only, avoid installing the full training stack. With `uv` installed and models extracted:

```bash
uv run --no-project --python 3.12 --with numpy --with pillow \
  --with opencv-python-headless --with onnxruntime \
  python src/compclass/demo.py --source apps/gpui-demo/demo_stack/01_battery_holder.jpg \
  --no-window --out demo_out/
```

For full development, use `uv sync --locked --group dev`, then `uv run pytest -q`.
Python is pinned to 3.12; Linux training dependencies use CUDA 12.8 Torch wheels and require a substantial download.
The tests use synthetic data. For webcam, GUI, CUDA and model contracts see [DEMO.md](DEMO.md).
For Rust builds/options see [the app README](../apps/gpui-demo/README.md); use `cargo --locked`.

## Reproduce results and documents

Read [the project README](../README.md#itri626-submission) for dataset retrieval and fixed-split reproduction.
A source clone plus release supports inference, but not every training command: raw/cleaned training images,
feature caches and checkpoints are excluded. Model selection uses development folds only. The original sealed
test and Wikimedia holdout were already evaluated; read the committed metrics for routine report work.

Reports and generated tables can be refreshed without retraining:

```bash
uv run python scripts/make_report_figures.py
uv run --no-project --with numpy --with scipy python docs/itri626_report/make_tables.py
(cd docs/itri626_report && latexmk -pdf main.tex)
```

The last command requires a TeX installation. IEEE paper instructions and remaining placeholders are in
[paper/README.md](paper/README.md); explorer instructions are in [explorer/README.md](explorer/README.md).
Record changed research conclusions in `EXPERIMENT_LOG.md`; update `STATUS.md` when the state changes.

## Maintain the release

ONNX files and `dist/` remain ignored by Git. This project distributes inference binaries as Release assets rather
than Git LFS pointers. The bundle is below GitHub's per-asset limit; LFS is also possible, but adds client setup and
storage/bandwidth accounting. See [GitHub large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
and [LFS limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

For a changed app/model, follow the packaging prerequisites in the app README, then run from the repo root:

```bash
apps/gpui-demo/scripts/package_linux.sh
```

The script builds a glibc-2.28-compatible binary and bundles onnxruntime, models, both launchers, credits and
checksums. Verify the archive/checksums, extract to a temporary directory and run a headless prediction with all
models. Test the window on a supported desktop before claiming GUI validation. Publish under a new dated tag,
attach the `.tar.gz` and `.tar.gz.sha256`, and update the links above and in README. Preserve prior releases so
colleagues can reproduce their downloads. Publish source changes and the matching release together.
