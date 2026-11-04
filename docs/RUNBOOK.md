# Runbook

Updated 2026-11-04. Source: https://github.com/Vaperizer2/626-Project (private).
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
It includes CPU onnxruntime, four model choices and 28 credited demo photos (the share-2026-10-30
release; later bundles carry 68); no Python/Rust install is needed.
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
Try the Demo tab, then Benchmark to compare A/B/C/Deployed. These photos are a presentation set, not a new
independent accuracy estimate. Allow several seconds for the initial SigLIP load.

## Run the Fast model on an NVIDIA GPU

Fast is the deployed model at 308 px with INT8 weights for TensorRT: the same cross-validated accuracy on the
development folds (0.8268) at 14 ms per picture instead of 113 ms for Deployed on CUDA (RTX 3060 Laptop, in the app).
The share-2026-10-30 release predates Fast and the GPU launcher: use a bundle built from `main`
(`apps/gpui-demo/scripts/package_linux.sh`, see below for Fast's model files) or a source build. It needs the GPU
runtime, which the bundle does not ship (about 7 GB, no root):

```bash
uv venv ~/.local/share/onnxruntime-gpu/1.31 --python 3.12
uv pip install --python ~/.local/share/onnxruntime-gpu/1.31/bin/python \
  'onnxruntime-gpu[cuda,cudnn]==1.31.1' 'tensorrt-cu13-libs==10.16.1.11'
./run.sh models        # first line: "run.sh: onnxruntime on GPU (onnxruntime 1.31.1, TensorRT from ...)"
./run.sh --tab demo    # starts on Fast; the first start builds the TensorRT engine (about 46 s), later ones ~1-2 s
```

The driver must support CUDA 13. That folder is one install per machine, shared by every app built from the template.
`run.sh` also finds the runtime through `COMPCLASS_GPU_VENV=<venv>`, the older `~/.local/share/component-classifier/gpu`
or an active virtualenv; `COMPCLASS_GPU=0` forces the CPU. Without TensorRT the app starts on Deployed (CUDA); without any GPU
runtime it runs on the CPU and says so on the launcher's first line and in `./run.sh models`.

Fast's ONNX files are not in any release yet. Copy `model.onnx` and `model.onnx.data` (440 MB) from the export (on
legion: `/home/vaper/Work/Prototypes/626-artifacts/compose/exports/fast/`) into `exports/fast/` of a source
checkout, whose other files (`model.json`, `preprocess.json`, `labels.json`, `runtime.json`) are tracked in git,
then build the app or the bundle (`apps/gpui-demo/scripts/package_linux.sh` includes Fast).
`gpui-demo models --check` verifies the copied files against the checksums in `exports/fast/model.json`.

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

## Build the desktop app from source

The Rust code is one Cargo workspace at the repo root: `crates/inference` (the models, the inference service and
onnxruntime, with no UI) and `apps/gpui-demo` (the window and the command line). Run cargo from the repo root. The
binary lands in `target/release/`, not under `apps/gpui-demo/`.

```bash
export CARGO_BUILD_JOBS=8
cargo build --release --locked -p gpui-demo --features turbojpeg   # libjpeg-turbo: the same JPEG pixels as Pillow
cargo test --locked                                                 # needs no ONNX files and no display
target/release/gpui-demo models                                     # finds exports/models.json by walking up
```

`--features turbojpeg` needs the system libjpeg-turbo 3.0 or newer (`pkg-config --modversion libturbojpeg`).
Ubuntu 24.04 ships 2.1.5, so there build with `--features turbojpeg-vendored` instead (it compiles libjpeg-turbo
3.1 from source and needs cmake and nasm). Without either feature the build works, but its JPEG pixels differ
slightly from Pillow's and the golden comparison does not match. The portable bundle always uses
`turbojpeg-vendored`, so its users need no libjpeg-turbo.

A clean build takes about 2.5 to 3 minutes on legion, nearly all of it GPUI Kit. The laptop is short of RAM, so run
release builds and full test suites on a Boat.dev CPU sandbox (`boat` CLI) when you can. `cargo test -p inference`
builds the crate without GPUI in under a minute. Options, keys and the profiling setup are in
[the app README](../apps/gpui-demo/README.md). A new desktop AI app starts from this one with `scripts/new_app.sh`;
[TEMPLATE.md](TEMPLATE.md) has the procedure.

## Reproduce results and documents

Read [the project README](../README.md#itri626-submission) for dataset retrieval and fixed-split reproduction.
A source clone plus release supports inference, but not every training command: raw/cleaned training images,
feature caches and checkpoints are excluded. Model selection uses development folds only. The original sealed
test and Wikimedia holdout were already evaluated; read the committed metrics for routine report work.

Reports and generated tables can be refreshed without retraining, using the tracked result files and review CSVs:

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
than Git LFS pointers. The share-2026-10-30 archive (1.74 GB) is below GitHub's 2 GiB per-asset limit. LFS is also
possible, but adds client setup and storage/bandwidth accounting. See [GitHub large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
and [LFS limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

For a changed app/model, follow the packaging prerequisites in the app README, then run from the repo root:

```bash
apps/gpui-demo/scripts/package_linux.sh
```

The script builds a glibc-2.28-compatible binary and bundles onnxruntime, both launchers, credits, checksums and
what `apps/gpui-demo/packaging/bundle.txt` lists (the models, the demo stack and the dataset note). Add a new model's
files to that list. Verify the archive/checksums, extract to a temporary directory and run a headless prediction with all
models. Test the window on a supported desktop before claiming GUI validation. Publish under a new dated tag,
attach the `.tar.gz` and `.tar.gz.sha256`, and update the links above and in README. A bundle with Fast can be
larger than 1900 MiB; the script then also writes `.tar.gz.part-*` files, and the bundled `README.txt` explains how
to join them. Preserve prior releases so
colleagues can reproduce their downloads. Publish source changes and the matching release together.
