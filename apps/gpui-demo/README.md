# gpui-demo — Component Classifier desktop app (Rust · GPUI Kit · onnxruntime)

A native desktop demo of the project's models (20 electronic-component classes), built with
[GPUI Kit](https://gpui-kit.com/docs/) (`gpui-kit` 0.7.3: GPUI plus a styled component library and themes). Three tabs:

* **Classify**: open, drop or paste a picture (file, URL, or clipboard image). It shows the picture with the top-3
  classes as probability bars and all 20 probabilities. The deployed model shows an **Unsure** state when its top
  calibrated probability is below `abstain_threshold` (0.50).
* **Demo**: a curated stack of 28 Wikimedia Commons photos that were never used for training or model selection
  (`demo_stack/`). Click a card, or step with the arrow keys and press Enter. Each result is marked correct or wrong
  against the true label, and the photo's author, licence and source are shown. **Run all** classifies the stack in
  order and keeps a running "21/28 correct" count.
* **Benchmark**: runs every available model over the stack, live and cancellable. It then compares accuracy,
  mean confidence and ms per image, and shows a per-image matrix (✓ bluish green = correct, ✕ vermillion = wrong,
  ? violet = unsure; colour-blind safe, and always with an icon). A switch shows only the images where the models
  disagree.

A model picker switches between the report's models (**A · Custom CNN**, **B · ConvNeXt-T**, **C · SigLIP + linear**),
the **Deployed** fused SigLIP model and **Fast**, the deployed model at 308 px with INT8 weights for TensorRT (same
cross-validated accuracy, 14 ms per picture instead of 113 ms on the RTX 3060 Laptop). Models are data: each export
folder has a `model.json` manifest and `exports/models.json` lists them, so a new model needs no Rust change. Models
load on first use. The 1.7 GB SigLIP weights are loaded once and shared by C and Deployed. The same binary also has
headless `classify`, `bench` and `models` modes. It implements the ONNX contract in `docs/DEMO.md` §8 and mirrors the
Python reference `src/compclass/demo.py`.

The models, the inference service and onnxruntime live in the GPUI-free crate `crates/inference`; this folder is the
window, the command line and the classifier's own parts. The app is also the template for new desktop AI apps:
`scripts/new_app.sh` copies it without the classifier's parts, as [docs/TEMPLATE.md](../../docs/TEMPLATE.md) describes.

## Download and run

For colleagues, use the [private release](https://github.com/Vaperizer2/626-Project/releases/tag/share-2026-10-30)
and follow the [runbook](../../docs/RUNBOOK.md). It includes all models, the CPU runtime and the demo photos.
On ordinary x86_64 Linux run `./run.sh --tab demo`; on NixOS run `./run-nixos.sh --tab demo`.
The first NixOS launch needs network access to fetch runtime libraries. Windows/macOS users can use the Python
ONNX demo from source; this release's desktop binary is Linux-only.

## Build from source

```bash
# 0. one-time: a Rust toolchain (rustup is installed on Arch; it just needs a default toolchain)
rustup default stable                   # needs Rust >= 1.92; tested with 1.99.1

# 1. one-time: the onnxruntime shared library (any onnxruntime >= 1.22).
#    A library next to the binary is detected automatically. Options: `uv sync` in the repo (gpui-demo finds .venv/.../onnxruntime/capi/), or
#      uv venv ~/.local/share/compclass-ort && uv pip install --python ~/.local/share/compclass-ort onnxruntime
#      export ORT_DYLIB_PATH=$(ls ~/.local/share/compclass-ort/lib/python3.*/site-packages/onnxruntime/capi/libonnxruntime.so.1.*)

# 2. download the models first (see ../../docs/RUNBOOK.md), then build and run, from the repo root: the Cargo
#    workspace (crates/inference and this app) builds into <repo>/target/
cargo build --release --locked -p gpui-demo --features turbojpeg   # Arch has libjpeg-turbo: bit-identical JPEG decoding to Pillow
target/release/gpui-demo                              # Classify tab
target/release/gpui-demo --tab demo                   # start on the Demo tab (for presenting)
target/release/gpui-demo --tab benchmark --autorun    # start the benchmark straight away
target/release/gpui-demo ~/Pictures/arduino.jpg "https://upload.wikimedia.org/...jpg"   # pre-load pictures

# headless (no window), e.g. for scripts/tests
target/release/gpui-demo classify photo.jpg                       # the default model, one text line per image
target/release/gpui-demo classify --json a.jpg b.png https://...  # JSON with all 20 probabilities
target/release/gpui-demo classify --json --model all apps/gpui-demo/demo_stack/*.jpg   # every model (a, b, c, deployed, fast)
target/release/gpui-demo models                                   # the registry and which providers register
target/release/gpui-demo models --check                           # also sha256 every file and load every graph
target/release/gpui-demo bench --model fast apps/gpui-demo/demo_stack/*.jpg   # latency by stage and streamed FPS (JSON)
```

**Output change since the registry (2026-10-31).** Each `classify` text line now has the model's head id as its second
column (`source, model, label, p, top3, ms`), and each JSON row has `model` and `low_confidence` fields (error rows
too, with `timing: null`). Scripts that split the old
five-column lines or keyed rows by source alone need updating. Without `--model` the head is the machine's default
(Fast where TensorRT registers, else Deployed), so pass `--model deployed` when output must match across machines.
`--model-dir` and `--demo-dir` are gone (2026-11-01) and are now unknown options (exit 64). Pass `--models` with the
index or its folder, and set `COMPCLASS_DEMO_DIR` for the demo stack.

The default model is the first entry of `exports/models.json`'s `default` list whose preferred execution provider
registers here: Fast when TensorRT loads, else Deployed. `--ep` does not change that choice: with `--ep cuda` or
`--ep cpu` Fast would not run on TensorRT, so Deployed is the default (`--model fast --ep cpu` still runs Fast on the
CPU). The window and the CLI say which one was passed over and why. For the GPU see "GPU: CUDA and TensorRT" below.

`cargo build --release --locked` (without `turbojpeg`) also works; it uses the pure-Rust JPEG decoder (see
"Preprocessing parity"). A clean build takes about 2.5 to 3 minutes on legion (GPUI Kit is about 500 of its 520
crates); later builds take seconds. The deployed
export folder must contain `model.onnx` **and** `model.onnx.data` (1,714,593,792 bytes ≈ 1.7 GB of external weights),
`preprocess.json` and `labels.json`.

### Options

| option | meaning |
|---|---|
| `--models PATH` | the model index `exports/models.json` or its folder. Default: `$COMPCLASS_MODELS`, else the first `exports/models.json` found walking up from the current directory, then from the executable's directory. |
| `--model HEADS` | `classify`, `bench` and `preprocess`: these models (`a`, `b`, `c`, `deployed`, `fast`, comma-separated, or `all`; `bench` and `preprocess` take one). Default: the first default whose preferred provider registers (see above). JSON rows carry a `model` field. |
| `--tab TAB` | window: start on `classify` (default), `demo` or `benchmark`. |
| `--autorun` | window: start **Run all** (Demo tab) or the benchmark (Benchmark tab) immediately. |
| `--dark`, `--light` | window: initial theme (default: follow the system; Ctrl+T toggles). |
| `--ort-lib PATH` | `libonnxruntime.so` to load. Default: `$ORT_DYLIB_PATH`, else `<repo>/.venv/lib/python3.*/site-packages/onnxruntime/capi/libonnxruntime.so*`, else a copy next to the executable, else the system loader path. |
| `--threads N` | onnxruntime intra-op threads (default: onnxruntime's choice, one per physical core). |
| `--ep LIST` | execution providers for every model, in order: `tensorrt`, `cuda`, `cpu`, comma-separated. Default: each manifest's `runtime.providers` (`cuda,cpu`; `tensorrt,cuda,cpu` for Fast). `--ep cpu` gives bit-reproducible output. |
| `--json` | `classify`: print a JSON array (`source, model, label, top3, unsure, low_confidence, ms, probs, error, timing`). `unsure` is an abstaining head's UNSURE; `low_confidence` marks an answer from a head that always answers (A, B, C) below its bound. An error row has the same keys, with `label` and `timing` null. |
| `bench [--rounds N] [--warmup N] IMG…` | one model through the inference service: per-image latency by stage (read+decode, preprocess, queue, run, readout, end to end), then rounds × images submitted at once for streamed FPS. JSON on stdout. |
| `models [--check]` | every model, whether it is available, and per graph which providers register and why the others do not. `--check` also verifies every file's sha256 and loads each graph. The heads print first; the provider and check lines follow once every graph is done. |
| `preprocess IMG OUT.f32 [--model HEAD]` | write the exact `[1,3,H,W]` float32 model input (little-endian), for parity tests. |

The Demo and Benchmark tabs read the demo stack from `$COMPCLASS_DEMO_DIR`, else the first `demo_stack/` (or
`apps/gpui-demo/demo_stack/`) with a `demo_stack.json` found walking up from the current directory, then from the
executable. `run.sh` sets the variable to the bundle's `demo_stack/`.

### Using the window

| key | action |
|---|---|
| Ctrl+1 / Ctrl+2 / Ctrl+3 | Classify / Demo / Benchmark tab (or click the tabs) |
| Ctrl+M | next available model (or click the model picker; hover a model for its state) |
| Ctrl+T | light / dark theme (button in the title bar) |
| Ctrl+O | Classify: open files (system dialog via xdg-desktop-portal, multi-select) |
| Ctrl+V | Classify: paste an image, a URL, a `file://` URI or a path (inside the URL field it pastes text) |
| Ctrl+L, Enter | Classify: focus the URL field; Enter fetches |
| Up / Down | Classify: previous / next recent picture |
| Left / Right or J / K | Demo: previous / next image (wraps around) |
| Enter | Demo: classify the selected image with the selected model |
| Ctrl+R | Demo: Run all · Benchmark: run |
| Esc | stop Run all / the benchmark after the current image |
| Ctrl+Q | quit |

The status bar shows the selected model and the execution providers it actually runs on (for example
"Fast on TensorRT + CUDA + CPU", or "Deployed on CPU (CUDA unavailable)"), the onnxruntime version, the JPEG decoder
and the keys of the current tab (as Kit `Kbd` chips). Hover a model in the picker for its load time, providers and why
a provider was skipped.

* **Drag and drop** image files anywhere on the window (the border highlights); they open in the Classify tab.
* **URL fetch**: HTTP(S) GET with a descriptive User-Agent, 20 s timeout, 20 MB cap. Proxies come from the
  environment as with curl (`https_proxy`/`http_proxy`, then `all_proxy`; `no_proxy` is honoured). Wikimedia throttles
  bursts (HTTP 429); wait a minute.
* **Recent** list (left, last 40): each entry remembers which model classified it.
* **Model loading** never blocks the window: each graph has its own worker thread that loads it on first use (the
  default model is pre-loaded at start-up), so a slow load blocks only that model. The model chip next to the picker
  shows "loading… n s", or "building TensorRT engine, first run only · n s" while TensorRT compiles Fast's engine,
  then the load time. Pictures added meanwhile are queued.
* Supported formats: JPEG, PNG, WebP, GIF (first frame), BMP, TIFF. EXIF orientation is applied, and transparency is
  flattened onto white. AVIF is not supported.

## Models

| key | name | files | input | test accuracy (418, report) |
|---|---|---|---|---|
| `a` | A · Custom CNN (scratch) | `exports/itri626/A_scratch_cnn/{model.onnx, preprocess.json, labels.json}` | 160 px, **bilinear** | 148/418 |
| `b` | B · ConvNeXt-T (fine-tuned) | `exports/itri626/B_convnext_tiny/…` (112 MB) | 224 px, ImageNet mean/std | 266/418 |
| `c` | C · SigLIP ViT + linear head | output `probs_c` of `exports/final/model_multi.onnx` | 384 px | 343/418 (ONNX) |
| `deployed` | Deployed (fused SigLIP) | output `probs` of `exports/final/model_multi.onnx` | 384 px | 350/418 |
| `fast` | Fast (fused SigLIP, 308 px INT8) | `exports/fast/{model.onnx, model.onnx.data}` (440 MB), INT8 QDQ GEMMs, head in the graph | 308 px | not measured: the sealed test is spent. Dev-fold CV 0.8268, the same as Deployed's |

Each folder's `model.json` (written by `tools/gen_manifests.py`) names the graph and its files with sizes and sha256,
the input (name, size, preprocessing), one entry per head (output, labels, calibration, decision rule) and the
execution providers in order of preference. `model_multi.onnx` is the deployed graph plus the model-C head on the
same pooled feature, so one forward pass gives both C and Deployed; it reads the unchanged `model.onnx.data`. Fast
comes from the speed track's `compose_int8_res308` export (`exports/fast/fit_report.json`). Its ONNX files are not in
the share-2026-10-30 release yet: copy them into `exports/fast/` from the export (see the runbook). Missing files make
a model show as unavailable (greyed out, with the reason in the tooltip and in `gpui-demo models`); the app never
crashes on a missing model. A model whose files were never checksummed here, or changed size or modification time
since, shows "checksums not verified"; `gpui-demo models --check` hashes them and records the result.

**Unsure only for calibrated models.** Deployed and Fast output temperature-calibrated probabilities, and the 0.5
abstain threshold was set for them. A, B and C output plain softmax probabilities, and at 0.5 A would abstain on
73 % and B on 35 % of the test images. So A, B and C always commit to their top-1. When it is below 50 %, the result
panel adds a "low confidence" note. In the Demo and Benchmark tabs, an out-of-scope photo therefore counts as correct
only when a calibrated model abstains.

## Demo stack (`demo_stack/`, 1.8 MB)

28 photos, none of which was used for training or model selection. 26 come from the independent Wikimedia Commons web
holdout `data/web_holdout_v1` (evaluated once). The 2 out-of-scope photos are the credited demo images in
`results/demo/wikimedia_credits.json`. `demo_stack/demo_stack.json` stores file, section, true label (`null` = not
a component), title, author, licence, licence URL, source URL, a note, and the deployed model's prediction and
confidence as verified by this binary. `demo_stack/ATTRIBUTION.md` holds the CC credits; the app shows them under
each photo (hover a card for the source URL, click the licence or "Wikimedia Commons"). The stack is rebuilt and
re-verified with

```bash
uv run --no-config apps/gpui-demo/scripts/build_demo_stack.py --bin target/release/gpui-demo
```

Sections: **One per class** (20 clean photos the deployed model gets right), **Hard cases** (5 instructive images:
LED↔diode, toroidal transformer↔inductor, DIP switch↔IC, SMD inductor↔diode, a cluttered Arduino + breadboard scene)
and **Not a component / abstains** (a coin-cell holder on a motherboard, a multimeter, a soldering iron). Verified
results with the `turbojpeg` build (`classify --json --model all`):

| model | One per class (20) | Hard cases (5) | Not a component / abstains (3) | total (28) |
|---|---|---|---|---|
| A · Custom CNN | 11 | 0 | 0 | 11 |
| B · ConvNeXt-T | 16 | 1 | 0 | 17 |
| C · SigLIP + linear | 20 | 1 | 0 | 21 |
| Deployed | 20 | 0 | 1 | 21 |

Fast is left out of this table on purpose: 26 of these photos come from the web holdout, which is spent, so no new
model is scored on it. The Benchmark tab still runs Fast over the stack.

The soldering iron is a deliberate borderline case. After downscaling to 1024 px the deployed model says "jumper wire"
at 50.6 %, just above the threshold, so it is not flagged. This shows that a confidence threshold is not a complete
out-of-distribution detector.

## What was verified (2026-10-20, on this laptop's CPU, in a sandbox without a GPU or a desktop display)

| check | result |
|---|---|
| `cargo build --release --locked` and `--features turbojpeg` (Rust 1.99.1) | ✓ |
| `cargo clippy --release --locked --all-targets` (default; `turbojpeg,cuda`; `hotpath,hotpath-alloc`) | no warnings |
| `cargo test --release --locked` | 32/32 pass. They cover the earlier preprocessing, contract and fetch tests, the bilinear kernel, demo_stack.json parsing and locating, the model registry and `model_multi.json`, benchmark statistics, the embedded theme file, a check that every Lucide icon used in the GUI is bundled, and a **headless GUI smoke test** (GPUI Kit's test platform: builds the real window, renders all three tabs, keyboard stepping) |
| parity, 64 dev images vs PyTorch `row_probs.npz` (`scripts/parity.py`, turbojpeg build) | argmax 64/64, UNSURE 64/64, input tensors bit-identical, max \|Δp\| 0.0029: unchanged from 2026-10-18 |
| A and B input tensors vs Pillow (`compclass.demo.preprocess`), every 10th test image | 42/42 bit-identical each (A uses the new bilinear resize) |
| A and B on all 418 test images (`classify --model a,b`) | 148/418 and 266/418, exactly the exported ONNX accuracy; argmax agreement with the report's PyTorch predictions 417/418 and 418/418 |
| Deployed from `model_multi.onnx` vs `model.onnx` on the demo stack | same top-1 on 28/28, \|Δp\| ≤ 5·10⁻⁵ (JSON rounding) |
| GUI window | rendered in a virtual X server (Xvfb + Mesa software GL); every tab was screenshotted in light and dark themes, with Run all and the benchmark run to completion. Not tested: a real Wayland/X11 desktop, the GPU renderer, drag and drop, the file dialog and the clipboard. |

## Profiling

Build with the optional `hotpath` feature ([hotpath](https://hotpath.rs)). Without it, the crate is not in the build,
and every instrumented block (`prof!` in the source) compiles to plain code.

```bash
cargo build --release -p gpui-demo --features turbojpeg,hotpath   # add hotpath-alloc for per-block allocation counts
target/release/gpui-demo classify --model all apps/gpui-demo/demo_stack/*.jpg > /dev/null   # report printed on exit
HOTPATH_SHUTDOWN_MS=90000 HOTPATH_OUTPUT_PATH=gui.txt target/release/gpui-demo --tab benchmark --autorun
```

Instrumented blocks: `job::read`, `job::decode`, `job::decode_bytes`, `job::display_copy` (inference service io
threads), `preprocess::to_tensor`, `graph::load`, `graph::run` (a graph's worker) and `demo::decode_and_thumbnail`.
The table below predates the inference service (2026-10-20). For latency through the service use `gpui-demo bench`.
The system tools work on the headless path (perf needs access to hardware counters, `perf_event_paranoid` ≤ 1 or
root):

```bash
perf stat -d target/release/gpui-demo classify apps/gpui-demo/demo_stack/*.jpg > /dev/null
perf record --call-graph dwarf -o perf.data target/release/gpui-demo classify --model all apps/gpui-demo/demo_stack/0*.jpg > /dev/null
perf report -i perf.data --no-children | head -60
target/release/gpui-demo --tab benchmark --autorun & pidstat -r -u -t -p $! 5     # CPU / RSS per thread
```

Measured in the sandbox (8 CPUs visible; `perf` could not run there because no PMU events were available, so these
numbers come from hotpath and pidstat):

| what | number |
|---|---|
| UI thread: building a frame's element tree (1,149 frames during a benchmark run) | mean 0.44 ms, p95 0.84 ms |
| demo stack: decode once + 360 px thumbnail + display texture, 28 images in parallel | 11 ms each, 37 ms wall |
| benchmark, 28 images × 4 models (3 graph runs per image: A, B, shared SigLIP) | 2.06 s per image, 57–62 s total |
| inference per image (preprocess + run): A / B / SigLIP (C + Deployed together) | ≈ 3–5 ms / 100–126 ms / 1.84–1.99 s |
| memory with all four models loaded (SigLIP once) | 2.4 GB RSS |
| idle after a run (only a 250 ms tick that stops when nothing is busy) | < 1 % CPU |
| display texture (BGRA copy), 12 MP photo → 1280 px / → 360 px | 65 → 52 ms / 31 → 17 ms: now resized in place without a full-size copy |
| headless `classify`, 8 images, deployed model: previous binary vs this one | 13.3 s vs 13.8 s wall (same code path; difference is noise) |

What keeps the live demo smooth: all decoding, fetching, model loading and inference run off the UI thread
(`inference` and `thumbnails` threads). Thumbnails are decoded once and kept as textures at display size. A demo image
is decoded once and its RGB pixels are reused by every later classification. C and Deployed share one SigLIP session
and one forward pass. Run all and the benchmark submit one image at a time, so they stay cancellable and the
Classify tab can still interleave its own jobs.

## Packaging (portable Linux bundle for another machine)

`scripts/package_linux.sh` builds a self-contained folder plus archive, for testing on any x86_64 Linux desktop
with glibc 2.28 or newer (Ubuntu 20.04+, Debian 10+, Fedora 29+, RHEL 8+, Arch):

```
dist/component-classifier-linux-x86_64/       2.2 GB: binary, CPU onnxruntime, models A/B/C/Deployed/Fast, demo_stack,
                                              run.sh, run-nixos.sh, app.env, gpu-env.sh, README.txt (for the
                                              recipient), LICENSES/, SHA256SUMS
dist/component-classifier-linux-x86_64.tar.gz (+ .sha256; split into 1900 MiB .part-* files if it is larger)
```

```bash
# one-time build tools: cargo-zigbuild, zig 0.14 from PyPI, cmake + nasm
cargo install cargo-zigbuild --locked
uv venv ~/.local/share/zig-venv && uv pip install --python ~/.local/share/zig-venv ziglang==0.14.1
export CARGO_ZIGBUILD_ZIG_PATH=$(ls -d ~/.local/share/zig-venv/lib/python3.*/site-packages/ziglang)/zig
ORT_LIB=<CPU libonnxruntime.so.1.31.0> apps/gpui-demo/scripts/package_linux.sh   # GLIBC=2.31, PART_MIB=1000, --skip-build, --no-archive
```

Use zig 0.14 even when a newer zig is installed. `CARGO_ZIGBUILD_ZIG_PATH` makes cargo-zigbuild take it over a `zig`
on `PATH`. Arch's zig 0.17 compiles everything, but its linker rejects the system `/usr/lib/libxkbcommon.so`, which
needs `GLIBC_2.33` and `GLIBC_2.38` symbols that the 2.28 stubs lack (2026-11-01, cargo-zigbuild 0.24).

The bundle's name, binary and environment variables come from `packaging/app.env` (`APP_NAME`, `APP_TITLE`,
`ENV_PREFIX`, `BIN`, and `COMMANDS`, the subcommands the launchers start without a window), which the script and both launchers source and which ships in the bundle. Keep it in step with
`src/app.rs`.

How the bundle stays portable:

* **glibc**: `cargo zigbuild --target x86_64-unknown-linux-gnu.2.28` links against zig's glibc 2.28 stubs, so the
  binary runs on glibc ≥ 2.28 (the same floor as onnxruntime's manylinux_2_28 library). The script fails if any
  symbol needs a newer `GLIBC_` version. It uses its own target dir (`dist/.build-target`), so
  `target/release/` is untouched.
* **JPEG**: feature `turbojpeg-vendored` builds the libjpeg-turbo 3.1 source bundled in `turbojpeg-sys` with
  CMake/NASM and links it statically. The decoder is the same as the system-library `turbojpeg` build:
  on the 28 demo images × 4 models the probabilities are identical (max |Δp| = 0), and `scripts/parity.py` gives the
  same result as before: 64/64 input tensors bit-identical to Pillow, argmax 64/64.
* **Runtime dependencies** (`readelf -d`): libc, libm, libdl, libpthread, ld-linux, libxcb, libxkbcommon,
  libxkbcommon-x11. Loaded with dlopen when present: libwayland-client/-egl (Wayland), libvulkan or libEGL
  (rendering), and the bundled `libonnxruntime.so.1.31.0` (which needs libstdc++ ≥ GCC 6 and libgcc_s).
  The script rejects any other `NEEDED` library.
* **Fonts**: Noto Sans Regular + Bold (SIL OFL 1.1, `assets/fonts/`) are embedded and registered with GPUI's
  text system at start-up (`theme::register_bundled_fonts`), so the type hierarchy is the same on machines
  whose UI font is a variable font or that lack Noto Sans.
* **Paths**: the app already finds `exports/`, `demo_stack/` and `libonnxruntime` relative to its executable;
  `run.sh` additionally sets `ORT_DYLIB_PATH`, `<ENV_PREFIX>_MODELS` and `<ENV_PREFIX>_DEMO_DIR` to the bundle
  (caller-set values win), passes all arguments through, and explains a missing display.
* **GPU runtime**: not bundled (about 7 GB). Both launchers source `gpu-env.sh`, which looks for an onnxruntime-gpu
  install in this order: `$COMPCLASS_GPU_VENV`; `${XDG_DATA_HOME:-~/.local/share}/onnxruntime-gpu/1.31`, one
  install per machine shared by every app built from this template; the app's older
  `~/.local/share/component-classifier/gpu`; an active virtualenv; then the bundle folder. It puts the runtime's CUDA,
  cuDNN and TensorRT library folders on the loader path and prints one line saying which runtime it chose and why.
  Without one, or with `COMPCLASS_GPU=0`, the bundled CPU library is used. `scripts/test_gpu_env.sh` (run by
  `cargo test`) checks that order against fake venvs, and the variable names against a second `app.env`.
* **Models and data**: what `packaging/bundle.txt` lists, one path from the repo root per line, with its place in
  the bundle when that differs. Here that is `exports/models.json` and per model folder `model.json`, the files it
  names and `preprocess.json`: `exports/final/{model_multi.onnx, model.onnx.data, labels.json, ...}` (the 1.7 GB
  `model.onnx.data` is shared), `exports/fast/{model.onnx, model.onnx.data, runtime.json, ...}` and
  `exports/itri626/{A_scratch_cnn, B_convnext_tiny}/{model.onnx, preprocess.json, labels.json}`, plus `demo_stack/`
  and the dataset note in `LICENSES/`. A new model in `exports/models.json` needs its lines there too.
* **Licences** (`LICENSES/`): onnxruntime (MIT + ThirdPartyNotices), Noto Sans (OFL), libjpeg-turbo (IJG/BSD),
  Lucide icons (ISC), Apache-2.0 / MIT texts, a dataset and photo-credit note, and `THIRD-PARTY-CRATES.txt`
  (616 crates with SPDX licence, generated from `cargo metadata` by `packaging/crate_licenses.py`).

Tested (2026-10-30) from a clean extraction under `env -i HOME=… PATH=/usr/bin:/bin`, started from an unrelated
directory: `run.sh classify --model a|b|c|deployed` on three demo images, `run.sh classify --json --model all` on all
28 (A 11, B 17, C 21, Deployed 21 of 28), the missing-display message, and the window under Xvfb (Demo tab, Run all).
Not tested: a real second machine or distribution, a GPU renderer, Wayland.

The GPU launcher logic (2026-11-01) was tested on a folder with the bundle's layout (the release binary, the bundle's
CPU onnxruntime, symlinked exports and launchers), not on a `package_linux.sh` build. From `env -i` shells, `run.sh`
chose the CPU library with no GPU runtime or with `COMPCLASS_GPU=0` (Deployed on CPU), CUDA without TensorRT for a
venv without `tensorrt-cu13-libs` (Deployed on CUDA, Fast passed over with its reason), and TensorRT for a full venv
found through `COMPCLASS_GPU_VENV`, `~/.local/share/component-classifier/gpu` or `VIRTUAL_ENV` (Fast on TensorRT).
`run-nixos.sh` was run on Arch with Nix: `--doctor` and `models` work, and with the GPU venv but no
`/run/opengl-driver` the app fell back to the CPU naming the missing `libcuda.so.1`. Not tested: a real NixOS
machine with the NVIDIA driver.

`package_linux.sh --no-archive` itself was run on legion on 2026-11-01 (zig 0.14.1): GLIBC ceiling 2.28, the
`DT_NEEDED` list unchanged, and the same files as before `app.env` existed, plus `app.env`. From `env -i` shells in
that bundle, `run.sh` found a GPU runtime in the shared `onnxruntime-gpu/1.31` folder (Fast: TensorRT registers),
`COMPCLASS_GPU=0` gave the CPU, and `classify` answered "battery holder" with Deployed on CPU and on CUDA.

## Design choices and pinned versions

| piece | pinned | why |
|---|---|---|
| UI | `gpui-kit = "=0.7.3"` | GPUI Kit (longbridge, formerly `gpui-component`) is the component library: `TitleBar`, `TabBar`, `Button`/`ButtonGroup`, `Input`, `Tag`, `Spinner`, `Alert`, `Kbd`, `Tooltip`, `Switch`, `Empty`, `StatusBar`, notifications, scrollbars and its theme system (the app ships its own two theme files). It pins and re-exports its own GPUI snapshot (`gpui-pre 0.3.9`), so the app has no direct GPUI dependency. Exact pin because the API changes between pre-1.0 releases. (0.8.0 was published while this port was made; it was not tested.) |
| inference | `ort = "=2.0.0-rc.14"`, features `load-dynamic`, `api-22` | `load-dynamic` dlopens libonnxruntime at runtime, so nothing is downloaded at build time and the CUDA build is just a different `.so`. `api-22` requires only the onnxruntime 1.22 C API. ort allows one environment per process; all sessions share it. |
| decode | `image = "0.25.10"` (+ optional `turbojpeg = "=1.5.1"`) | see "Preprocessing parity" |
| resize | own code (`crates/inference/src/preprocess.rs`) | a port of Pillow's `Resample.c` (bicubic and bilinear); see below |
| HTTP | `ureq = "3.4"` (rustls) | small and blocking (it runs on the worker thread), no OpenSSL needed |
| profiling | `hotpath = "0.28"`, optional | see "Profiling" |
| libc | normal (0.2.191 in `Cargo.lock`) | The old pin to 0.2.189 is no longer needed: it existed only for `xattr 0.2.3`, which GPUI 0.2.2 pulled in and GPUI Kit's GPUI does not. |

| look | own GPUI Kit themes "Bench Light" / "Bench Dark" (`src/gui/themes/bench.json`, loaded by `src/gui/theme.rs`) | Built for a projector: chrome / canvas / card layering, one blue accent for "the answer", colour-blind-safe verdicts with icons, large numerals. The rationale, the reference apps studied and the token table are in [`docs/VISUAL_DIRECTION.md`](docs/VISUAL_DIRECTION.md). Views use theme roles only (no hex in views). The default follows the system theme: light reads better on a projector in a lit room; use `--dark` in a dim one. |
| fonts | system fonts, nothing bundled | The UI prefers Noto Sans (or DejaVu Sans, Liberation Sans, Ubuntu) because GPUI's Linux text system draws only the regular instance of a *variable* system font (GNOME's Adwaita Sans), which would make all bold text regular. Numbers in columns use the system monospace font. |
| icons | Lucide via `gpui_kit::assets` | The Kit's `Assets` bundles only its ~100 component icons; the extra Lucide icons the app uses are embedded with `icon_assets!` (`widgets::ExtraIcons`, ~20 KB). A unit test fails if an icon used in the source is not bundled (it would render blank). |

The `Cargo.lock` is committed; build with `--locked`.

### Preprocessing parity

Each model folder's `preprocess.json` gives its own transform. Deployed and C: short side 427 (Pillow bicubic),
centre-crop 384, ×1/255, (x−0.5)/0.5. B: 256 bicubic, crop 224, ImageNet mean/std. A: 183 **bilinear**, crop 160,
(x−0.5)/0.5. All are RGB NCHW. Rust's `image` CatmullRom filter is not bit-identical to Pillow's. So
`crates/inference/src/preprocess.rs` ports Pillow's algorithm directly:
* Keys cubic with a = −0.5 (support 2) or the triangle filter 1 − |x| (bilinear, support 1), the support widened by
  the downscale factor (antialias), coefficients normalised in double precision and then quantised to 22-bit fixed
  point;
* horizontal pass first, over only the rows the vertical pass reads, each pass rounded and clipped to u8;
* torchvision's truncated long side, and `CenterCrop`'s Python `round()` (half-to-even);
* the same float32 arithmetic as numpy for the normalisation.
On Pillow-decoded pixels the Rust tensor is **bit-identical** to `compclass.demo.preprocess`.

The remaining difference is the JPEG **decoder**. The default build uses the `image` crate's pure-Rust decoder
(zune-jpeg), which differs from Pillow's libjpeg-turbo by a few grey levels on some pixels. Build with
`--features turbojpeg` (requires the system `libturbojpeg`: `pkg-config --modversion libturbojpeg`; Arch package
`libjpeg-turbo`) to decode JPEG pixels with libjpeg-turbo, as Pillow does. The model input is then bit-identical to
the Python pipeline. `gpui-demo --version` prints which decoder a binary uses.

Measured parity on 64 dev images is in `results/demo/gpui_parity.json`; reproduce it with

```bash
python apps/gpui-demo/scripts/parity.py --bin target/release/gpui-demo [--ort-lib ...]   # needs numpy + Pillow
```

### GPU: CUDA and TensorRT

Every build can use the GPU: whether CUDA or TensorRT runs is decided by the libonnxruntime that is loaded, and each
model's manifest lists the providers to try (`--ep` overrides). What you need is the GPU build of onnxruntime with its
CUDA 13 and cuDNN 9 libraries, plus TensorRT 10 for Fast, all from pip (about 7 GB):

```bash
uv venv ~/.local/share/onnxruntime-gpu/1.31 --python 3.12
uv pip install --python ~/.local/share/onnxruntime-gpu/1.31/bin/python \
    'onnxruntime-gpu[cuda,cudnn]==1.31.1' 'tensorrt-cu13-libs==10.16.1.11'
```

That folder does not depend on the app, so one install serves every app made from this template. The bundle's
`run.sh` finds it by itself (`packaging/gpu-env.sh`: `$COMPCLASS_GPU_VENV`, that folder, the older
`~/.local/share/component-classifier/gpu`, an active virtualenv) and prints which runtime it chose. From a source
build, point at it yourself:

```bash
SP=$(ls -d ~/.local/share/onnxruntime-gpu/1.31/lib/python3.*/site-packages)
export ORT_DYLIB_PATH=$(ls $SP/onnxruntime/capi/libonnxruntime.so.1.*)
export LD_LIBRARY_PATH=$SP/onnxruntime/capi:$SP/nvidia/cu13/lib:$SP/nvidia/cudnn/lib:$SP/tensorrt_libs
target/release/gpui-demo models          # which providers register for each graph, and why the others do not
target/release/gpui-demo --tab demo      # starts on Fast when TensorRT registers
```

The first load of Fast builds a TensorRT engine for this GPU, onnxruntime and TensorRT version: 46 s (45 to 48 s over
three runs) on the RTX 3060 Laptop with a peak of about 6 GB RAM, and 91 s once in the window while RAM was short. The
window shows "building TensorRT engine, first run only" with a seconds counter, the CLI prints `building the TensorRT
engine (first run only)`. The engine and timing cache go to `~/.cache/component-classifier/tensorrt/<graph>-<sha256 of
the graph files>-ort<version>-trt<version>-<gpu>/` (443 MB), so a changed graph, library or GPU never meets a stale
engine. Fast's graph has a static batch of 1. In the speed track, TensorRT returned wrong outputs at batch above 1 until the
cause was found (an onnxruntime rewrite; `docs/TEMPLATE.md` has the `session_config` that turns it off).

Measured on legion (RTX 3060 Laptop 6 GB, onnxruntime 1.31.1, TensorRT 10.16.1, driver 615.78), through the app's
inference service, medians over 5 alternated rounds (`results/demo/gpui_legion_speed.json`, scripts in
`626-artifacts/app-core/measure/`):

| | Fast, TensorRT | Deployed, CUDA | Fast, CUDA only |
|---|---|---|---|
| model ready at start-up (cached engine) | 1.0 to 1.5 s | 1.1 s | 0.5 s |
| first start, engine build | 46 s | | |
| per picture, run | 12.1 ms | 110.2 ms | 91.6 ms¹ |
| per picture, end to end (read, decode, preprocess, run) | 14.4 ms | 113.0 ms | 94.8 ms¹ |
| streamed (pictures submitted at once) | 78 / s | 9.0 / s | 10.8 / s¹ |

¹ measured before the watcher fix (ef94d84), which took 2.0 ms (Deployed) to 3.3 ms (Fast) off every run.

The previous app binary (07e569c, one serial loop) took 112.6 ms per picture for Deployed on CUDA. Fast's run stage
matches the graph timed alone (the harness's 11.9 ms; 12.7 ms in Python with IO binding the same day). Reading,
decoding and preprocessing on the CPU add about 2 ms per picture end to end. Fast's batch is 1 (see above), so
streaming is bound by the engine: 78 / s is 12.8 ms per picture, with decoding and preprocessing overlapped.

When a provider cannot be used (no driver, no GPU runtime, a missing library), the graph falls back to the next one
in its list and the reason is shown: in the model picker's tooltip, in `gpui-demo models` and on the CLI's stderr.

## Troubleshooting

* **"cannot load the onnxruntime library"**: run `uv sync` in the repo, or pass `--ort-lib`/`ORT_DYLIB_PATH` to any
  `libonnxruntime.so.1.2x`/`1.3x`. An onnxruntime older than 1.22 fails with an API-version error.
* **"loading …/model.onnx … (is model.onnx.data next to it and complete?)"**: the external-weights file is
  missing or truncated. It must be 1,714,593,792 bytes.
* **Memory**: the session needs roughly 3 GB of RAM. Close other large processes on an 8 GB machine.
* **"no display"**: the window needs a Wayland (`WAYLAND_DISPLAY`) or X11 (`DISPLAY`) session. Over SSH, use
  `classify`.
* **Wayland vs X11**: GPUI uses Wayland when `WAYLAND_DISPLAY` is set, otherwise X11 (XWayland works). To force
  X11 on a Wayland desktop, run `WAYLAND_DISPLAY= ./gpui-demo`. On compositors without server-side decorations
  (GNOME), the Kit `TitleBar` draws the window controls (drag the bar to move, double-click to maximise).
* **GPU / renderer**: GPUI Kit's GPUI renders with wgpu, which tries Vulkan first and falls back to OpenGL (EGL).
  Check Vulkan with `vulkaninfo --summary` (package `vulkan-tools`); the RTX 3060 needs `nvidia-utils`. On an
  Optimus laptop, either the iGPU's Mesa driver or `prime-run ./gpui-demo` works. Without any GPU driver Mesa's
  software GL (llvmpipe) still renders: `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
  LIBGL_ALWAYS_SOFTWARE=1 ./gpui-demo` (this is how the screenshots in the sandbox were made).
* **Ctrl+O does nothing / "File dialog unavailable"**: GPUI's Linux file dialog goes through
  xdg-desktop-portal. Install `xdg-desktop-portal` plus a backend (`-gtk`, `-kde`, `-gnome` or `-wlr`).
  Drag-and-drop and paste work without it.
* **Fonts look wrong / missing glyphs**: GPUI needs fontconfig and at least one sans font (e.g. `noto-fonts`).
* **onnxruntime prints `pthread_setaffinity_np failed`**: harmless (seen in containers and sandboxes). Pass
  `--threads N` to silence it.

* **"A · CNN … not exported" / buttons greyed out**: the model files are missing. `gpui-demo models` lists what the
  registry found and why the rest is unavailable (see "Models" above).
* **Demo tab says "Demo stack not found"**: start the app from inside the repository, or set
  `COMPCLASS_DEMO_DIR=apps/gpui-demo/demo_stack`.

## Layout

The workspace root holds `Cargo.toml` (members `crates/inference` and `apps/gpui-demo`, and the profiles),
`Cargo.lock` and `target/`. `crates/inference` is the models without a UI: manifests and registry, the inference
service, onnxruntime with execution-provider fallback and the TensorRT engine cache, weight integrity, image decoding
and preprocessing, fetching. Its crate doc (`cargo doc -p inference --open`) is the interface. This folder:

```
src/main.rs            the window, or one subcommand of app::COMMANDS; the log! macro
src/app.rs             the identity (NAME, TITLE, ABOUT, ENV_PREFIX, ICON, UNSURE, subtitle), TABS, COMMANDS, Shared
src/cli.rs             options, --help, and what the subcommands share (index, service, default head, results)
src/cli/classify.rs    classify: text or JSON rows, one per image and head
src/cli/bench.rs       bench: one head's latency by stage, then streamed FPS (JSON)
src/cli/models.rs      models [--check]: prints inference::check::models
src/cli/preprocess.rs  preprocess: the exact model input tensor, for parity tests (classifier only)
src/demo.rs            demo_stack.json: parsing, locating, the verdict (classifier only)
src/gui.rs             the window: actions, key bindings, opening the window over the service
src/gui/shell.rs       Shell (title bar, tab row with the model picker, status bar, alerts) and the Tab protocol
src/gui/models.rs      Models: the service, each graph's load state and the selected head, shared by every tab
src/gui/classify.rs    Classify tab: open, drop, paste or fetch pictures; recent list, picture, result panel
src/gui/demo_tab.rs    Demo tab: sectioned thumbnail grid, detail pane with verdict + attribution, Run all (classifier only)
src/gui/bench.rs       Benchmark tab: every model over the demo stack, stats, matrix, disagreement filter (classifier only)
src/gui/stack.rs       the decoded demo stack, its thumbnails, the verdict's looks, the tab header (classifier only)
src/gui/results.rs     one input's answers per head
src/gui/widgets.rs     bars, prediction block, tags, model-state chip, Kbd hints, extra Lucide icon assets
src/gui/images.rs      display textures from the service's BGRA copies
src/gui/theme.rs       loads the Bench Light / Bench Dark themes; app roles (canvas/chrome/surface, verdict colours)
src/gui/themes/        bench.json: the two GPUI Kit theme files (see docs/VISUAL_DIRECTION.md)
src/gui/tests.rs       the headless window tests' harness (GPUI Kit test platform) and ProbeTab
src/gui/tests/shell.rs the shell's tests, over ProbeTab and app::TABS
src/gui/*/tests.rs     each tab's tests (classify, demo_tab, bench)
assets/fonts/          Noto Sans Regular + Bold (embedded) and OFL.txt
packaging/             app.env (the bundle's identity), bundle.txt (what the bundle carries), run.sh, run-nixos.sh,
                       gpu-env.sh, README.txt (for the recipient) and licence inputs
scripts/package_linux.sh   builds dist/component-classifier-linux-x86_64[.tar.gz] (see "Packaging")
scripts/test_gpu_env.sh    gpu-env.sh and run.sh against fake GPU runtimes (run by cargo test)
scripts/golden.py      compares classify --json output with results/demo/golden/ (row by row)
scripts/parity.py      parity vs the Python/PyTorch reference on dev images
scripts/build_demo_stack.py   rebuilds and re-verifies demo_stack/
docs/VISUAL_DIRECTION.md   reference-app study, design synthesis and token table for the look
demo_stack/            the 28 demo photos + demo_stack.json + ATTRIBUTION.md
```

The parts marked "classifier only", with `demo_stack/`, the golden and parity scripts and the dataset note, are what
`scripts/new_app.sh` leaves out of a new app.

`cargo test --release` needs neither the model files nor a display. The golden comparison needs
`-p gpui-demo --features turbojpeg`, because the golden rows were made with libjpeg-turbo.
