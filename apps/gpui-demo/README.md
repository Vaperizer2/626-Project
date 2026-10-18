# gpui-demo — Component Classifier desktop app (Rust · GPUI · onnxruntime)

A native desktop demo of the project's final model (`exports/final/model.onnx`, SigLIP So400m/384 + fused heads,
20 electronic-component classes). Open, drop or paste a picture (file, URL, or clipboard image) and the app shows
it with the top-3 classes as probability bars. A prominent **UNSURE** banner appears when the top calibrated
probability is below `abstain_threshold` (0.50). The same binary also has a headless `classify` mode.

It implements the ONNX contract in `docs/DEMO.md` §8 ("Deploying elsewhere") and mirrors the Python reference
`src/compclass/demo.py`.

## Quick start (on the laptop)

```bash
# 0. one-time: a Rust toolchain (rustup is installed on Arch; it just needs a default toolchain)
rustup default stable                   # needs Rust >= 1.92 (ort 2.0.0-rc.14); tested with 1.99.1

# 1. one-time: the onnxruntime shared library. Easiest: the project's own uv venv already has it
cd ~/Work/Prototypes/626-Project
uv sync                                 # creates .venv with onnxruntime (1.31.0 in uv.lock)
#    gpui-demo finds .venv/lib/python3.*/site-packages/onnxruntime/capi/libonnxruntime.so.* by itself.
#    Any other onnxruntime >= 1.22 works too:  --ort-lib /path/to/libonnxruntime.so.1.xx  (or ORT_DYLIB_PATH=...)
#    Lighter alternative (the full `uv sync` also pulls CUDA torch, several GB): a tiny venv with only onnxruntime
#      uv venv ~/.local/share/compclass-ort && uv pip install --python ~/.local/share/compclass-ort onnxruntime
#      export ORT_DYLIB_PATH=$(ls ~/.local/share/compclass-ort/lib/python3.*/site-packages/onnxruntime/capi/libonnxruntime.so.1.*)

# 2. build and run (from anywhere inside the repo: exports/final/ is found by walking up)
cd apps/gpui-demo
cargo run --release --locked            # first build ~5-10 min (GPUI is large), then seconds
# recommended on Arch (libjpeg-turbo is installed): bit-identical JPEG decoding to the Python pipeline
cargo run --release --locked --features turbojpeg
cargo run --release -- ~/Pictures/arduino.jpg "https://upload.wikimedia.org/...jpg"   # pre-load images

# headless (no window), e.g. for scripts/tests
cargo run --release -- classify ../../path/to/photo.jpg            # text line per image
cargo run --release -- classify --json a.jpg b.png https://...     # JSON with all 20 probabilities
```

The release binary is `target/release/gpui-demo` (or `$CARGO_TARGET_DIR/release/gpui-demo`). The model folder
must contain `model.onnx` **and** `model.onnx.data` (1,714,593,792 bytes ≈ 1.7 GB of external weights), `preprocess.json` and `labels.json`.

### Options

| option | meaning |
|---|---|
| `--model-dir DIR` | export folder. Default: `$COMPCLASS_MODEL_DIR`, else the first `exports/final/` with a `model.onnx` found walking up from the current directory, then from the executable's directory. |
| `--ort-lib PATH` | `libonnxruntime.so` to load. Default: `$ORT_DYLIB_PATH`, else `<repo>/.venv/lib/python3.*/site-packages/onnxruntime/capi/libonnxruntime.so*` (searched above the model dir, the CWD and the executable), else a copy next to the executable, else the system loader path (`libonnxruntime.so`). |
| `--threads N` | onnxruntime intra-op threads (default: onnxruntime picks; one per physical core). |
| `--cuda` | request the CUDA execution provider (needs a `--features cuda` build, see below). |
| `--json` | `classify`: print a JSON array (`source, label, top3, unsure, ms, probs, error`). |
| `preprocess IMG OUT.f32` | write the exact `[1,3,384,384]` float32 model input (little-endian), for parity tests. |

### Using the window

* **Open image… / Ctrl+O**: system file dialog (multi-select). It uses xdg-desktop-portal on Linux, see
  Troubleshooting.
* **Drag and drop** one or more image files anywhere on the window (the window border highlights).
* **URL field + Enter / Fetch**: HTTP(S) GET with a descriptive User-Agent
  (`compclass-gpui-demo/0.1.0 (educational image-classification demo; Rust ureq)`). Wikimedia returns 403 to
  default agents. Timeout 20 s, 20 MB cap. **Ctrl+L** focuses the field, **Esc** leaves it. Proxies are taken
  from the environment like curl does: `https_proxy`/`http_proxy` first, then `all_proxy` (http or socks5), and
  `no_proxy` is honoured. Wikimedia throttles bursts, so a quick series of fetches can return `HTTP 429`; wait a
  minute and retry.
* **Paste / Ctrl+V**: if the clipboard holds an image (e.g. "Copy image" in a browser), it is classified directly.
  If it holds text, each line that is an `http(s)://` URL, a `file://` URI or a path is classified. Pasting while
  the URL field is focused inserts the text instead.
* **Recent** list (left, scrollable, last 40): click an entry, or use **Up/Down**, to view an earlier result again.
* **Ctrl+T** or the button toggles the light/dark theme (the initial theme follows the system). **Ctrl+Q** quits.
* The model loads once at startup on a dedicated inference thread; the title bar shows "Loading model… n s" and
  then the load time. Images added meanwhile are queued. Each job runs off the UI thread: read/fetch, decode,
  then the picture appears, then the prediction. Shown latency = preprocessing + inference, the same span as the
  Python demo's `ms`.
* Supported formats: JPEG, PNG, WebP, GIF (first frame), BMP, TIFF. EXIF orientation is applied, and transparency
  is flattened onto white. AVIF is **not** supported (the Python demo reads it via pillow-avif).

## What was verified (2026-10-18, on this laptop's CPU, inside a sandbox without GPU or display)

| check | result |
|---|---|
| `cargo build --release --locked` (Rust 1.99.1) | ✓ (also with `--features turbojpeg`, and `clippy` with `--features turbojpeg,cuda`) |
| `cargo test --release` | 16/16 unit tests pass |
| `cargo clippy --release` | no warnings in this crate |
| parity, 64 dev images vs PyTorch `row_probs.npz` (default build) | argmax 64/64, UNSURE decision 64/64, max \|Δp\| 0.024 (mean of per-image max 0.0023) |
| parity, same images, `--features turbojpeg` | argmax 64/64, input tensor bit-identical to Pillow, max \|Δp\| 0.0029 (onnxruntime vs PyTorch) |
| model load (onnxruntime 1.31.0 CPU, 1.7 GB weights, file in page cache) | 1.1–1.9 s |
| latency per image (preprocess + inference, CPU) | median 1.20 s, p90 1.25 s (default build); turbojpeg build median 1.26 s |
| URL fetch | Wikimedia resistor photo → `resistor 0.998` |
| GUI window | **not launched here**: no display server, no Xvfb/weston, and no software Vulkan (lavapipe) in the sandbox, and conda-forge was unreachable. The no-display error path was checked. The GUI code compiles and type-checks against gpui 0.2.2, but its first real run is on your desktop. |

Full numbers: `results/demo/gpui_parity.json`.

## Design choices and pinned versions

| piece | pinned | why |
|---|---|---|
| UI | `gpui = "=0.2.2"` (crates.io, published by zed-industries on 2025-10-22) | The official Zed release, and the newest one on crates.io. GPUI lives in Zed's monorepo and changes API between Zed releases; a git pin of zed-industries/zed would pull the whole monorepo and still break. Community forks (`gpui-ce`, `gpui-pre`, which `gpui-component` 0.7 depends on, and `gpui-unofficial`) are not maintained by Zed. So this app uses only plain GPUI elements, with no widget library; the URL field is a small hand-written key handler. |
| inference | `ort = "=2.0.0-rc.14"`, features `load-dynamic`, `api-22` | `load-dynamic` dlopens libonnxruntime at runtime: nothing is downloaded at build time and the CUDA build is just a different `.so`. `api-22` asks only for the onnxruntime 1.22 C API (the rc.14 default would require 1.31), so any onnxruntime ≥ 1.22 works. |
| decode | `image = "0.25.10"` (+ optional `turbojpeg = "=1.5.1"`) | see "Preprocessing parity" |
| resize | own code (`src/preprocess.rs`) | a port of Pillow's `Resample.c`; see below |
| HTTP | `ureq = "3.4"` (rustls) | small and blocking (it runs on the worker thread), no OpenSSL needed |
| libc | locked to **0.2.189** in `Cargo.lock` | libc 0.2.191 removed the deprecated Linux `ENOATTR` constant, and `xattr 0.2.3` (pulled in by `gpui 0.2.2 → gpui_http_client → zed-async-tar`) no longer compiles against it. Keep the committed `Cargo.lock`, and do not run `cargo update` without re-checking this. |

The `Cargo.lock` is committed. Build with `cargo build --release --locked` to make sure the exact versions are used.

### Preprocessing parity

`preprocess.json` says: resize so the short side is 427 (Pillow bicubic), centre-crop 384, RGB, ×1/255,
(x−0.5)/0.5, NCHW. Rust's `image` CatmullRom filter is not bit-identical to Pillow's. So
`src/preprocess.rs` ports Pillow's algorithm directly:
* Keys cubic with a = −0.5, the support widened by the downscale factor (antialias), coefficients normalised in
  double precision and then quantised to 22-bit fixed point;
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

### CUDA (optional)

```bash
cargo build --release --features cuda
# a CUDA build of onnxruntime: e.g. in a uv venv `uv pip install onnxruntime-gpu` (+ the nvidia-* CUDA 12/cuDNN 9
# wheels, which torch already installs), then make the CUDA/cuDNN libraries findable and point at that library:
export LD_LIBRARY_PATH=$(python -c "import nvidia, os; print(':'.join(os.path.join(p, d, 'lib') for p in nvidia.__path__ for d in os.listdir(p)))")
./target/release/gpui-demo --cuda --ort-lib .venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.xx
```

If the CUDA provider cannot be registered, onnxruntime logs a warning and runs on the CPU. The header reads
"CUDA (requested) + CPU" whenever `--cuda` was given. Check `nvidia-smi` while classifying to see whether the
GPU is in use. This path was not tested (no GPU access on the build machine).

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
  (GNOME), the app draws its own title bar (drag it to move, ✕ to close).
* **Vulkan**: GPUI renders with Vulkan (blade-graphics). Check with `vulkaninfo --summary` (package
  `vulkan-tools`). The RTX 3060 needs `nvidia-utils` (the ICD `/usr/share/vulkan/icd.d/nvidia_icd.json`). On an
  Optimus laptop, either works: the iGPU's Mesa driver (`vulkan-intel`/`vulkan-radeon`) or
  `prime-run ./gpui-demo`. If no hardware Vulkan driver is present, install `vulkan-swrast` (lavapipe): slow
  but works. A crash at startup that mentions `blade`/`vk` almost always means a missing Vulkan ICD.
* **Ctrl+O does nothing / "File dialog unavailable"**: GPUI's Linux file dialog goes through
  xdg-desktop-portal. Install `xdg-desktop-portal` plus a backend (`-gtk`, `-kde`, `-gnome` or `-wlr`).
  Drag-and-drop and paste work without it.
* **Fonts look wrong / missing glyphs**: GPUI needs fontconfig and at least one sans font (e.g. `noto-fonts`).
* **First build fails on `xattr`**: `Cargo.lock` was regenerated with libc ≥ 0.2.191. Run
  `cargo update -p libc --precise 0.2.189`.
* **onnxruntime prints `pthread_setaffinity_np failed`**: harmless (seen in containers and sandboxes). Pass
  `--threads N` to silence it.

## Layout

```
src/main.rs        CLI (window / classify / preprocess), help text
src/gui.rs         GPUI view, actions, key bindings, inference worker thread
src/model.rs       model dir + libonnxruntime discovery, ort session, predict()
src/preprocess.rs  decode (EXIF, alpha-on-white, gray), Pillow-exact bicubic, crop, normalise; unit tests
src/contract.rs    preprocess.json / labels.json, top-3 + UNSURE rule; unit tests
src/fetch.rs       path / file:// / URL sources, HTTP with UA, timeout, size cap; unit tests
scripts/parity.py  parity vs the Python/PyTorch reference on dev images
```

`cargo test --release` runs the unit tests: preprocessing shape and normalisation, the Pillow kernel, crop
rounding, alpha and grayscale handling, the label/threshold rule, source parsing. They need neither the model nor a
display. `cargo clippy --release` should report no warnings in this crate.
