# Making a new desktop AI app from this one

This repo's Rust app, `apps/gpui-demo` (the component classifier), is also the template for new desktop AI apps.
A new app gets the inference crate, a GPUI window with one tab (Classify), a command line with `classify`, `bench` and `models`, and the Linux packaging.
It gets them under its own name, title, binary and environment variables.
The classifier's Demo and Benchmark tabs, its demo stack, its `preprocess` subcommand, its golden scripts and its models stay here.

The steps are below, in order.
Each later section covers one kind of change: a model, a tab, a subcommand, a new task, or an input source.

## 1. Make the app

```sh
scripts/new_app.sh ../hello-ai hello-ai "Hello AI" HELLOAI
cd ../hello-ai && git init && git add . && git commit -m "From the template"
```

The arguments are the destination folder (it must not exist), the name, the title and the environment prefix.
The name is the Cargo package, the binary and the folder names (`apps/<name>/`, `~/.cache/<name>/`, the bundle). It takes `a-z`, `0-9` and `-`.
The prefix starts the environment variables (`HELLOAI_MODELS`, `HELLOAI_GPU`, `HELLOAI_GPU_VENV`). It takes `A-Z`, `0-9` and `_`.

The script copies what `HEAD` holds, so commit template fixes before you run it.
After a change to `new_app.sh` or to a file it copies, `scripts/check_new_app.sh` makes an app from `HEAD` in a temporary folder and runs its tests.
It copies the workspace `Cargo.toml` and `Cargo.lock`, `rustfmt.toml`, `crates/inference`, `apps/gpui-demo` as `apps/<name>`, and this file as `docs/TEMPLATE.md`.
Then it deletes the classifier's files and their module lines, and writes the identity files.
The new app's layout is the same as here.

```
Cargo.toml  Cargo.lock  .gitignore   workspace: crates/inference and apps/<name>
crates/inference/                    the inference crate, with its test fixtures in tests/fixtures/
apps/<name>/src/app.rs               identity, tabs, subcommands
apps/<name>/src/{main,cli,gui}.rs    the binary, the command line, the window
apps/<name>/packaging/               app.env, bundle.txt, README.txt, launchers, licences
apps/<name>/scripts/                 package_linux.sh, test_gpu_env.sh
docs/TEMPLATE.md                     this file
```

**The identity.** These three places name the app. `new_app.sh` writes all three, and they must agree.

| Where | What |
|---|---|
| `apps/<name>/src/app.rs` | `NAME` (cache folder), `TITLE` (window title, `--help`), `ABOUT` (one line in `--help`), `ENV_PREFIX`, `ICON` (title bar), `UNSURE` (the label of an abstaining head in `classify` output), `subtitle` (the title bar's note on the selected head). `BIN` is `env!("CARGO_BIN_NAME")`. It is the Wayland app id and the prefix of every log line. |
| `apps/<name>/packaging/app.env` | `APP_NAME`, `APP_TITLE`, `ENV_PREFIX`, `BIN` and `COMMANDS`, for the launchers and `package_linux.sh`. |
| `apps/<name>/Cargo.toml` | the package name and the `[[bin]]` name. |

Edit `ABOUT` first. The script sets it to "classify pictures".

**The first build.** A clean build compiles GPUI Kit, about 500 crates, which takes 2.5 to 3 minutes on legion. Two apps that share one target dir share that work. Point `CARGO_TARGET_DIR` at another app's `target/` and the new app's first build compiles only the inference crate and the app.

## 2. Add a model

A model is data: a folder in `exports/` with a hand-written `model.json`, listed in `exports/models.json`.
The app finds `exports/models.json` through `--models`, then `$<PREFIX>_MODELS`, then by walking up from the current directory and from the executable.

```
exports/models.json              {"schema": "model-index/1", "default": ["mine"], "manifests": ["mine/model.json"]}
exports/mine/model.json          the manifest
exports/mine/model.onnx          the graph (and its external data, if any)
exports/mine/labels.json         the class names, a JSON array of strings
```

The manifest has schema `model-manifest/1`. `crates/inference/src/manifest.rs` parses it, and `crates/inference/tests/fixtures/` holds four real ones.

- `id`, then `graph { file, external_data[], precision }`. Precision is `fp32`, `fp16` or `int8_qdq`.
- `files { <name>: { bytes, sha256 } }` lists every file the graph and its heads read, relative to the manifest's folder. The registry marks a graph unavailable when a file's size differs. `models --check` hashes every file and records a stamp in `~/.cache/<name>/verified.json`, and a load then only checks that size and mtime still match the stamp.
- `input { name, dtype: "f32", layout: "NCHW", size: [h, w], batch { max }, preprocess }`. The one `preprocess` kind is `resize_short_center_crop { resize_short_side, filter, color: "RGB", scale, mean, std }`, with filter `pillow_bicubic` or `pillow_bilinear`.
- `heads[] { id, title, short, task }`. Head ids are unique across the index. A graph with two outputs can carry two heads.
- A classify `task` is `{ kind: "classify", readout, labels, calibration, decision }`. The readout is `{ kind: "probs" | "logits", output }`, the output's name in the graph. Calibration is `{ kind: "none" }` or `{ kind: "temperature", t, in_graph }`. The decision is `{ kind: "commit", low_confidence_below }` or `{ kind: "abstain", below }`, and abstaining needs a calibrated head.
- `runtime { graph_optimization_level, session_config, providers }`. The providers are `[{ ep: "tensorrt" | "cuda" | "cpu", options {...} }]` in order of preference, with onnxruntime's own option names. With `trt_engine_cache_enable: true` the app keeps TensorRT engines in `~/.cache/<name>/tensorrt/`.
- `provenance` is free-form. The app never reads it.

Then check the model and record its checksums:

```sh
hello-ai models --check
```

The parser rejects what the app cannot honour, and says which file and field.
`models` without `--check` lists the heads and which execution providers register on this machine.

Two known onnxruntime problems with TensorRT set rules for TensorRT manifests. TensorRT with batch above 1 returns wrong outputs on these graphs, so keep `batch.max` at 1. A ToMe plus INT8 engine needs `"graph_optimization_level": "ORT_DISABLE_ALL"`.

`.gitignore` ignores `*.onnx` and `*.onnx.data`. Keep the weights somewhere else as well, and commit the JSON files.

## 3. Tabs and subcommands

**A tab** is a module with a `gui::Tab` impl and a line in `app::TABS`. Ctrl+1 to Ctrl+9 show the tabs in that order. `--help`, `--tab` and the status bar read their titles and hints from the same list.
This tab shows the selected head's title:

```rust
// apps/<name>/src/app/hello_tab.rs, with `mod hello_tab;` in app.rs and
// `TabSpec::of::<hello_tab::HelloTab>()` in app::TABS
use gpui_kit::{Context, Entity, FocusHandle, Window, div, prelude::*};

use crate::app::Shared;
use crate::gui::models::Models;
use crate::gui::{Hint, Lucide, Tab};

pub struct HelloTab {
    models: Entity<Models>,
    focus: FocusHandle,
}

impl Tab for HelloTab {
    const TITLE: &'static str = "Hello";
    const ICON: Lucide = Lucide::Microchip;
    const HINTS: &'static [Hint] = &[];

    fn new(models: Entity<Models>, _: &Shared, _: &mut Window, cx: &mut Context<Self>) -> Self {
        Self { models, focus: cx.focus_handle() }
    }

    fn focus_handle(&self) -> FocusHandle {
        self.focus.clone()
    }
}

impl Render for HelloTab {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let m = self.models.read(cx);
        let title = m.selected().map(|k| m.title(k)).unwrap_or_default();
        div().id("hello-tab").track_focus(&self.focus).size_full().p_8().child(format!("Hello from {title}"))
    }
}
```

A tab that runs a model makes its own lane in `new`, with `models.read(cx).inference().lane(Priority::Interactive, Policy::Keep)`.
It submits with `lane.submit(input, &[head], display)`, keeps the returned ticket (dropping it cancels the request), and reads the updates with `gui::pump`.
The updates are `Update::Decoded`, then one `Update::Head` per head with a `HeadOutput`, then `Update::Done`.
`gui/classify.rs` is the full example. `Tab`'s other methods are optional. `TAKES_INPUTS` and `take` receive dropped, opened and pasted pictures, `run` is Ctrl+R and `--autorun`, and `follow_model` runs when the selected model changes.
State that several tabs share goes in `app::Shared`, which `app::shared` builds once per window.

**A subcommand** is a module with a `cli::Command { name, args, about, run }`, a line in `app::COMMANDS`, and its name in `COMMANDS` in `packaging/app.env`, which the launchers start without a window. A test fails when the two lists differ. `--help` lists the commands in that order. `cli.rs` has what the commands share: the registry, the service with its log lines, the start-up default head, and one request's results.

**Speed.** New apps measure latency with the command line, not with a GUI tab:

```sh
hello-ai bench --model mine --rounds 20 picture1.jpg picture2.jpg
```

It prints JSON with the latency per stage (read, decode, preprocess, run) and the streamed images per second.

## 4. Build, test and package

```sh
export CARGO_BUILD_JOBS=8
cargo build --release
cargo test
```

`cargo test` runs the crate's tests on its own fixtures and on a fake backend (`inference::testing`), the window tests headless, and `scripts/test_gpu_env.sh`.
None of them need a model in `exports/` or a GPU.
`test_gpu_env.sh` checks the GPU launcher with fake runtimes. Without an NVIDIA driver it prints `skip: no NVIDIA driver` and passes, so run it by hand (`sh apps/<name>/scripts/test_gpu_env.sh`) on a machine with one, and look for 17 `ok` lines.

The binary decodes JPEGs with the pure-Rust zune-jpeg unless you build with `--features turbojpeg` (the system libjpeg-turbo) or `turbojpeg-vendored` (built in, for the bundle). Only libjpeg-turbo matches Pillow's pixels exactly. Use it when you compare the app's outputs with a Python pipeline.

**The bundle.** `apps/<name>/scripts/package_linux.sh` builds `dist/<name>-linux-x86_64/` and a `.tar.gz`.
It needs cargo-zigbuild, zig 0.14 (zig 0.17 fails the final link), cmake, nasm and a CPU `libonnxruntime.so` 1.31:

```sh
cargo install cargo-zigbuild --locked
uv venv ~/.local/share/zig-venv && uv pip install --python ~/.local/share/zig-venv/bin/python ziglang==0.14.1
export CARGO_ZIGBUILD_ZIG_PATH=$(ls -d ~/.local/share/zig-venv/lib/python3.*/site-packages/ziglang)/zig
export ORT_LIB=/path/to/libonnxruntime.so.1.31.0
apps/hello-ai/scripts/package_linux.sh --no-archive
```

The bundle holds the binary (built for glibc 2.28), onnxruntime, the launchers, `app.env`, `README.txt`, the licences and what `packaging/bundle.txt` lists.
A new app's `bundle.txt` lists `exports`, the whole folder. List single files instead when `exports/` holds more than the app ships.
`packaging/README.txt` is a stub. Write it before you hand the bundle to anyone. The classifier's `apps/gpui-demo/packaging/README.txt` has GPU, NixOS and troubleshooting sections to copy from.

The GPU runtime is not bundled, because it is several GB. `run.sh` looks for an onnxruntime-gpu 1.31 install in `$<PREFIX>_GPU_VENV`, then in `~/.local/share/onnxruntime-gpu/1.31` (shared by every app), then in `~/.local/share/<name>/gpu`, then in an active virtualenv. `packaging/README.txt` in this repo has the install command.

## Add a non-classification task

Do this only when a real exported graph exists. The first time, do steps 1 to 3 once. After that each new task is steps 4 to 7.

1. Change the `Graph` seam to return `Vec<Tensor { shape: Vec<usize>, data: Vec<f32> }>`, and slice rows by `shape[0]` in `run_batch` (`crates/inference/src/service.rs`). The adapter widens f16 and int64 outputs to f32.
2. Add `Task::outputs() -> Vec<&str>`. Use it in `run_batch` and in `check_io` (`runtime.rs`) in place of the irrefutable `let Task::Classify(t)` patterns. Once `Task` gets a second variant, those patterns stop compiling, and that lists every site to fix.
3. Return `(Vec<f32>, Frame)` from `preprocess::to_tensor`. `Frame` is the inverse of the transform (source size, scale, crop or pad offset), so boxes and masks map back onto the source picture. Add `Preprocess::Letterbox { size, pad, filter, scale, mean, std }`, because a center crop drops the edges where objects sit.
4. Add a `WireTask` variant and its parse-time rules in `manifest.rs`.
5. Add the `Task` variant, its `check_io` rule, and a pure `read_<task>(task, rows, &Frame) -> HeadOutput` in a new module next to `classification.rs`, with unit tests on fixed rows.
6. Add a `service::HeadOutput` variant. The compiler then lists each tab and subcommand that has to show the new variant or skip it, `app::subtitle` among them.
7. Add one widget in the app that renders it.

| Task | Manifest `task` | Graph contract (post-processing is in the graph) | `HeadOutput` | Renderer |
|---|---|---|---|---|
| classify (exists) | `readout`, `labels`, `calibration`, `decision` | `[N, C]` probs or logits | `Classify(Classification)` | `widgets::prediction_block` |
| embed | `{ output, normalize }` | `[N, D]` | `Embed(Arc<[f32]>)` | dims and norm. Similarity search is app code over the app's own gallery. |
| detect | `{ boxes, scores, classes, labels, min_score }` | NMS in the graph, padded `[N, K, 4]` xyxy in input pixels, `[N, K]` scores, `[N, K]` classes | `Detect(Vec<Det { rect, score, class }>)`, `rect` in source pixels through `Frame` | bordered divs over the picture, plus a label tag |
| segment | `{ output, labels }` | `[N, C, h, w]` logits or `[N, h, w]` class ids | `Segment(Mask { w, h, class: Arc<[u16]> })` | a palette RGBA texture over the picture through `gui/images.rs` |
| OCR (one text line) | `{ output, charset, blank }` | `[N, T, V]` CTC logits | `Text { text, confidence }` | text plus confidence |

Full-page OCR needs a detector, then crops, then a recognizer. That is routing between models, which the service does not do.

## Plug in an input source

- Files, paths, `file://` and http(s) URLs already work through `ImageInput::Source(fetch::Source)`. The service's io threads read and decode them.
- Pasted image bytes already work through `ImageInput::Bytes`.
- A folder is enumerated in the app and submitted as one `Source::File` per picture, on a lane with `Priority::Bulk` and `Policy::Keep`.
- A camera or a video needs no crate change. A producer thread in the app decodes frames and submits `ImageInput::Decoded(Arc<RgbImage>)` on a lane with `Policy::Latest`. That policy supersedes queued and running work. The tab draws the latest frame and the latest result.

## Fixes that two apps need

Each app has its own copy of `crates/inference` and of the window shell, and a fix is ported by hand.
Compare a file with `diff -u ../ITRI626/crates/inference/src/service.rs crates/inference/src/service.rs`.
The first time one fix has to land in two apps, make the crate a git dependency pinned to a tag instead.

## What the template leaves out

Each item has a trigger. Build it when the trigger happens.

- A shared UI crate. Apps copy the shell. GPUI Kit is pinned `=0.7.3` and pre-1.0, so apps that shared a shell crate would have to upgrade together. Trigger: a third app copies the shell unchanged.
- Task kinds beyond classify. Trigger: a real exported graph of that task. The design is above.
- Per-tab model selection. Trigger: one index mixes task kinds.
- Routing or pipelines, such as a detector feeding a recognizer.
- A second backend or a non-image model. `Loader` is the seam for a backend that takes and returns tensors (native TensorRT, a Triton server). An LLM or an audio model is a different service.
