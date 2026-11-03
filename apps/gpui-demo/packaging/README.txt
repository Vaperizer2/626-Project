Component Classifier: portable Linux build
==========================================

A desktop demo of an image classifier for 20 kinds of electronic components (battery holder, breadboard,
capacitor, ..., transistor). It was made for an honours project (ITRI626, deep learning for image analysis).
You give it a photo; it says which component it sees and how sure it is. It runs fully offline, on the CPU, or
on an NVIDIA GPU when you add the GPU runtime (see GPU below). Five models are included, so you can compare them:

  A  Custom CNN, trained from scratch       (160 px input, ~5 MB)
  B  ConvNeXt-T, fine-tuned                 (224 px input, ~110 MB)
  C  SigLIP ViT + linear head               (384 px input, shares the 1.7 GB SigLIP weights)
  Deployed  fused SigLIP model, calibrated; it answers "Unsure" when its top probability is below 50%
  Fast      the Deployed model at 308 px with INT8 weights for NVIDIA TensorRT: same cross-validated accuracy,
            under 20 ms per picture instead of about 115 ms on an RTX 3060 Laptop (440 MB; needs the GPU runtime)


Requirements
------------
* Linux on x86_64 (64-bit Intel/AMD) with glibc 2.28 or newer: Ubuntu 20.04+, Debian 10+, Fedora 29+,
  RHEL/Rocky/Alma 8+, openSUSE Leap 15.3+, Arch/Manjaro/CachyOS or similar. (Check: ldd --version)
* About 4 GB of free RAM (the SigLIP models use about 2.5 GB) and 2 GB of disk.
* For the window: a Wayland or X11 desktop session with working Vulkan or OpenGL drivers (any normal desktop
  install has them). The command-line "classify" mode needs no display at all.
* System libraries every desktop already has: glibc, libstdc++ and libgcc_s (for onnxruntime), libxcb,
  libxkbcommon, libxkbcommon-x11; loaded when available: libwayland-client + libwayland-egl (Wayland),
  libvulkan (Vulkan) or libEGL (OpenGL, e.g. Mesa). The UI font (Noto Sans) is built in. Nothing needs to be
  installed with root, and nothing is downloaded at run time.


Running it
----------
  tar xzf component-classifier-linux-x86_64.tar.gz      (or reassemble the parts first, see below)
  cd component-classifier-linux-x86_64
  ./run.sh                     opens the window
  ./run.sh --tab demo          start on the Demo tab
  ./run.sh --dark / --light    choose the theme (default follows your desktop; Ctrl+T toggles)
  ./run.sh --help              all options

Quick check without a window (prints the prediction, about 5 s including model loading):
  ./run.sh classify demo_stack/01_battery_holder.jpg
  ./run.sh classify --model all demo_stack/01_battery_holder.jpg      (all five models)

The first prediction with C or Deployed takes a few seconds while the 1.7 GB weights load; after that a
picture takes about 1-2 s with SigLIP on the CPU and well under 0.2 s with A or B.


GPU (optional, NVIDIA)
----------------------
The bundle ships the CPU build of ONNX Runtime. The GPU build (onnxruntime-gpu with the CUDA 13 and cuDNN 9
libraries, plus TensorRT 10 for the Fast model) is a separate install of about 7 GB. Install it once, no root,
with uv (https://docs.astral.sh/uv/). The folder is shared: other apps that use onnxruntime-gpu 1.31 find it too.
    uv venv ~/.local/share/onnxruntime-gpu/1.31 --python 3.12
    uv pip install --python ~/.local/share/onnxruntime-gpu/1.31/bin/python \
        'onnxruntime-gpu[cuda,cudnn]==1.31.1' 'tensorrt-cu13-libs==10.16.1.11'
It needs an NVIDIA driver that supports CUDA 13: the header of nvidia-smi shows a CUDA version of 13.0 or newer.
run.sh looks for it in $COMPCLASS_GPU_VENV, then in that folder, then in ~/.local/share/component-classifier/gpu
(where earlier versions said to install it), then in an active virtualenv, and says on its first line which runtime
it uses:
    run.sh: onnxruntime on GPU (onnxruntime 1.31.1, TensorRT from .../tensorrt_libs)
    run.sh: onnxruntime on GPU without TensorRT (...)      CUDA only: the app starts on Deployed
    run.sh: onnxruntime on CPU (no onnxruntime-gpu found: ...)
COMPCLASS_GPU=0 ./run.sh forces the CPU.

With TensorRT the app starts on Fast. The first time Fast loads, TensorRT builds an engine for your GPU: about
45 s on an RTX 3060 Laptop (longer when RAM is short; it needs about 6 GB at its peak). The model chip and the
status bar say "building TensorRT engine, first run only" with a seconds counter while it works. The engine is
kept in ~/.cache/component-classifier/tensorrt/ (about 450 MB), so later starts load it in about 1-2 s; a new GPU,
driver library or model version builds a new one. When a GPU provider cannot be used, the app runs on the CPU and
says why: hover the model picker, or run ./run.sh models.


What to try (the window)
------------------------
Classify tab   Open (Ctrl+O), drag and drop, or paste (Ctrl+V) a picture, a file path or an image URL.
               It shows the top class, its confidence, the top 3 as bars and all 20 probabilities.
Demo tab       68 Wikimedia Commons photos the models never saw in training. Click a card, or step with
               Left/Right and press Enter. "Run all" (Ctrl+R) classifies every photo. Each result is marked
               correct (green tick), wrong (red cross) or unsure (violet question mark).
Benchmark tab  Runs every available model over the 68 photos and compares them. A switch shows only the
               photos where the models disagree.

Model picker (top right): A / B / C / Deployed / Fast, or Ctrl+M to cycle.
Other keys: Ctrl+1/2/3 tabs, Ctrl+T light/dark, Esc stop a run, Ctrl+Q quit.

Expected results on the 68 demo photos (Benchmark tab or Demo "Run all"):
  A 19/68    B 38/68    C 53/68    Deployed 53/68 (10 photos answered "Unsure")
If you get these numbers, everything works. Slightly different numbers would mean a problem; please report them.


Sending a large file? Reassembling the parts
--------------------------------------------
If you received component-classifier-linux-x86_64.tar.gz.part-aa, -ab, ... instead of one .tar.gz:
  cat component-classifier-linux-x86_64.tar.gz.part-* > component-classifier-linux-x86_64.tar.gz
Check the download with:   sha256sum -c component-classifier-linux-x86_64.tar.gz.sha256
Check the extracted folder: cd component-classifier-linux-x86_64 && sha256sum -c SHA256SUMS


Troubleshooting
---------------
* "no graphical session found": run it from a terminal inside your desktop session. Over SSH use classify.
* The window does not open on Wayland, or looks wrong there: force X11 (XWayland):
      WAYLAND_DISPLAY= ./run.sh
* Black window or a GPU/Vulkan error: try Mesa's software renderer (slow but always works):
      LIBGL_ALWAYS_SOFTWARE=1 ./run.sh
  On NVIDIA laptops (Optimus) "prime-run ./run.sh" may also help.
* The app is killed or the machine freezes while loading C/Deployed: not enough free RAM. Close large
  programs; about 4 GB free is needed. A and B alone need much less.
* "loading .../model.onnx ... (is model.onnx.data next to it and complete?)": the archive was not fully
  extracted or a part was missing. exports/final/model.onnx.data must be 1,714,593,792 bytes; run
  sha256sum -c SHA256SUMS.
* Ctrl+O does nothing: the file dialog needs xdg-desktop-portal (installed on GNOME/KDE by default).
  Drag and drop and paste work without it.
* Messages like "pthread_setaffinity_np failed" from onnxruntime are harmless.
* "run.sh: onnxruntime on CPU" although you installed the GPU runtime: check the folder (COMPCLASS_GPU_VENV) and
  nvidia-smi. ./run.sh models lists, per model, which execution providers register and why the others do not.
* Lots of output on the terminal? Only lines starting with "error" matter.


What is inside
--------------
gpui-demo                  the app (Rust; GPUI Kit UI, onnxruntime inference), built for glibc >= 2.28
libonnxruntime.so.1.31.0   ONNX Runtime 1.31.0 (CPU), loaded by the app at start-up unless a GPU runtime is found
exports/models.json        the model index; each model folder has a model.json (files, checksums, preprocessing)
exports/final/             Deployed + C: model_multi.onnx (one graph, both outputs), model.onnx.data
                           (the 1.7 GB weights), model.json, preprocess.json, labels.json
exports/fast/              Fast: model.onnx, model.onnx.data, preprocess.json, runtime.json, labels.json
exports/itri626/           A_scratch_cnn/ and B_convnext_tiny/: model.onnx, preprocess.json, labels.json
demo_stack/                the 68 demo photos, demo_stack.json, ATTRIBUTION.md (author/licence of each photo)
run.sh                     launcher (sets the paths above; passes arguments through)
app.env                    the app's name, environment variable prefix, binary and subcommands, read by the launchers
gpu-env.sh                 used by run.sh and run-nixos.sh: finds the GPU runtime (see GPU)
LICENSES/                  licences: onnxruntime (MIT), Noto Sans (SIL OFL 1.1, embedded in the binary),
                           GPUI Kit (Apache-2.0), Rust crates, dataset note, photo credits pointer
SHA256SUMS                 checksums of every file

The app keeps no settings. It writes only to ~/.cache/component-classifier/: verified.json (which model files
passed a checksum, so they are not hashed again) and the TensorRT engines (see GPU).

NixOS (niri, sway, Hyprland, ...)
---------------------------------
The plain ./run.sh cannot start on NixOS (no /lib64 loader, no /usr/lib), and if the Wayland libraries are not found
the app falls back to X11, which niri does not provide. Use the NixOS launcher instead; it takes the libraries from
your nixpkgs (first run downloads them once, cached in ~/.cache/component-classifier) and starts the app on Wayland:
    ./run-nixos.sh                  open the window
    ./run-nixos.sh --doctor         show which libraries/drivers were found (send this output if it fails)
    ./run-nixos.sh --x11            use X11 via xwayland-satellite instead
Requires hardware.graphics.enable = true (GPU drivers in /run/opengl-driver).
