Component Classifier: portable Linux build
==========================================

A desktop demo of an image classifier for 20 kinds of electronic components (battery holder, breadboard,
capacitor, ..., transistor). It was made for an honours project (ITRI626, deep learning for image analysis).
You give it a photo; it says which component it sees and how sure it is. It runs fully offline on the CPU.
Four models are included, so you can compare them:

  A  Custom CNN, trained from scratch       (160 px input, ~5 MB)
  B  ConvNeXt-T, fine-tuned                 (224 px input, ~110 MB)
  C  SigLIP ViT + linear head               (384 px input, shares the 1.7 GB SigLIP weights)
  Deployed  fused SigLIP model, calibrated; it answers "Unsure" when its top probability is below 50%


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
  ./run.sh classify --model all demo_stack/01_battery_holder.jpg      (all four models)

The first prediction with C or Deployed takes a few seconds while the 1.7 GB weights load; after that a
picture takes about 1-2 s with SigLIP and well under 0.2 s with A or B.


What to try (the window)
------------------------
Classify tab   Open (Ctrl+O), drag and drop, or paste (Ctrl+V) a picture, a file path or an image URL.
               It shows the top class, its confidence, the top 3 as bars and all 20 probabilities.
Demo tab       28 Wikimedia Commons photos the models never saw in training. Click a card, or step with
               Left/Right and press Enter. "Run all" (Ctrl+R) classifies every photo. Each result is marked
               correct (green tick), wrong (red cross) or unsure (violet question mark).
Benchmark tab  Runs all four models over the 28 photos (about 1 minute) and compares them. A switch shows
               only the photos where the models disagree.

Model picker (top right): A / B / C / Deployed, or Ctrl+M to cycle.
Other keys: Ctrl+1/2/3 tabs, Ctrl+T light/dark, Esc stop a run, Ctrl+Q quit.

Expected results on the 28 demo photos (Benchmark tab or Demo "Run all"):
  A 11/28    B 17/28    C 21/28    Deployed 21/28 (2 of them "Unsure")
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
* Lots of output on the terminal? Only lines starting with "error" matter.


What is inside
--------------
gpui-demo                  the app (Rust; GPUI Kit UI, onnxruntime inference), built for glibc >= 2.28
libonnxruntime.so.1.31.0   ONNX Runtime 1.31.0 (CPU), loaded by the app at start-up
exports/final/             Deployed + C: model.onnx, model_multi.onnx, model.onnx.data (shared weights),
                           preprocess.json, labels.json, model_multi.json
exports/itri626/           A_scratch_cnn/ and B_convnext_tiny/: model.onnx, preprocess.json, labels.json
demo_stack/                the 28 demo photos, demo_stack.json, ATTRIBUTION.md (author/licence of each photo)
run.sh                     launcher (sets the paths above; passes arguments through)
LICENSES/                  licences: onnxruntime (MIT), Noto Sans (SIL OFL 1.1, embedded in the binary),
                           GPUI Kit (Apache-2.0), Rust crates, dataset note, photo credits pointer
SHA256SUMS                 checksums of every file

The app writes nothing to disk except what you ask it to (it keeps no settings or caches).

NixOS (niri, sway, Hyprland, ...)
---------------------------------
The plain ./run.sh cannot start on NixOS (no /lib64 loader, no /usr/lib), and if the Wayland libraries are not found
the app falls back to X11, which niri does not provide. Use the NixOS launcher instead; it takes the libraries from
your nixpkgs (first run downloads them once, cached in ~/.cache/component-classifier) and starts the app on Wayland:
    ./run-nixos.sh                  open the window
    ./run-nixos.sh --doctor         show which libraries/drivers were found (send this output if it fails)
    ./run-nixos.sh --x11            use X11 via xwayland-satellite instead
Requires hardware.graphics.enable = true (GPU drivers in /run/opengl-driver).
