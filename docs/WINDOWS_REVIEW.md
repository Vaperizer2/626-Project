# Windows review build for Meiring

Download the assets from [the Windows review prerelease](https://github.com/Vaperizer2/626-Project/releases/tag/meiring-windows-review-2026-11-05).
Repository access is required because this project is private.

1. Download both `component-classifier-windows-x86_64.zip.part-aa` and `.part-ab`, plus `Extract-Windows-App.bat`, into one folder. GitHub limits individual release asset sizes, so the ZIP is split.
2. Double-click `Extract-Windows-App.bat`. It joins the parts, verifies the complete ZIP's SHA-256 and extracts the app. Allow roughly 9 GB free disk for the parts, assembled ZIP and extracted folder. Re-running extraction requires moving or removing the previous extracted folder first.
3. Open `component-classifier-windows-x86_64`, then double-click `run.bat`.
4. If Windows reports missing MSVCP or VCRUNTIME DLLs, install the included Microsoft `VC_redist.x64.exe`, then retry.

This is an x64 Windows 10/11 test build with CPU ONNX Runtime 1.31.0. Models A, B, C, Deployed, Fast and Small, 68 demo photos and their licences are included. Fast is intended for TensorRT; on this CPU-only package it can be slow. The optional Deployed at crop 1.0 is excluded, matching the packaging manifest.

## What to review and test

- Open each of Classify, Demo and Benchmark. Confirm labels, text and icons render correctly.
- Open a local image, then test drag/drop, clipboard input and a URL.
- Select Small and confirm it loads without a checksum error. Switch among all six models.
- In Demo, step through cards and try Run all. These are demonstration images, not a new accuracy study.
- Start and cancel Benchmark. Confirm the window remains responsive.
- In Command Prompt, run `run.bat models --check`, then `run.bat classify --model all demo_stack\01_battery_holder.jpg`.
- Report the Windows version, GPU, failing action and any terminal output. CPU inference can take several seconds for the larger models.

## Build and validation limits

Source is main `17aff8b` plus the PR's Small labels-manifest checksum correction and registry-test head-count correction. The manifest now hashes the committed labels file; no model weights or label order changed.

The executable was cross-compiled on Boat with Rust 1.97.1 and cargo-xwin 0.23.1:

```sh
CARGO_TARGET_DIR=target-windows CARGO_BUILD_JOBS=6 \
CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true \
cargo xwin build --release --locked --target x86_64-pc-windows-msvc \
  -p gpui-demo --features turbojpeg-vendored
```

Debug assertions enable GPUI's runtime shader compilation while retaining release optimizations. This review package has not been validated on native Windows. Wine could not load it because that Wine installation lacks the Windows ICU system component (`icuuc.dll`).

The Linux build passed 87 Rust tests (two ignored), loaded every bundled graph with valid checksums, and returned predictions from all six heads on one demonstration image. These checks do not establish Windows GUI correctness. Build metadata and file checksums are included in the package.
