# Windows review build for Meiring

Download the assets from [the Windows review prerelease](https://github.com/Vaperizer2/626-Project/releases/tag/meiring-windows-review-2026-11-05).
Repository access is required because this project is private.

1. Download both `component-classifier-windows-x86_64.zip.part-aa` and `.part-ab`, plus `Extract-Windows-App.bat`, into one folder. GitHub limits individual release asset sizes, so the ZIP is split.
2. Double-click `Extract-Windows-App.bat`. It joins the parts, verifies the complete ZIP's SHA-256 and extracts the app. Allow roughly 9 GB free disk for the parts, assembled ZIP and extracted folder. Re-running extraction requires moving or removing the previous extracted folder first.
3. Apply the application-only fix below before testing the GUI. The original executable has a shader source-path bug.
4. If Windows reports missing MSVCP or VCRUNTIME DLLs, install the included Microsoft `VC_redist.x64.exe`, then retry.

This is an x64 Windows 10/11 test build with CPU ONNX Runtime 1.31.0. Models A, B, C, Deployed, Fast and Small, 68 demo photos and their licences are included. Fast is intended for TensorRT; on this CPU-only package it can be slow. The optional Deployed at crop 1.0 is excluded, matching the packaging manifest.

## Application-only fix1: keep your extracted models

Meiring confirmed that extraction and inference with all six models work. The GUI crash is separate: our
debug-assertion-enabled cross-build tried to read GPUI's HLSL source from an absolute Boat build-machine path.
The fix embeds that source in the executable; it does not change the model graphs, weights or label order.
Native Windows GUI validation of the fix is **pending**.

Download `component-classifier-windows-fix1.zip` and its `.zip.sha256` from the same
prerelease. This is a small executable-only update, not another model download.

1. Close the app. Optionally back up the existing `gpui-demo.exe` and `run.bat`.
2. Verify the downloaded ZIP against its `.zip.sha256` (PowerShell: `Get-FileHash .\component-classifier-windows-fix1.zip -Algorithm SHA256`).
3. Extract **all contents of the fix ZIP INTO THE ALREADY EXTRACTED original app folder**,
   `component-classifier-windows-x86_64`. Replace `gpui-demo.exe` and `run.bat`. The replacement executable must
   sit beside `onnxruntime.dll` and `exports`, not in another nested folder. Keep the original DLLs, models,
   demo photos and licences. No fake `C:\home\user` directory or Boat source tree is needed.
4. In Command Prompt, run `run.bat --tab demo`. From elsewhere, a path with spaces works when quoted:

   ```bat
   "C:\Users\Meiring\Downloads\Component Classifier\run.bat" --tab demo
   ```

The launcher intentionally uses the bundled CPU runtime. No GPU launcher is included; the reviewer's
working external GPU launcher can remain separate. Arguments are forwarded unchanged and the application's exit
code is preserved. A no-argument/double-click launch pauses on exit, and other GUI launches pause on failure so
errors stay readable. Headless commands (`classify`, `preprocess`, `bench`, `models`, help/version) do not pause.
Set `COMPCLASS_NO_PAUSE=1` for automated GUI runs; use `call` to invoke the launcher from another batch file.

The ZIP contains exactly `gpui-demo.exe`, `run.bat`, `BUILD_INFO_FIX1.txt`, `UPDATE_WINDOWS_FIX1.txt` and
`SHA256SUMS-WINDOWS-FIX1.txt`, all at its root. The new checksum list covers only the update. The original bundle's
`SHA256SUMS` still describes the old executable and launcher, so those two entries are expected to differ after
updating; all other entries still apply. To roll back, restore the two backups or extract the original archive
into a separate folder.

## What to review and test

- Open each of Classify, Demo and Benchmark. Confirm labels, text and icons render correctly.
- Open a local image, then test drag/drop, clipboard input and a URL.
- Select Small and confirm it loads without a checksum error. Switch among all six models.
- In Demo, step through cards and try Run all. These are demonstration images, not a new accuracy study.
- Start and cancel Benchmark. Confirm the window remains responsive.
- In Command Prompt, run `run.bat models --check`, then `run.bat classify --model all demo_stack\01_battery_holder.jpg`.
- Report the Windows version, GPU, failing action and any terminal output. CPU inference can take several seconds for the larger models.

## Build and validation limits

The original package was main `17aff8b` plus the PR's Small labels-manifest checksum correction and registry-test
head-count correction. Fix1 adds the local `gpui-pre-windows` shader-source patch and the launcher update;
`BUILD_INFO_FIX1.txt` records the final executable's declared source commit, byte count and SHA-256. Package only
a binary built from that commit, including the local crate patch.

Meiring also identified a Windows checkout hazard: the old Small labels file had 328 CRLF bytes instead of the
committed 307 LF bytes, which changes its manifest hash. Root `.gitattributes` keeps `exports/**/*.json` at LF
even with `core.autocrlf=true`. This protects fresh source checkouts; it does not rewrite an already-extracted
review bundle. Meiring's existing extracted package already passed all six model checks; do not replace its
model metadata as part of this update.

Canonical cross-build (run on Boat, not the laptop), retaining the original Rust 1.97.1 / cargo-xwin 0.23.1
configuration and the workspace's patched `gpui-pre-windows` crate:

```sh
CARGO_TARGET_DIR=target-windows CARGO_BUILD_JOBS=6 \
CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true \
cargo xwin build --release --locked --target x86_64-pc-windows-msvc \
  -p gpui-demo --features turbojpeg-vendored
```

Debug assertions enable GPUI's runtime shader compilation while retaining release optimizations. With the patch,
that compiler reads embedded HLSL rather than the absolute source path. Merely keeping debug assertions enabled
without the patch reproduces the original problem. The normal native release/precompiled-shader branch is not
changed by this fix.

The focused shader regression check imports the actual patched loader and compiles all 18 vertex/fragment
programs, without opening a window or loading ONNX Runtime:

```sh
# Linux/Boat with Wine and Microsoft's compiler DLL, fetched and verified by the script:
bash tools/windows-shader-check/check-wine.sh
```

Or on native Windows with Rust/MSVC and the Windows SDK:

```powershell
cargo run --locked --manifest-path tools/windows-shader-check/Cargo.toml
```

See [the shader-check README](../tools/windows-shader-check/README.md) for the relocated-executable check.
Passing this proves shader compilation, not native Windows GUI correctness. The original full-app Wine attempt
could not load Windows ICU (`icuuc.dll`); native Windows GUI validation is still pending.

The original Linux build passed 87 Rust tests (two ignored), loaded every bundled graph with valid checksums,
and returned predictions from all six heads on one demonstration image. Meiring subsequently confirmed Windows
extraction and inference. Neither establishes correctness of the patched Windows GUI; report that only after a
native smoke test. Build metadata and checksums are included in each package.

## Package the small update (maintainer)

Wait for the patched Windows binary and its matching final source commit. From the repo root:

```sh
uv run --no-project python apps/gpui-demo/scripts/package_windows_update.py \
  --exe target-windows/x86_64-pc-windows-msvc/release/gpui-demo.exe \
  --source-commit "$(git rev-parse HEAD)"
```

This writes `dist/local-main/component-classifier-windows-fix1.zip` and `.zip.sha256`; it does not rebuild,
run the executable, modify the original extracted package, publish assets or read any models. It rejects a
non-x64 PE input or abbreviated commit, and refuses existing outputs unless `--force` is supplied. `--output`
can select a temporary fixture destination. ZIP member order, timestamps and permissions are fixed, so equal
inputs produce equal archives with the same Python/zlib version. Source provenance is declared by the packager,
not inferred or independently verified from executable bytes. Only package the final matching build.

Before publication, inspect the five ZIP members, verify the ZIP checksum and overlay onto a copy of the original
bundle. The source-side launcher is normalized to CRLF in the ZIP regardless of the checkout's line endings.
Do not attach replacement models, graph files or a new full bundle. Keep the prerelease status until native
Windows GUI testing is complete.
