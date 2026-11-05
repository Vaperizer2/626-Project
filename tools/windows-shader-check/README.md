# Windows runtime shader regression check

This standalone crate imports `vendor/gpui-pre-windows/src/shader_resources.rs`
directly. It calls the application's actual `RawShaderBytes::new` for all nine
modules, both vertex and fragment targets, and requires DXBC output for all 18.
It does not initialize GPUI, Direct3D, a display, ONNX Runtime or ICU.
Debug assertions stay enabled even for an optimized harness build: this check
exercises the runtime compiler, not precompiled bytecode.

## Windows

With Rust's MSVC toolchain and the Windows SDK installed:

```powershell
cargo run --locked --manifest-path tools/windows-shader-check/Cargo.toml
```

To check relocation, copy only `windows-shader-check.exe` from this crate's
`target/debug` directory to an unrelated directory, change into that directory,
and run it. Do not copy any HLSL files. Windows supplies `d3dcompiler_47.dll`.
The final line must be `PASS: all 18 shader programs compiled to DXBC`.

## Linux + Wine

Requires `cargo-xwin`, Wine with x64 support, `curl`, `cabextract`, and `sha256sum`:

```bash
bash tools/windows-shader-check/check-wine.sh
```

The script downloads a pinned Microsoft Windows SDK cabinet from
`download.microsoft.com`, verifies both cabinet and DLL SHA-256 values, and
extracts the x64 `d3dcompiler_47.dll` to a temporary directory. It copies only the
cross-built harness executable beside the DLL and runs from that unrelated
directory, forcing Wine to use the native Microsoft compiler. Both downloaded
and copied files are removed afterward; nothing is installed or redistributed.
`CARGO_TARGET_DIR` may point at an existing cross-build cache.

Wine 9's builtin HLSL compiler rejects valid upstream HLSL with `E5000` at line
576. That is not the relocation regression: do not simplify the shaders to
accommodate Wine. Microsoft's compiler successfully compiles all 18 programs.
The original path-based loader instead fails with `File not found` when the
build-machine source directory is unavailable.

This is shader-compilation coverage, not a complete Windows GUI smoke test.
The application's normal native release build still uses upstream `build.rs`
and `fxc.exe` to generate and embed precompiled bytecode when debug assertions
are disabled; that branch and all three HLSL sources are unchanged.
