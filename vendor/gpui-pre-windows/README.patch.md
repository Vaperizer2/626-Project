# Local GPUI Windows shader patch

Upstream: crates.io gpui-pre-windows 0.3.9, Zed revision 93d745668805029476e6d7fbb19652116fed51de. Apache-2.0; original licence is included.

The runtime shader loader used by debug-assertion builds canonicalized CARGO_MANIFEST_DIR/src/*.hlsl on the user's machine. Our cross-built release enabled that path, so the GUI failed outside Boat.

Changes from the published crate:

- Move directx_renderer.rs's shader_resources module unchanged into shader_resources.rs, so the same loader can be exercised by the standalone Windows shader check without initializing the GUI.
- In that module, embed shaders.hlsl, color_text_raster.hlsl and alpha_correction.hlsl with include_str!, expand the one shared include, and pass the complete source to D3DCompile in memory. No user/build-machine path or external shader files are needed.
- Preserve the existing native release/precompiled shader path and all other upstream behavior.

Regression: tools/windows-shader-check imports this exact module and compiles all 18 vertex/fragment programs, checking for DXBC output. Run it on Windows, or run `bash tools/windows-shader-check/check-wine.sh` from the repository root. The Wine check uses a checksum-verified Microsoft SDK compiler DLL, not Wine's incomplete builtin HLSL compiler, and runs with only the executable and compiler DLL in an unrelated temporary directory. See the harness README for requirements. The unmodified loader fails with File not found when its build-machine source directory is unavailable.

Remove the patch when upgrading to an upstream implementation with relocatable runtime shader loading. Keep Cargo.toml pinned to the supported version until then.
