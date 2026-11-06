#!/usr/bin/env bash
# Cross-build and run the actual GPUI runtime shader loader without a GUI/GPU.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
for tool in cargo curl cabextract wine sha256sum; do
    command -v "$tool" >/dev/null || { echo "Missing dependency: $tool" >&2; exit 1; }
done

work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT

# Microsoft's Windows 8.1 SDK x64 cabinet, also used by winetricks d3dcompiler_47.
# Wine's builtin compiler does not implement all the HLSL these shaders use.
curl --fail --location --silent --show-error \
    'https://download.microsoft.com/download/B/0/C/B0C80BA3-8AD6-4958-810B-6882485230B5/standalonesdk/Installers/61d57a7a82309cd161a854a6f4619e52.cab' \
    --output "$work/compiler.cab"
(
    cd -- "$work"
    echo 'f736e161547095bb8d98c636b85fdfeb4070fefeee3c3745db3ce88f6eb1d9de  compiler.cab' | sha256sum --check
    cabextract -q -F fil3585cb2ea5db13cc0838f8d06b5c9679 compiler.cab
    mv fil3585cb2ea5db13cc0838f8d06b5c9679 d3dcompiler_47.dll
    echo 'e994847e01a6f1e4cbdc5a864616ac262f67ee4f14db194984661a8d927ab7f4  d3dcompiler_47.dll' | sha256sum --check
)

export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-"$root/target/windows-shader-check"}
if [[ $CARGO_TARGET_DIR != /* ]]; then
    CARGO_TARGET_DIR="$PWD/$CARGO_TARGET_DIR"
fi
cargo xwin build --locked --manifest-path "$root/tools/windows-shader-check/Cargo.toml" \
    --target x86_64-pc-windows-msvc
cp -- "$CARGO_TARGET_DIR/x86_64-pc-windows-msvc/debug/windows-shader-check.exe" "$work/"

# The executable and compiler DLL are the only runtime inputs. No HLSL is copied.
# Native-only override prevents silently falling back to Wine's builtin compiler.
cd -- "$work"
WINEDLLOVERRIDES=d3dcompiler_47=n WINEDEBUG=-all wine ./windows-shader-check.exe
