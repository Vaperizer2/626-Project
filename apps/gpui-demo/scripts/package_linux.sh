#!/usr/bin/env bash
# Build the portable Linux x86_64 bundle of gpui-demo ("Component Classifier") for another machine:
#   dist/component-classifier-linux-x86_64/          binary + onnxruntime + models + demo stack + run.sh + docs
#   dist/component-classifier-linux-x86_64.tar.gz    (+ .sha256, + .part-aa/ab/... of 1900 MiB for 2 GB upload caps)
#
# The binary is linked against an old glibc with cargo-zigbuild (default 2.28, the same floor as the manylinux_2_28 onnxruntime; GLIBC=2.31 etc. also work) and decodes
# JPEGs with a statically linked, vendored libjpeg-turbo (feature `turbojpeg-vendored`), so it has no
# libjpeg-turbo runtime dependency and keeps bit-exact parity with the Python/Pillow pipeline.
#
# Needs on the build machine: rustup/cargo, cargo-zigbuild (`cargo install cargo-zigbuild`), zig (on PATH, or
# `python -m ziglang` from `uv pip install ziglang==0.14.1` with CARGO_ZIGBUILD_PYTHON_PATH=<that python>),
# cmake + nasm (libjpeg-turbo), python3, binutils (readelf/objdump/strip), tar, gzip, split, sha256sum.
# Inputs: exports/final, exports/itri626/{A_scratch_cnn,B_convnext_tiny}, apps/gpui-demo/demo_stack and an
# onnxruntime 1.31 library (ORT_LIB, default apps/gpui-demo/target/release/libonnxruntime.so.1.31.0).
#
# Usage: apps/gpui-demo/scripts/package_linux.sh [--skip-build] [--no-archive]
# Env:   GLIBC=2.28  PART_MIB=1900  CARGO_BUILD_JOBS=4  CARGO_TARGET_DIR=dist/.build-target  ORT_LIB=...  DIST=<repo>/dist
set -euo pipefail

APP=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
REPO=$(CDPATH='' cd -- "$APP/../.." && pwd)
DIST=${DIST:-$REPO/dist}
NAME=component-classifier-linux-x86_64
OUT=$DIST/$NAME
GLIBC=${GLIBC:-2.28}
TRIPLE=x86_64-unknown-linux-gnu
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$DIST/.build-target}
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4}
ORT_LIB=${ORT_LIB:-$APP/target/release/libonnxruntime.so.1.31.0}
PART_MIB=${PART_MIB:-1900}
SKIP_BUILD=0; ARCHIVE=1
for a in "$@"; do
    case $a in
        --skip-build) SKIP_BUILD=1 ;;
        --no-archive) ARCHIVE=0 ;;
        *) echo "unknown option $a" >&2; exit 2 ;;
    esac
done
say() { printf '\n== %s\n' "$*"; }
die() { echo "package_linux.sh: $*" >&2; exit 1; }

mkdir -p "$DIST"
BIN=$CARGO_TARGET_DIR/$TRIPLE/release/gpui-demo

# ---------------------------------------------------------------------------------------------- 1. build
if [ "$SKIP_BUILD" = 0 ]; then
    say "building with cargo-zigbuild for glibc $GLIBC"
    (cd "$APP" && cargo zigbuild --release --locked --features turbojpeg-vendored --target "$TRIPLE.$GLIBC")
fi
[ -x "$BIN" ] || die "no binary at $BIN"

# ---------------------------------------------------------------------------------------------- 2. verify
say "checking the binary"
maxglibc=$(objdump -T "$BIN" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -uV | tail -1)
echo "max GLIBC symbol version: $maxglibc (limit $GLIBC)"
[ "$(printf '%s\n%s\n' "$maxglibc" "$GLIBC" | sort -V | tail -1)" = "$GLIBC" ] || die "binary needs GLIBC_$maxglibc > $GLIBC"
needed=$(readelf -d "$BIN" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | tr '\n' ' ')
echo "DT_NEEDED: $needed"
for lib in $needed; do
    case $lib in
        libc.so.6 | libm.so.6 | libdl.so.2 | libpthread.so.0 | librt.so.1 | libgcc_s.so.1 | ld-linux-x86-64.so.2 | \
            libxcb.so.1 | libxkbcommon.so.0 | libxkbcommon-x11.so.0) ;;
        *) die "unexpected runtime dependency $lib (only base system libraries are allowed)" ;;
    esac
done
[ -f "$ORT_LIB" ] || die "onnxruntime library not found: $ORT_LIB (set ORT_LIB)"
for f in "$REPO/exports/final/model.onnx.data" "$REPO/exports/final/model_multi.onnx" \
    "$REPO/exports/itri626/A_scratch_cnn/model.onnx" "$REPO/exports/itri626/B_convnext_tiny/model.onnx" \
    "$APP/demo_stack/demo_stack.json"; do
    [ -f "$f" ] || die "missing input $f"
done

# ---------------------------------------------------------------------------------------------- 3. bundle
say "assembling $OUT"
case $OUT in "$DIST"/component-classifier-linux-x86_64) ;; *) die "refusing to replace $OUT" ;; esac
rm -rf "$OUT"
mkdir -p "$OUT/exports/final" "$OUT/exports/itri626" "$OUT/LICENSES"
cp "$BIN" "$OUT/gpui-demo" && strip --strip-all "$OUT/gpui-demo"
cp "$ORT_LIB" "$OUT/libonnxruntime.so.1.31.0"
# only what the app reads: model.onnx is required (folder detection + the CLI default), model_multi.onnx
# carries Deployed + C in one graph over the shared model.onnx.data
for f in model.onnx model.onnx.data model_multi.onnx model_multi.json preprocess.json labels.json; do
    cp --reflink=auto "$REPO/exports/final/$f" "$OUT/exports/final/"
done
for m in A_scratch_cnn B_convnext_tiny; do
    mkdir -p "$OUT/exports/itri626/$m"
    for f in model.onnx preprocess.json labels.json; do cp "$REPO/exports/itri626/$m/$f" "$OUT/exports/itri626/$m/"; done
done
cp -r "$APP/demo_stack" "$OUT/demo_stack"
cp "$APP/packaging/run.sh" "$OUT/run.sh" && cp "$APP/packaging/run-nixos.sh" "$OUT/run-nixos.sh" && chmod 755 "$OUT/run.sh" "$OUT/run-nixos.sh" "$OUT/gpui-demo"
cp "$APP/packaging/README.txt" "$OUT/README.txt"

# ---------------------------------------------------------------------------------------------- 4. licences
say "licences"
L=$OUT/LICENSES
REG=$(cargo metadata --format-version 1 --manifest-path "$APP/Cargo.toml" --locked --filter-platform "$TRIPLE" | python3 -c \
    'import json,sys,os; d=json.load(sys.stdin); p=[x for x in d["packages"] if x["name"]=="gpui-kit"][0]; print(os.path.dirname(os.path.dirname(p["manifest_path"])))')
cp "$APP/assets/fonts/OFL.txt" "$L/NotoSans-OFL.txt"
cp "$APP/packaging/licenses/onnxruntime-LICENSE.txt" "$L/onnxruntime-LICENSE.txt"
cp "$APP/packaging/licenses/onnxruntime-ThirdPartyNotices.txt" "$L/onnxruntime-ThirdPartyNotices.txt"
cp "$REG"/turbojpeg-sys-*/libjpeg-turbo/LICENSE.md "$L/libjpeg-turbo-LICENSE.md"
cp "$REG"/turbojpeg-sys-*/libjpeg-turbo/README.ijg "$L/libjpeg-turbo-README.ijg"
cp "$REG"/gpui-kit-assets-*/LICENSE-LUCIDE "$L/Lucide-icons-LICENSE.txt"
cp "$(find "$REG" -maxdepth 2 -path "*/serde-1.*/LICENSE-APACHE" | sort -V | tail -1)" "$L/Apache-2.0.txt"
cp "$(find "$REG" -maxdepth 2 -path "*/serde-1.*/LICENSE-MIT" | sort -V | tail -1)" "$L/MIT-template.txt"
cp "$APP/packaging/licenses/DATASET.txt" "$L/DATASET.txt"
cargo metadata --format-version 1 --manifest-path "$APP/Cargo.toml" --locked --features turbojpeg-vendored \
    --filter-platform "$TRIPLE" | python3 "$APP/packaging/crate_licenses.py" > "$L/THIRD-PARTY-CRATES.txt"
echo "$(grep -c '^[a-z0-9]' "$L/THIRD-PARTY-CRATES.txt") crates listed"

(cd "$OUT" && find . -type f -printf '%P\n' | LC_ALL=C sort | xargs -d '\n' sha256sum) > "$DIST/.SHA256SUMS.tmp"
mv "$DIST/.SHA256SUMS.tmp" "$OUT/SHA256SUMS"
du -sh "$OUT"

# ---------------------------------------------------------------------------------------------- 5. archive
if [ "$ARCHIVE" = 1 ]; then
    say "archive"
    TGZ=$DIST/$NAME.tar.gz
    rm -f "$TGZ" "$TGZ".sha256 "$TGZ".part-*
    tar -C "$DIST" --owner=0 --group=0 --numeric-owner --sort=name -cf - "$NAME" | gzip -1 > "$TGZ"
    # parts of PART_MIB MiB (default 1900) for upload services with a 2 GB cap; skipped when the archive fits
    if [ "$(stat -c %s "$TGZ")" -gt $((PART_MIB * 1024 * 1024)) ]; then
        split -b "${PART_MIB}M" "$TGZ" "$TGZ.part-"
        (cd "$DIST" && sha256sum "$NAME.tar.gz" "$NAME".tar.gz.part-* > "$NAME.tar.gz.sha256")
        echo "reassemble: cat $NAME.tar.gz.part-* > $NAME.tar.gz && sha256sum -c $NAME.tar.gz.sha256 --ignore-missing" \
            > "$DIST/$NAME.REASSEMBLE.txt"
    else
        (cd "$DIST" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
        rm -f "$DIST/$NAME.REASSEMBLE.txt"
        echo "archive is below $PART_MIB MiB: no parts needed"
    fi
    ls -l "$TGZ"*
    cat "$DIST/$NAME.tar.gz.sha256"
fi
