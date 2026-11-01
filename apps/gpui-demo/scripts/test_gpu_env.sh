#!/bin/sh
# Checks packaging/gpu-env.sh against fake GPU venvs under a path with spaces (run by `cargo test`, gpu_env tests).
#   sh scripts/test_gpu_env.sh     exit 0 when every case passes; skips without an NVIDIA driver (gpu-env.sh needs one)
set -eu
ENV_SH="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)/packaging/gpu-env.sh"
if ! ls /proc/driver/nvidia/gpus/* >/dev/null 2>&1; then
    echo "skip: no NVIDIA driver"
    exit 0
fi
T=$(mktemp -d "${TMPDIR:-/tmp}/gpu env test.XXXXXX")
trap 'rm -rf "$T"' EXIT

# venv VERSION DIR [trt]: a fake onnxruntime-gpu venv (and TensorRT's pip libraries with trt)
venv() {
    capi="$2/lib/python3.12/site-packages/onnxruntime/capi"
    mkdir -p "$capi"
    : > "$capi/libonnxruntime_providers_cuda.so"
    : > "$capi/libonnxruntime.so.$1"
    if [ "${3:-}" = trt ]; then
        mkdir -p "$2/lib/python3.12/site-packages/tensorrt_libs"
        : > "$2/lib/python3.12/site-packages/tensorrt_libs/libnvinfer.so.10"
    fi
}
venv 1.30.0 "$T/old venv"
venv 1.31.1 "$T/good venv" trt
mkdir -p "$T/bundle" "$T/data"

# case NAME EXPECTED-ORT EXPECTED-STATE-PATTERN [VAR=VALUE ...]: source gpu-env.sh in a clean shell and compare
fails=0
case_() {
    name=$1 want_ort=$2 want_state=$3
    shift 3
    got=$(env -i PATH="$PATH" HOME="$T" XDG_DATA_HOME="$T/data" HERE="$T/bundle" ORT_CPU="$T/bundle/libonnxruntime.so.1.31.0" "$@" \
        sh -c '. "$0"; printf "%s\n%s\n" "$ORT_DYLIB_PATH" "$GPU_STATE"' "$ENV_SH")
    ort=$(printf '%s\n' "$got" | sed -n 1p)
    state=$(printf '%s\n' "$got" | sed -n 2p)
    case "$state" in
        $want_state) ok=1 ;;
        *) ok=0 ;;
    esac
    if [ "$ort" = "$want_ort" ] && [ $ok = 1 ]; then
        echo "ok   $name"
    else
        echo "FAIL $name: ORT_DYLIB_PATH=$ort GPU_STATE=$state"
        fails=$((fails + 1))
    fi
}
GOOD="$T/good venv/lib/python3.12/site-packages"
case_ "a venv with another onnxruntime version is passed over" "$GOOD/onnxruntime/capi/libonnxruntime.so.1.31.1" \
    "GPU (onnxruntime 1.31.1, TensorRT from $GOOD/tensorrt_libs)" COMPCLASS_GPU_VENV="$T/old venv" VIRTUAL_ENV="$T/good venv"
case_ "only a wrong version: the CPU library, saying why" "$T/bundle/libonnxruntime.so.1.31.0" "CPU (*1.30.0*1.31*)" \
    COMPCLASS_GPU_VENV="$T/old venv"
# bundle VERSION DIR: a bundle folder carrying onnxruntime-gpu VERSION beside the CPU library
bundle() {
    mkdir -p "$2"
    : > "$2/libonnxruntime_providers_cuda.so"
    : > "$2/libonnxruntime.so.1.31.0"
    : > "$2/libonnxruntime.so.$1"
}
bundle 1.30.0 "$T/old bundle"
bundle 1.31.1 "$T/good bundle"
case_ "a bundled onnxruntime-gpu of another version is passed over" "$T/old bundle/libonnxruntime.so.1.31.0" \
    "CPU (*bundle folder*1.30.0*1.31*)" HERE="$T/old bundle" ORT_CPU="$T/old bundle/libonnxruntime.so.1.31.0"
case_ "a bundled onnxruntime-gpu of the right version is used" "$T/good bundle/libonnxruntime.so.1.31.1" "GPU*" \
    HERE="$T/good bundle" ORT_CPU="$T/good bundle/libonnxruntime.so.1.31.0"
exit $fails
