#!/bin/sh
# Checks packaging/gpu-env.sh and run.sh against fake GPU venvs under a path with spaces (run by `cargo test`,
# gpu_env tests).
#   sh scripts/test_gpu_env.sh     exit 0 when every case passes; skips without an NVIDIA driver (gpu-env.sh needs one)
set -eu
PKG="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)/packaging"
ENV_SH="$PKG/gpu-env.sh"
. "$PKG/app.env"
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
    got=$(env -i PATH="$PATH" HOME="$T" XDG_DATA_HOME="$T/data" HERE="$T/bundle" ORT_CPU="$T/bundle/libonnxruntime.so.1.31.0" \
        APP_ENV="$PKG/app.env" "$@" sh -c '. "$APP_ENV"; . "$0"; printf "%s\n%s\n" "$ORT_DYLIB_PATH" "$GPU_STATE"' "$ENV_SH")
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
    "GPU (onnxruntime 1.31.1, TensorRT from $GOOD/tensorrt_libs)" "${ENV_PREFIX}_GPU_VENV=$T/old venv" VIRTUAL_ENV="$T/good venv"
case_ "only a wrong version: the CPU library, saying why" "$T/bundle/libonnxruntime.so.1.31.0" "CPU (*1.30.0*1.31*)" \
    "${ENV_PREFIX}_GPU_VENV=$T/old venv"
case_ "${ENV_PREFIX}_GPU=0 forces the CPU" "$T/bundle/libonnxruntime.so.1.31.0" "CPU (${ENV_PREFIX}_GPU=0)" \
    "${ENV_PREFIX}_GPU=0" "${ENV_PREFIX}_GPU_VENV=$T/good venv"
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

# The search order: each place in turn holds a good runtime, and every place after it holds one too.
O="$T/order"
venv 1.31.1 "$O/named venv" trt
venv 1.31.1 "$O/data/onnxruntime-gpu/1.31" trt
venv 1.31.1 "$O/data/$APP_NAME/gpu" trt
venv 1.31.1 "$O/active venv" trt
bundle 1.31.1 "$O/bundle"
ort_in() { echo "$1/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.31.1"; }
order() {
    case_ "order: $1" "$2" "GPU*" "${ENV_PREFIX}_GPU_VENV=$O/named venv" XDG_DATA_HOME="$O/data" \
        VIRTUAL_ENV="$O/active venv" HERE="$O/bundle" ORT_CPU="$O/bundle/libonnxruntime.so.1.31.0"
}
order "\$${ENV_PREFIX}_GPU_VENV first" "$(ort_in "$O/named venv")"
rm -rf "$O/named venv"
order "then the shared onnxruntime-gpu/1.31" "$(ort_in "$O/data/onnxruntime-gpu/1.31")"
rm -rf "$O/data/onnxruntime-gpu"
order "then the app's own $APP_NAME/gpu" "$(ort_in "$O/data/$APP_NAME/gpu")"
rm -rf "$O/data/$APP_NAME"
order "then \$VIRTUAL_ENV" "$(ort_in "$O/active venv")"
rm -rf "$O/active venv"
order "then the bundle folder" "$O/bundle/libonnxruntime.so.1.31.1"
venv 1.31.1 "$T/.local/share/onnxruntime-gpu/1.31" trt
case_ "the shared runtime without XDG_DATA_HOME is under ~/.local/share" \
    "$(ort_in "$T/.local/share/onnxruntime-gpu/1.31")" "GPU*" XDG_DATA_HOME=

# Another app's identity: its variables and its folder are the ones read, not this app's.
ON=$APP_NAME-other OP=${ENV_PREFIX}_OTHER
printf '%s\n' "APP_NAME=$ON" 'APP_TITLE="Other"' "ENV_PREFIX=$OP" "BIN=$ON" > "$T/other.env"
venv 1.31.1 "$T/data/$ON/gpu" trt
case_ "another app: ${OP}_GPU_VENV is read, ${ENV_PREFIX}_GPU_VENV is not" "$(ort_in "$T/good venv")" "GPU*" \
    APP_ENV="$T/other.env" "${OP}_GPU_VENV=$T/good venv" "${ENV_PREFIX}_GPU_VENV=$T/old venv"
case_ "another app: its own older folder $ON/gpu" "$(ort_in "$T/data/$ON/gpu")" "GPU*" APP_ENV="$T/other.env"
case_ "another app: ${OP}_GPU=0" "$T/bundle/libonnxruntime.so.1.31.0" "CPU (${OP}_GPU=0)" \
    APP_ENV="$T/other.env" "${OP}_GPU=0"

# run.sh in a bundle of another app: it starts app.env's BIN with that app's variables pointing into the bundle.
B="$T/other bundle"
mkdir -p "$B"
cp "$PKG/run.sh" "$PKG/gpu-env.sh" "$B/" && cp "$T/other.env" "$B/app.env"
printf '%s\n' '#!/bin/sh' "printf '%s|%s|%s|%s\\n' \"\$1\" \"\$${OP}_MODELS\" \"\$${OP}_DEMO_DIR\" \"\${${ENV_PREFIX}_MODELS-unset}\"" \
    > "$B/$ON"
chmod +x "$B/$ON"
want="models|$B/exports|$B/demo_stack|unset"
got=$(env -i PATH="$PATH" HOME="$T" XDG_DATA_HOME="$T/data" "${OP}_GPU=0" sh "$B/run.sh" models 2>/dev/null)
if [ "$got" = "$want" ]; then echo "ok   run.sh starts BIN with the app's variables"; else
    echo "FAIL run.sh: got $got, want $want"; fails=$((fails + 1)); fi
got=$(env -i PATH="$PATH" HOME="$T" "${OP}_GPU=0" "${OP}_MODELS=$T/x" sh "$B/run.sh" models 2>/dev/null)
case "$got" in "models|$T/x|"*) echo "ok   run.sh keeps a variable the caller set" ;; *)
    echo "FAIL run.sh overrode ${OP}_MODELS: $got"; fails=$((fails + 1)) ;; esac
got=$(env -i PATH="$PATH" HOME="$T" "${OP}_GPU=0" sh "$B/run.sh" models 2>&1 >/dev/null)
case "$got" in "run.sh: onnxruntime on CPU (${OP}_GPU=0)") echo "ok   run.sh names the runtime" ;; *)
    echo "FAIL run.sh first line: $got"; fails=$((fails + 1)) ;; esac
exit $fails
