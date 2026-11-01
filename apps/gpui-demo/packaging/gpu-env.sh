# Find the GPU runtime for the Component Classifier; sourced by run.sh and run-nixos.sh (POSIX sh, set -eu safe).
#
# The bundle ships a CPU onnxruntime. The GPU runtime (onnxruntime-gpu 1.31 with its CUDA 13 + cuDNN 9 libraries,
# and TensorRT 10 for the fast model) is a separate download of several GB, usually a Python venv:
#     uv venv ~/.local/share/component-classifier/gpu --python 3.12
#     uv pip install --python ~/.local/share/component-classifier/gpu/bin/python \
#         'onnxruntime-gpu[cuda,cudnn]==1.31.1' 'tensorrt-cu13-libs==10.16.1.11'
# It is looked for in $COMPCLASS_GPU_VENV, that default location, an active $VIRTUAL_ENV, then the bundle folder.
# Found: ORT_DYLIB_PATH points at its libonnxruntime and GPU_LIBPATH lists the directories the loader needs. Not
# found: the bundled CPU library is used. GPU_STATE says which, and why, in one line.
# COMPCLASS_GPU=0 skips the search; an ORT_DYLIB_PATH you set yourself is kept.
#
# In: HERE (bundle folder), ORT_CPU (the bundled CPU library). Out: ORT_DYLIB_PATH, GPU_LIBPATH, GPU_STATE.

GPU_LIBPATH=""
GPU_STATE=""
# the onnxruntime series the app and its CPU library are built and checked against
GPU_ORT_SERIES=1.31
# why the last venv with an onnxruntime-gpu was passed over
gpu_why=""

# gpu_series_ok LIB WHERE: LIB (libonnxruntime.so.X.Y.Z) is of the expected series; else say why WHERE was passed over.
gpu_series_ok() {
    case "${1##*.so.}" in
        "$GPU_ORT_SERIES" | "$GPU_ORT_SERIES".*) return 0 ;;
    esac
    gpu_why="$2 has onnxruntime-gpu ${1##*.so.}, the app needs $GPU_ORT_SERIES.x"
    return 1
}

# gpu_try VENV: use VENV's onnxruntime-gpu (and the pip CUDA, cuDNN and TensorRT libraries beside it) if it has one
# of the expected version.
gpu_try() {
    [ -n "$1" ] || return 1
    gpu_sp=""
    for p in "$1"/lib/python3.*/site-packages; do
        if [ -d "$p/onnxruntime/capi" ]; then gpu_sp="$p"; fi
    done
    [ -n "$gpu_sp" ] || return 1
    [ -f "$gpu_sp/onnxruntime/capi/libonnxruntime_providers_cuda.so" ] || return 1
    gpu_ort=""
    for lib in "$gpu_sp"/onnxruntime/capi/libonnxruntime.so.*; do
        if [ -f "$lib" ]; then gpu_ort="$lib"; fi
    done
    [ -n "$gpu_ort" ] || return 1
    if ! gpu_series_ok "$gpu_ort" "$1"; then
        gpu_ort=""
        return 1
    fi
    GPU_LIBPATH="$gpu_sp/onnxruntime/capi"
    for d in "$gpu_sp"/nvidia/*/lib "$gpu_sp"/tensorrt_libs; do
        if [ -d "$d" ]; then GPU_LIBPATH="$GPU_LIBPATH:$d"; fi
    done
    return 0
}

gpu_find() {
    if [ "${COMPCLASS_GPU:-1}" = 0 ]; then
        GPU_STATE="CPU (COMPCLASS_GPU=0)"
        return 0
    fi
    if ! ls /proc/driver/nvidia/gpus/* >/dev/null 2>&1; then
        GPU_STATE="CPU (no NVIDIA GPU or driver found)"
        return 0
    fi
    gpu_ort=""
    if gpu_try "${COMPCLASS_GPU_VENV:-}" || gpu_try "${XDG_DATA_HOME:-$HOME/.local/share}/component-classifier/gpu" \
        || gpu_try "${VIRTUAL_ENV:-}"; then
        :
    elif [ -f "$HERE/libonnxruntime_providers_cuda.so" ]; then
        for lib in "$HERE"/libonnxruntime.so.*; do
            if [ -f "$lib" ] && [ "$lib" != "$ORT_CPU" ]; then gpu_ort="$lib"; fi
        done
        if [ -n "$gpu_ort" ] && gpu_series_ok "$gpu_ort" "the bundle folder"; then
            GPU_LIBPATH="$HERE"
        else
            gpu_ort=""
        fi
    fi
    if [ -z "$gpu_ort" ] && [ -n "$gpu_why" ]; then
        GPU_STATE="CPU ($gpu_why: see README.txt, GPU)"
        return 0
    fi
    if [ -z "$gpu_ort" ] && [ -n "${COMPCLASS_GPU_VENV:-}" ]; then
        GPU_STATE="CPU (COMPCLASS_GPU_VENV=$COMPCLASS_GPU_VENV holds no onnxruntime-gpu: see README.txt, GPU)"
        return 0
    fi
    if [ -z "$gpu_ort" ]; then
        GPU_STATE="CPU (no onnxruntime-gpu found: install one as README.txt says under GPU, or set COMPCLASS_GPU_VENV)"
        return 0
    fi
    ORT_DYLIB_PATH="$gpu_ort"
    gpu_trt=""
    # split on ':' only, so directories with spaces stay whole
    gpu_rest="$GPU_LIBPATH:/usr/lib/x86_64-linux-gnu:/usr/lib:/opt/tensorrt/lib"
    while [ -z "$gpu_trt" ] && [ -n "$gpu_rest" ]; do
        d=${gpu_rest%%:*}
        case "$gpu_rest" in *:*) gpu_rest=${gpu_rest#*:} ;; *) gpu_rest="" ;; esac
        if [ -f "$d/libnvinfer.so.10" ]; then gpu_trt="$d"; fi
    done
    if [ -n "$gpu_trt" ]; then
        case ":$GPU_LIBPATH:" in *":$gpu_trt:"*) ;; *) GPU_LIBPATH="$GPU_LIBPATH:$gpu_trt" ;; esac
        GPU_STATE="GPU (onnxruntime ${gpu_ort##*.so.}, TensorRT from $gpu_trt)"
    else
        GPU_STATE="GPU without TensorRT (no libnvinfer.so.10: Fast is not the default, Deployed runs on CUDA; README.txt, GPU)"
    fi
    return 0
}

if [ -n "${ORT_DYLIB_PATH:-}" ]; then
    GPU_STATE="onnxruntime as set by ORT_DYLIB_PATH ($ORT_DYLIB_PATH)"
else
    gpu_find
    ORT_DYLIB_PATH="${ORT_DYLIB_PATH:-$ORT_CPU}"
fi
export ORT_DYLIB_PATH
