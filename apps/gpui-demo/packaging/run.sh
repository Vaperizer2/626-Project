#!/bin/sh
# Launcher for the portable Linux bundle; the app's name and variables come from app.env.
# Everything (binary, onnxruntime, models, demo pictures) is found relative to this script, so the folder can
# live anywhere and be started from any directory. All arguments are passed through to the app:
#   ./run.sh                          open the window
#   ./run.sh --tab classify --dark    start on the Classify tab, dark theme
#   ./run.sh classify picture.jpg     headless: print the prediction (no display needed)
#   ./run.sh --help                   all options
set -eu
HERE=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/app.env"
BIN="$HERE/$BIN"

# onnxruntime: an installed onnxruntime-gpu (CUDA, and TensorRT for the fast model) when there is one, else the
# bundled CPU library. gpu-env.sh says which in one line.
ORT_CPU="$HERE/libonnxruntime.so.1.31.0"
. "$HERE/gpu-env.sh"

# Explicit paths (an environment variable already set by the caller wins).
models=$(app_env MODELS)
demo_dir=$(app_env DEMO_DIR)
export "${ENV_PREFIX}_MODELS=${models:-$HERE/exports}" "${ENV_PREFIX}_DEMO_DIR=${demo_dir:-$HERE/demo_stack}"
if [ -n "$GPU_LIBPATH" ]; then
    export LD_LIBRARY_PATH="$GPU_LIBPATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
echo "run.sh: onnxruntime on $GPU_STATE" >&2

if [ "$(uname -m)" != "x86_64" ]; then
    echo "run.sh: this bundle is built for x86_64 Linux; this machine is $(uname -m)." >&2
    exit 1
fi
if [ ! -x "$BIN" ]; then
    echo "run.sh: $BIN is missing or not executable (was the archive fully extracted?)." >&2
    exit 1
fi

if [ -n "${1:-}" ]; then
    case " $COMMANDS -h --help -V --version " in *" $1 "*) exec "$BIN" "$@" ;; esac
fi

if [ -z "${WAYLAND_DISPLAY:-}" ] && [ -z "${DISPLAY:-}" ]; then
    cat >&2 <<'MSG'
run.sh: no graphical session found (neither WAYLAND_DISPLAY nor DISPLAY is set).
  * Start it from a terminal inside your desktop session (not over plain SSH).
  * Over SSH without a display you can still test the models:
        ./run.sh classify picture.jpg
MSG
    exit 1
fi

"$BIN" "$@" || {
    status=$?
    if [ -n "${WAYLAND_DISPLAY:-}" ]; then
        echo "run.sh: the app exited with status $status. If the window did not open on Wayland, try X11 instead:" >&2
        echo "        WAYLAND_DISPLAY= $0 $*" >&2
    fi
    echo "run.sh: see README.txt, section Troubleshooting." >&2
    exit "$status"
}
