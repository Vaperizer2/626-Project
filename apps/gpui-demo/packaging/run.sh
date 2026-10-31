#!/bin/sh
# Component Classifier: launcher for the portable Linux bundle.
# Everything (binary, onnxruntime, models, demo pictures) is found relative to this script, so the folder can
# live anywhere and be started from any directory. All arguments are passed through to the app:
#   ./run.sh                          open the window
#   ./run.sh --tab demo --dark        start on the Demo tab, dark theme
#   ./run.sh classify picture.jpg     headless: print the prediction (no display needed)
#   ./run.sh --help                   all options
set -eu
HERE=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
BIN="$HERE/gpui-demo"

# Explicit paths (an environment variable already set by the caller wins).
export ORT_DYLIB_PATH="${ORT_DYLIB_PATH:-$HERE/libonnxruntime.so.1.31.0}"
export COMPCLASS_MODELS="${COMPCLASS_MODELS:-$HERE/exports}"
export COMPCLASS_DEMO_DIR="${COMPCLASS_DEMO_DIR:-$HERE/demo_stack}"

if [ "$(uname -m)" != "x86_64" ]; then
    echo "run.sh: this bundle is built for x86_64 Linux; this machine is $(uname -m)." >&2
    exit 1
fi
if [ ! -x "$BIN" ]; then
    echo "run.sh: $BIN is missing or not executable (was the archive fully extracted?)." >&2
    exit 1
fi

case "${1:-}" in
    classify | preprocess | models | -h | --help | -V | --version)
        exec "$BIN" "$@" ;;
esac

if [ -z "${WAYLAND_DISPLAY:-}" ] && [ -z "${DISPLAY:-}" ]; then
    cat >&2 <<'MSG'
run.sh: no graphical session found (neither WAYLAND_DISPLAY nor DISPLAY is set).
  * Start it from a terminal inside your desktop session (not over plain SSH).
  * Over SSH without a display you can still test the models:
        ./run.sh classify demo_stack/01_battery_holder.jpg
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
