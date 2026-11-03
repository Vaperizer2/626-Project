#!/usr/bin/env bash
# new_app.sh's output builds and passes its tests. Run it after a change to new_app.sh or to a file it copies.
# The app is made from HEAD in a temporary folder and built in this repo's target dir, so only its own crate compiles.
# The tests that load onnxruntime need ORT_DYLIB_PATH, as in this repo.
#   scripts/check_new_app.sh
set -euo pipefail
SRC=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
T=$(mktemp -d "${TMPDIR:-/tmp}/check-new-app.XXXXXX")
trap 'rm -rf "$T"' EXIT
"$SRC/scripts/new_app.sh" "$T/app" check-app "Check App" CHECKAPP
cd "$T/app"
CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$SRC/target} cargo test --locked
