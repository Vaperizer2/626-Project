#!/usr/bin/env bash
# new_app.sh's output builds and passes its tests. Run it after a change to new_app.sh or to a file it copies.
# The app is made from HEAD in target/check-new-app/app and built in target/check-new-app/target. It needs a target
# dir of its own: its crates/inference has the same name and relative path as this repo's, cargo names a path
# crate's artifacts by that relative path, and in a shared target dir one copy's artifacts would stand in for the
# other's. The first run builds every dependency; later runs rebuild only the app and its inference copy.
# The tests that load onnxruntime need ORT_DYLIB_PATH, as in this repo.
#   scripts/check_new_app.sh
set -euo pipefail
SRC=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
W=$SRC/target/check-new-app
rm -rf "$W/app"
"$SRC/scripts/new_app.sh" "$W/app" check-app "Check App" CHECKAPP
cd "$W/app"
CARGO_TARGET_DIR=$W/target cargo test --locked
