#!/usr/bin/env bash
# A new desktop AI app from what HEAD holds: the inference crate, the window with its Classify tab, the command line
# and the packaging, under a new identity, without the classifier's parts. docs/TEMPLATE.md is the procedure.
#   scripts/new_app.sh DEST NAME "Title" PREFIX     e.g. scripts/new_app.sh ../hello-ai hello-ai "Hello AI" HELLOAI
set -euo pipefail
die() { echo "new_app.sh: $*" >&2; exit 64; }
[ $# = 4 ] || die 'usage: new_app.sh DEST NAME "Title" PREFIX'
DEST=$1 NAME=$2 TITLE=$3 PREFIX=$4 APP=apps/$2
[[ $NAME =~ ^[a-z][a-z0-9-]*$ && $NAME != gpui-demo ]] || die "NAME '$NAME': a-z, 0-9 and -, from a letter"
[[ $PREFIX =~ ^[A-Z][A-Z0-9_]*$ ]] || die "PREFIX '$PREFIX': A-Z, 0-9 and _, from a letter"
[[ -n $TITLE && $TITLE != *[\"\\\$\`]* ]] || die "Title '$TITLE': no quotes, backslashes, \$ or backticks"
[ ! -e "$DEST" ] || die "$DEST exists"
SRC=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
mkdir -p "$DEST"
git -C "$SRC" archive HEAD Cargo.toml Cargo.lock rustfmt.toml crates/inference apps/gpui-demo docs/TEMPLATE.md |
    tar -x -C "$DEST"
cd "$DEST"
mv apps/gpui-demo "$APP"
sed -i "s|\"apps/gpui-demo\"|\"$APP\"|" Cargo.toml
printf '%s\n' /target/ /dist/ '*.onnx' '*.onnx.data' > .gitignore

cd "$APP"
rm -r demo_stack README.md scripts/{golden,parity,build_demo_stack}.py packaging/licenses/DATASET.txt src/demo.rs \
    src/cli/preprocess.rs src/gui/demo_tab.rs src/gui/demo_tab src/gui/bench.rs src/gui/bench src/gui/stack.rs
sed -i '/^mod demo;$/d' src/main.rs
sed -i '/^pub mod \(bench\|demo_tab\|stack\);$/d' src/gui.rs
sed -i '/^pub mod preprocess;$/d' src/cli.rs
sed -i -e "s/^name = \"gpui-demo\"$/name = \"$NAME\"/" -e '/^rayon = /d' Cargo.toml
# packaging/bundle.txt keeps its header and ships the whole exports/ folder
sed -i -n '/^[^#]/q;p' packaging/bundle.txt && echo exports >> packaging/bundle.txt
cat > packaging/app.env <<EOF
# The app's identity for packaging: sourced by run.sh, run-nixos.sh and scripts/package_linux.sh (POSIX sh).
# Keep it in step with src/app.rs (NAME, TITLE, ENV_PREFIX) and the binary name in Cargo.toml.
APP_NAME=$NAME
APP_TITLE="$TITLE"
ENV_PREFIX=$PREFIX
BIN=$NAME
EOF
cat > packaging/README.txt <<EOF
$TITLE: portable Linux build (x86_64, glibc 2.28 or newer)
  ./run.sh (./run-nixos.sh on NixOS)  the window;  ./run.sh classify picture.jpg  the answer without one;  ./run.sh --help
No window? Try WAYLAND_DISPLAY= ./run.sh, or LIBGL_ALWAYS_SOFTWARE=1 ./run.sh. sha256sum -c SHA256SUMS checks every file.

GPU: the bundle runs on the CPU. For an NVIDIA GPU (a driver for CUDA 13), install onnxruntime-gpu once, without root:
  uv venv ~/.local/share/onnxruntime-gpu/1.31 --python 3.12 && uv pip install --python ~/.local/share/onnxruntime-gpu/1.31/bin/python \\
      'onnxruntime-gpu[cuda,cudnn]==1.31.1' 'tensorrt-cu13-libs==10.16.1.11'
run.sh then uses it and says so on its first line. ${PREFIX}_GPU=0 ./run.sh forces the CPU.
EOF
cat > src/app.rs <<EOF
//! The app's identity, its tabs and its subcommands. Made by new_app.sh; docs/TEMPLATE.md says what to change.

use gpui_kit::App;

use crate::cli::{self, Command};
use crate::gui::classify::ClassifyTab;
use crate::gui::{Lucide, TabSpec};
use inference::manifest::Task;
use inference::registry::Head;

/// The per-user cache is \`\$XDG_CACHE_HOME/<NAME>\` (or \`~/.cache/<NAME>\`).
pub const NAME: &str = "$NAME";
pub const TITLE: &str = "$TITLE";
/// What the app does, after its title in \`--help\`.
pub const ABOUT: &str = "classify pictures";
/// Environment variables are \`<ENV_PREFIX>_MODELS\`, and \`_GPU\` and \`_GPU_VENV\` in packaging/gpu-env.sh.
pub const ENV_PREFIX: &str = "$PREFIX";
/// The binary's name: the Wayland app id, and what \`--help\` and the log lines call the app.
pub const BIN: &str = env!("CARGO_BIN_NAME");
/// The title bar's icon.
pub const ICON: Lucide = Lucide::Microchip;
/// The \`label\` of an abstaining head in \`classify\` output.
pub const UNSURE: &str = "UNSURE";

/// The window's tabs, in order; Ctrl+1 to Ctrl+9 show them. A tab is a module with a \`gui::Tab\` impl and a line here.
pub const TABS: &[TabSpec] = &[TabSpec::of::<ClassifyTab>()];

/// The subcommands, in \`--help\`'s order. A subcommand is a module with a \`cli::Command\` and a line here.
pub const COMMANDS: &[Command] = &[cli::classify::COMMAND, cli::bench::COMMAND, cli::models::COMMAND];

/// The title bar's note on the selected head, after the app's title.
pub fn subtitle(head: &Head) -> Option<String> {
    let Task::Classify(c) = &head.spec.task;
    Some(format!("{} classes", c.labels.len()))
}

/// What the tabs share besides the models.
pub struct Shared {}

pub fn shared(_: &mut App) -> Shared {
    Shared {}
}
EOF
cd ../..
cargo update --workspace --offline --quiet
echo "made $DEST ($APP) from $(git -C "$SRC" rev-parse --short HEAD); next steps: $DEST/docs/TEMPLATE.md"
