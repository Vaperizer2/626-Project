//! The app's identity, its tabs and its subcommands. An app made from this one changes this file, its package name
//! and `packaging/`.

use gpui_kit::{App, AppContext as _, Entity};

use crate::cli::{self, Command};
use crate::gui::bench::BenchTab;
use crate::gui::classify::ClassifyTab;
use crate::gui::demo_tab::DemoTab;
use crate::gui::stack::Stack;
use crate::gui::{GuiOptions, Lucide, TabSpec};

/// The per-user cache is `$XDG_CACHE_HOME/<NAME>` (or `~/.cache/<NAME>`).
pub const NAME: &str = "component-classifier";
pub const TITLE: &str = "Component Classifier";
/// What the app does, after its title in `--help`.
pub const ABOUT: &str = "classify electronic components in pictures";
/// Environment variables are `<ENV_PREFIX>_MODELS` and `<ENV_PREFIX>_DEMO_DIR`.
pub const ENV_PREFIX: &str = "COMPCLASS";
/// The binary's name: the Wayland app id, and what `--help` and the log lines call the app.
pub const BIN: &str = env!("CARGO_BIN_NAME");
/// The title bar's icon.
pub const ICON: Lucide = Lucide::Microchip;
/// What the title bar calls the selected head's labels, after their count.
pub const LABELS: &str = "electronic-component classes";
/// The `label` of an abstaining head in `classify` output (the Python demo's spelling).
pub const UNSURE: &str = "UNSURE";

/// The window's tabs, in order; Ctrl+1 to Ctrl+9 show them. A tab is a module with a `gui::Tab` impl and a line here.
pub const TABS: &[TabSpec] = &[TabSpec::of::<ClassifyTab>(), TabSpec::of::<DemoTab>(), TabSpec::of::<BenchTab>()];

/// The subcommands, in `--help`'s order. A subcommand is a module with a `cli::Command` and a line here.
pub const COMMANDS: &[Command] =
    &[cli::classify::COMMAND, cli::preprocess::COMMAND, cli::bench::COMMAND, cli::models::COMMAND];

/// What the tabs share besides the models: the demo stack, for the Demo and Benchmark tabs.
pub struct Shared {
    pub stack: Entity<Stack>,
}

pub fn shared(gui: &GuiOptions, cx: &mut App) -> Shared {
    Shared { stack: cx.new(|cx| Stack::new(gui.demo_dir.clone(), cx)) }
}
