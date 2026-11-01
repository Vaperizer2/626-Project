//! The desktop window, built with GPUI Kit (`gpui-kit`: GPUI + the Kit's components and themes).
//!
//! Layout: Kit `TitleBar` (app name, theme toggle) · tab row (`TabBar` with the tabs of `app::TABS`, plus the model
//! picker) · the active tab · Kit `StatusBar` (model / onnxruntime state, shortcuts as `Kbd`).
//!
//! `shell::Shell` is the root view and draws the chrome. Every tab is an entity of its own (`shell::Tab`) with its
//! own focus, keys and lanes. The models the tabs share (the service, load states, the selected head) are one
//! entity, `models::Models`. Inference, model loading, image reading and decoding all run in the inference service
//! (`inference::service`); results come back over channels that GPUI tasks await.

pub mod bench;
pub mod classify;
pub mod demo_tab;
mod images;
pub mod models;
mod results;
mod shell;
pub mod stack;
mod theme;
pub mod widgets;

pub use shell::{Hint, Tab, TabSpec, keys_help, tab_index};
pub use widgets::Lucide;

use std::sync::Arc;
use std::time::Duration;

use futures::channel::mpsc as fmpsc;
use gpui_kit::component::TitleBar;
use gpui_kit::{
    Action, App, AppContext as _, Bounds, Context, KeyBinding, WindowBounds, WindowOptions, actions, px, size,
};

use inference::registry::Registry;
use inference::runtime::{OrtLoader, Settings};
use inference::service::{Inference, ServiceEvent};
use shell::Shell;

actions!(
    app,
    [Open, Paste, FocusUrl, ToggleTheme, Quit, NextModel, StepPrev, StepNext, Activate, RunAll, Cancel]
);

/// Ctrl+1 to Ctrl+9: show the tab at this index.
#[derive(Clone, PartialEq, Action)]
#[action(namespace = app, no_json)]
pub struct ShowTab(pub usize);

/// Key context of the root view; arrows, j/k and Enter only fire when no text field has focus.
const CTX: &str = "Shell";
const CTX_NAV: &str = "Shell && !Input";

/// Redraw `cx`'s view every 250 ms while `busy` holds, for the elapsed-time counters.
fn tick_while<T: 'static>(cx: &mut Context<T>, busy: impl Fn(&T, &App) -> bool + 'static) {
    cx.spawn(async move |this, cx| {
        loop {
            cx.background_executor().timer(Duration::from_millis(250)).await;
            let Ok(busy) = this.update(cx, |this, cx| busy(this, cx)) else { break };
            if busy {
                this.update(cx, |_, cx| cx.notify()).ok();
            }
        }
    })
    .detach();
}

/// Window start-up options (`--tab`, `--autorun`, `--dark`/`--light`).
#[derive(Debug, Clone, Default)]
pub struct GuiOptions {
    /// index into `app::TABS`
    pub tab: usize,
    pub autorun: bool,
    /// Some(true) = dark, Some(false) = light, None = follow the system
    pub theme: Option<bool>,
}

pub fn keymap() -> Vec<KeyBinding> {
    let mut keys = vec![
        KeyBinding::new("ctrl-o", Open, Some(CTX)),
        KeyBinding::new("ctrl-v", Paste, Some(CTX_NAV)),
        KeyBinding::new("ctrl-l", FocusUrl, Some(CTX)),
        KeyBinding::new("ctrl-t", ToggleTheme, Some(CTX)),
        KeyBinding::new("ctrl-m", NextModel, Some(CTX)),
        KeyBinding::new("ctrl-r", RunAll, Some(CTX)),
        // also while a text field has focus: the field handles Esc first, then it reaches Cancel
        KeyBinding::new("escape", Cancel, Some(CTX)),
        KeyBinding::new("left", StepPrev, Some(CTX_NAV)),
        KeyBinding::new("right", StepNext, Some(CTX_NAV)),
        KeyBinding::new("up", StepPrev, Some(CTX_NAV)),
        KeyBinding::new("down", StepNext, Some(CTX_NAV)),
        KeyBinding::new("k", StepPrev, Some(CTX_NAV)),
        KeyBinding::new("j", StepNext, Some(CTX_NAV)),
        KeyBinding::new("enter", Activate, Some(CTX_NAV)),
        KeyBinding::new("ctrl-q", Quit, None),
        KeyBinding::new("cmd-o", Open, Some(CTX)),
        KeyBinding::new("cmd-v", Paste, Some(CTX_NAV)),
        KeyBinding::new("cmd-q", Quit, None),
    ];
    keys.extend((1..=9).map(|n| KeyBinding::new(&format!("ctrl-{n}"), ShowTab(n - 1), Some(CTX))));
    keys
}

/// The model index (an empty one when it cannot be read, with the reason) and the service over it, backed by
/// onnxruntime. Reading the index is a few small files; nothing is loaded until a model is used.
fn service(opts: &Settings) -> (Inference, fmpsc::UnboundedReceiver<ServiceEvent>) {
    let registry = Arc::new(Registry::open(opts).unwrap_or_else(|e| Registry::empty(e.to_string())));
    Inference::spawn(registry.clone(), OrtLoader::new(opts, &registry))
}

/// Open the window. `initial` are image paths / URLs for the tab that takes pictures.
pub fn run(opts: Settings, gui: GuiOptions, initial: Vec<String>) {
    let unset = |v: &str| std::env::var_os(v).is_none_or(|s| s.is_empty());
    if cfg!(target_os = "linux") && unset("DISPLAY") && unset("WAYLAND_DISPLAY") {
        eprintln!(
            "{bin}: no display (neither WAYLAND_DISPLAY nor DISPLAY is set). Run it from a desktop session, \
             or use `{bin} classify <images>` for headless output.",
            bin = crate::app::BIN
        );
        std::process::exit(1);
    }
    gpui_kit::application().with_assets(widgets::AppAssets).run(move |cx: &mut App| {
        gpui_kit::init(cx);
        theme::register_bundled_fonts(cx);
        theme::install(cx);
        cx.bind_keys(keymap());
        let service = service(&opts);
        let inference = service.0.clone();
        let quit = move |cx: &mut App| {
            // release every session before exit handlers run (see Inference::shutdown)
            inference.shutdown_or_exit(Duration::from_secs(3), 0, || {
                log!("a model is still loading; quitting without waiting for it")
            });
            cx.quit();
        };
        let on_quit = quit.clone();
        cx.on_action(move |_: &Quit, cx: &mut App| on_quit(cx));
        let bounds = Bounds::centered(None, size(px(1360.), px(860.)), cx);
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            app_id: Some(crate::app::BIN.into()),
            window_min_size: Some(size(px(1024.), px(640.))),
            ..TitleBar::window_options()
        };
        let opened = gpui_kit::open_window(options, cx, |window, cx| {
            theme::apply_initial(gui.theme, window, cx);
            window.set_window_title(crate::app::TITLE);
            cx.new(|cx| Shell::new(service, &gui, initial, window, cx))
        });
        match opened {
            Ok((handle, view)) => {
                handle
                    .update(cx, |_, window, cx| {
                        let focus = view.read(cx).active_focus();
                        window.focus(&focus, cx);
                    })
                    .ok();
                cx.activate(true);
            }
            Err(e) => {
                eprintln!(
                    "{}: failed to open the window: {e:#} (see README troubleshooting: Wayland/X11/Vulkan)",
                    crate::app::BIN
                );
                quit(cx);
                return;
            }
        }
        cx.on_window_closed(move |cx, _| quit(cx)).detach();
    });
}

#[cfg(test)]
mod tests;
