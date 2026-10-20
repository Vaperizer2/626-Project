//! The desktop window, built with GPUI Kit (`gpui-kit`: GPUI + the Kit's components and themes).
//!
//! Layout: Kit `TitleBar` (app name, theme toggle) · tab row (`TabBar`: Classify / Demo / Benchmark, plus the
//! model picker) · the active tab · Kit `StatusBar` (model / onnxruntime state, shortcuts as `Kbd`).
//!
//! One view (`DemoApp`) owns all state; each tab's state and rendering lives in its own module
//! (`classify`, `demo_tab`, `bench`). Inference, model loading, image reading and decoding all run on
//! background threads (`worker`); results come back over channels that GPUI tasks await.

mod bench;
mod classify;
mod demo_tab;
mod theme;
mod widgets;
mod worker;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use futures::StreamExt;
use gpui_kit::component::{
    ActiveTheme as _, Icon, IconName, Sizable as _, Theme, ThemeMode, TitleBar,
    alert::Alert,
    button::{Button, ButtonGroup, ButtonVariants as _},
    h_flex,
    status_bar::StatusBar,
    tab::{Tab, TabBar},
};
use gpui_kit::component::{Disableable as _, Selectable as _};
use gpui_kit::{
    App, Bounds, Context, FocusHandle, FontWeight, KeyBinding, RenderImage, SharedString, Window, WindowBounds,
    WindowOptions, actions, div, prelude::*, px, size,
};

use crate::model::ModelOptions;
use crate::models::ModelKey;
use widgets::Lucide;
use worker::{Cmd, Event, ModelState};

actions!(
    gpui_demo,
    [
        Open,
        Paste,
        FocusUrl,
        ToggleTheme,
        Quit,
        ShowClassify,
        ShowDemo,
        ShowBench,
        NextModel,
        StepPrev,
        StepNext,
        Activate,
        RunAll,
        Cancel
    ]
);

/// Key context of the root view; demo keys (arrows, j/k, Enter) only fire when no text field has focus.
const CTX: &str = "GpuiDemo";
const CTX_NAV: &str = "GpuiDemo && !Input";

/// Window start-up options (`--tab`, `--autorun`, `--dark`/`--light`).
#[derive(Debug, Clone, Default)]
pub struct GuiOptions {
    pub tab: TabKind,
    pub autorun: bool,
    /// Some(true) = dark, Some(false) = light, None = follow the system
    pub theme: Option<bool>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TabKind {
    #[default]
    Classify,
    Demo,
    Bench,
}

impl TabKind {
    const ALL: [TabKind; 3] = [TabKind::Classify, TabKind::Demo, TabKind::Bench];
    fn index(self) -> usize {
        Self::ALL.iter().position(|t| *t == self).unwrap_or(0)
    }
}

pub struct DemoApp {
    focus: FocusHandle,
    tab: TabKind,
    jobs: mpsc::Sender<Cmd>,
    /// selected model for Classify and Demo
    model: ModelKey,
    states: HashMap<ModelKey, ModelState>,
    /// registry could not be built at all (no exports/final)
    registry_error: Option<String>,
    labels: Vec<String>,
    /// deployed backbone (preprocess.json `model`)
    backbone: Option<String>,
    runtime: Option<(String, String)>,
    classify: classify::ClassifyState,
    demo: demo_tab::DemoState,
    bench: bench::BenchState,
    /// textures evicted from the recent list, released on the next frame
    to_drop: Vec<std::sync::Arc<RenderImage>>,
    _subscriptions: Vec<gpui_kit::Subscription>,
}

impl DemoApp {
    pub fn new(
        opts: ModelOptions,
        demo_dir: Option<PathBuf>,
        initial: Vec<String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let (jobs, mut events) = worker::spawn(opts);
        // Model registry + load-state events from the inference thread.
        cx.spawn(async move |this, cx| {
            while let Some(ev) = events.next().await {
                if this.update(cx, |this, cx| this.on_event(ev, cx)).is_err() {
                    break;
                }
            }
        })
        .detach();
        // Tick while something is loading or running (elapsed-time counters, spinners).
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(Duration::from_millis(250)).await;
                let Ok(busy) = this.update(cx, |this, _| this.is_busy()) else { break };
                if busy {
                    this.update(cx, |_, cx| cx.notify()).ok();
                }
            }
        })
        .detach();
        let (classify, subs) = classify::ClassifyState::new(window, cx);
        let mut this = Self {
            focus: cx.focus_handle(),
            tab: TabKind::Classify,
            jobs,
            model: ModelKey::Deployed,
            states: HashMap::new(),
            registry_error: None,
            labels: Vec::new(),
            backbone: None,
            runtime: None,
            classify,
            demo: demo_tab::DemoState::default(),
            bench: bench::BenchState::default(),
            to_drop: Vec::new(),
            _subscriptions: subs,
        };
        // Warm up the default model while the window appears.
        let _ = this.jobs.send(Cmd::Preload(ModelKey::Deployed));
        this.demo_init(demo_dir, cx);
        for s in initial {
            if let Some(src) = crate::fetch::Source::parse(&s) {
                this.submit_source(src, cx);
            }
        }
        this
    }

    fn is_busy(&self) -> bool {
        self.states.values().any(|s| matches!(s, ModelState::Loading(_)))
            || self.classify.busy()
            || self.demo.running()
            || self.bench.running()
    }

    fn on_event(&mut self, ev: Event, cx: &mut Context<Self>) {
        match ev {
            Event::Registry(Ok(reg)) => {
                self.labels = reg.labels;
                self.backbone = reg.backbone;
                for (k, r) in reg.models {
                    let st = match r {
                        Ok(_) => self.states.get(&k).cloned().unwrap_or(ModelState::Idle),
                        Err(e) => ModelState::Unavailable(e),
                    };
                    self.states.insert(k, st);
                }
            }
            Event::Registry(Err(e)) => {
                self.registry_error = Some(e.clone());
                for k in ModelKey::ALL {
                    self.states.insert(k, ModelState::Unavailable(e.clone()));
                }
            }
            Event::Model(k, st) => {
                self.states.insert(k, st);
            }
            Event::Runtime(lib, provider) => self.runtime = Some((lib, provider)),
        }
        cx.notify();
    }

    fn state(&self, k: ModelKey) -> Option<&ModelState> {
        self.states.get(&k)
    }

    fn model_usable(&self, k: ModelKey) -> bool {
        self.state(k).is_none_or(ModelState::usable)
    }

    fn select_model(&mut self, k: ModelKey, window: &mut Window, cx: &mut Context<Self>) {
        if !self.model_usable(k) {
            let why = match self.state(k) {
                Some(ModelState::Unavailable(e)) | Some(ModelState::Failed(e)) => e.clone(),
                _ => String::new(),
            };
            use gpui_kit::component::WindowExt as _;
            window.push_notification(
                (
                    gpui_kit::component::notification::NotificationType::Warning,
                    SharedString::from(format!("{} is unavailable. {why}", k.name())),
                ),
                cx,
            );
            return;
        }
        self.model = k;
        if !matches!(self.state(k), Some(ModelState::Ready(_)) | Some(ModelState::Loading(_))) {
            let _ = self.jobs.send(Cmd::Preload(k));
        }
        cx.notify();
    }

    fn next_model(&mut self, _: &NextModel, window: &mut Window, cx: &mut Context<Self>) {
        let i = ModelKey::ALL.iter().position(|k| *k == self.model).unwrap_or(0);
        for step in 1..=ModelKey::ALL.len() {
            let k = ModelKey::ALL[(i + step) % ModelKey::ALL.len()];
            if self.model_usable(k) {
                return self.select_model(k, window, cx);
            }
        }
    }

    fn set_tab(&mut self, tab: TabKind, window: &mut Window, cx: &mut Context<Self>) {
        self.tab = tab;
        window.focus(&self.focus, cx);
        cx.notify();
    }

    fn toggle_theme(&mut self, _: &ToggleTheme, window: &mut Window, cx: &mut Context<Self>) {
        let mode = if cx.theme().is_dark() { ThemeMode::Light } else { ThemeMode::Dark };
        Theme::change(mode, Some(window), cx);
        cx.notify();
    }

    fn threshold(&self) -> f32 {
        0.5
    }

    // ------------------------------------------------------------------------------------------ chrome
    fn render_title_bar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let dark = t.is_dark();
        TitleBar::new()
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .size(px(22.))
                            .flex()
                            .items_center()
                            .justify_center()
                            .rounded(t.radius)
                            .bg(t.primary)
                            .text_color(t.primary_foreground)
                            .child(Icon::new(Lucide::Microchip).small()),
                    )
                    .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child("Component Classifier"))
                    .child(div().text_xs().text_color(t.muted_foreground).child("· 20 electronic-component classes")),
            )
            .child(
                h_flex().pr_2().child(
                    Button::new("theme")
                        .ghost()
                        .small()
                        .icon(if dark { IconName::Sun } else { IconName::Moon })
                        .tooltip(if dark { "Light theme (Ctrl+T)" } else { "Dark theme (Ctrl+T)" })
                        .on_click(cx.listener(|this, _, w, cx| this.toggle_theme(&ToggleTheme, w, cx))),
                ),
            )
    }

    fn render_tab_row(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let models = ButtonGroup::new("model-picker")
            .small()
            .outline()
            .children(ModelKey::ALL.iter().map(|&k| {
                let usable = self.model_usable(k);
                let tip = match self.state(k) {
                    Some(ModelState::Unavailable(e)) => format!("{} — unavailable: {e}", k.name()),
                    Some(ModelState::Failed(e)) => format!("{} — failed to load: {e}", k.name()),
                    Some(ModelState::Ready(ms)) => format!("{} — loaded in {:.1} s", k.name(), ms / 1000.0),
                    Some(ModelState::Loading(_)) => format!("{} — loading…", k.name()),
                    _ => format!("{} — loads on first use (Ctrl+M cycles models)", k.name()),
                };
                Button::new(SharedString::from(format!("model-{}", k.id())))
                    .label(k.short())
                    .selected(self.model == k)
                    // outline + primary: the chosen model reads as an accent tint, not a grey press state
                    .when(self.model == k, |b| b.primary())
                    .disabled(!usable)
                    .tooltip(tip)
            }))
            .on_click(cx.listener(|this, sel: &Vec<usize>, w, cx| {
                if let Some(&i) = sel.first() {
                    this.select_model(ModelKey::ALL[i], w, cx);
                }
            }));
        h_flex()
            .justify_between()
            .items_center()
            .gap_4()
            .px_4()
            .bg(t.tab_bar)
            .border_b_1()
            .border_color(t.border)
            .child(
                TabBar::new("tabs")
                    .underline()
                    .selected_index(self.tab.index())
                    .on_click(cx.listener(|this, ix: &usize, w, cx| this.set_tab(TabKind::ALL[*ix], w, cx)))
                    .child(Tab::new().prefix(Icon::new(Lucide::ScanSearch).small()).label("Classify"))
                    .child(Tab::new().prefix(Icon::new(Lucide::LayoutGrid).small()).label("Demo"))
                    .child(Tab::new().prefix(Icon::new(Lucide::ChartNoAxesColumn).small()).label("Benchmark")),
            )
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .when(self.tab != TabKind::Bench, |d| {
                        d.child(div().text_xs().text_color(t.muted_foreground).child("Model"))
                            .child(models)
                            .child(widgets::model_state_chip(self.state(self.model), cx))
                    })
                    .when(self.tab == TabKind::Bench, |d| {
                        d.child(div().text_xs().text_color(t.muted_foreground).child("Runs every available model"))
                    }),
            )
    }

    fn render_status_bar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let mut left = match &self.runtime {
            Some((lib, provider)) => {
                let name =
                    std::path::Path::new(lib).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                format!("onnxruntime {name} · {provider} · jpeg: {}", crate::preprocess::JPEG_DECODER)
            }
            None => "onnxruntime not loaded yet".to_string(),
        };
        if let Some(b) = &self.backbone {
            left = format!("deployed backbone {b} · {left}");
        }
        let hints: Vec<gpui_kit::AnyElement> = match self.tab {
            TabKind::Classify => vec![
                widgets::hint(&["ctrl-o"], "Open", cx).into_any_element(),
                widgets::hint(&["ctrl-v"], "Paste", cx).into_any_element(),
                widgets::hint(&["ctrl-l"], "URL", cx).into_any_element(),
                widgets::hint(&["up", "down"], "Recent", cx).into_any_element(),
            ],
            TabKind::Demo => vec![
                widgets::hint(&["left", "right"], "Step", cx).into_any_element(),
                widgets::hint(&["enter"], "Classify", cx).into_any_element(),
                widgets::hint(&["ctrl-r"], "Run all", cx).into_any_element(),
                widgets::hint(&["escape"], "Stop", cx).into_any_element(),
            ],
            TabKind::Bench => vec![
                widgets::hint(&["ctrl-r"], "Run", cx).into_any_element(),
                widgets::hint(&["escape"], "Stop", cx).into_any_element(),
            ],
        };
        StatusBar::new().left(div().text_xs().text_color(t.muted_foreground).child(left)).right(
            h_flex().gap_4().children(hints).child(widgets::hint(&["ctrl-m"], "Model", cx)).child(widgets::hint(
                &["ctrl-t"],
                "Theme",
                cx,
            )),
        )
    }
}

impl Render for DemoApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        for im in self.to_drop.drain(..) {
            window.drop_image(im).ok();
        }
        let t = cx.theme();
        let (bg, fg, drop_border) = (theme::roles(cx).canvas, t.foreground, t.drag_border);
        let body = prof!(
            "gui::build_tab_elements",
            match self.tab {
                TabKind::Classify => self.render_classify(window, cx).into_any_element(),
                TabKind::Demo => self.render_demo(window, cx).into_any_element(),
                TabKind::Bench => self.render_bench(cx).into_any_element(),
            }
        );
        let registry_alert = self.registry_error.clone().map(|e| {
            div()
                .px_4()
                .pt_3()
                .child(Alert::error("registry-error", SharedString::from(e)).title("Model folder not found").small())
        });
        div()
            .id("root")
            .track_focus(&self.focus)
            .key_context(CTX)
            .on_action(cx.listener(Self::open))
            .on_action(cx.listener(Self::paste))
            .on_action(cx.listener(Self::focus_url))
            .on_action(cx.listener(Self::toggle_theme))
            .on_action(cx.listener(Self::next_model))
            .on_action(cx.listener(|this, _: &ShowClassify, w, cx| this.set_tab(TabKind::Classify, w, cx)))
            .on_action(cx.listener(|this, _: &ShowDemo, w, cx| this.set_tab(TabKind::Demo, w, cx)))
            .on_action(cx.listener(|this, _: &ShowBench, w, cx| this.set_tab(TabKind::Bench, w, cx)))
            .on_action(cx.listener(|this, _: &StepPrev, w, cx| this.step(-1, w, cx)))
            .on_action(cx.listener(|this, _: &StepNext, w, cx| this.step(1, w, cx)))
            .on_action(cx.listener(|this, _: &Activate, _, cx| this.activate(cx)))
            .on_action(cx.listener(|this, _: &RunAll, _, cx| this.run_all(cx)))
            .on_action(cx.listener(|this, _: &Cancel, _, cx| this.cancel(cx)))
            .on_mouse_down(
                gpui_kit::MouseButton::Left,
                cx.listener(|this, _, window, cx| {
                    if !this.focus.contains_focused(window, cx) {
                        window.focus(&this.focus, cx);
                    }
                }),
            )
            .on_drop(cx.listener(|this, paths: &gpui_kit::ExternalPaths, w, cx| {
                this.set_tab(TabKind::Classify, w, cx);
                for p in paths.paths() {
                    this.submit_source(crate::fetch::Source::File(p.clone()), cx);
                }
            }))
            .drag_over::<gpui_kit::ExternalPaths>(move |style, _, _, _| style.border_2().border_color(drop_border))
            .size_full()
            .flex()
            .flex_col()
            .bg(bg)
            .text_color(fg)
            .child(self.render_title_bar(cx))
            .child(self.render_tab_row(cx))
            .children(registry_alert)
            .child(div().flex_1().min_h_0().child(body))
            .child(self.render_status_bar(cx))
    }
}

impl DemoApp {
    /// Left/Right, j/k, Up/Down: per tab.
    fn step(&mut self, delta: isize, window: &mut Window, cx: &mut Context<Self>) {
        match self.tab {
            TabKind::Classify => self.classify_step(delta, cx),
            TabKind::Demo => self.demo_step(delta, window, cx),
            TabKind::Bench => {}
        }
    }

    fn activate(&mut self, cx: &mut Context<Self>) {
        if self.tab == TabKind::Demo {
            self.demo_classify_selected(cx);
        }
    }

    fn run_all(&mut self, cx: &mut Context<Self>) {
        match self.tab {
            TabKind::Demo => self.demo_run_all(cx),
            TabKind::Bench => self.bench_start(cx),
            TabKind::Classify => {}
        }
    }

    fn cancel(&mut self, cx: &mut Context<Self>) {
        self.demo_cancel(cx);
        self.bench_cancel(cx);
    }
}

pub fn keymap() -> Vec<KeyBinding> {
    vec![
        KeyBinding::new("ctrl-o", Open, Some(CTX)),
        KeyBinding::new("ctrl-v", Paste, Some(CTX_NAV)),
        KeyBinding::new("ctrl-l", FocusUrl, Some(CTX)),
        KeyBinding::new("ctrl-t", ToggleTheme, Some(CTX)),
        KeyBinding::new("ctrl-m", NextModel, Some(CTX)),
        KeyBinding::new("ctrl-1", ShowClassify, Some(CTX)),
        KeyBinding::new("ctrl-2", ShowDemo, Some(CTX)),
        KeyBinding::new("ctrl-3", ShowBench, Some(CTX)),
        KeyBinding::new("ctrl-r", RunAll, Some(CTX)),
        KeyBinding::new("escape", Cancel, Some(CTX_NAV)),
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
    ]
}

/// Open the window. `initial` are image paths / URLs to classify straight away.
pub fn run(opts: ModelOptions, demo_dir: Option<PathBuf>, gui: GuiOptions, initial: Vec<String>) {
    let unset = |v: &str| std::env::var_os(v).is_none_or(|s| s.is_empty());
    if cfg!(target_os = "linux") && unset("DISPLAY") && unset("WAYLAND_DISPLAY") {
        eprintln!(
            "gpui-demo: no display (neither WAYLAND_DISPLAY nor DISPLAY is set). Run it from a desktop session, \
             or use `gpui-demo classify <images>` for headless output."
        );
        std::process::exit(1);
    }
    gpui_kit::application().with_assets(widgets::AppAssets).run(move |cx: &mut App| {
        gpui_kit::init(cx);
        theme::install(cx);
        cx.bind_keys(keymap());
        cx.on_action(|_: &Quit, cx: &mut App| cx.quit());
        let bounds = Bounds::centered(None, size(px(1360.), px(860.)), cx);
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            app_id: Some("gpui-demo".into()),
            window_min_size: Some(size(px(1024.), px(640.))),
            ..TitleBar::window_options()
        };
        let opened = gpui_kit::open_window(options, cx, |window, cx| {
            theme::apply_initial(gui.theme, window, cx);
            window.set_window_title("Component Classifier");
            cx.new(|cx| {
                let mut app = DemoApp::new(opts, demo_dir, initial, window, cx);
                app.tab = gui.tab;
                if gui.autorun {
                    app.run_all(cx);
                }
                app
            })
        });
        match opened {
            Ok((handle, view)) => {
                handle
                    .update(cx, |_, window, cx| {
                        let focus = view.read(cx).focus.clone();
                        window.focus(&focus, cx);
                    })
                    .ok();
                cx.activate(true);
            }
            Err(e) => {
                eprintln!(
                    "gpui-demo: failed to open the window: {e:#} (see README troubleshooting: Wayland/X11/Vulkan)"
                );
                cx.quit();
            }
        }
        cx.on_window_closed(|cx, _| cx.quit()).detach();
    });
}

#[cfg(test)]
mod tests;
