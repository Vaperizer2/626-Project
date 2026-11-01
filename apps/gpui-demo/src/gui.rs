//! The desktop window, built with GPUI Kit (`gpui-kit`: GPUI + the Kit's components and themes).
//!
//! Layout: Kit `TitleBar` (app name, theme toggle) · tab row (`TabBar`: Classify / Demo / Benchmark, plus the
//! model picker) · the active tab · Kit `StatusBar` (model / onnxruntime state, shortcuts as `Kbd`).
//!
//! One view (`DemoApp`) owns all state; each tab's state and rendering lives in its own module
//! (`classify`, `demo_tab`, `bench`). Inference, model loading, image reading and decoding all run in the
//! inference service (`crate::inference`); results come back over channels that GPUI tasks await.

mod bench;
mod classify;
mod demo_tab;
mod theme;
mod images;
mod results;
mod widgets;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use futures::channel::mpsc as fmpsc;
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
    App, Bounds, Context, FocusHandle, FontWeight, KeyBinding, RenderImage, SharedString, TestSupportExt as _, Window,
    WindowBounds, WindowOptions, actions, div, prelude::*, px, size,
};

use crate::inference::{GraphState, Inference, Lane, LoadStage, Loaded, Policy, Priority, ServiceEvent};
use crate::manifest::{GraphId, HeadId, Task};
use crate::registry::{Head, Registry};
use crate::runtime::{ModelOptions, OrtLoader};
use widgets::Lucide;

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

/// What the UI shows for one head: its graph's state, or why it cannot be used at all.
#[derive(Clone, Debug)]
pub enum ModelState {
    Unavailable(String),
    Idle,
    Loading { since: Instant, stage: LoadStage },
    Ready(Loaded),
    Failed(String),
}

impl ModelState {
    pub fn usable(&self) -> bool {
        !matches!(self, ModelState::Unavailable(_) | ModelState::Failed(_))
    }
}

/// Where the selected model came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pick {
    /// the start-up probe has not picked the default yet: the fallback is shown, nothing runs on it
    Probing,
    Default,
    /// the user picked; the start-up default no longer applies
    User,
}

pub struct DemoApp {
    focus: FocusHandle,
    tab: TabKind,
    inference: Inference,
    /// Classify submissions and single Demo classifications
    interactive: Lane,
    /// Demo "Run all" and the benchmark
    bulk: Lane,
    /// selected head for Classify and Demo (none only when no model is available)
    model: Option<HeadId>,
    pick: Pick,
    /// `--autorun`, waiting for the start-up default
    autorun: bool,
    graphs: HashMap<GraphId, GraphState>,
    /// the preferred default heads passed over at start-up, and why
    passed_over: Vec<(HeadId, String)>,
    /// why there is no model at all (the index could not be read, or none of its manifests parsed)
    registry_error: Option<String>,
    classify: classify::ClassifyState,
    demo: demo_tab::DemoState,
    bench: bench::BenchState,
    /// textures evicted from the recent list, released on the next frame
    to_drop: Vec<std::sync::Arc<RenderImage>>,
    _subscriptions: Vec<gpui_kit::Subscription>,
}

impl DemoApp {
    pub fn new(
        (inference, mut events): (Inference, fmpsc::UnboundedReceiver<ServiceEvent>),
        demo_dir: Option<PathBuf>,
        initial: Vec<String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
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
        // until the start-up probe picks the default, the fallback (the last default) is selected; nothing loads yet
        let reg = inference.registry();
        let model = reg.default_order().last().cloned().or_else(|| reg.first_default());
        let registry_error = (reg.heads().next().is_none()).then(|| reg.problems().join("\n"));
        let mut this = Self {
            focus: cx.focus_handle(),
            tab: TabKind::Classify,
            interactive: inference.lane(Priority::Interactive, Policy::Keep),
            bulk: inference.lane(Priority::Bulk, Policy::Keep),
            inference,
            model,
            pick: Pick::Probing,
            autorun: false,
            graphs: HashMap::new(),
            passed_over: Vec::new(),
            registry_error,
            classify,
            demo: demo_tab::DemoState::default(),
            bench: bench::BenchState::default(),
            to_drop: Vec::new(),
            _subscriptions: subs,
        };
        this.demo_init(demo_dir, cx);
        for s in initial {
            if let Some(src) = crate::fetch::Source::parse(&s) {
                this.submit_source(src, cx);
            }
        }
        this
    }

    fn registry(&self) -> &Arc<Registry> {
        self.inference.registry()
    }

    fn is_busy(&self) -> bool {
        self.graphs.values().any(|s| matches!(s, GraphState::Loading { .. }))
            || self.classify.busy()
            || self.demo.running()
            || self.bench.running()
    }

    fn on_event(&mut self, ev: ServiceEvent, cx: &mut Context<Self>) {
        match ev {
            ServiceEvent::Graph(g, st) => {
                self.graphs.insert(g, st);
            }
            ServiceEvent::Default { head, passed_over } => {
                if self.pick != Pick::User {
                    // with no head that can run, the fallback stays selected and says why it cannot answer
                    if let Some(head) = head {
                        self.inference.preload(&head);
                        self.model = Some(head);
                    }
                    self.pick = Pick::Default;
                }
                self.passed_over = passed_over;
                self.follow_model(cx);
                if std::mem::take(&mut self.autorun) {
                    self.run_all(cx);
                }
            }
        }
        cx.notify();
    }

    fn state(&self, k: &HeadId) -> ModelState {
        let Some(h) = self.head(k) else { return ModelState::Unavailable("unknown model".into()) };
        if let Err(e) = h.available() {
            return ModelState::Unavailable(e.to_string());
        }
        match self.graphs.get(&h.manifest().id) {
            None => ModelState::Idle,
            Some(GraphState::Loading { since, stage }) => ModelState::Loading { since: *since, stage: *stage },
            Some(GraphState::Ready(l)) => ModelState::Ready(l.clone()),
            Some(GraphState::Failed(e)) => ModelState::Failed(e.clone()),
        }
    }

    fn model_usable(&self, k: &HeadId) -> bool {
        self.state(k).usable()
    }

    /// Every head in UI order (manifest order, then head order).
    fn head_ids(&self) -> Vec<HeadId> {
        self.registry().ids()
    }

    fn head(&self, id: &HeadId) -> Option<Head<'_>> {
        self.registry().head(id)
    }

    fn title(&self, id: &HeadId) -> String {
        self.head(id).map(|h| h.spec.title.clone()).unwrap_or_else(|| id.to_string())
    }

    fn short(&self, id: &HeadId) -> String {
        self.head(id).map(|h| h.spec.short.clone()).unwrap_or_else(|| id.to_string())
    }

    fn select_model(&mut self, k: HeadId, window: &mut Window, cx: &mut Context<Self>) {
        let state = self.state(&k);
        if !state.usable() {
            let why = match state {
                ModelState::Unavailable(e) | ModelState::Failed(e) => e,
                _ => String::new(),
            };
            use gpui_kit::component::WindowExt as _;
            window.push_notification(
                (
                    gpui_kit::component::notification::NotificationType::Warning,
                    SharedString::from(format!("{} is unavailable. {why}", self.title(&k))),
                ),
                cx,
            );
            return;
        }
        if matches!(state, ModelState::Idle) {
            self.inference.preload(&k);
        }
        self.model = Some(k);
        self.pick = Pick::User;
        self.follow_model(cx);
        cx.notify();
    }

    /// The model to run: the selected one, once the start-up default is known or the user picked.
    fn run_model(&self) -> Option<HeadId> {
        if self.pick == Pick::Probing { None } else { self.model.clone() }
    }

    /// The input on screen gets the selected model's answer.
    fn follow_model(&mut self, cx: &mut Context<Self>) {
        match self.tab {
            TabKind::Classify => self.classify_follow_model(cx),
            TabKind::Demo => self.demo_follow_model(cx),
            TabKind::Bench => {}
        }
    }

    fn next_model(&mut self, _: &NextModel, window: &mut Window, cx: &mut Context<Self>) {
        let ids = self.head_ids();
        let i = ids.iter().position(|k| Some(k) == self.model.as_ref()).unwrap_or(0);
        for step in 1..=ids.len() {
            let k = &ids[(i + step) % ids.len()];
            if self.model_usable(k) {
                return self.select_model(k.clone(), window, cx);
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
                    .children(self.model.as_ref().and_then(|m| self.head(m)).map(|h| {
                        let Task::Classify(c) = &h.spec.task;
                        div()
                            .text_xs()
                            .text_color(t.muted_foreground)
                            .child(format!("· {} electronic-component classes", c.labels.len()))
                    })),
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
            .children(self.head_ids().into_iter().map(|k| {
                let usable = self.model_usable(&k);
                let name = self.title(&k);
                let tip = match self.state(&k) {
                    ModelState::Unavailable(e) => format!("{name} — unavailable: {e}"),
                    ModelState::Failed(e) => format!("{name} — failed to load: {e}"),
                    ModelState::Ready(l) => {
                        let skipped: String =
                            l.placement.skipped.iter().map(|(ep, why)| format!("\n{} skipped: {why}", ep.name())).collect();
                        let checked = if l.verified {
                            "checksums verified"
                        } else {
                            "checksums not verified since the files changed (gpui-demo models --check)"
                        };
                        format!("{name} — on {}, loaded in {}, {checked}{skipped}", l.placement, widgets::duration(l.ms))
                    }
                    ModelState::Loading { stage: LoadStage::BuildingEngine, .. } => {
                        format!("{name} — building its TensorRT engine (first run only)")
                    }
                    ModelState::Loading { .. } => format!("{name} — loading…"),
                    ModelState::Idle => format!("{name} — loads on first use (Ctrl+M cycles models)"),
                };
                let selected = self.model.as_ref() == Some(&k);
                Button::new(SharedString::from(format!("model-{k}")))
                    .label(self.short(&k))
                    .selected(selected)
                    // outline + primary: the chosen model reads as an accent tint, not a grey press state
                    .when(selected, |b| b.primary())
                    .disabled(!usable)
                    .tooltip(tip)
            }))
            .on_click(cx.listener(|this, sel: &Vec<usize>, w, cx| {
                if let Some(k) = sel.first().and_then(|&i| this.head_ids().get(i).cloned()) {
                    this.select_model(k, w, cx);
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
                            .children(self.model.as_ref().map(|m| widgets::model_state_chip(&self.state(m), cx)))
                    })
                    .when(self.tab == TabKind::Bench, |d| {
                        d.child(div().text_xs().text_color(t.muted_foreground).child("Runs every available model"))
                    }),
            )
    }

    fn render_status_bar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let mut left = Vec::new();
        if let Some(m) = &self.model {
            let short = self.short(m);
            match self.state(m) {
                ModelState::Ready(l) => {
                    let skipped: Vec<&str> = l.placement.skipped.iter().map(|(ep, _)| ep.name()).collect();
                    left.push(if skipped.is_empty() {
                        format!("{short} on {}", l.placement)
                    } else {
                        format!("{short} on {} ({} unavailable)", l.placement, skipped.join(", "))
                    });
                    left.push(l.runtime);
                }
                ModelState::Loading { since, stage: LoadStage::BuildingEngine } => left.push(format!(
                    "{short}: building the TensorRT engine, first run only ({:.0} s)",
                    since.elapsed().as_secs_f64()
                )),
                ModelState::Loading { .. } => left.push(format!("{short} loading…")),
                ModelState::Idle => left.push(format!("{short} loads on first use")),
                ModelState::Failed(_) => left.push(format!("{short} failed to load")),
                ModelState::Unavailable(_) => left.push(format!("{short} unavailable")),
            }
        }
        left.push(format!("jpeg: {}", crate::preprocess::JPEG_DECODER));
        let left = left.join(" · ");
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
        // laid out here rather than in the Kit's left/right slots, which shrink alike and clip the shortcuts
        StatusBar::new().child(
            h_flex()
                .w_full()
                .gap_4()
                .justify_between()
                .child(div().flex_1().min_w_0().truncate().text_color(t.muted_foreground).child(left))
                .child(
                    h_flex()
                        .id("status-hints")
                        .flex_none()
                        .gap_4()
                        .children(hints)
                        .child(widgets::hint(&["ctrl-m"], "Model", cx))
                        .child(widgets::hint(&["ctrl-t"], "Theme", cx)),
                ),
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
                .child(Alert::error("registry-error", SharedString::from(e)).title("Model index not found").small())
        });
        let fallback_alert = (self.pick == Pick::Default && !self.passed_over.is_empty()).then(|| {
            let using = self.model.as_ref().filter(|m| self.model_usable(m)).map(|m| self.title(m));
            let why: Vec<String> = self
                .passed_over
                .iter()
                .map(|(h, why)| {
                    let why = why.trim_end_matches('.');
                    if using.is_some() {
                        format!("{} is not the default here: {why}.", self.title(h))
                    } else {
                        format!("{}: {why}.", self.title(h))
                    }
                })
                .collect();
            let title = using.map_or_else(|| "No model can run here".to_string(), |u| format!("Using {u}"));
            div().id("fallback-notice").test_support().px_4().pt_3().child(
                Alert::warning("default-fallback", SharedString::from(why.join(" "))).title(title).small(),
            )
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
            .on_action(cx.listener(|this, _: &Cancel, w, cx| this.cancel(w, cx)))
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
            .children(fallback_alert)
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

    /// Esc steps back one level: out of a text field first, then it stops the visible tab's run.
    fn cancel(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.focus.is_focused(window) {
            return window.focus(&self.focus, cx);
        }
        match self.tab {
            TabKind::Demo => self.demo_cancel(cx),
            TabKind::Bench => self.bench_cancel(cx),
            TabKind::Classify => {}
        }
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
    ]
}

/// The model index (an empty one when it cannot be read, with the reason) and the service over it, backed by
/// onnxruntime. Reading the index is a few small files; nothing is loaded until a model is used.
fn service(opts: &ModelOptions) -> (Inference, fmpsc::UnboundedReceiver<ServiceEvent>) {
    let registry = match Registry::locate(opts.models.as_deref()).and_then(|i| Registry::load(&i)) {
        Ok(r) => r,
        Err(e) => Registry::empty(e.to_string()),
    };
    let exports = registry.index_path().parent().map(PathBuf::from);
    Inference::spawn(Arc::new(registry), OrtLoader::new(opts, exports.as_deref()))
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
        theme::register_bundled_fonts(cx);
        theme::install(cx);
        cx.bind_keys(keymap());
        let service = service(&opts);
        let inference = service.0.clone();
        let quit = move |cx: &mut App| {
            // release every session before exit handlers run (see Inference::shutdown)
            if !inference.shutdown(Duration::from_secs(3)) {
                eprintln!("[gpui-demo] a model is still loading; quitting without waiting for it");
                crate::runtime::exit_now(0);
            }
            cx.quit();
        };
        let on_quit = quit.clone();
        cx.on_action(move |_: &Quit, cx: &mut App| on_quit(cx));
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
                let mut app = DemoApp::new(service, demo_dir, initial, window, cx);
                app.tab = gui.tab;
                app.autorun = gui.autorun;
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
                quit(cx);
                return;
            }
        }
        cx.on_window_closed(move |cx, _| quit(cx)).detach();
    });
}

#[cfg(test)]
mod tests;
