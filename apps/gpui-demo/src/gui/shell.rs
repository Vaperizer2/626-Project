//! The window's chrome and the tab protocol. `Shell` draws the title bar, the tab row with the model picker, the
//! registry and fallback alerts, the active tab and the status bar. It knows tabs only through the list it opens
//! (`app::TABS`): each is an entity implementing [`Tab`], with its own focus, its own keys and its own lanes. Pictures that reach the
//! window (dropped, opened, pasted, given on the command line) go to the tab that takes them.

use gpui_kit::component::{
    ActiveTheme as _, Icon, IconName, Sizable as _, Theme, ThemeMode, TitleBar, WindowExt as _,
    alert::Alert,
    button::{Button, ButtonGroup, ButtonVariants as _},
    h_flex,
    notification::NotificationType,
    status_bar::StatusBar,
    tab::{Tab as TabButton, TabBar},
};
use gpui_kit::component::{Disableable as _, Selectable as _};
use gpui_kit::{
    AnyView, App, ClipboardEntry, ClipboardItem, Context, Entity, ExternalPaths, FocusHandle, FontWeight,
    PathPromptOptions, SharedString, TestSupportExt as _, Window, div, prelude::*, px,
};

use super::models::{ModelEvent, ModelState, Models};
use super::widgets::{self, Lucide};
use super::{CTX, Cancel, FocusUrl, GuiOptions, NextModel, Open, Paste, ShowTab, ToggleTheme, theme};
use crate::app;
use inference::fetch::Source;
use inference::service::{ImageInput, Inference, LoadStage, ServiceEvent};

/// A shortcut the status bar shows: its keys and what they do.
pub type Hint = (&'static [&'static str], &'static str);

/// One tab of the window: a view under the tab row with its own focus, so it handles its own keys. Adding a tab is
/// a module with a `Tab` impl and a line in `app::TABS`.
pub trait Tab: Render + Sized {
    const TITLE: &'static str;
    const ICON: Lucide;
    /// The status bar's shortcuts while the tab is on screen; `--help` lists them too.
    const HINTS: &'static [Hint];
    /// The tab takes the window's pictures: dropped and opened files, pasted images, paths and URLs, and the ones on
    /// the command line, through [`Tab::take`].
    const TAKES_INPUTS: bool = false;
    /// Shown in the tab row instead of the model picker, by a tab that does not use the selected model.
    const MODEL_NOTE: Option<&'static str> = None;

    fn new(models: Entity<Models>, shared: &app::Shared, window: &mut Window, cx: &mut Context<Self>) -> Self;
    /// Has the keyboard while the tab is on screen.
    fn focus_handle(&self) -> FocusHandle;
    /// The tab came on screen, or the model to run changed while it was: the input on screen gets its answer.
    fn follow_model(&mut self, _cx: &mut Context<Self>) {}
    /// Start the tab's run, as its Ctrl+R does. `--autorun` calls it once the start-up default is known.
    fn run(&mut self, _cx: &mut Context<Self>) {}
    /// Pictures for a tab that sets `TAKES_INPUTS`.
    fn take(&mut self, _inputs: Vec<ImageInput>, _window: &mut Window, _cx: &mut Context<Self>) {}
    /// Focus the tab's own text input (a URL or path field), as Ctrl+L does from any tab, for a tab that sets
    /// `TAKES_INPUTS`.
    fn focus_input(&mut self, _window: &mut Window, _cx: &mut Context<Self>) {}
}

/// A tab before it exists: what `--help` and the shell need to know, and how to make it. `app::TABS` lists them.
pub struct TabSpec {
    pub title: &'static str,
    pub icon: Lucide,
    pub hints: &'static [Hint],
    pub takes_inputs: bool,
    pub model_note: Option<&'static str>,
    open: fn(Entity<Models>, &app::Shared, &mut Window, &mut App) -> (AnyView, FocusHandle),
    follow_model: fn(&AnyView, &mut App),
    run: fn(&AnyView, &mut App),
    take: fn(&AnyView, Vec<ImageInput>, &mut Window, &mut App),
    focus_input: fn(&AnyView, &mut Window, &mut App),
}

impl TabSpec {
    pub const fn of<T: Tab>() -> Self {
        TabSpec {
            title: T::TITLE,
            icon: T::ICON,
            hints: T::HINTS,
            takes_inputs: T::TAKES_INPUTS,
            model_note: T::MODEL_NOTE,
            open: open::<T>,
            follow_model: follow_model::<T>,
            run: run::<T>,
            take: take::<T>,
            focus_input: focus_input::<T>,
        }
    }
}

fn open<T: Tab>(
    models: Entity<Models>,
    shared: &app::Shared,
    window: &mut Window,
    cx: &mut App,
) -> (AnyView, FocusHandle) {
    let view = cx.new(|cx| T::new(models, shared, window, cx));
    let focus = view.read(cx).focus_handle();
    (view.into(), focus)
}

/// `view` was made by `open::<T>`.
fn typed<T: Tab>(view: &AnyView) -> Entity<T> {
    view.clone().downcast::<T>().unwrap_or_else(|_| unreachable!("a tab's view has its spec's type"))
}

fn follow_model<T: Tab>(view: &AnyView, cx: &mut App) {
    typed::<T>(view).update(cx, |t, cx| t.follow_model(cx));
}

fn run<T: Tab>(view: &AnyView, cx: &mut App) {
    typed::<T>(view).update(cx, |t, cx| t.run(cx));
}

fn take<T: Tab>(view: &AnyView, inputs: Vec<ImageInput>, window: &mut Window, cx: &mut App) {
    typed::<T>(view).update(cx, |t, cx| t.take(inputs, window, cx));
}

fn focus_input<T: Tab>(view: &AnyView, window: &mut Window, cx: &mut App) {
    typed::<T>(view).update(cx, |t, cx| t.focus_input(window, cx));
}

/// The index in `tabs` of the tab `--tab` names: a title, or the start of one, in any case.
pub fn tab_index(tabs: &[TabSpec], name: &str) -> Option<usize> {
    let name = name.to_lowercase();
    tabs.iter().position(|t| !name.is_empty() && t.title.to_lowercase().starts_with(&name))
}

/// The window's keys for `--help`: the shell's own, then each tab's hints.
pub fn keys_help(tabs: &[TabSpec]) -> String {
    let titles: Vec<&str> = tabs.iter().map(|t| t.title).collect();
    let show = if tabs.len() == 1 { "Ctrl+1".to_string() } else { format!("Ctrl+1..{}", tabs.len()) };
    let mut lines =
        vec![format!("{show} {} tab, Ctrl+M next model, Ctrl+T light/dark theme, Ctrl+Q quit,", titles.join(" / "))];
    lines.push("Ctrl+O open pictures, Ctrl+V paste a picture, path or URL, or drop files on the window.".into());
    for t in tabs.iter().filter(|t| !t.hints.is_empty()) {
        let hints: Vec<String> = t
            .hints
            .iter()
            .map(|(keys, label)| format!("{} {label}", keys.iter().map(|k| key_name(k)).collect::<Vec<_>>().join("/")))
            .collect();
        lines.push(format!("{}: {}", t.title, hints.join(", ")));
    }
    lines.join("\n               ")
}

/// "ctrl-o" as "Ctrl+O", "escape" as "Esc".
fn key_name(keys: &str) -> String {
    let part = |p: &str| match p {
        "escape" => "Esc".to_string(),
        p => p[..1].to_uppercase() + &p[1..],
    };
    keys.split('-').map(part).collect::<Vec<_>>().join("+")
}

/// One tab, open.
pub(super) struct TabEntry {
    spec: &'static TabSpec,
    pub(super) view: AnyView,
    pub(super) focus: FocusHandle,
}

pub struct Shell {
    pub(super) tabs: Vec<TabEntry>,
    active: usize,
    pub(super) models: Entity<Models>,
    /// `--autorun`, waiting for the start-up default
    autorun: bool,
    /// manifests that could not be read and defaults that name no model, beside the models that work; dismissible
    registry_problems: Vec<String>,
    _model_events: gpui_kit::Subscription,
}

impl Shell {
    pub fn new(
        service: (Inference, futures::channel::mpsc::UnboundedReceiver<ServiceEvent>),
        gui: &GuiOptions,
        initial: Vec<String>,
        tabs: &'static [TabSpec],
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        // the status bar and the model chip count a load's seconds
        super::tick_while(cx, |this, cx| this.models.read(cx).loading());
        let models = cx.new(|cx| Models::new(service, cx));
        let shared = app::shared(cx);
        let open: Vec<TabEntry> = tabs
            .iter()
            .map(|spec| {
                let (view, focus) = (spec.open)(models.clone(), &shared, window, cx);
                TabEntry { spec, view, focus }
            })
            .collect();
        let model_events = cx.subscribe(&models, |this, _, ev, cx| this.on_model_event(ev, cx));
        let m = models.read(cx);
        let registry_problems =
            if m.registry_error().is_some() { Vec::new() } else { m.registry().problems().to_vec() };
        let this = Self {
            tabs: open,
            active: gui.tab.min(tabs.len() - 1),
            models,
            autorun: gui.autorun,
            registry_problems,
            _model_events: model_events,
        };
        let inputs: Vec<ImageInput> = initial.iter().filter_map(|s| Source::parse(s)).map(ImageInput::Source).collect();
        if let (Some(ix), false) = (this.input_tab(), inputs.is_empty()) {
            let tab = &this.tabs[ix];
            (tab.spec.take)(&tab.view, inputs, window, cx);
        }
        this
    }

    fn on_model_event(&mut self, ev: &ModelEvent, cx: &mut Context<Self>) {
        let ModelEvent::Picked { startup } = ev else { return };
        self.follow_model(cx);
        if *startup && std::mem::take(&mut self.autorun) {
            let tab = &self.tabs[self.active];
            (tab.spec.run)(&tab.view, cx);
        }
        cx.notify();
    }

    fn follow_model(&mut self, cx: &mut Context<Self>) {
        let tab = &self.tabs[self.active];
        (tab.spec.follow_model)(&tab.view, cx);
    }

    /// The index of the tab on screen.
    #[cfg(test)]
    pub fn active_tab(&self) -> usize {
        self.active
    }

    /// Has the keyboard while the active tab is on screen.
    pub fn active_focus(&self) -> FocusHandle {
        self.tabs[self.active].focus.clone()
    }

    pub fn set_tab(&mut self, ix: usize, window: &mut Window, cx: &mut Context<Self>) {
        if ix >= self.tabs.len() {
            return;
        }
        self.active = ix;
        // the model may have changed on another tab
        self.follow_model(cx);
        window.focus(&self.active_focus(), cx);
        cx.notify();
    }

    /// Where the window's pictures go: the active tab when it takes them, else the first tab that does.
    fn input_tab(&self) -> Option<usize> {
        if self.tabs[self.active].spec.takes_inputs {
            return Some(self.active);
        }
        self.tabs.iter().position(|t| t.spec.takes_inputs)
    }

    /// Show the tab that takes pictures and hand it `inputs`.
    pub fn send_inputs(&mut self, inputs: Vec<ImageInput>, window: &mut Window, cx: &mut Context<Self>) {
        let Some(ix) = self.input_tab() else { return };
        self.set_tab(ix, window, cx);
        let tab = &self.tabs[ix];
        (tab.spec.take)(&tab.view, inputs, window, cx);
    }

    fn open(&mut self, _: &Open, window: &mut Window, cx: &mut Context<Self>) {
        let Some(ix) = self.input_tab() else { return };
        self.set_tab(ix, window, cx);
        let rx = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: true,
            prompt: Some(self.tabs[ix].spec.title.into()),
        });
        cx.spawn_in(window, async move |this, cx| {
            let result = rx.await;
            this.update_in(cx, |this, window, cx| match result {
                Ok(Ok(Some(paths))) => {
                    let inputs = paths.into_iter().map(|p| ImageInput::Source(Source::File(p))).collect();
                    this.send_inputs(inputs, window, cx);
                }
                Ok(Ok(None)) => {}
                Ok(Err(e)) => window.push_notification(
                    (
                        NotificationType::Warning,
                        SharedString::from(format!(
                            "File dialog unavailable ({e}). It needs xdg-desktop-portal; drag files onto the window \
                             or paste a path with Ctrl+V instead."
                        )),
                    ),
                    cx,
                ),
                Err(_) => window.push_notification("The file dialog closed unexpectedly.", cx),
            })
            .ok();
        })
        .detach();
    }

    fn focus_url(&mut self, _: &FocusUrl, window: &mut Window, cx: &mut Context<Self>) {
        let Some(ix) = self.input_tab() else { return };
        self.set_tab(ix, window, cx);
        let tab = &self.tabs[ix];
        (tab.spec.focus_input)(&tab.view, window, cx);
    }

    fn paste(&mut self, _: &Paste, window: &mut Window, cx: &mut Context<Self>) {
        let Some(item) = cx.read_from_clipboard() else {
            window.push_notification("The clipboard is empty.", cx);
            return;
        };
        let inputs = clipboard_inputs(&item);
        if inputs.is_empty() {
            window.push_notification("The clipboard has no URL, path or image.", cx);
        }
        self.send_inputs(inputs, window, cx);
    }

    fn next_model(&mut self, _: &NextModel, window: &mut Window, cx: &mut Context<Self>) {
        self.models.update(cx, |m, cx| m.next_model(window, cx));
    }

    pub fn dismiss_problems(&mut self, cx: &mut Context<Self>) {
        self.registry_problems.clear();
        cx.notify();
    }

    fn toggle_theme(&mut self, _: &ToggleTheme, window: &mut Window, cx: &mut Context<Self>) {
        let mode = if cx.theme().is_dark() { ThemeMode::Light } else { ThemeMode::Dark };
        Theme::change(mode, Some(window), cx);
        cx.notify();
    }

    /// Esc steps back one level: a text field gives the keyboard back to its tab first. Then the tab handles Esc.
    fn leave_text_field(&mut self, _: &Cancel, window: &mut Window, cx: &mut Context<Self>) {
        let focus = self.active_focus();
        if !focus.is_focused(window) {
            window.focus(&focus, cx);
            cx.stop_propagation();
        }
    }

    // ------------------------------------------------------------------------------------------ chrome
    fn render_title_bar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let dark = t.is_dark();
        let m = self.models.read(cx);
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
                            .child(Icon::new(app::ICON).small()),
                    )
                    .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(app::TITLE))
                    .children(
                        m.selected()
                            .and_then(|k| m.head(k))
                            .and_then(|h| app::subtitle(&h))
                            .map(|note| div().text_xs().text_color(t.muted_foreground).child(format!("· {note}"))),
                    ),
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
        let m = self.models.read(cx);
        let models = ButtonGroup::new("model-picker")
            .small()
            .outline()
            .children(m.head_ids().into_iter().map(|k| {
                let state = m.state(&k);
                let pickable = !matches!(state, ModelState::Unavailable(_));
                let name = m.title(&k);
                let tip = match state {
                    ModelState::Unavailable(e) => format!("{name} — unavailable: {e}"),
                    ModelState::Failed(e) => format!("{name} — failed to load: {e}. Pick it to try again."),
                    ModelState::Ready(l) => {
                        let skipped: String = l
                            .placement
                            .skipped
                            .iter()
                            .map(|(ep, why)| format!("\n{} skipped: {why}", ep.name()))
                            .collect();
                        let checked = if l.verified {
                            "checksums verified".to_string()
                        } else {
                            format!("checksums not verified since the files changed ({} models --check)", app::BIN)
                        };
                        format!(
                            "{name} — on {}, loaded in {}, {checked}{skipped}",
                            l.placement,
                            widgets::duration(l.ms)
                        )
                    }
                    ModelState::Loading { stage: LoadStage::BuildingEngine, .. } => {
                        format!("{name} — building its TensorRT engine (first run only)")
                    }
                    ModelState::Loading { .. } => format!("{name} — loading…"),
                    ModelState::Idle => format!("{name} — loads on first use (Ctrl+M cycles models)"),
                };
                let selected = m.selected() == Some(&k);
                Button::new(SharedString::from(format!("model-{k}")))
                    .label(m.short(&k))
                    .selected(selected)
                    // outline + primary: the chosen model reads as an accent tint, not a grey press state
                    .when(selected, |b| b.primary())
                    .disabled(!pickable)
                    .tooltip(tip)
            }))
            .on_click(cx.listener(|this, sel: &Vec<usize>, w, cx| {
                this.models.update(cx, |m, cx| {
                    if let Some(k) = sel.first().and_then(|&i| m.head_ids().get(i).cloned()) {
                        m.select_model(k, w, cx);
                    }
                });
            }));
        let note = self.tabs[self.active].spec.model_note;
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
                    .selected_index(self.active)
                    .on_click(cx.listener(|this, ix: &usize, w, cx| this.set_tab(*ix, w, cx)))
                    .children(
                        self.tabs
                            .iter()
                            .map(|tab| TabButton::new().prefix(Icon::new(tab.spec.icon).small()).label(tab.spec.title)),
                    ),
            )
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .when(note.is_none(), |d| {
                        d.child(div().text_xs().text_color(t.muted_foreground).child("Model"))
                            .child(models)
                            .children(m.selected().map(|k| widgets::model_state_chip(&m.state(k), cx)))
                    })
                    .children(note.map(|note| div().text_xs().text_color(t.muted_foreground).child(note))),
            )
    }

    fn render_status_bar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let mut left = Vec::new();
        let m = self.models.read(cx);
        if let Some(k) = m.selected() {
            let short = m.short(k);
            match m.state(k) {
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
        left.push(format!("jpeg: {}", inference::preprocess::JPEG_DECODER));
        let left = left.join(" · ");
        let hints = self.tabs[self.active].spec.hints;
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
                        .test_support()
                        .flex_none()
                        .gap_4()
                        .children(hints.iter().map(|(keys, label)| widgets::hint(keys, label, cx)))
                        .child(widgets::hint(&["ctrl-m"], "Model", cx))
                        .child(widgets::hint(&["ctrl-t"], "Theme", cx)),
                ),
        )
    }
}

/// The pictures on the clipboard: an image, copied files, or one path or URL per line of text.
fn clipboard_inputs(item: &ClipboardItem) -> Vec<ImageInput> {
    for entry in item.entries() {
        match entry {
            ClipboardEntry::Image(image) => return vec![ImageInput::Bytes(image.bytes.clone())],
            ClipboardEntry::ExternalPaths(paths) => {
                return paths.paths().iter().map(|p| ImageInput::Source(Source::File(p.clone()))).collect();
            }
            _ => {}
        }
    }
    item.text().map(|text| text.lines().filter_map(Source::parse).map(ImageInput::Source).collect()).unwrap_or_default()
}

impl Render for Shell {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let (bg, fg, drop_border) = (theme::roles(cx).canvas, t.foreground, t.drag_border);
        let m = self.models.read(cx);
        let registry_alert = m.registry_error().map(|e| {
            let found = !m.registry().index_path().as_os_str().is_empty();
            let title = if found { "No model in the index could be read" } else { "Model index not found" };
            div()
                .px_4()
                .pt_3()
                .child(Alert::error("registry-error", SharedString::from(e.to_string())).title(title).small())
        });
        let problems_alert = (!self.registry_problems.is_empty()).then(|| {
            div().id("registry-problems").test_support().px_4().pt_3().child(
                Alert::warning("registry-problems-alert", SharedString::from(self.registry_problems.join("\n")))
                    .title("Some models in the index could not be read")
                    .small()
                    .on_close(cx.listener(|this, _, _, cx| this.dismiss_problems(cx))),
            )
        });
        let fallback_alert = m.fallback_notice().map(|(title, why)| {
            div()
                .id("fallback-notice")
                .test_support()
                .px_4()
                .pt_3()
                .child(Alert::warning("default-fallback", SharedString::from(why)).title(title).small())
        });
        div()
            .id("root")
            .key_context(CTX)
            .capture_action(cx.listener(Self::leave_text_field))
            .on_action(cx.listener(Self::open))
            .on_action(cx.listener(Self::paste))
            .on_action(cx.listener(Self::focus_url))
            .on_action(cx.listener(Self::toggle_theme))
            .on_action(cx.listener(Self::next_model))
            .on_action(cx.listener(|this, tab: &ShowTab, w, cx| this.set_tab(tab.0, w, cx)))
            .on_mouse_down(
                gpui_kit::MouseButton::Left,
                cx.listener(|this, _, window, cx| {
                    let focus = this.active_focus();
                    if !focus.contains_focused(window, cx) {
                        window.focus(&focus, cx);
                    }
                }),
            )
            .on_drop(cx.listener(|this, paths: &ExternalPaths, w, cx| {
                let inputs = paths.paths().iter().map(|p| ImageInput::Source(Source::File(p.clone()))).collect();
                this.send_inputs(inputs, w, cx);
            }))
            .drag_over::<ExternalPaths>(move |style, _, _, _| style.border_2().border_color(drop_border))
            .size_full()
            .flex()
            .flex_col()
            .bg(bg)
            .text_color(fg)
            .child(self.render_title_bar(cx))
            .child(self.render_tab_row(cx))
            .children(registry_alert)
            .children(problems_alert)
            .children(fallback_alert)
            .child(div().flex_1().min_h_0().child(self.tabs[self.active].view.clone()))
            .child(self.render_status_bar(cx))
    }
}
