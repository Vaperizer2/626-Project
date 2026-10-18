//! The GPUI window: open / drop / paste images, show the picture, top-3 bars, UNSURE banner and a recent list.
//!
//! Threading: one dedicated `inference` thread owns the onnxruntime session. It loads the model at startup and
//! then serves jobs one at a time (read/fetch -> decode -> display copy -> preprocess -> run). The UI thread
//! never blocks: each job reports back over a futures channel that a GPUI task awaits.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use futures::channel::{mpsc as fmpsc, oneshot};
use gpui::{
    Animation, AnimationExt, App, Application, Bounds, ClickEvent, ClipboardEntry, Context, Decorations, ExternalPaths,
    FocusHandle, FontWeight, Hsla, KeyBinding, KeyDownEvent, MouseButton, ObjectFit, PathPromptOptions, RenderImage,
    ScrollHandle, SharedString, TitlebarOptions, Window, WindowAppearance, WindowBounds, WindowOptions, actions, div,
    img, prelude::*, pulsating_between, px, relative, rgb, size,
};

use crate::contract::Prediction;
use crate::fetch::Source;
use crate::model::{Classifier, ModelOptions};
use crate::preprocess;

actions!(gpui_demo, [Open, Paste, FocusUrl, ToggleTheme, SelectPrev, SelectNext, Quit]);

const MAX_ENTRIES: usize = 40;
const DISPLAY_MAX_SIDE: u32 = 1280;

// ------------------------------------------------------------------------------------------- worker
enum JobInput {
    Source(Source),
    Bytes(Vec<u8>),
}

struct Job {
    input: JobInput,
    updates: fmpsc::UnboundedSender<Update>,
}

enum Update {
    Image(Arc<RenderImage>, (u32, u32)),
    Done(Result<Prediction, String>, f64),
}

#[derive(Clone)]
struct ModelInfo {
    name: String,
    load_ms: f64,
    provider: String,
    threshold: f32,
}

fn worker(opts: ModelOptions, jobs: mpsc::Receiver<Job>, ready: oneshot::Sender<Result<ModelInfo, String>>) {
    let (mut clf, load_err) = match Classifier::load(&opts) {
        Ok(c) => {
            let info = ModelInfo {
                name: c.contract.pre.model.clone().unwrap_or_else(|| "model.onnx".into()),
                load_ms: c.load_ms,
                provider: c.provider.clone(),
                threshold: c.threshold(),
            };
            eprintln!("[gpui-demo] model loaded from {} in {:.0} ms", c.model_dir.display(), c.load_ms);
            let _ = ready.send(Ok(info));
            (Some(c), String::new())
        }
        Err(e) => {
            let msg = format!("{e:#}");
            eprintln!("[gpui-demo] model failed to load: {msg}");
            let _ = ready.send(Err(msg.clone()));
            (None, msg)
        }
    };
    while let Ok(job) = jobs.recv() {
        let t0 = Instant::now();
        let bytes = match job.input {
            JobInput::Source(s) => s.read(),
            JobInput::Bytes(b) => Ok(b),
        };
        let rgb = match bytes.and_then(|b| preprocess::decode(&b)) {
            Ok(rgb) => rgb,
            Err(e) => {
                let _ = job.updates.unbounded_send(Update::Done(Err(format!("{e:#}")), 0.0));
                continue;
            }
        };
        let frame = image::Frame::new(preprocess::display_bgra(&rgb, DISPLAY_MAX_SIDE));
        let render = Arc::new(RenderImage::new(smallvec::smallvec![frame]));
        let _ = job.updates.unbounded_send(Update::Image(render, rgb.dimensions()));
        let result = match clf.as_mut() {
            Some(c) => c.predict(&rgb).map_err(|e| format!("{e:#}")),
            None => Err(format!("model not loaded: {load_err}")),
        };
        let _ = job.updates.unbounded_send(Update::Done(result, t0.elapsed().as_secs_f64() * 1000.0));
    }
}

// -------------------------------------------------------------------------------------------- theme
#[derive(Clone, Copy)]
struct Theme {
    bg: Hsla,
    panel: Hsla,
    raised: Hsla,
    border: Hsla,
    text: Hsla,
    muted: Hsla,
    accent: Hsla,
    accent_text: Hsla,
    bar_bg: Hsla,
    bar_2: Hsla,
    warn_bg: Hsla,
    warn_text: Hsla,
    err: Hsla,
    selected: Hsla,
}

fn theme(dark: bool) -> Theme {
    let c = |hex: u32| -> Hsla { rgb(hex).into() };
    if dark {
        Theme {
            bg: c(0x16181d),
            panel: c(0x1e2128),
            raised: c(0x272b34),
            border: c(0x343944),
            text: c(0xe6e8ec),
            muted: c(0x9aa1ad),
            accent: c(0x4c9aff),
            accent_text: c(0x0b1220),
            bar_bg: c(0x2d323c),
            bar_2: c(0x5b6577),
            warn_bg: c(0x4a3410),
            warn_text: c(0xffc864),
            err: c(0xff6b6b),
            selected: c(0x2a3a55),
        }
    } else {
        Theme {
            bg: c(0xf3f4f6),
            panel: c(0xffffff),
            raised: c(0xeef0f3),
            border: c(0xd7dbe1),
            text: c(0x1b1f24),
            muted: c(0x5f6773),
            accent: c(0x1f6feb),
            accent_text: c(0xffffff),
            bar_bg: c(0xe3e6ea),
            bar_2: c(0x9aa4b2),
            warn_bg: c(0xfff1d6),
            warn_text: c(0x8a5300),
            err: c(0xc62828),
            selected: c(0xdbe8ff),
        }
    }
}

// --------------------------------------------------------------------------------------------- view
enum EntryState {
    Working,
    Done(Prediction, f64),
    Failed(String),
}

struct Entry {
    id: u64,
    name: SharedString,
    full: SharedString,
    image: Option<Arc<RenderImage>>,
    dims: Option<(u32, u32)>,
    state: EntryState,
}

enum ModelStatus {
    Loading(Instant),
    Ready(ModelInfo),
    Failed(String),
}

struct DemoApp {
    focus: FocusHandle,
    url_focus: FocusHandle,
    jobs: mpsc::Sender<Job>,
    model: ModelStatus,
    url: String,
    entries: Vec<Entry>,
    selected: Option<u64>,
    next_id: u64,
    notice: Option<String>,
    dark: bool,
    list_scroll: ScrollHandle,
    to_drop: Vec<Arc<RenderImage>>,
}

impl DemoApp {
    fn new(opts: ModelOptions, initial: Vec<String>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = oneshot::channel();
        std::thread::Builder::new()
            .name("inference".into())
            .spawn(move || worker(opts, job_rx, ready_tx))
            .expect("spawn inference thread");
        cx.spawn(async move |this, cx| {
            let status = match ready_rx.await {
                Ok(Ok(info)) => ModelStatus::Ready(info),
                Ok(Err(e)) => ModelStatus::Failed(e),
                Err(_) => ModelStatus::Failed("inference thread exited".into()),
            };
            this.update(cx, |this, cx| {
                this.model = status;
                cx.notify();
            })
            .ok();
        })
        .detach();
        // Re-render once a second while the model loads (elapsed-time counter).
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(Duration::from_millis(250)).await;
                let loading = this
                    .update(cx, |this, cx| {
                        cx.notify();
                        matches!(this.model, ModelStatus::Loading(_))
                    })
                    .unwrap_or(false);
                if !loading {
                    break;
                }
            }
        })
        .detach();
        let dark = matches!(window.appearance(), WindowAppearance::Dark | WindowAppearance::VibrantDark);
        let mut this = Self {
            focus: cx.focus_handle(),
            url_focus: cx.focus_handle(),
            jobs: job_tx,
            model: ModelStatus::Loading(Instant::now()),
            url: String::new(),
            entries: Vec::new(),
            selected: None,
            next_id: 1,
            notice: None,
            dark,
            list_scroll: ScrollHandle::new(),
            to_drop: Vec::new(),
        };
        for s in initial {
            if let Some(src) = Source::parse(&s) {
                this.submit_source(src, cx);
            }
        }
        this
    }

    fn submit_source(&mut self, src: Source, cx: &mut Context<Self>) {
        let (name, full) = (src.display_name(), src.full());
        self.submit(JobInput::Source(src), name, full, cx);
    }

    fn submit(&mut self, input: JobInput, name: String, full: String, cx: &mut Context<Self>) {
        let id = self.next_id;
        self.next_id += 1;
        self.notice = None;
        let (tx, mut rx) = fmpsc::unbounded();
        let state = if self.jobs.send(Job { input, updates: tx }).is_ok() {
            EntryState::Working
        } else {
            EntryState::Failed("inference thread is not running".into())
        };
        self.entries.insert(0, Entry { id, name: name.into(), full: full.into(), image: None, dims: None, state });
        while self.entries.len() > MAX_ENTRIES {
            if let Some(e) = self.entries.pop() {
                self.to_drop.extend(e.image);
            }
        }
        self.selected = Some(id);
        self.list_scroll.scroll_to_item(0);
        cx.spawn(async move |this, cx| {
            while let Some(update) = rx.next().await {
                let alive = this
                    .update(cx, |this, cx| {
                        if let Some(e) = this.entries.iter_mut().find(|e| e.id == id) {
                            match update {
                                Update::Image(img, dims) => {
                                    e.image = Some(img);
                                    e.dims = Some(dims);
                                }
                                Update::Done(Ok(p), total) => e.state = EntryState::Done(p, total),
                                Update::Done(Err(msg), _) => e.state = EntryState::Failed(msg),
                            }
                        }
                        cx.notify();
                    })
                    .is_ok();
                if !alive {
                    break;
                }
            }
        })
        .detach();
        cx.notify();
    }

    fn fetch_typed_url(&mut self, cx: &mut Context<Self>) {
        match Source::parse(&self.url) {
            Some(src) => self.submit_source(src, cx),
            None => {
                self.notice = Some("Type or paste an image URL (or a file path) first.".into());
                cx.notify();
            }
        }
    }

    fn open(&mut self, _: &Open, _window: &mut Window, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: true,
            prompt: Some("Classify".into()),
        });
        cx.spawn(async move |this, cx| {
            let result = rx.await;
            this.update(cx, |this, cx| {
                match result {
                    Ok(Ok(Some(paths))) => {
                        for p in paths {
                            this.submit_source(Source::File(p), cx);
                        }
                    }
                    Ok(Ok(None)) => {}
                    Ok(Err(e)) => {
                        this.notice = Some(format!(
                            "File dialog unavailable ({e}). It needs xdg-desktop-portal; drag files onto the \
                             window or paste a path with Ctrl+V instead."
                        ))
                    }
                    Err(_) => this.notice = Some("File dialog was closed unexpectedly.".into()),
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn paste(&mut self, _: &Paste, window: &mut Window, cx: &mut Context<Self>) {
        let Some(item) = cx.read_from_clipboard() else {
            self.notice = Some("Clipboard is empty.".into());
            cx.notify();
            return;
        };
        for entry in item.entries() {
            if let ClipboardEntry::Image(image) = entry {
                self.submit(JobInput::Bytes(image.bytes.clone()), "pasted image".into(), "clipboard".into(), cx);
                return;
            }
        }
        let Some(text) = item.text() else { return };
        if self.url_focus.is_focused(window) {
            self.url.push_str(text.trim());
            cx.notify();
            return;
        }
        let sources: Vec<Source> = text.lines().filter_map(Source::parse).collect();
        if sources.is_empty() {
            self.notice = Some("Clipboard has no URL, path or image.".into());
            cx.notify();
        }
        for s in sources {
            if let Source::Url(u) = &s {
                self.url = u.clone();
            }
            self.submit_source(s, cx);
        }
    }

    fn focus_url(&mut self, _: &FocusUrl, window: &mut Window, cx: &mut Context<Self>) {
        window.focus(&self.url_focus);
        cx.notify();
    }

    fn toggle_theme(&mut self, _: &ToggleTheme, _: &mut Window, cx: &mut Context<Self>) {
        self.dark = !self.dark;
        cx.notify();
    }

    fn select_step(&mut self, delta: isize, cx: &mut Context<Self>) {
        if self.entries.is_empty() {
            return;
        }
        let cur = self.selected.and_then(|id| self.entries.iter().position(|e| e.id == id)).unwrap_or(0) as isize;
        let next = (cur + delta).clamp(0, self.entries.len() as isize - 1) as usize;
        self.selected = Some(self.entries[next].id);
        self.list_scroll.scroll_to_item(next);
        cx.notify();
    }

    fn on_url_key(&mut self, ev: &KeyDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        let ks = &ev.keystroke;
        if ks.modifiers.control || ks.modifiers.platform || ks.modifiers.alt {
            return; // let key bindings (Ctrl+V, Ctrl+O, ...) handle it
        }
        match ks.key.as_str() {
            "backspace" => {
                self.url.pop();
            }
            "enter" => self.fetch_typed_url(cx),
            "escape" => window.focus(&self.focus),
            "up" | "down" | "left" | "right" | "tab" => return,
            _ => match &ks.key_char {
                Some(ch) if !ch.chars().any(char::is_control) => self.url.push_str(ch),
                _ => return,
            },
        }
        cx.stop_propagation();
        cx.notify();
    }

    // ------------------------------------------------------------------------------------- rendering
    fn button(id: &'static str, label: impl Into<SharedString>, t: Theme, primary: bool) -> gpui::Stateful<gpui::Div> {
        div()
            .id(id)
            .px_3()
            .py_1p5()
            .rounded_md()
            .text_sm()
            .cursor_pointer()
            .when(primary, |d| d.bg(t.accent).text_color(t.accent_text).font_weight(FontWeight::SEMIBOLD))
            .when(!primary, |d| d.bg(t.raised).text_color(t.text).border_1().border_color(t.border))
            .hover(|s| s.opacity(0.85))
            .child(label.into())
    }

    fn model_chip(&self, t: Theme) -> impl IntoElement {
        let (text, color): (String, Hsla) = match &self.model {
            ModelStatus::Loading(t0) => (format!("Loading model… {:.1} s", t0.elapsed().as_secs_f64()), t.warn_text),
            ModelStatus::Ready(i) => {
                (format!("{} · {} · loaded in {:.1} s", i.name, i.provider, i.load_ms / 1000.0), t.muted)
            }
            ModelStatus::Failed(_) => ("Model failed to load (see panel)".into(), t.err),
        };
        div().text_xs().text_color(color).child(text)
    }

    fn render_title_bar(&self, t: Theme, window: &Window, cx: &mut Context<Self>) -> impl IntoElement {
        let csd = matches!(window.window_decorations(), Decorations::Client { .. });
        div()
            .flex()
            .flex_row()
            .items_center()
            .gap_3()
            .px_4()
            .py_2()
            .bg(t.panel)
            .border_b_1()
            .border_color(t.border)
            .child(
                div()
                    .flex_1()
                    .flex()
                    .flex_row()
                    .items_baseline()
                    .gap_3()
                    .when(csd, |d| d.on_mouse_down(MouseButton::Left, |_, window, _| window.start_window_move()))
                    .child(
                        div().text_lg().font_weight(FontWeight::BOLD).text_color(t.text).child("Component Classifier"),
                    )
                    .child(self.model_chip(t)),
            )
            .child(
                Self::button("theme", if self.dark { "Light theme" } else { "Dark theme" }, t, false)
                    .on_click(cx.listener(|this, _: &ClickEvent, w, cx| this.toggle_theme(&ToggleTheme, w, cx))),
            )
            .when(csd, |d| {
                d.child(
                    Self::button("close", "✕", t, false).on_click(|_: &ClickEvent, window, _| window.remove_window()),
                )
            })
    }

    fn render_toolbar(&self, t: Theme, window: &Window, cx: &mut Context<Self>) -> impl IntoElement {
        let focused = self.url_focus.is_focused(window);
        let field_text: SharedString = if self.url.is_empty() && !focused {
            "Paste or type an image URL, then Enter…".into()
        } else if focused {
            format!("{}▏", self.url).into()
        } else {
            self.url.clone().into()
        };
        div()
            .flex()
            .flex_row()
            .items_center()
            .gap_2()
            .px_4()
            .py_2()
            .bg(t.panel)
            .border_b_1()
            .border_color(t.border)
            .child(
                Self::button("open", "Open image…  Ctrl+O", t, true)
                    .on_click(cx.listener(|this, _: &ClickEvent, w, cx| this.open(&Open, w, cx))),
            )
            .child(
                div()
                    .id("url")
                    .track_focus(&self.url_focus)
                    .flex_1()
                    .min_w_0()
                    .px_3()
                    .py_1p5()
                    .rounded_md()
                    .bg(t.bg)
                    .border_1()
                    .border_color(if focused { t.accent } else { t.border })
                    .text_sm()
                    .text_color(if self.url.is_empty() && !focused { t.muted } else { t.text })
                    .overflow_hidden()
                    .whitespace_nowrap()
                    .cursor_text()
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|this, _, window, cx| {
                            window.focus(&this.url_focus);
                            cx.stop_propagation();
                            cx.notify();
                        }),
                    )
                    .on_key_down(cx.listener(Self::on_url_key))
                    .child(field_text),
            )
            .child(
                Self::button("fetch", "Fetch", t, false)
                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| this.fetch_typed_url(cx))),
            )
            .child(
                Self::button("paste", "Paste  Ctrl+V", t, false)
                    .on_click(cx.listener(|this, _: &ClickEvent, w, cx| this.paste(&Paste, w, cx))),
            )
    }

    fn render_recent(&self, t: Theme, cx: &mut Context<Self>) -> impl IntoElement {
        let rows = self.entries.iter().map(|e| {
            let id = e.id;
            let sel = self.selected == Some(id);
            let (line, color): (String, Hsla) = match &e.state {
                EntryState::Working => ("classifying…".into(), t.muted),
                EntryState::Failed(_) => ("error".into(), t.err),
                EntryState::Done(p, _) if p.unsure => {
                    (format!("UNSURE ({} {:.0}%)", p.top3[0].0, p.top3[0].1 * 100.0), t.warn_text)
                }
                EntryState::Done(p, _) => (format!("{} {:.0}%", p.top3[0].0, p.top3[0].1 * 100.0), t.text),
            };
            div()
                .id(("entry", id as usize))
                .flex()
                .flex_row()
                .gap_2()
                .items_center()
                .p_1p5()
                .rounded_md()
                .cursor_pointer()
                .when(sel, |d| d.bg(t.selected))
                .when(!sel, |d| d.hover(|s| s.bg(t.raised)))
                .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                    this.selected = Some(id);
                    cx.notify();
                }))
                .child(
                    div()
                        .w(px(56.))
                        .h(px(42.))
                        .flex_none()
                        .rounded_sm()
                        .overflow_hidden()
                        .bg(t.raised)
                        .when_some(e.image.clone(), |d, im| d.child(img(im).size_full().object_fit(ObjectFit::Cover))),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .min_w_0()
                        .flex_1()
                        .child(
                            div()
                                .text_xs()
                                .text_color(t.muted)
                                .overflow_hidden()
                                .whitespace_nowrap()
                                .text_ellipsis()
                                .child(e.name.clone()),
                        )
                        .child(
                            div()
                                .text_sm()
                                .text_color(color)
                                .overflow_hidden()
                                .whitespace_nowrap()
                                .text_ellipsis()
                                .child(line),
                        ),
                )
        });
        div()
            .w(px(250.))
            .flex_none()
            .flex()
            .flex_col()
            .bg(t.panel)
            .border_r_1()
            .border_color(t.border)
            .child(
                div()
                    .px_3()
                    .py_2()
                    .text_xs()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(t.muted)
                    .child(format!("RECENT ({})", self.entries.len())),
            )
            .child(
                div()
                    .id("recent")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .track_scroll(&self.list_scroll)
                    .px_2()
                    .pb_2()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .children(rows)
                    .when(self.entries.is_empty(), |d| {
                        d.child(div().p_2().text_xs().text_color(t.muted).child("Classified images appear here."))
                    }),
            )
    }

    fn render_image(&self, t: Theme, entry: Option<&Entry>) -> impl IntoElement {
        let body = match entry.and_then(|e| e.image.clone()) {
            Some(im) => div().size_full().p_3().child(img(im).size_full().object_fit(ObjectFit::Contain)),
            None => div().size_full().flex().flex_col().items_center().justify_center().gap_2().child(
                div()
                    .flex()
                    .flex_col()
                    .items_center()
                    .gap_2()
                    .p_8()
                    .rounded_lg()
                    .border_2()
                    .border_dashed()
                    .border_color(t.border)
                    .child(div().text_xl().text_color(t.text).child(if entry.is_some() {
                        "Loading image…"
                    } else {
                        "Drop component pictures here"
                    }))
                    .child(
                        div()
                            .text_sm()
                            .text_color(t.muted)
                            .child("or press Ctrl+O to open files, or paste a URL / path / image with Ctrl+V"),
                    ),
            ),
        };
        div().flex_1().min_w_0().h_full().bg(t.bg).child(body)
    }

    fn render_results(&self, t: Theme, entry: Option<&Entry>) -> impl IntoElement {
        let threshold = match &self.model {
            ModelStatus::Ready(i) => i.threshold,
            _ => 0.5,
        };
        let mut panel = div()
            .w(px(380.))
            .flex_none()
            .h_full()
            .flex()
            .flex_col()
            .gap_4()
            .p_4()
            .bg(t.panel)
            .border_l_1()
            .border_color(t.border);
        if let ModelStatus::Failed(msg) = &self.model {
            panel = panel.child(
                div()
                    .p_3()
                    .rounded_md()
                    .bg(t.raised)
                    .border_1()
                    .border_color(t.err)
                    .text_sm()
                    .text_color(t.err)
                    .child(format!("Model failed to load:\n{msg}")),
            );
        }
        let Some(e) = entry else {
            return panel.child(div().text_sm().text_color(t.muted).child("No image selected yet."));
        };
        panel = panel.child(
            div()
                .flex()
                .flex_col()
                .gap_0p5()
                .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).text_color(t.text).child(e.name.clone()))
                .child(div().text_xs().text_color(t.muted).child(e.full.clone()))
                .when_some(e.dims, |d, (w, h)| {
                    d.child(div().text_xs().text_color(t.muted).child(format!("{w} × {h} px")))
                }),
        );
        match &e.state {
            EntryState::Working => panel.child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap_2()
                    .child(div().size_3().rounded_full().bg(t.accent).with_animation(
                        "busy-dot",
                        Animation::new(Duration::from_millis(900)).repeat().with_easing(pulsating_between(0.2, 1.0)),
                        |d, delta| d.opacity(delta),
                    ))
                    .child(div().text_lg().text_color(t.text).child(
                        if matches!(self.model, ModelStatus::Loading(_)) {
                            "Waiting for the model to load…"
                        } else {
                            "Classifying…"
                        },
                    )),
            ),
            EntryState::Failed(msg) => panel.child(
                div()
                    .p_3()
                    .rounded_md()
                    .bg(t.raised)
                    .border_1()
                    .border_color(t.err)
                    .child(div().text_sm().font_weight(FontWeight::BOLD).text_color(t.err).child("Could not classify"))
                    .child(div().text_sm().text_color(t.text).child(msg.clone())),
            ),
            EntryState::Done(p, total_ms) => {
                let headline = if p.unsure {
                    div()
                        .p_3()
                        .rounded_md()
                        .bg(t.warn_bg)
                        .border_1()
                        .border_color(t.warn_text)
                        .child(div().text_3xl().font_weight(FontWeight::BOLD).text_color(t.warn_text).child("UNSURE"))
                        .child(div().text_sm().text_color(t.warn_text).child(format!(
                            "Top probability {:.0}% is below the {:.0}% threshold. Best guesses:",
                            p.top3[0].1 * 100.0,
                            threshold * 100.0
                        )))
                } else {
                    div()
                        .flex()
                        .flex_col()
                        .child(div().text_xs().text_color(t.muted).child("PREDICTION"))
                        .child(
                            div()
                                .text_3xl()
                                .font_weight(FontWeight::BOLD)
                                .text_color(t.text)
                                .child(p.top3[0].0.clone()),
                        )
                        .child(
                            div()
                                .text_sm()
                                .text_color(t.muted)
                                .child(format!("{:.1}% confidence (calibrated)", p.top3[0].1 * 100.0)),
                        )
                };
                let bars = p.top3.iter().enumerate().map(|(i, (label, prob))| {
                    div()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(
                            div()
                                .flex()
                                .flex_row()
                                .justify_between()
                                .text_sm()
                                .child(div().text_color(t.text).child(format!("{}. {label}", i + 1)))
                                .child(div().text_color(t.muted).child(format!("{:.1}%", prob * 100.0))),
                        )
                        .child(
                            div().w_full().h(px(12.)).rounded_full().bg(t.bar_bg).child(
                                div()
                                    .h_full()
                                    .rounded_full()
                                    .w(relative(prob.clamp(0.0, 1.0)))
                                    .bg(if i == 0 && !p.unsure { t.accent } else { t.bar_2 }),
                            ),
                        )
                });
                panel.child(headline).child(div().flex().flex_col().gap_3().children(bars)).child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_0p5()
                        .text_xs()
                        .text_color(t.muted)
                        .child(format!("Inference (preprocess + model): {:.0} ms", p.ms))
                        .child(format!("Total incl. load/decode: {:.0} ms", total_ms)),
                )
            }
        }
    }
}

impl Render for DemoApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        for im in self.to_drop.drain(..) {
            window.drop_image(im).ok();
        }
        let t = theme(self.dark);
        let entry = self.selected.and_then(|id| self.entries.iter().find(|e| e.id == id));
        let accent = t.accent;
        div()
            .id("root")
            .track_focus(&self.focus)
            .key_context("GpuiDemo")
            .on_action(cx.listener(Self::open))
            .on_action(cx.listener(Self::paste))
            .on_action(cx.listener(Self::focus_url))
            .on_action(cx.listener(Self::toggle_theme))
            .on_action(cx.listener(|this, _: &SelectPrev, _, cx| this.select_step(-1, cx)))
            .on_action(cx.listener(|this, _: &SelectNext, _, cx| this.select_step(1, cx)))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, window, cx| {
                    if this.url_focus.is_focused(window) {
                        window.focus(&this.focus);
                        cx.notify();
                    }
                }),
            )
            .on_drop(cx.listener(|this, paths: &ExternalPaths, _, cx| {
                for p in paths.paths() {
                    this.submit_source(Source::File(PathBuf::from(p)), cx);
                }
            }))
            .drag_over::<ExternalPaths>(move |style, _, _, _| style.border_2().border_color(accent))
            .size_full()
            .flex()
            .flex_col()
            .bg(t.bg)
            .text_color(t.text)
            .child(self.render_title_bar(t, window, cx))
            .child(self.render_toolbar(t, window, cx))
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .flex()
                    .flex_row()
                    .child(self.render_recent(t, cx))
                    .child(self.render_image(t, entry))
                    .child(self.render_results(t, entry)),
            )
            .when_some(self.notice.clone(), |d, n| {
                d.child(
                    div()
                        .px_4()
                        .py_1p5()
                        .bg(t.raised)
                        .border_t_1()
                        .border_color(t.border)
                        .text_sm()
                        .text_color(t.warn_text)
                        .child(n),
                )
            })
    }
}

/// Open the window. `initial` are image paths / URLs to classify straight away.
pub fn run(opts: ModelOptions, initial: Vec<String>) {
    if cfg!(target_os = "linux")
        && std::env::var_os("DISPLAY").is_none()
        && std::env::var_os("WAYLAND_DISPLAY").is_none()
    {
        eprintln!(
            "gpui-demo: no display (neither WAYLAND_DISPLAY nor DISPLAY is set). Run it from a desktop session, \
             or use `gpui-demo classify <images>` for headless output."
        );
        std::process::exit(1);
    }
    Application::new().run(move |cx: &mut App| {
        cx.bind_keys([
            KeyBinding::new("ctrl-o", Open, None),
            KeyBinding::new("ctrl-v", Paste, None),
            KeyBinding::new("ctrl-l", FocusUrl, None),
            KeyBinding::new("ctrl-t", ToggleTheme, None),
            KeyBinding::new("up", SelectPrev, Some("GpuiDemo")),
            KeyBinding::new("down", SelectNext, Some("GpuiDemo")),
            KeyBinding::new("ctrl-q", Quit, None),
            KeyBinding::new("cmd-o", Open, None),
            KeyBinding::new("cmd-v", Paste, None),
            KeyBinding::new("cmd-q", Quit, None),
        ]);
        cx.on_action(|_: &Quit, cx: &mut App| cx.quit());
        let bounds = Bounds::centered(None, size(px(1280.), px(820.)), cx);
        let window = cx
            .open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    titlebar: Some(TitlebarOptions {
                        title: Some("Component Classifier".into()),
                        ..Default::default()
                    }),
                    app_id: Some("gpui-demo".into()),
                    window_min_size: Some(size(px(960.), px(600.))),
                    ..Default::default()
                },
                |window, cx| cx.new(|cx| DemoApp::new(opts, initial, window, cx)),
            )
            .expect("failed to open the window (see README troubleshooting: Wayland/X11/Vulkan)");
        window
            .update(cx, |view, window, cx| {
                window.set_window_title("Component Classifier");
                window.focus(&view.focus);
                cx.activate(true);
            })
            .ok();
        cx.on_window_closed(|cx| cx.quit()).detach();
    });
}
