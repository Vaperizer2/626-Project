//! Classify tab: open / drop / paste / fetch pictures, the recent list, the picture and the result panel.

use std::path::PathBuf;
use std::sync::Arc;

use futures::StreamExt;
use gpui_kit::component::{
    ActiveTheme as _, Icon, IconName, Sizable as _, WindowExt as _,
    alert::Alert,
    button::{Button, ButtonVariants as _},
    empty::{Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle},
    h_flex,
    input::{Input, InputEvent, InputState},
    notification::NotificationType,
    scroll::ScrollableElement as _,
    spinner::Spinner,
    v_flex,
};
use gpui_kit::{
    ClipboardEntry, Context, Entity, FontWeight, ObjectFit, PathPromptOptions, RenderImage, ScrollHandle, SharedString,
    Subscription, Window, div, img, prelude::*, px,
};

use super::widgets::Lucide;
use super::images::{DISPLAY_MAX_SIDE, render};
use super::results::{HeadResult, Results};
use super::{DemoApp, FocusUrl, ModelState, Open, Paste, theme, widgets};
use crate::fetch::Source;
use crate::inference::{ImageInput, RequestHandle, Ticket, Update};

const MAX_ENTRIES: usize = 40;

pub struct Entry {
    id: u64,
    name: SharedString,
    full: SharedString,
    /// what was opened, to run it again with another model
    input: ImageInput,
    image: Option<Arc<RenderImage>>,
    dims: Option<(u32, u32)>,
    results: Results,
    /// dropping the entry (evicted from the list) cancels its requests
    tickets: Vec<Ticket>,
}

pub struct ClassifyState {
    url: Entity<InputState>,
    entries: Vec<Entry>,
    selected: Option<u64>,
    next_id: u64,
    list_scroll: ScrollHandle,
}

impl ClassifyState {
    pub fn new(window: &mut Window, cx: &mut Context<DemoApp>) -> (Self, Vec<Subscription>) {
        let url =
            cx.new(|cx| InputState::new(window, cx).placeholder("Paste or type an image URL or file path, then Enter"));
        let sub = cx.subscribe_in(&url, window, |this: &mut DemoApp, _, ev: &InputEvent, window, cx| {
            if let InputEvent::PressEnter { .. } = ev {
                this.fetch_typed_url(window, cx);
            }
        });
        let s = Self { url, entries: Vec::new(), selected: None, next_id: 1, list_scroll: ScrollHandle::new() };
        (s, vec![sub])
    }

    pub fn busy(&self) -> bool {
        self.entries.iter().any(|e| e.results.working())
    }

    fn selected_ix(&self) -> Option<usize> {
        self.selected.and_then(|id| self.entries.iter().position(|e| e.id == id))
    }

    /// The selected entry has `head`'s answer, its wall time and its display picture.
    #[cfg(test)]
    pub fn answered(&self, head: &crate::manifest::HeadId) -> bool {
        self.selected_ix()
            .map(|i| &self.entries[i])
            .is_some_and(|e| matches!(e.results.get(head), Some(HeadResult::Done(_, _, Some(_)))) && e.image.is_some())
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

impl DemoApp {
    pub(super) fn submit_source(&mut self, src: Source, cx: &mut Context<Self>) {
        let (name, full) = (src.display_name(), src.full());
        self.submit(ImageInput::Source(src), name, full, cx);
    }

    fn submit(&mut self, input: ImageInput, name: String, full: String, cx: &mut Context<Self>) {
        if self.model.is_none() {
            return;
        }
        let c = &mut self.classify;
        let id = c.next_id;
        c.next_id += 1;
        let entry = Entry {
            id,
            name: name.into(),
            full: full.into(),
            input,
            image: None,
            dims: None,
            results: Results::default(),
            tickets: Vec::new(),
        };
        c.entries.insert(0, entry);
        while c.entries.len() > MAX_ENTRIES {
            if let Some(e) = c.entries.pop() {
                self.to_drop.extend(e.image);
            }
        }
        c.selected = Some(id);
        c.list_scroll.scroll_to_item(0);
        self.classify_follow_model(cx);
    }

    /// The selected entry gets the selected model's answer when it has none yet: on submission, after a model
    /// switch, and when another entry is selected.
    pub(super) fn classify_follow_model(&mut self, cx: &mut Context<Self>) {
        let (Some(model), Some(ix)) = (self.model.clone(), self.classify.selected_ix()) else { return };
        let c = &mut self.classify;
        let e = &mut c.entries[ix];
        if e.results.get(&model).is_some() {
            return;
        }
        let (id, token) = (e.id, c.next_id);
        c.next_id += 1;
        let display = e.image.is_none().then_some(DISPLAY_MAX_SIDE);
        let RequestHandle { updates: mut rx, ticket } =
            self.interactive.submit(e.input.clone(), std::slice::from_ref(&model), display);
        e.tickets.push(ticket);
        e.results.start(model.clone(), token);
        cx.spawn(async move |this, cx| {
            while let Some(update) = rx.next().await {
                let alive = this
                    .update(cx, |this, cx| {
                        if let Some(e) = this.classify.entries.iter_mut().find(|e| e.id == id) {
                            match update {
                                Update::Decoded { size, display } => {
                                    e.dims = Some(size);
                                    if let Some(d) = display {
                                        e.image = Some(render(d));
                                    }
                                }
                                u => e.results.apply(&model, token, u),
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

    fn fetch_typed_url(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let text = self.classify.url.read(cx).value().to_string();
        match Source::parse(&text) {
            Some(src) => self.submit_source(src, cx),
            None => window.push_notification("Type or paste an image URL (or a file path) first.", cx),
        }
    }

    pub(super) fn open(&mut self, _: &Open, window: &mut Window, cx: &mut Context<Self>) {
        self.set_tab(super::TabKind::Classify, window, cx);
        let rx = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: true,
            prompt: Some("Classify".into()),
        });
        cx.spawn_in(window, async move |this, cx| {
            let result = rx.await;
            this.update_in(cx, |this, window, cx| match result {
                Ok(Ok(Some(paths))) => {
                    for p in paths {
                        this.submit_source(Source::File(p), cx);
                    }
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

    pub(super) fn paste(&mut self, _: &Paste, window: &mut Window, cx: &mut Context<Self>) {
        let Some(item) = cx.read_from_clipboard() else {
            window.push_notification("The clipboard is empty.", cx);
            return;
        };
        self.set_tab(super::TabKind::Classify, window, cx);
        for entry in item.entries() {
            match entry {
                ClipboardEntry::Image(image) => {
                    self.submit(ImageInput::Bytes(image.bytes.clone()), "pasted image".into(), "clipboard".into(), cx);
                    return;
                }
                ClipboardEntry::ExternalPaths(paths) => {
                    for p in paths.paths() {
                        self.submit_source(Source::File(PathBuf::from(p)), cx);
                    }
                    return;
                }
                _ => {}
            }
        }
        let Some(text) = item.text() else { return };
        let sources: Vec<Source> = text.lines().filter_map(Source::parse).collect();
        if sources.is_empty() {
            window.push_notification("The clipboard has no URL, path or image.", cx);
        }
        for s in sources {
            if let Source::Url(u) = &s {
                let u = u.clone();
                self.classify.url.update(cx, |st, cx| st.set_value(u, window, cx));
            }
            self.submit_source(s, cx);
        }
    }

    pub(super) fn focus_url(&mut self, _: &FocusUrl, window: &mut Window, cx: &mut Context<Self>) {
        self.set_tab(super::TabKind::Classify, window, cx);
        self.classify.url.update(cx, |st, cx| st.focus(window, cx));
    }

    pub(super) fn classify_step(&mut self, delta: isize, cx: &mut Context<Self>) {
        let c = &mut self.classify;
        if c.entries.is_empty() {
            return;
        }
        let cur = c.selected.and_then(|id| c.entries.iter().position(|e| e.id == id)).unwrap_or(0) as isize;
        let next = (cur + delta).clamp(0, c.entries.len() as isize - 1) as usize;
        c.selected = Some(c.entries[next].id);
        c.list_scroll.scroll_to_item(next);
        self.classify_follow_model(cx);
        cx.notify();
    }

    // ------------------------------------------------------------------------------------------ render
    pub(super) fn render_classify(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let entry_ix = self.classify.selected_ix();
        v_flex().size_full().child(self.render_toolbar(window, cx)).child(
            h_flex()
                .flex_1()
                .min_h_0()
                .items_start()
                .child(self.render_recent(cx))
                .child(self.render_picture(entry_ix, cx))
                .child(self.render_result(entry_ix, cx)),
        )
    }

    fn render_toolbar(&self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        h_flex()
            .gap_2()
            .px_4()
            .py_2()
            .bg(theme::roles(cx).chrome)
            .border_b_1()
            .border_color(t.border)
            .child(
                Button::new("open")
                    .primary()
                    .icon(IconName::FolderOpen)
                    .label("Open image…")
                    .tooltip("Open files (Ctrl+O)")
                    .on_click(cx.listener(|this, _, w, cx| this.open(&Open, w, cx))),
            )
            .child(
                div().flex_1().min_w_0().child(
                    Input::new(&self.classify.url)
                        .prefix(Icon::new(IconName::Globe).small().text_color(t.muted_foreground))
                        .cleanable(true),
                ),
            )
            .child(
                Button::new("fetch")
                    .label("Fetch")
                    .tooltip("Download and classify the URL (Enter)")
                    .on_click(cx.listener(|this, _, w, cx| this.fetch_typed_url(w, cx))),
            )
            .child(
                Button::new("paste")
                    .icon(IconName::Copy)
                    .label("Paste")
                    .tooltip("Classify the clipboard: image, URL or path (Ctrl+V)")
                    .on_click(cx.listener(|this, _, w, cx| this.paste(&Paste, w, cx))),
            )
    }

    fn render_recent(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let c = &self.classify;
        let rows = c.entries.iter().map(|e| {
            let id = e.id;
            let sel = c.selected == Some(id);
            let status = match self.model.as_ref().and_then(|m| e.results.get(m)) {
                None => div().text_color(t.muted_foreground).child("not classified yet").into_any_element(),
                Some(HeadResult::Working(_)) => h_flex()
                    .gap_1()
                    .items_center()
                    .child(Spinner::new().xsmall().color(t.muted_foreground))
                    .child(div().text_color(t.muted_foreground).child("classifying…"))
                    .into_any_element(),
                Some(HeadResult::Failed(_)) => h_flex()
                    .gap_1()
                    .items_center()
                    .text_color(t.danger)
                    .child(Icon::new(IconName::CircleAlert).xsmall())
                    .child("could not read")
                    .into_any_element(),
                Some(HeadResult::Done(c, ..)) if c.abstained() => h_flex()
                    .gap_1()
                    .items_center()
                    .child(widgets::verdict_tag(crate::demo::Verdict::Unsure, true, cx))
                    .child(div().text_xs().text_color(t.muted_foreground).truncate().child(c.best_label().to_string()))
                    .into_any_element(),
                Some(HeadResult::Done(c, ..)) => h_flex()
                    .gap_2()
                    .justify_between()
                    .min_w_0()
                    .child(
                        div()
                            .text_color(t.foreground)
                            .font_weight(FontWeight::MEDIUM)
                            .truncate()
                            .child(c.best_label().to_string()),
                    )
                    .child(div().flex_none().text_xs().text_color(t.muted_foreground).child(widgets::pct(c.best().1)))
                    .into_any_element(),
            };
            h_flex()
                .id(("entry", id as usize))
                .gap_2()
                .p_1p5()
                .rounded(t.radius)
                .cursor_pointer()
                .when(sel, |d| d.bg(t.list_active).border_1().border_color(t.list_active_border))
                .when(!sel, |d| d.border_1().border_color(gpui_kit::transparent_black()).hover(|s| s.bg(t.list_hover)))
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.classify.selected = Some(id);
                    this.classify_follow_model(cx);
                    cx.notify();
                }))
                .child(
                    div()
                        .w(px(56.))
                        .h(px(42.))
                        .flex_none()
                        .rounded(t.radius)
                        .overflow_hidden()
                        .bg(t.muted)
                        .when_some(e.image.clone(), |d, im| d.child(img(im).size_full().object_fit(ObjectFit::Cover))),
                )
                .child(
                    v_flex()
                        .min_w_0()
                        .flex_1()
                        .text_sm()
                        .child(div().text_xs().text_color(t.muted_foreground).truncate().child(e.name.clone()))
                        .child(status),
                )
        });
        v_flex()
            .w(px(272.))
            .h_full()
            .flex_none()
            .bg(t.sidebar)
            .border_r_1()
            .border_color(t.border)
            .child(
                h_flex()
                    .px_3()
                    .pt_3()
                    .pb_2()
                    .justify_between()
                    .child(widgets::caption("Recent", cx))
                    .child(div().text_xs().text_color(t.muted_foreground).child(format!("{}", c.entries.len()))),
            )
            .child(
                div()
                    .relative()
                    .flex_1()
                    .min_h_0()
                    .child(
                        v_flex()
                            .id("recent")
                            .size_full()
                            .overflow_y_scroll()
                            .track_scroll(&c.list_scroll)
                            .px_2()
                            .pb_2()
                            .gap_1()
                            .children(rows)
                            .when(c.entries.is_empty(), |d| {
                                d.child(
                                    h_flex()
                                        .gap_2()
                                        .p_2()
                                        .items_center()
                                        .text_xs()
                                        .text_color(t.muted_foreground)
                                        .child(Icon::new(Lucide::Images).small())
                                        .child("Nothing classified yet."),
                                )
                            }),
                    )
                    .vertical_scrollbar(&c.list_scroll),
            )
    }

    fn render_picture(&self, entry_ix: Option<usize>, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let entry = entry_ix.map(|i| &self.classify.entries[i]);
        let failed = entry.and_then(|e| match self.model.as_ref().and_then(|m| e.results.get(m)) {
            Some(HeadResult::Failed(msg)) => Some(msg.clone()),
            _ => None,
        });
        let body = match entry.and_then(|e| e.image.clone()) {
            Some(im) => div().size_full().p_6().child(img(im).size_full().object_fit(ObjectFit::Contain)).into_any_element(),
            None if failed.is_some() => div()
                .size_full()
                .p_8()
                .child(
                    Empty::new().size_full().header(
                        EmptyHeader::new()
                            .media(EmptyMedia::new().child(Icon::new(Lucide::ImageOff).large().text_color(t.danger)))
                            .title(EmptyTitle::new().child("This picture could not be read"))
                            .description(EmptyDescription::new().child(
                                "Check the path or URL, or try another picture. The details are on the right.",
                            )),
                    ),
                )
                .into_any_element(),
            None if entry.is_some() => v_flex()
                .size_full()
                .items_center()
                .justify_center()
                .gap_2()
                .child(Spinner::new().large().color(t.muted_foreground))
                .child(div().text_sm().text_color(t.muted_foreground).child("Reading picture…"))
                .into_any_element(),
            None => div()
                .size_full()
                .p_8()
                .child(
                    Empty::new()
                        .size_full()
                        .border_2()
                        .border_dashed()
                        .border_color(theme::roles(cx).border_strong)
                        .rounded(t.radius_lg)
                        .bg(theme::roles(cx).surface)
                        .header(
                            EmptyHeader::new()
                                .media(EmptyMedia::new().child(Icon::new(Lucide::ImagePlus).large()))
                                .title(EmptyTitle::new().child("Drop component pictures here"))
                                .description(EmptyDescription::new().child(
                                    "Or open files, paste a picture, URL or path, or try the curated images in the Demo tab.",
                                )),
                        )
                        .child(
                            h_flex()
                                .gap_4()
                                .child(widgets::hint(&["ctrl-o"], "Open", cx))
                                .child(widgets::hint(&["ctrl-v"], "Paste", cx))
                                .child(widgets::hint(&["ctrl-2"], "Demo tab", cx)),
                        ),
                )
                .into_any_element(),
        };
        div().flex_1().min_w_0().h_full().bg(theme::roles(cx).canvas).child(body)
    }

    fn render_result(&self, entry_ix: Option<usize>, cx: &mut Context<Self>) -> impl IntoElement {
        let border = cx.theme().border;
        div()
            .id("result-panel")
            .w(px(400.))
            .h_full()
            .flex_none()
            .bg(theme::roles(cx).surface)
            .border_l_1()
            .border_color(border)
            .overflow_y_scroll()
            // block child keeps its natural height, so long content scrolls instead of being squeezed
            .child(div().flex_none().child(self.render_result_body(entry_ix, cx)))
    }

    fn render_result_body(&self, entry_ix: Option<usize>, cx: &mut Context<Self>) -> gpui_kit::Div {
        let t = cx.theme();
        let entry = entry_ix.map(|i| &self.classify.entries[i]);
        let mut panel = v_flex().gap_5().p_5();
        if let Some(m) = &self.model
            && let ModelState::Failed(msg) = self.state(m)
        {
            panel = panel.child(
                Alert::error("model-failed", SharedString::from(msg.clone()))
                    .title(format!("{} failed to load", self.title(m))),
            );
        }
        let Some(e) = entry else {
            return panel.child(
                v_flex()
                    .gap_2()
                    .pt_8()
                    .items_center()
                    .text_color(t.muted_foreground)
                    .child(Icon::new(Lucide::ScanSearch).large())
                    .child(div().text_sm().font_weight(FontWeight::MEDIUM).child("No picture yet"))
                    .child(
                        div().text_xs().text_center().child("The top class, its confidence and the top 3 appear here."),
                    ),
            );
        };
        panel = panel.child(
            v_flex()
                .gap_0p5()
                .child(div().text_base().font_weight(FontWeight::SEMIBOLD).truncate().child(e.name.clone()))
                .child(div().text_xs().text_color(t.muted_foreground).truncate().child(e.full.clone()))
                .when_some(e.dims, |d, (w, h)| {
                    d.child(div().text_xs().text_color(t.muted_foreground).child(format!("{w} × {h} px")))
                }),
        );
        let Some(m) = &self.model else { return panel };
        match e.results.get(m) {
            None | Some(HeadResult::Working(_)) => {
                panel.child(h_flex().gap_2().items_center().child(Spinner::new().color(t.primary)).child(
                    div().text_lg().child(if matches!(self.state(m), ModelState::Loading { .. }) {
                        format!("Loading {}…", self.short(m))
                    } else {
                        "Classifying…".to_string()
                    }),
                ))
            }
            Some(HeadResult::Failed(msg)) => panel.child(
                Alert::error(("classify-error", e.id as usize), SharedString::from(msg.clone()))
                    .title("Could not classify"),
            ),
            Some(HeadResult::Done(c, timing, wall)) => {
                let block = widgets::prediction_block(c, timing, &self.title(m), None, cx);
                panel.child(block).children(wall.map(|ms| {
                    div().text_xs().text_color(t.muted_foreground).child(format!("Total incl. read and decode: {ms:.0} ms"))
                }))
            }
        }
    }
}
