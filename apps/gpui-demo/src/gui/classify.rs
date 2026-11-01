//! Classify tab: the pictures the window gets (opened, dropped, pasted, fetched from a typed URL, or given on the
//! command line), the recent list, the picture and the result panel.

use std::sync::Arc;

use futures::StreamExt;
use gpui_kit::component::{
    ActiveTheme as _, Icon, IconName, Sizable as _, WindowExt as _,
    alert::Alert,
    button::{Button, ButtonVariants as _},
    empty::{Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle},
    h_flex,
    input::{Input, InputEvent, InputState},
    scroll::ScrollableElement as _,
    spinner::Spinner,
    v_flex,
};
use gpui_kit::{
    Context, Entity, FocusHandle, FontWeight, ObjectFit, RenderImage, ScrollHandle, SharedString, Subscription, Window,
    div, img, prelude::*, px,
};

use super::images::{DISPLAY_MAX_SIDE, render};
use super::models::{ModelEvent, ModelState, Models};
use super::results::{HeadResult, Results};
use super::widgets::Lucide;
use super::{Hint, Open, Paste, StepNext, StepPrev, Tab, theme, widgets};
use crate::app;
use inference::fetch::Source;
use inference::manifest::HeadId;
use inference::prof;
use inference::service::{ImageInput, Lane, Policy, Priority, RequestHandle, RunError, Ticket, Update};

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

pub struct ClassifyTab {
    models: Entity<Models>,
    lane: Lane,
    focus: FocusHandle,
    url: Entity<InputState>,
    entries: Vec<Entry>,
    selected: Option<u64>,
    next_id: u64,
    list_scroll: ScrollHandle,
    /// textures evicted from the recent list, released on the next frame
    to_drop: Vec<Arc<RenderImage>>,
    _subscriptions: [Subscription; 2],
}

impl Tab for ClassifyTab {
    const TITLE: &'static str = "Classify";
    const ICON: Lucide = Lucide::ScanSearch;
    const HINTS: &'static [Hint] =
        &[(&["ctrl-o"], "Open"), (&["ctrl-v"], "Paste"), (&["ctrl-l"], "URL"), (&["up", "down"], "Recent")];
    const TAKES_INPUTS: bool = true;

    fn new(models: Entity<Models>, _: &app::Shared, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let url =
            cx.new(|cx| InputState::new(window, cx).placeholder("Paste or type an image URL or file path, then Enter"));
        let enter = cx.subscribe_in(&url, window, |this, _, ev: &InputEvent, window, cx| {
            if let InputEvent::PressEnter { .. } = ev {
                this.fetch_typed_url(window, cx);
            }
        });
        // a retry of a failed graph: its heads' "failed to load" answers are stale
        let retries = cx.subscribe(&models, |this, _, ev, _| {
            if let ModelEvent::Retrying(heads) = ev {
                for e in &mut this.entries {
                    for h in heads {
                        e.results.forget_load_failure(h);
                    }
                }
            }
        });
        let lane = models.read(cx).inference().lane(Priority::Interactive, Policy::Keep);
        Self {
            models,
            lane,
            focus: cx.focus_handle(),
            url,
            entries: Vec::new(),
            selected: None,
            next_id: 1,
            list_scroll: ScrollHandle::new(),
            to_drop: Vec::new(),
            _subscriptions: [enter, retries],
        }
    }

    fn focus_handle(&self) -> FocusHandle {
        self.focus.clone()
    }

    /// The selected entry gets the selected model's answer when it has none yet: on submission, after a model
    /// switch, and when another entry is selected. Entries never submitted (they arrived while the start-up default
    /// was probed) get theirs too.
    fn follow_model(&mut self, cx: &mut Context<Self>) {
        let Some(model) = self.models.read(cx).run_model() else { return };
        let waiting = self.entries.iter().enumerate().filter(|(_, e)| e.tickets.is_empty()).map(|(i, _)| i);
        let todo: Vec<usize> = self.selected_ix().into_iter().chain(waiting).collect();
        for ix in todo {
            self.answer(ix, &model, cx);
        }
    }

    fn take(&mut self, inputs: Vec<ImageInput>, window: &mut Window, cx: &mut Context<Self>) {
        for input in inputs {
            if let ImageInput::Source(Source::Url(u)) = &input {
                let u = u.clone();
                self.url.update(cx, |st, cx| st.set_value(u, window, cx));
            }
            self.submit(input, cx);
        }
    }

    fn focus_input(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.url.update(cx, |st, cx| st.focus(window, cx));
    }
}

impl ClassifyTab {
    /// Something is classifying (the spinners animate themselves; the tests wait for this).
    #[cfg(test)]
    pub fn busy(&self) -> bool {
        self.entries.iter().any(|e| e.results.working())
    }

    fn selected_ix(&self) -> Option<usize> {
        self.selected.and_then(|id| self.entries.iter().position(|e| e.id == id))
    }

    /// The selected entry has `head`'s answer, its wall time and its display picture.
    #[cfg(test)]
    pub fn answered(&self, head: &HeadId) -> bool {
        self.selected_ix()
            .map(|i| &self.entries[i])
            .is_some_and(|e| matches!(e.results.get(head), Some(HeadResult::Done(_, _, Some(_)))) && e.image.is_some())
    }

    /// How many entries have `head`'s answer and their display picture.
    #[cfg(test)]
    pub fn answered_entries(&self, head: &HeadId) -> usize {
        self.entries
            .iter()
            .filter(|e| matches!(e.results.get(head), Some(HeadResult::Done(_, _, Some(_)))) && e.image.is_some())
            .count()
    }

    /// The short status the Recent list shows for the selected entry's failed answer from `head`.
    #[cfg(test)]
    pub fn failure(&self, head: &HeadId) -> Option<&'static str> {
        match self.entries[self.selected_ix()?].results.get(head) {
            Some(HeadResult::Failed(e)) => Some(widgets::failure_label(e)),
            _ => None,
        }
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    fn submit(&mut self, input: ImageInput, cx: &mut Context<Self>) {
        if self.models.read(cx).selected().is_none() {
            return;
        }
        let (name, full) = match &input {
            ImageInput::Source(src) => (src.display_name(), src.full()),
            ImageInput::Bytes(_) | ImageInput::Decoded(_) => ("pasted image".to_string(), "clipboard".to_string()),
        };
        let id = self.next_id;
        self.next_id += 1;
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
        self.entries.insert(0, entry);
        while self.entries.len() > MAX_ENTRIES {
            if let Some(e) = self.entries.pop() {
                self.to_drop.extend(e.image);
            }
        }
        self.selected = Some(id);
        self.list_scroll.scroll_to_item(0);
        self.follow_model(cx);
    }

    /// Entry `ix` gets `model`'s answer unless it has one or is waiting for it.
    fn answer(&mut self, ix: usize, model: &HeadId, cx: &mut Context<Self>) {
        let token = self.next_id;
        let e = &mut self.entries[ix];
        if e.results.get(model).is_some() {
            return;
        }
        self.next_id += 1;
        let id = e.id;
        let display = e.image.is_none().then_some(DISPLAY_MAX_SIDE);
        let RequestHandle { updates: mut rx, ticket } =
            self.lane.submit(e.input.clone(), std::slice::from_ref(model), display);
        e.tickets.push(ticket);
        e.results.start(model.clone(), token);
        let model = model.clone();
        cx.spawn(async move |this, cx| {
            while let Some(update) = rx.next().await {
                let alive = this
                    .update(cx, |this, cx| {
                        if let Some(e) = this.entries.iter_mut().find(|e| e.id == id) {
                            match update {
                                Update::Decoded { size, display } => {
                                    e.dims = Some(size);
                                    // two requests (a model switch before the first decode) both carry a copy;
                                    // keep the first, so no painted texture is replaced without drop_image
                                    if e.image.is_none() {
                                        e.image = display.map(render);
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
        let text = self.url.read(cx).value().to_string();
        match Source::parse(&text) {
            Some(src) => self.submit(ImageInput::Source(src), cx),
            None => window.push_notification("Type or paste an image URL (or a file path) first.", cx),
        }
    }

    fn step(&mut self, delta: isize, cx: &mut Context<Self>) {
        if self.entries.is_empty() {
            return;
        }
        let cur = self.selected_ix().unwrap_or(0) as isize;
        let next = (cur + delta).clamp(0, self.entries.len() as isize - 1) as usize;
        self.selected = Some(self.entries[next].id);
        self.list_scroll.scroll_to_item(next);
        self.follow_model(cx);
        cx.notify();
    }

    // ------------------------------------------------------------------------------------------ render
    fn render_body(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let entry_ix = self.selected_ix();
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
                    .on_click(|_, window, cx| window.dispatch_action(Box::new(Open), cx)),
            )
            .child(
                div().flex_1().min_w_0().child(
                    Input::new(&self.url)
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
                    .on_click(|_, window, cx| window.dispatch_action(Box::new(Paste), cx)),
            )
    }

    fn render_recent(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let c = self;
        let model = self.models.read(cx).selected();
        let rows = c.entries.iter().map(|e| {
            let id = e.id;
            let sel = c.selected == Some(id);
            let status = match model.and_then(|m| e.results.get(m)) {
                None => div().text_color(t.muted_foreground).child("not classified yet").into_any_element(),
                Some(HeadResult::Working(_)) => h_flex()
                    .gap_1()
                    .items_center()
                    .child(Spinner::new().xsmall().color(t.muted_foreground))
                    .child(div().text_color(t.muted_foreground).child("classifying…"))
                    .into_any_element(),
                Some(HeadResult::Failed(e)) => h_flex()
                    .gap_1()
                    .items_center()
                    .text_color(t.danger)
                    .child(Icon::new(IconName::CircleAlert).xsmall())
                    .child(widgets::failure_label(e))
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
                    this.selected = Some(id);
                    this.follow_model(cx);
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
        let entry = entry_ix.map(|i| &self.entries[i]);
        let model = self.models.read(cx).selected();
        let unreadable = matches!(
            entry.and_then(|e| model.and_then(|m| e.results.get(m))),
            Some(HeadResult::Failed(RunError::Read(_) | RunError::Decode(_)))
        );
        let body = match entry.and_then(|e| e.image.clone()) {
            Some(im) => div().size_full().p_6().child(img(im).size_full().object_fit(ObjectFit::Contain)).into_any_element(),
            None if unreadable => div()
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
        let models = self.models.read(cx);
        let entry = entry_ix.map(|i| &self.entries[i]);
        let mut panel = v_flex().gap_5().p_5();
        let Some(e) = entry else {
            // with a picture on screen, its answer from the failed model says this instead
            if let Some(m) = models.selected()
                && let ModelState::Failed(msg) = models.state(m)
            {
                panel = panel.child(
                    Alert::error("model-failed", SharedString::from(msg)).title(format!("{} failed to load", models.title(m))),
                );
            }
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
        let Some(m) = models.selected() else { return panel };
        match e.results.get(m) {
            None | Some(HeadResult::Working(_)) => {
                panel.child(h_flex().gap_2().items_center().child(Spinner::new().color(t.primary)).child(
                    div().text_lg().child(if matches!(models.state(m), ModelState::Loading { .. }) {
                        format!("Loading {}…", models.short(m))
                    } else {
                        "Classifying…".to_string()
                    }),
                ))
            }
            Some(HeadResult::Failed(err)) => {
                panel.child(widgets::failure_alert(("classify-error", e.id as usize), err, &models.title(m)))
            }
            Some(HeadResult::Done(c, timing, wall)) => {
                let block = widgets::prediction_block(c, timing, &models.title(m), None, cx);
                panel.child(block).children(wall.map(|ms| {
                    div().text_xs().text_color(t.muted_foreground).child(format!("Total incl. read and decode: {ms:.0} ms"))
                }))
            }
        }
    }
}

impl Render for ClassifyTab {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        for im in self.to_drop.drain(..) {
            window.drop_image(im).ok();
        }
        div()
            .id("classify-tab")
            .track_focus(&self.focus)
            .key_context("Classify")
            .on_action(cx.listener(|this, _: &StepPrev, _, cx| this.step(-1, cx)))
            .on_action(cx.listener(|this, _: &StepNext, _, cx| this.step(1, cx)))
            .size_full()
            .child(prof!("gui::build_tab_elements", self.render_body(window, cx)))
    }
}
