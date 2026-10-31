//! Demo tab: the curated stack as a sectioned thumbnail grid, a detail pane with the result against the true
//! label and the photo's attribution, "Run all" with progress and an accuracy summary, keyboard stepping.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::StreamExt;
use futures::channel::mpsc as fmpsc;
use gpui_kit::component::{
    ActiveTheme as _, Disableable as _, Icon, IconName, Sizable as _,
    alert::Alert,
    button::{Button, ButtonVariants as _},
    h_flex,
    scroll::ScrollableElement as _,
    spinner::Spinner,
    tag::Tag,
    tooltip::Tooltip,
    v_flex,
};
use gpui_kit::{
    AnyElement, Context, FontWeight, ObjectFit, RenderImage, ScrollHandle, SharedString, TestSupportExt as _, Window,
    div, img, prelude::*, px,
};
use image::RgbImage;

use super::worker::{self, Cmd, Job, JobInput, Update};
use super::{DemoApp, theme, widgets};
use crate::classification::{Classification, HeadOutput};
use crate::demo::{DemoStack, Verdict};
use crate::manifest::HeadId;
use crate::models::Timing;

/// Cards are at least this wide; the grid fits as many columns as the window allows and widens them evenly.
const CARD_MIN_W: f32 = 176.;
const GRID_GAP: f32 = 12.;
const GRID_PAD: f32 = 16.;
const DETAIL_W: f32 = 440.;

pub enum ItemResult {
    Working,
    Done(Classification, Timing),
    Failed(String),
}

#[derive(Default)]
pub struct Item {
    pub thumb: Option<Arc<RenderImage>>,
    pub full: Option<Arc<RenderImage>>,
    pub rgb: Option<Arc<RgbImage>>,
    pub load_error: Option<String>,
    pub results: HashMap<HeadId, ItemResult>,
}

struct RunState {
    model: HeadId,
    queue: VecDeque<usize>,
    total: usize,
    cancel: Arc<AtomicBool>,
}

pub struct DemoState {
    pub stack: Option<Result<DemoStack, String>>,
    pub items: Vec<Item>,
    pub selected: usize,
    run: Option<RunState>,
    scroll: ScrollHandle,
}

impl Default for DemoState {
    fn default() -> Self {
        Self { stack: None, items: Vec::new(), selected: 0, run: None, scroll: ScrollHandle::new() }
    }
}

impl DemoState {
    pub fn running(&self) -> bool {
        self.run.is_some() || self.items.iter().any(|i| i.results.values().any(|r| matches!(r, ItemResult::Working)))
    }

    pub fn stack(&self) -> Option<&DemoStack> {
        self.stack.as_ref().and_then(|s| s.as_ref().ok())
    }

    /// Child index of image `i` in the flattened grid (one header element precedes each section).
    fn child_index(&self, i: usize) -> usize {
        let Some(s) = self.stack() else { return i };
        let sec = s.sections.iter().position(|x| *x == s.images[i].section).unwrap_or(0);
        i + sec + 1
    }

    /// (correct, judged, unsure) for `model` over the images classified so far.
    pub fn summary(&self, model: Option<&HeadId>) -> (usize, usize, usize) {
        let (Some(s), Some(model)) = (self.stack(), model) else { return (0, 0, 0) };
        let (mut ok, mut n, mut unsure) = (0, 0, 0);
        for (im, it) in s.images.iter().zip(&self.items) {
            if let Some(ItemResult::Done(c, _)) = it.results.get(model) {
                n += 1;
                match im.judge(c.best_label(), c.abstained()) {
                    Verdict::Correct => ok += 1,
                    Verdict::Unsure => unsure += 1,
                    Verdict::Wrong => {}
                }
            }
        }
        (ok, n, unsure)
    }
}

impl DemoApp {
    pub(super) fn demo_init(&mut self, dir: Option<PathBuf>, cx: &mut Context<Self>) {
        let stack = crate::demo::locate(dir.as_deref()).and_then(|d| DemoStack::load(&d)).map_err(|e| format!("{e:#}"));
        if let Ok(s) = &stack {
            eprintln!("[gpui-demo] demo stack: {} images in {}", s.images.len(), s.dir.display());
            self.demo.items = (0..s.images.len()).map(|_| Item::default()).collect();
            let paths: Vec<PathBuf> = (0..s.images.len()).map(|i| s.path(i)).collect();
            let mut rx = worker::load_thumbnails(paths);
            cx.spawn(async move |this, cx| {
                while let Some((i, r)) = rx.next().await {
                    let alive = this
                        .update(cx, |this, cx| {
                            if let Some(it) = this.demo.items.get_mut(i) {
                                match r {
                                    Ok(d) => {
                                        it.thumb = Some(d.thumb);
                                        it.full = Some(d.full);
                                        it.rgb = Some(d.rgb);
                                    }
                                    Err(e) => it.load_error = Some(e),
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
        } else if let Err(e) = &stack {
            eprintln!("[gpui-demo] demo stack unavailable: {e}");
        }
        self.demo.stack = Some(stack);
    }

    /// Classify image `i` with `model`. `advance`: continue the running "Run all" when it finishes.
    fn demo_classify(
        &mut self,
        i: usize,
        model: HeadId,
        advance: bool,
        cancel: Arc<AtomicBool>,
        cx: &mut Context<Self>,
    ) {
        let Some(stack) = self.demo.stack() else { return };
        let path = stack.path(i);
        let Some(it) = self.demo.items.get_mut(i) else { return };
        let input = match &it.rgb {
            Some(rgb) => JobInput::Decoded(rgb.clone()),
            None => JobInput::Source(crate::fetch::Source::File(path)),
        };
        let (tx, mut rx) = fmpsc::unbounded();
        let job = Job { input, models: vec![model.clone()], display: false, cancel, updates: tx };
        if self.jobs.send(Cmd::Job(job)).is_err() {
            it.results.insert(model, ItemResult::Failed("inference thread is not running".into()));
            return;
        }
        it.results.insert(model.clone(), ItemResult::Working);
        cx.spawn(async move |this, cx| {
            while let Some(u) = rx.next().await {
                let alive = this
                    .update(cx, |this, cx| {
                        if let Some(it) = this.demo.items.get_mut(i) {
                            match u {
                                Update::Result(k, Ok((HeadOutput::Classify(c), timing))) => {
                                    it.results.insert(k, ItemResult::Done(c, timing));
                                }
                                Update::Result(k, Err(e)) => {
                                    it.results.insert(k, ItemResult::Failed(e));
                                }
                                Update::Failed(e) => {
                                    it.results.insert(model.clone(), ItemResult::Failed(e));
                                }
                                Update::Image(..) => {}
                                Update::Done(_) => {
                                    if matches!(it.results.get(&model), Some(ItemResult::Working)) {
                                        it.results.remove(&model); // cancelled before it ran
                                    }
                                    if advance {
                                        this.demo_next(cx);
                                    }
                                }
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

    /// Item `it`'s result for the selected head.
    fn selected_result<'a>(&self, it: &'a Item) -> Option<&'a ItemResult> {
        self.model.as_ref().and_then(|m| it.results.get(m))
    }

    pub(super) fn demo_classify_selected(&mut self, cx: &mut Context<Self>) {
        if self.demo.stack().is_none() {
            return;
        }
        let i = self.demo.selected;
        let Some(model) = self.model.clone() else { return };
        if matches!(self.demo.items.get(i).and_then(|it| it.results.get(&model)), Some(ItemResult::Working)) {
            return;
        }
        self.demo_classify(i, model, false, Arc::new(AtomicBool::new(false)), cx);
    }

    pub(super) fn demo_run_all(&mut self, cx: &mut Context<Self>) {
        let Some(n) = self.demo.stack().map(|s| s.images.len()) else { return };
        if self.demo.run.is_some() {
            return self.demo_cancel(cx);
        }
        let Some(model) = self.model.clone() else { return };
        for it in &mut self.demo.items {
            it.results.remove(&model);
        }
        self.demo.run =
            Some(RunState { model, queue: (0..n).collect(), total: n, cancel: Arc::new(AtomicBool::new(false)) });
        self.demo_next(cx);
    }

    fn demo_next(&mut self, cx: &mut Context<Self>) {
        let Some(run) = self.demo.run.as_mut() else { return };
        if run.cancel.load(Ordering::Relaxed) {
            self.demo.run = None;
            return cx.notify();
        }
        match run.queue.pop_front() {
            Some(i) => {
                let (model, cancel) = (run.model.clone(), run.cancel.clone());
                self.demo.selected = i;
                let ci = self.demo.child_index(i);
                self.demo.scroll.scroll_to_item(ci);
                self.demo_classify(i, model, true, cancel, cx);
            }
            None => {
                self.demo.run = None;
                cx.notify();
            }
        }
    }

    pub(super) fn demo_cancel(&mut self, cx: &mut Context<Self>) {
        if let Some(run) = self.demo.run.take() {
            run.cancel.store(true, Ordering::Relaxed);
            cx.notify();
        }
    }

    pub(super) fn demo_step(&mut self, delta: isize, _window: &mut Window, cx: &mut Context<Self>) {
        let n = self.demo.items.len();
        if n == 0 {
            return;
        }
        self.demo.selected = (self.demo.selected as isize + delta).rem_euclid(n as isize) as usize;
        let ci = self.demo.child_index(self.demo.selected);
        self.demo.scroll.scroll_to_item(ci);
        cx.notify();
    }

    // ------------------------------------------------------------------------------------------ render
    pub(super) fn render_demo(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = cx.theme();
        match &self.demo.stack {
            None => return div().into_any_element(),
            Some(Err(e)) => {
                return div()
                    .p_6()
                    .child(
                        Alert::warning("demo-missing", SharedString::from(e.clone()))
                            .title("Demo stack not found")
                            .icon(IconName::Info),
                    )
                    .child(div().pt_3().text_sm().text_color(t.muted_foreground).child(
                        "Run the app from inside the repository, or pass --demo-dir apps/gpui-demo/demo_stack \
                         (or set COMPCLASS_DEMO_DIR).",
                    ))
                    .into_any_element();
            }
            Some(Ok(_)) => {}
        }
        v_flex()
            .size_full()
            .child(self.render_demo_header(cx))
            .child(
                h_flex()
                    .flex_1()
                    .min_h_0()
                    .items_start()
                    .child(self.render_grid(window, cx))
                    .child(self.render_detail(cx)),
            )
            .into_any_element()
    }

    fn render_demo_header(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let r = theme::roles(cx);
        let n = self.demo.items.len();
        let (ok, judged, _) = self.demo.summary(self.model.as_ref());
        let short = self.model.as_ref().map(|m| self.short(m)).unwrap_or_default();
        let running = self.demo.run.as_ref();
        let progress = running.map(|run| {
            let done = (run.total - run.queue.len()).saturating_sub(1); // one image is in flight
            h_flex()
                .gap_2()
                .items_center()
                .child(div().w(px(200.)).child(widgets::bar(
                    done as f32 / run.total.max(1) as f32,
                    r.accent,
                    px(6.),
                    &r,
                )))
                .child(div().text_sm().text_color(t.muted_foreground).child(format!("{done}/{}", run.total)))
        });
        let summary = (judged > 0).then(|| {
            h_flex()
                .gap_2()
                .items_baseline()
                .child(div().text_3xl().font_weight(FontWeight::BOLD).child(format!("{ok}/{judged}")))
                .child(div().text_sm().text_color(t.muted_foreground).child(if judged < n {
                    format!("correct · {judged} of {n} classified · {short}")
                } else {
                    format!("correct · {short}")
                }))
        });
        h_flex()
            .justify_between()
            .items_center()
            .gap_4()
            .px_4()
            .py_3()
            .bg(r.chrome)
            .border_b_1()
            .border_color(t.border)
            .child(
                v_flex()
                    .child(div().text_lg().font_weight(FontWeight::SEMIBOLD).child("Demo stack"))
                    .child(div().text_xs().text_color(t.muted_foreground).child(format!(
                        "{n} Wikimedia Commons photos never used for training or model selection · click a card or press Enter"
                    ))),
            )
            .child(
                h_flex()
                    .gap_4()
                    .items_center()
                    .children(summary)
                    .children(progress)
                    .child(if running.is_some() {
                        Button::new("demo-run")
                            .danger()
                            .outline()
                            .icon(IconName::Square)
                            .label("Stop")
                            .tooltip("Stop after the current image (Esc)")
                            .on_click(cx.listener(|this, _, _, cx| this.demo_cancel(cx)))
                    } else {
                        Button::new("demo-run")
                            .primary()
                            .icon(IconName::Play)
                            .label("Run all")
                            .tooltip("Classify every image in order (Ctrl+R)")
                            .on_click(cx.listener(|this, _, _, cx| this.demo_run_all(cx)))
                    }),
            )
    }

    fn render_grid(&self, window: &Window, cx: &mut Context<Self>) -> impl IntoElement {
        let muted_fg = cx.theme().muted_foreground;
        // even columns: as many as fit at CARD_MIN_W, sharing the surplus (scrollbar gutter reserved)
        let avail = f32::from(window.viewport_size().width) - DETAIL_W - 2.0 * GRID_PAD - 12.0;
        let cols = ((avail + GRID_GAP) / (CARD_MIN_W + GRID_GAP)).floor().max(1.0);
        let card_w = ((avail - GRID_GAP * (cols - 1.0)) / cols).floor().max(CARD_MIN_W);
        let Some(stack) = self.demo.stack() else { return div().into_any_element() };
        let mut children: Vec<AnyElement> = Vec::new();
        for (si, section) in stack.sections.iter().enumerate() {
            let idx = stack.section_indices(section);
            let (ok, n) = idx.iter().fold((0, 0), |(ok, n), &i| match self.selected_result(&self.demo.items[i]) {
                Some(ItemResult::Done(c, _)) => {
                    (ok + usize::from(stack.images[i].judge(c.best_label(), c.abstained()) == Verdict::Correct), n + 1)
                }
                _ => (ok, n),
            });
            children.push(
                h_flex()
                    .w_full()
                    .justify_between()
                    .items_baseline()
                    .when(si > 0, |d| d.pt_5())
                    .child(div().text_base().font_weight(FontWeight::SEMIBOLD).child(section.clone()))
                    .child(div().text_xs().text_color(muted_fg).child(if n > 0 {
                        format!("{ok}/{n} correct · {} images", idx.len())
                    } else {
                        format!("{} images", idx.len())
                    }))
                    .into_any_element(),
            );
            for i in idx {
                children.push(self.render_card(i, card_w, cx).into_any_element());
            }
        }
        div()
            .relative()
            .flex_1()
            .min_w_0()
            .h_full()
            .bg(theme::roles(cx).canvas)
            .child(
                div()
                    .id("demo-grid")
                    .test_support()
                    .size_full()
                    .overflow_y_scroll()
                    .track_scroll(&self.demo.scroll)
                    .p(px(GRID_PAD))
                    .flex()
                    .flex_row()
                    .flex_wrap()
                    .content_start()
                    .gap(px(GRID_GAP))
                    .children(children),
            )
            .vertical_scrollbar(&self.demo.scroll)
            .into_any_element()
    }

    fn render_card(&self, i: usize, card_w: f32, cx: &mut Context<Self>) -> impl IntoElement {
        let thumb_h = (card_w * 0.72).round();
        let t = cx.theme();
        let r = theme::roles(cx);
        let stack = self.demo.stack().expect("stack loaded");
        let im = &stack.images[i];
        let it = &self.demo.items[i];
        let sel = self.demo.selected == i;
        let verdict = match self.selected_result(it) {
            Some(ItemResult::Done(c, _)) => Some(im.judge(c.best_label(), c.abstained())),
            _ => None,
        };
        let result: AnyElement = match self.selected_result(it) {
            Some(ItemResult::Working) => h_flex()
                .gap_1()
                .items_center()
                .child(Spinner::new().xsmall().color(t.muted_foreground))
                .child(div().text_xs().text_color(t.muted_foreground).child("classifying…"))
                .into_any_element(),
            Some(ItemResult::Done(c, _)) => {
                let v = im.judge(c.best_label(), c.abstained());
                let text = if c.abstained() { "unsure".to_string() } else { c.best_label().to_string() };
                h_flex()
                    .gap_1()
                    .items_center()
                    .min_w_0()
                    .text_xs()
                    .child(Icon::new(widgets::verdict_icon(v)).xsmall().text_color(r.verdict(v).0))
                    .child(div().flex_1().min_w_0().truncate().text_color(r.fg).child(text))
                    .child(div().flex_none().text_color(r.muted_fg).child(widgets::pct(c.best().1)))
                    .into_any_element()
            }
            Some(ItemResult::Failed(_)) => h_flex()
                .gap_1()
                .items_center()
                .text_xs()
                .text_color(t.danger)
                .child(Icon::new(IconName::CircleAlert).xsmall())
                .child("error")
                .into_any_element(),
            None => div().text_xs().text_color(t.muted_foreground).child("not classified yet").into_any_element(),
        };
        let credit = im.credit();
        let tip = format!("{}\n{}\n{}", im.title, credit, im.source_url);
        v_flex()
            .id(("demo-card", i))
            .w(px(card_w))
            .rounded(t.radius_lg)
            .border_1()
            .border_color(if sel { r.accent } else { r.border })
            .when(sel, |d| d.border_2())
            .bg(r.surface)
            .overflow_hidden()
            .cursor_pointer()
            .hover(|s| s.border_color(r.accent.opacity(0.6)))
            .tooltip(move |window, cx| Tooltip::new(tip.clone()).build(window, cx))
            .on_click(cx.listener(move |this, _, _, cx| {
                this.demo.selected = i;
                this.demo_classify_selected(cx);
            }))
            .child(div().w_full().h(px(thumb_h)).overflow_hidden().bg(t.muted).child(match &it.thumb {
                Some(th) => img(th.clone()).size_full().object_fit(ObjectFit::Cover).into_any_element(),
                None => div().size_full().into_any_element(),
            }))
            // verdict strip: colour echo of the icon below (never the only cue)
            .child(div().w_full().h(px(3.)).bg(match verdict {
                Some(v) => r.verdict(v).0,
                None => r.border,
            }))
            .child(
                v_flex()
                    .gap_1()
                    .px_2()
                    .py_2()
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::SEMIBOLD)
                            .truncate()
                            .when(im.true_label.is_none(), |d| d.text_color(t.muted_foreground))
                            .child(im.true_text().to_string()),
                    )
                    .child(result)
                    .child(div().text_xs().text_color(t.muted_foreground).truncate().child(credit)),
            )
    }

    fn render_detail(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let r = theme::roles(cx);
        let stack = self.demo.stack().expect("stack loaded");
        let i = self.demo.selected.min(stack.images.len() - 1);
        let im = &stack.images[i];
        let it = &self.demo.items[i];
        let picture = div().w_full().h(px(280.)).rounded(t.radius_lg).bg(r.canvas).p_2().overflow_hidden().child(
            match &it.full {
                Some(full) => img(full.clone()).size_full().object_fit(ObjectFit::Contain).into_any_element(),
                None => div().size_full().into_any_element(),
            },
        );
        let heading = v_flex()
            .gap_1()
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(Tag::secondary().small().rounded_full().child(im.section.clone()))
                    .child(div().text_xs().text_color(t.muted_foreground).child(format!(
                        "{} / {}",
                        i + 1,
                        stack.images.len()
                    ))),
            )
            .child(div().text_xl().font_weight(FontWeight::BOLD).child(format!("True label: {}", im.true_text())))
            .when(!im.note.is_empty(), |d| {
                d.child(div().text_sm().text_color(t.muted_foreground).child(im.note.clone()))
            })
            .when_some(im.model_pred.clone().zip(im.model_conf), |d, (p, c)| {
                d.child(
                    div()
                        .text_xs()
                        .text_color(t.muted_foreground)
                        .child(format!("Reference prediction in demo_stack.json: {p} {}", widgets::pct(c))),
                )
            });
        let short = self.model.as_ref().map(|m| self.short(m)).unwrap_or_default();
        let result: AnyElement = match (self.selected_result(it), &self.model) {
            (Some(ItemResult::Done(c, timing)), Some(m)) => {
                let v = im.judge(c.best_label(), c.abstained());
                widgets::prediction_block(c, timing, &self.title(m), Some((v, im.true_text())), cx)
            }
            (Some(ItemResult::Working), _) => h_flex()
                .gap_2()
                .items_center()
                .py_4()
                .child(Spinner::new().color(t.primary))
                .child(div().text_lg().child(format!("Classifying with {short}…")))
                .into_any_element(),
            (Some(ItemResult::Failed(e)), _) => {
                Alert::error("demo-error", SharedString::from(e.clone())).title("Could not classify").into_any_element()
            }
            _ => v_flex()
                .gap_2()
                .items_start()
                .child(
                    Button::new("demo-classify")
                        .primary()
                        .large()
                        .icon(IconName::Play)
                        .label(format!("Classify with {short}"))
                        .on_click(cx.listener(|this, _, _, cx| this.demo_classify_selected(cx))),
                )
                .child(widgets::hint(&["enter"], "classify · ← → step through the stack", cx))
                .into_any_element(),
        };
        let (src, lic) = (im.source_url.clone(), im.licence_url.clone());
        let attribution = v_flex()
            .gap_1()
            .pt_3()
            .border_t_1()
            .border_color(t.border)
            .child(widgets::caption("Photo credit", cx))
            .child(div().text_sm().child(im.title.clone()))
            .child(div().text_xs().text_color(t.muted_foreground).child(format!("by {}", im.author)))
            .child(
                h_flex()
                    .gap_3()
                    .child(
                        Button::new("demo-licence")
                            .link()
                            .xsmall()
                            .label(im.licence.clone())
                            .disabled(lic.is_empty())
                            .on_click(move |_, _, cx| cx.open_url(&lic)),
                    )
                    .child(
                        Button::new("demo-source")
                            .link()
                            .xsmall()
                            .icon(IconName::ExternalLink)
                            .label("Wikimedia Commons")
                            .on_click(move |_, _, cx| cx.open_url(&src)),
                    ),
            );
        div()
            .id("demo-detail")
            .w(px(DETAIL_W))
            .h_full()
            .flex_none()
            .bg(r.surface)
            .border_l_1()
            .border_color(t.border)
            .overflow_y_scroll()
            // block child: keeps its natural height (a flex column would shrink the picture instead of scrolling)
            .child(
                div()
                    .flex_none()
                    .p_4()
                    .child(v_flex().gap_4().child(picture.flex_none()).child(heading).child(result).child(attribution)),
            )
    }
}
