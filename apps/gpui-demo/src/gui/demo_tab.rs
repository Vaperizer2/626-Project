//! Demo tab: the curated stack as a sectioned thumbnail grid, a detail pane with the result against the true
//! label and the photo's attribution, "Run all" with progress and an accuracy summary, keyboard stepping.

use std::collections::HashMap;

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
    AnyElement, Context, Entity, FocusHandle, FontWeight, ObjectFit, ScrollHandle, SharedString, Subscription,
    TestSupportExt as _, Window, div, img, prelude::*, px,
};

use super::models::{Models, on_retry};
use super::widgets::Lucide;
use super::results::HeadResult;
use super::stack::{Item, Stack, tab_header, verdict_colors, verdict_icon, verdict_tag};
use super::{Activate, Cancel, Hint, RunAll, StepNext, StepPrev, Tab, theme, widgets};
use crate::app;
use crate::demo::Verdict;
use inference::prof;
use inference::service::{Lane, Policy, Priority, RequestHandle, Ticket, Update};
use inference::manifest::HeadId;

/// Cards are at least this wide; the grid fits as many columns as the window allows and widens them evenly.
const CARD_MIN_W: f32 = 176.;
const GRID_GAP: f32 = 12.;
const GRID_PAD: f32 = 16.;
const DETAIL_W: f32 = 440.;

/// "Run all": the whole stack was submitted at once; dropping the run cancels what has not run yet.
struct RunState {
    token: u64,
    total: usize,
    done: usize,
    _tickets: Vec<Ticket>,
}

pub struct DemoTab {
    models: Entity<Models>,
    pub(super) stack: Entity<Stack>,
    /// single classifications
    interactive: Lane,
    /// "Run all"
    bulk: Lane,
    focus: FocusHandle,
    pub selected: usize,
    run: Option<RunState>,
    next_token: u64,
    scroll: ScrollHandle,
    /// the latest single classification of each image (a new one replaces and cancels it)
    single: HashMap<usize, Ticket>,
    _retries: Subscription,
}

/// Child index of image `i` in the flattened grid (one header element precedes each section).
fn child_index(stack: &Stack, i: usize) -> usize {
    let Some(s) = stack.stack() else { return i };
    let sec = s.sections.iter().position(|x| *x == s.images[i].section).unwrap_or(0);
    i + sec + 1
}

/// (correct, judged, unsure) for `model` over the images classified so far.
fn summary(stack: &Stack, model: Option<&HeadId>) -> (usize, usize, usize) {
    let (Some(s), Some(model)) = (stack.stack(), model) else { return (0, 0, 0) };
    let (mut ok, mut n, mut unsure) = (0, 0, 0);
    for (im, it) in s.images.iter().zip(&stack.items) {
        if let Some(HeadResult::Done(c, ..)) = it.results.get(model) {
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

/// Item `it`'s result for the selected head `model`.
fn selected_result<'a>(model: Option<&HeadId>, it: &'a Item) -> Option<&'a HeadResult> {
    model.and_then(|m| it.results.get(m))
}

impl Tab for DemoTab {
    const TITLE: &'static str = "Demo";
    const ICON: Lucide = Lucide::LayoutGrid;
    const HINTS: &'static [Hint] =
        &[(&["left", "right"], "Step"), (&["enter"], "Classify"), (&["ctrl-r"], "Run all"), (&["escape"], "Stop")];

    fn new(models: Entity<Models>, shared: &app::Shared, _: &mut Window, cx: &mut Context<Self>) -> Self {
        let stack = shared.stack.clone();
        let inference = models.read(cx).inference();
        let (interactive, bulk) =
            (inference.lane(Priority::Interactive, Policy::Keep), inference.lane(Priority::Bulk, Policy::Keep));
        let retries = on_retry(&models, cx, |this, heads, cx| {
            this.stack.update(cx, |s, _| {
                for it in &mut s.items {
                    it.results.forget_load_failures(heads);
                }
            });
        });
        Self {
            models,
            stack,
            interactive,
            bulk,
            focus: cx.focus_handle(),
            selected: 0,
            run: None,
            next_token: 1,
            scroll: ScrollHandle::new(),
            single: HashMap::new(),
            _retries: retries,
        }
    }

    fn focus_handle(&self) -> FocusHandle {
        self.focus.clone()
    }

    /// Submit the whole stack at once; the graph takes the images as fast as it can (batched when it allows).
    fn run(&mut self, cx: &mut Context<Self>) {
        let Some(n) = self.stack.read(cx).stack().map(|s| s.images.len()) else { return };
        if self.run.is_some() {
            return self.cancel(cx);
        }
        let Some(model) = self.models.read(cx).run_model() else { return };
        // each submission's `start` replaces the card's earlier answer from this model
        let token = self.next_token;
        let tickets = (0..n).filter_map(|i| self.submit(i, model.clone(), Priority::Bulk, cx)).map(|t| t.1).collect();
        // the run counts the requests submitted from `token` on
        self.run = Some(RunState { token, total: n, done: 0, _tickets: tickets });
        cx.notify();
    }

    /// After a model switch: a card on screen that another model answered gets the selected model's answer too.
    fn follow_model(&mut self, cx: &mut Context<Self>) {
        let Some(it) = self.stack.read(cx).items.get(self.selected) else { return };
        let models = self.models.read(cx);
        let answered = models.head_ids().iter().any(|h| it.results.get(h).is_some());
        if answered && selected_result(models.selected(), it).is_none() {
            self.classify_selected(cx);
        }
    }
}

impl DemoTab {
    /// "Run all" is under way.
    #[cfg(test)]
    pub fn running(&self) -> bool {
        self.run.is_some()
    }

    /// Submit image `i` for `model`; its card shows the result. Bulk submissions count towards the running
    /// "Run all" with `token`.
    fn submit(&mut self, i: usize, model: HeadId, priority: Priority, cx: &mut Context<Self>) -> Option<(u64, Ticket)> {
        let input = self.stack.read(cx).input(i)?;
        let token = self.next_token;
        self.next_token += 1;
        let lane = if priority == Priority::Bulk { &self.bulk } else { &self.interactive };
        let RequestHandle { updates, ticket } = lane.submit(input, std::slice::from_ref(&model), None);
        self.stack.update(cx, |s, _| s.items[i].results.start(model.clone(), token));
        super::pump(cx, updates, move |this, u, cx| {
            let finished = matches!(u, Update::Done(_));
            let applied = this.stack.update(cx, |s, cx| {
                let Some(it) = s.items.get_mut(i) else { return false };
                it.results.apply(&model, token, u);
                cx.notify();
                true
            });
            if applied && finished {
                this.progress(token, i, cx);
            }
            cx.notify();
        });
        cx.notify();
        Some((token, ticket))
    }

    /// The selected card has `head`'s answer.
    #[cfg(test)]
    pub fn answered(&self, head: &HeadId, cx: &gpui_kit::App) -> bool {
        let items = &self.stack.read(cx).items;
        matches!(items.get(self.selected).and_then(|it| it.results.get(head)), Some(HeadResult::Done(..)))
    }


    pub fn classify_selected(&mut self, cx: &mut Context<Self>) {
        let stack = self.stack.read(cx);
        if stack.stack().is_none() {
            return;
        }
        let i = self.selected;
        let Some(model) = self.models.read(cx).run_model() else { return };
        if matches!(stack.items.get(i).and_then(|it| it.results.get(&model)), Some(HeadResult::Working(_))) {
            return;
        }
        if let Some((_, ticket)) = self.submit(i, model, Priority::Interactive, cx) {
            self.single.insert(i, ticket);
        }
    }


    /// Request `token` for image `i` finished; when it belongs to the running "Run all", follow it in the grid.
    fn progress(&mut self, token: u64, i: usize, cx: &mut Context<Self>) {
        let Some(run) = self.run.as_mut() else { return };
        if token < run.token || token >= run.token + run.total as u64 {
            return;
        }
        run.done += 1;
        if run.done >= run.total {
            self.run = None;
        }
        self.selected = i;
        let ci = child_index(self.stack.read(cx), i);
        self.scroll.scroll_to_item(ci);
        cx.notify();
    }

    pub fn cancel(&mut self, cx: &mut Context<Self>) {
        if self.run.take().is_some() {
            cx.notify();
        }
    }

    pub fn step(&mut self, delta: isize, cx: &mut Context<Self>) {
        let stack = self.stack.read(cx);
        let n = stack.items.len();
        if n == 0 {
            return;
        }
        self.selected = (self.selected as isize + delta).rem_euclid(n as isize) as usize;
        let ci = child_index(stack, self.selected);
        self.scroll.scroll_to_item(ci);
        cx.notify();
    }

    // ------------------------------------------------------------------------------------------ render
    fn render_body(&self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = cx.theme();
        if let Some(e) = self.stack.read(cx).error() {
            return div()
                .p_6()
                .child(
                    Alert::warning("demo-missing", SharedString::from(e.to_string()))
                        .title("Demo stack not found")
                        .icon(IconName::Info),
                )
                .child(div().pt_3().text_sm().text_color(t.muted_foreground).child(format!(
                    "Run the app from inside the repository, or set {}_DEMO_DIR to the folder with demo_stack.json.",
                    crate::app::ENV_PREFIX
                )))
                .into_any_element();
        }
        v_flex()
            .size_full()
            .child(self.render_header(cx))
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

    fn render_header(&self, cx: &Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let r = theme::roles(cx);
        let stack = self.stack.read(cx);
        let n = stack.items.len();
        let models = self.models.read(cx);
        let (ok, judged, _) = summary(stack, models.selected());
        let short = models.selected().map(|m| models.short(m)).unwrap_or_default();
        let running = self.run.as_ref();
        let progress = running.map(|run| {
            let done = run.done;
            h_flex()
                .gap_2()
                .items_center()
                .child(div().w(px(120.)).child(widgets::bar(
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
        tab_header(
            "Demo stack",
            format!("{n} Wikimedia Commons photos never used for training or model selection · click a card or press Enter"),
            h_flex().gap_4().items_center().children(summary).children(progress).child(if running.is_some() {
                Button::new("demo-run")
                    .danger()
                    .outline()
                    .icon(IconName::Square)
                    .label("Stop")
                    .tooltip("Stop: drop the images not classified yet (Esc)")
                    .on_click(cx.listener(|this, _, _, cx| this.cancel(cx)))
            } else {
                Button::new("demo-run")
                    .primary()
                    .icon(IconName::Play)
                    .label("Run all")
                    .tooltip("Classify every image in order (Ctrl+R)")
                    .on_click(cx.listener(|this, _, _, cx| this.run(cx)))
            }),
            cx,
        )
    }

    fn render_grid(&self, window: &Window, cx: &Context<Self>) -> impl IntoElement {
        let muted_fg = cx.theme().muted_foreground;
        // even columns: as many as fit at CARD_MIN_W, sharing the surplus (scrollbar gutter reserved)
        let avail = f32::from(window.viewport_size().width) - DETAIL_W - 2.0 * GRID_PAD - 12.0;
        let cols = ((avail + GRID_GAP) / (CARD_MIN_W + GRID_GAP)).floor().max(1.0);
        let card_w = ((avail - GRID_GAP * (cols - 1.0)) / cols).floor().max(CARD_MIN_W);
        let items = &self.stack.read(cx).items;
        let Some(stack) = self.stack.read(cx).stack() else { return div().into_any_element() };
        let model = self.models.read(cx).selected();
        let mut children: Vec<AnyElement> = Vec::new();
        for (si, section) in stack.sections.iter().enumerate() {
            let idx = stack.section_indices(section);
            let (ok, n) = idx.iter().fold((0, 0), |(ok, n), &i| match selected_result(model, &items[i]) {
                Some(HeadResult::Done(c, ..)) => {
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
                    .track_scroll(&self.scroll)
                    .p(px(GRID_PAD))
                    .flex()
                    .flex_row()
                    .flex_wrap()
                    .content_start()
                    .gap(px(GRID_GAP))
                    .children(children),
            )
            .vertical_scrollbar(&self.scroll)
            .into_any_element()
    }

    fn render_card(&self, i: usize, card_w: f32, cx: &Context<Self>) -> impl IntoElement {
        let thumb_h = (card_w * 0.72).round();
        let t = cx.theme();
        let r = theme::roles(cx);
        let stack = self.stack.read(cx).stack().expect("stack loaded");
        let im = &stack.images[i];
        let it = &self.stack.read(cx).items[i];
        let sel = self.selected == i;
        let result = selected_result(self.models.read(cx).selected(), it);
        let verdict = match result {
            Some(HeadResult::Done(c, ..)) => Some(im.judge(c.best_label(), c.abstained())),
            _ => None,
        };
        let result: AnyElement = match result {
            Some(HeadResult::Working(_)) => h_flex()
                .gap_1()
                .items_center()
                .child(Spinner::new().xsmall().color(t.muted_foreground))
                .child(div().text_xs().text_color(t.muted_foreground).child("classifying…"))
                .into_any_element(),
            Some(HeadResult::Done(c, ..)) => {
                let v = im.judge(c.best_label(), c.abstained());
                let text = if c.abstained() { "unsure".to_string() } else { c.best_label().to_string() };
                h_flex()
                    .gap_1()
                    .items_center()
                    .min_w_0()
                    .text_xs()
                    .child(Icon::new(verdict_icon(v)).xsmall().text_color(verdict_colors(v, cx).0))
                    .child(div().flex_1().min_w_0().truncate().text_color(r.fg).child(text))
                    .child(div().flex_none().text_color(r.muted_fg).child(widgets::pct(c.best().1)))
                    .into_any_element()
            }
            Some(HeadResult::Failed(e)) => h_flex()
                .gap_1()
                .items_center()
                .text_xs()
                .text_color(t.danger)
                .child(Icon::new(IconName::CircleAlert).xsmall())
                .child(widgets::failure_label(e))
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
                this.selected = i;
                this.classify_selected(cx);
            }))
            .child(div().w_full().h(px(thumb_h)).overflow_hidden().bg(t.muted).child(match &it.thumb {
                Some(th) => img(th.clone()).size_full().object_fit(ObjectFit::Cover).into_any_element(),
                None => div().size_full().into_any_element(),
            }))
            // verdict strip: colour echo of the icon below (never the only cue)
            .child(div().w_full().h(px(3.)).bg(match verdict {
                Some(v) => verdict_colors(v, cx).0,
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

    fn render_detail(&self, cx: &Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let r = theme::roles(cx);
        let stack = self.stack.read(cx).stack().expect("stack loaded");
        let i = self.selected.min(stack.images.len() - 1);
        let im = &stack.images[i];
        let it = &self.stack.read(cx).items[i];
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
        let models = self.models.read(cx);
        let short = models.selected().map(|m| models.short(m)).unwrap_or_default();
        let result: AnyElement = match (selected_result(models.selected(), it), models.selected()) {
            (Some(HeadResult::Done(c, timing, _)), Some(m)) => {
                let verdict = h_flex()
                    .gap_2()
                    .items_center()
                    .child(verdict_tag(im.judge(c.best_label(), c.abstained()), false, cx))
                    .child(div().text_sm().text_color(r.muted_fg).child(format!("True label: {}", im.true_text())));
                widgets::prediction_block(c, timing, &models.title(m), Some(verdict.into_any_element()), cx)
            }
            (Some(HeadResult::Working(_)), _) => h_flex()
                .gap_2()
                .items_center()
                .py_4()
                .child(Spinner::new().color(t.primary))
                .child(div().text_lg().child(format!("Classifying with {short}…")))
                .into_any_element(),
            (Some(HeadResult::Failed(e)), Some(m)) => widgets::failure_alert("demo-error", e, &models.title(m)).into_any_element(),
            _ => v_flex()
                .gap_2()
                .items_start()
                .child(
                    Button::new("demo-classify")
                        .primary()
                        .large()
                        .icon(IconName::Play)
                        .label(format!("Classify with {short}"))
                        .on_click(cx.listener(|this, _, _, cx| this.classify_selected(cx))),
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

impl Render for DemoTab {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("demo-tab")
            .track_focus(&self.focus)
            .key_context("Demo")
            .on_action(cx.listener(|this, _: &StepPrev, _, cx| this.step(-1, cx)))
            .on_action(cx.listener(|this, _: &StepNext, _, cx| this.step(1, cx)))
            .on_action(cx.listener(|this, _: &Activate, _, cx| this.classify_selected(cx)))
            .on_action(cx.listener(|this, _: &RunAll, _, cx| this.run(cx)))
            .on_action(cx.listener(|this, _: &Cancel, _, cx| this.cancel(cx)))
            .size_full()
            .child(prof!("gui::build_tab_elements", self.render_body(window, cx)))
    }
}

#[cfg(test)]
mod tests;
