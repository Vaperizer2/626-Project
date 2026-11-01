//! Benchmark tab: every available model over the demo stack, live (the whole stack is submitted at once, all models
//! per image, cancellable), then per-model accuracy / mean confidence / latency and a per-image matrix.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use futures::StreamExt;
use gpui_kit::component::{
    ActiveTheme as _, Disableable as _, Icon, IconName, Sizable as _,
    alert::Alert,
    button::{Button, ButtonVariants as _},
    h_flex,
    scroll::ScrollableElement as _,
    spinner::Spinner,
    switch::Switch,
    tooltip::Tooltip,
    v_flex,
};
use gpui_kit::{
    AnyElement, App, Context, Entity, FocusHandle, FontWeight, ObjectFit, ScrollHandle, SharedString,
    TestSupportExt as _, Window, div, img, prelude::*, px,
};

use super::models::{ModelState, Models};
use super::stack::{Stack, verdict_colors, verdict_icon};
use super::widgets::Lucide;
use super::{Cancel, Hint, RunAll, Tab, theme, widgets};
use crate::app;
use inference::classification::Classification;
use crate::demo::Verdict;
use inference::service::{Finish, HeadOutput, ImageInput, Lane, Policy, Priority, RequestHandle, RunError, Ticket, Timing, Update};
use inference::manifest::HeadId;
use inference::prof;

/// One benchmark run: every image was submitted at once; dropping the run cancels what has not run yet.
struct BenchRun {
    id: u64,
    total: usize,
    finished: HashSet<usize>,
    started: Instant,
    _tickets: Vec<Ticket>,
}

impl BenchRun {
    /// The first image still waiting for its results.
    fn current(&self) -> Option<usize> {
        (0..self.total).find(|i| !self.finished.contains(i))
    }
}

pub struct BenchTab {
    models: Entity<Models>,
    stack: Entity<Stack>,
    lane: Lane,
    focus: FocusHandle,
    run: Option<BenchRun>,
    next_id: u64,
    /// columns of the last run
    heads: Vec<HeadId>,
    pub(super) results: HashMap<(usize, HeadId), Result<(Classification, Timing), RunError>>,
    only_disagree: bool,
    wall_s: Option<f64>,
    cancelled: bool,
    scroll: ScrollHandle,
}

/// Per-model aggregate over the images it has classified.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Stats {
    pub correct: usize,
    pub judged: usize,
    pub unsure: usize,
    pub errors: usize,
    pub mean_conf: f32,
    pub mean_ms: f64,
}

/// Aggregate `results` for `model` over `stack`.
pub fn stats(
    stack: &crate::demo::DemoStack,
    results: &HashMap<(usize, HeadId), Result<(Classification, Timing), RunError>>,
    model: &HeadId,
) -> Stats {
    let mut s = Stats::default();
    let (mut conf, mut ms) = (0f32, 0f64);
    for (i, im) in stack.images.iter().enumerate() {
        match results.get(&(i, model.clone())) {
            Some(Ok((c, timing))) => {
                s.judged += 1;
                conf += c.best().1;
                ms += timing.ms();
                match im.judge(c.best_label(), c.abstained()) {
                    Verdict::Correct => s.correct += 1,
                    Verdict::Unsure => s.unsure += 1,
                    Verdict::Wrong => {}
                }
            }
            Some(Err(_)) => s.errors += 1,
            None => {}
        }
    }
    if s.judged > 0 {
        s.mean_conf = conf / s.judged as f32;
        s.mean_ms = ms / s.judged as f64;
    }
    s
}

/// Do the models' answers (the committed class, or abstaining) differ on image `i`?
pub fn disagree(
    results: &HashMap<(usize, HeadId), Result<(Classification, Timing), RunError>>,
    i: usize,
    models: &[HeadId],
) -> bool {
    let answers: HashSet<Option<&str>> = models
        .iter()
        .filter_map(|k| results.get(&(i, k.clone())).and_then(|r| r.as_ref().ok()))
        .map(|(c, _)| c.committed().map(|i| c.labels.name(i)))
        .collect();
    answers.len() > 1
}

impl Tab for BenchTab {
    const TITLE: &'static str = "Benchmark";
    const ICON: Lucide = Lucide::ChartNoAxesColumn;
    const HINTS: &'static [Hint] = &[(&["ctrl-r"], "Run"), (&["escape"], "Stop")];
    const MODEL_NOTE: Option<&'static str> = Some("Runs every available model");

    fn new(models: Entity<Models>, shared: &app::Shared, _: &mut Window, cx: &mut Context<Self>) -> Self {
        let stack = shared.stack.clone();
        let lane = models.read(cx).inference().lane(Priority::Bulk, Policy::Keep);
        // the header counts the run's seconds
        super::tick_while(cx, |this, _| this.running());
        Self {
            models,
            stack,
            lane,
            focus: cx.focus_handle(),
            run: None,
            next_id: 1,
            heads: Vec::new(),
            results: HashMap::new(),
            only_disagree: false,
            wall_s: None,
            cancelled: false,
            scroll: ScrollHandle::new(),
        }
    }

    fn focus_handle(&self) -> FocusHandle {
        self.focus.clone()
    }

    /// Every available model over the whole stack, submitted at once; again while it runs stops it.
    fn run(&mut self, cx: &mut Context<Self>) {
        if self.run.is_some() {
            return self.cancel(cx);
        }
        let stack = self.stack.read(cx);
        let Some(n) = stack.stack().map(|s| s.images.len()) else { return };
        let Some(inputs) = (0..n).map(|i| stack.input(i)).collect::<Option<Vec<ImageInput>>>() else { return };
        let m = self.models.read(cx);
        let heads: Vec<HeadId> = m.head_ids().into_iter().filter(|k| m.model_usable(k)).collect();
        if heads.is_empty() {
            return;
        }
        self.heads = heads.clone();
        self.results.clear();
        self.wall_s = None;
        self.cancelled = false;
        let id = self.next_id;
        self.next_id += 1;
        let mut tickets = Vec::new();
        for (i, input) in inputs.into_iter().enumerate() {
            let RequestHandle { updates: mut rx, ticket } = self.lane.submit(input, &heads, None);
            tickets.push(ticket);
            cx.spawn(async move |this, cx| {
                while let Some(u) = rx.next().await {
                    let alive = this
                        .update(cx, |this, cx| {
                            if this.run.as_ref().is_none_or(|r| r.id != id) {
                                return;
                            }
                            match u {
                                Update::Head { head, result, timing } => {
                                    let r = result.map(|HeadOutput::Classify(c)| (c, timing));
                                    // share with the Demo tab so its cards show the result too
                                    if let Ok((c, timing)) = &r {
                                        this.stack.update(cx, |s, cx| {
                                            if let Some(it) = s.items.get_mut(i) {
                                                it.results.done(head.clone(), c.clone(), *timing);
                                                cx.notify();
                                            }
                                        });
                                    }
                                    this.results.insert((i, head), r);
                                }
                                Update::Done(Finish::Failed(e)) => {
                                    for k in this.heads.clone() {
                                        this.results.insert((i, k), Err(e.clone()));
                                    }
                                    this.progress(i, cx);
                                }
                                Update::Done(_) => this.progress(i, cx),
                                Update::Decoded { .. } => {}
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
        }
        self.run = Some(BenchRun { id, total: n, finished: HashSet::new(), started: Instant::now(), _tickets: tickets });
        cx.notify();
    }
}

impl BenchTab {
    pub fn running(&self) -> bool {
        self.run.is_some()
    }


    pub fn cancel(&mut self, cx: &mut Context<Self>) {
        if let Some(run) = self.run.take() {
            self.wall_s = Some(run.started.elapsed().as_secs_f64());
            self.cancelled = true;
            cx.notify();
        }
    }

    /// Image `i` has all its results; follow the run in the matrix and close it after the last image.
    fn progress(&mut self, i: usize, cx: &mut Context<Self>) {
        let Some(run) = self.run.as_mut() else { return };
        run.finished.insert(i);
        let (done, next) = (run.finished.len() >= run.total, run.current());
        if done {
            self.wall_s = Some(run.started.elapsed().as_secs_f64());
            self.run = None;
        }
        if let Some(next) = next {
            let row = self.row_index(next, cx);
            self.scroll.scroll_to_item(row);
        }
        cx.notify();
    }

    /// "one pass with X" when other benchmarked heads read the same graph as `k`.
    fn shared_pass(&self, k: &HeadId, cx: &App) -> Option<String> {
        let m = self.models.read(cx);
        let others: Vec<String> = m
            .registry()
            .siblings(k)
            .iter()
            .filter(|s| *s != k && self.heads.contains(s))
            .map(|s| m.short(s))
            .collect();
        (!others.is_empty()).then(|| format!("one pass with {}", others.join(" and ")))
    }

    fn rows(&self, cx: &App) -> Vec<usize> {
        let n = self.stack.read(cx).items.len();
        (0..n).filter(|&i| !self.only_disagree || disagree(&self.results, i, &self.heads)).collect()
    }

    fn row_index(&self, i: usize, cx: &App) -> usize {
        self.rows(cx).iter().position(|&r| r == i).unwrap_or(0)
    }

    // ------------------------------------------------------------------------------------------ render
    fn render_body(&self, cx: &Context<Self>) -> AnyElement {
        let t = cx.theme();
        let Some(stack) = self.stack.read(cx).stack() else {
            return div()
                .p_6()
                .child(
                    Alert::warning("bench-missing", "The benchmark runs over the demo stack, which was not found.")
                        .title("No demo stack"),
                )
                .into_any_element();
        };
        let n = stack.images.len();
        let running = self.run.as_ref();
        let m = self.models.read(cx);
        let available: Vec<HeadId> = m.head_ids().into_iter().filter(|k| m.model_usable(k)).collect();
        let unavailable: Vec<String> = m
            .head_ids()
            .into_iter()
            .filter_map(|k| match m.state(&k) {
                ModelState::Unavailable(_) => Some(format!("{} (not exported)", m.short(&k))),
                ModelState::Failed(_) => Some(format!("{} (failed to load)", m.short(&k))),
                _ => None,
            })
            .collect();
        let roles = theme::roles(cx);
        let progress = running.map(|r| {
            let done = r.finished.len();
            h_flex()
                .gap_2()
                .items_center()
                .child(Spinner::new().small().color(roles.accent))
                .child(div().w(px(160.)).child(widgets::bar(
                    done as f32 / r.total.max(1) as f32,
                    roles.accent,
                    px(6.),
                    &roles,
                )))
                .child(div().text_sm().text_color(t.muted_foreground).child(format!(
                    "image {}/{} · {:.0} s",
                    (done + 1).min(r.total),
                    r.total,
                    r.started.elapsed().as_secs_f64()
                )))
        });
        let skipped =
            if unavailable.is_empty() { String::new() } else { format!(" · skipped: {}", unavailable.join(", ")) };
        let header = widgets::tab_header(
            "Benchmark on the demo stack",
            format!("{n} images × {} models{skipped}", available.len()),
            h_flex()
                .gap_4()
                .items_center()
                .children(progress)
                .when(running.is_none() && self.wall_s.is_some(), |d| {
                    d.child(
                        h_flex()
                            .gap_1()
                            .items_center()
                            .text_sm()
                            .text_color(t.muted_foreground)
                            .child(
                                Icon::new(if self.cancelled { Lucide::CircleStop } else { Lucide::Timer })
                                    .small(),
                            )
                            .child(format!(
                                "{} in {:.0} s",
                                if self.cancelled { "stopped" } else { "finished" },
                                self.wall_s.unwrap_or(0.0)
                            )),
                    )
                })
                .child(
                    Switch::new("bench-disagree")
                        .checked(self.only_disagree)
                        .label("Only where models disagree")
                        .on_click(cx.listener(|this, checked: &bool, _, cx| {
                            this.only_disagree = *checked;
                            cx.notify();
                        })),
                )
                .child(if running.is_some() {
                    Button::new("bench-run")
                        .danger()
                        .outline()
                        .icon(IconName::Square)
                        .label("Stop")
                        .tooltip("Stop: drop the images not classified yet (Esc)")
                        .on_click(cx.listener(|this, _, _, cx| this.cancel(cx)))
                } else {
                    Button::new("bench-run")
                        .primary()
                        .icon(IconName::Play)
                        .label(if self.results.is_empty() { "Run benchmark" } else { "Run again" })
                        .disabled(available.is_empty())
                        .tooltip("Classify every demo image with every available model (Ctrl+R)")
                        .on_click(cx.listener(|this, _, _, cx| this.run(cx)))
                }),
            cx,
        );
        let models = if self.heads.is_empty() { available.clone() } else { self.heads.clone() };
        // mid-run the counts favour whichever model got through more images
        let best = if running.is_some() {
            None
        } else {
            models.iter().map(|k| stats(stack, &self.results, k)).filter(|s| s.judged > 0).map(|s| s.correct).max()
        };
        let cards = h_flex()
            .items_stretch()
            .gap_3()
            .px_4()
            .pt_4()
            .children(models.iter().map(|k| self.render_stats_card(k, n, best, cx)));
        v_flex()
            .size_full()
            .bg(roles.canvas)
            .child(header)
            .child(cards)
            .child(self.render_matrix(&models, cx))
            .into_any_element()
    }

    fn render_stats_card(&self, k: &HeadId, n: usize, best: Option<usize>, cx: &App) -> impl IntoElement {
        let t = cx.theme();
        let r = theme::roles(cx);
        let stack = self.stack.read(cx).stack().expect("stack");
        let s = stats(stack, &self.results, k);
        let big = if s.judged == 0 { "–".to_string() } else { format!("{}/{}", s.correct, s.judged) };
        let acc = if s.judged == 0 { 0.0 } else { s.correct as f32 / s.judged as f32 };
        let leading = s.judged > 0 && best == Some(s.correct);
        let meta =
            |icon: Lucide, text: String| h_flex().gap_1().items_center().child(Icon::new(icon).xsmall()).child(text);
        let title = self.models.read(cx).title(k);
        v_flex()
            .id(SharedString::from(format!("bench-card-{k}")))
            .flex_1()
            .min_w_0()
            .gap_2()
            .p_4()
            .tooltip(move |window, cx| Tooltip::new(title.clone()).build(window, cx))
            .rounded(t.radius_lg)
            .border_1()
            .border_color(if leading { r.accent.opacity(0.6) } else { r.border })
            .bg(r.surface)
            .child(
                h_flex()
                    .justify_between()
                    .items_start()
                    .gap_2()
                    .child(div().flex_1().min_w_0().text_sm().font_weight(FontWeight::SEMIBOLD).truncate().child(self.models.read(cx).short(k)))
                    .when(leading, |d| {
                        d.child(
                            h_flex()
                                .id(SharedString::from(format!("bench-best-{k}")))
                                .test_support()
                                .flex_none()
                                .gap_1()
                                .items_center()
                                .px_2()
                                .rounded_full()
                                .bg(r.accent_soft)
                                .text_xs()
                                .text_color(r.accent)
                                .child(Icon::new(Lucide::Trophy).xsmall())
                                .child("best"),
                        )
                    }),
            )
            .child(
                h_flex()
                    .flex_wrap()
                    .gap_x_2()
                    .items_baseline()
                    .child(div().text_3xl().font_weight(FontWeight::BOLD).child(big))
                    .child(
                        div()
                            .text_lg()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(r.accent)
                            .when(s.judged > 0, |d| d.child(format!("{:.0}%", acc * 100.0))),
                    )
                    .child(div().text_sm().text_color(t.muted_foreground).child(if s.judged == 0 {
                        format!("of {n}")
                    } else {
                        "correct".to_string()
                    })),
            )
            .child(widgets::bar(acc, if leading { r.accent } else { r.bar_muted }, px(6.), &r))
            .child(
                h_flex()
                    .flex_wrap()
                    .gap_x_4()
                    .gap_y_1()
                    .text_xs()
                    .text_color(t.muted_foreground)
                    .child(meta(
                        Lucide::Gauge,
                        format!("mean conf. {}", if s.judged > 0 { widgets::pct(s.mean_conf) } else { "–".into() }),
                    ))
                    .child(meta(
                        Lucide::Timer,
                        if s.judged > 0 { format!("{:.0} ms / image", s.mean_ms) } else { "– ms / image".into() },
                    ))
                    .when_some(self.shared_pass(k, cx), |d, note| d.child(widgets::icon_line(Icon::new(Lucide::Layers).xsmall(), note).w_full()))
                    .when(s.unsure > 0, |d| d.child(meta(Lucide::CircleQuestionMark, format!("{} unsure", s.unsure))))
                    .when(s.errors > 0, |d| d.child(meta(Lucide::CircleAlert, format!("{} errors", s.errors)))),
            )
    }

    fn render_matrix(&self, models: &[HeadId], cx: &App) -> impl IntoElement {
        let t = cx.theme();
        let r = theme::roles(cx);
        let items = &self.stack.read(cx).items;
        let stack = self.stack.read(cx).stack().expect("stack");
        let current = self.run.as_ref().and_then(BenchRun::current);
        let head = h_flex()
            .px_4()
            .py_2()
            .gap_2()
            .text_xs()
            .font_weight(FontWeight::SEMIBOLD)
            .text_color(t.muted_foreground)
            .border_b_1()
            .border_color(t.border)
            .child(div().w(px(300.)).flex_none().child("Image · true label"))
            .children(models.iter().map(|k| div().flex_1().min_w_0().truncate().child(self.models.read(cx).short(k))));
        let rows: Vec<AnyElement> = self
            .rows(cx)
            .into_iter()
            .map(|i| {
                let im = &stack.images[i];
                let thumb = items.get(i).and_then(|it| it.thumb.clone());
                let cells = models.iter().map(|k| {
                    let cell = h_flex()
                        .flex_1()
                        .min_w_0()
                        .h(px(48.))
                        .rounded(t.radius)
                        .overflow_hidden()
                        .border_1()
                        .border_color(r.border);
                    let body = v_flex().flex_1().min_w_0().px_2().gap_1().justify_center();
                    match self.results.get(&(i, k.clone())) {
                        Some(Ok((out, _))) => {
                            let v = im.judge(out.best_label(), out.abstained());
                            let c = verdict_colors(v, cx).0;
                            let (best, p) = (out.best_label(), out.best().1);
                            let label = if out.abstained() { format!("unsure · {best}") } else { best.to_string() };
                            cell.bg(r.surface)
                                .child(div().w(px(4.)).h_full().flex_none().bg(c))
                                .child(
                                    body.child(
                                        h_flex()
                                            .gap_1()
                                            .items_center()
                                            .min_w_0()
                                            .child(Icon::new(verdict_icon(v)).small().text_color(c))
                                            .child(
                                                div()
                                                    .flex_1()
                                                    .min_w_0()
                                                    .text_sm()
                                                    .font_weight(FontWeight::MEDIUM)
                                                    .truncate()
                                                    .when(out.abstained(), |d| d.text_color(r.muted_fg))
                                                    .child(label),
                                            )
                                            .child(
                                                div()
                                                    .flex_none()
                                                    .text_xs()
                                                    .text_color(r.muted_fg)
                                                    .child(widgets::pct(p)),
                                            ),
                                    )
                                    .child(widgets::bar(
                                        p,
                                        r.bar_muted,
                                        px(3.),
                                        &r,
                                    )),
                                )
                                .into_any_element()
                        }
                        Some(Err(e)) => cell
                            .bg(r.surface)
                            .child(div().w(px(4.)).h_full().flex_none().bg(r.wrong))
                            .child(
                                body.child(
                                    h_flex()
                                        .gap_1()
                                        .items_center()
                                        .text_xs()
                                        .text_color(r.wrong)
                                        .child(Icon::new(IconName::CircleAlert).xsmall())
                                        .child(div().truncate().child(widgets::failure_label(e))),
                                ),
                            )
                            .into_any_element(),
                        None if current == Some(i) => cell
                            .bg(r.surface)
                            .border_color(r.accent.opacity(0.5))
                            .child(body.items_center().child(Spinner::new().small().color(r.accent)))
                            .into_any_element(),
                        None => cell.border_dashed().into_any_element(),
                    }
                });
                h_flex()
                    .id(("bench-row", i))
                    .px_4()
                    .py_1()
                    .gap_2()
                    .items_center()
                    .when(current == Some(i), |d| d.bg(r.accent_soft))
                    .child(
                        h_flex()
                            .w(px(300.))
                            .flex_none()
                            .gap_2()
                            .items_center()
                            .child(
                                div()
                                    .w(px(64.))
                                    .h(px(48.))
                                    .flex_none()
                                    .rounded(t.radius)
                                    .overflow_hidden()
                                    .bg(t.muted)
                                    .border_1()
                                    .border_color(r.border)
                                    .when_some(thumb, |d, th| {
                                        d.child(img(th).size_full().object_fit(ObjectFit::Cover))
                                    }),
                            )
                            .child(
                                v_flex()
                                    .min_w_0()
                                    .child(
                                        div()
                                            .text_sm()
                                            .font_weight(FontWeight::SEMIBOLD)
                                            .truncate()
                                            .child(im.true_text().to_string()),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(t.muted_foreground)
                                            .truncate()
                                            .child(im.file.clone()),
                                    ),
                            ),
                    )
                    .children(cells)
                    .into_any_element()
            })
            .collect();
        let empty = rows.is_empty().then(|| {
            div().p_6().text_sm().text_color(t.muted_foreground).child(if self.only_disagree {
                "All models agree on every image classified so far."
            } else {
                "No images."
            })
        });
        v_flex().flex_1().min_h_0().pt_3().child(head).child(
            div()
                .relative()
                .flex_1()
                .min_h_0()
                .child(
                    v_flex()
                        .id("bench-matrix")
                        .test_support()
                        .size_full()
                        .overflow_y_scroll()
                        .track_scroll(&self.scroll)
                        .pb_4()
                        .children(rows)
                        .children(empty),
                )
                .vertical_scrollbar(&self.scroll),
        )
    }
}

impl Render for BenchTab {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("bench-tab")
            .track_focus(&self.focus)
            .key_context("Bench")
            .on_action(cx.listener(|this, _: &RunAll, _, cx| this.run(cx)))
            .on_action(cx.listener(|this, _: &Cancel, _, cx| this.cancel(cx)))
            .size_full()
            .child(prof!("gui::build_tab_elements", self.render_body(cx)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference::classification::read_head;
    use crate::demo::DemoStack;
    use inference::manifest::{Calibration, ClassifyTask, Decision, Labels, Prob, Readout, Task};
    use std::path::Path;
    use std::sync::Arc;

    /// A calibrated head that abstains below 0.5, answering class `label_ix` with probability `p`.
    fn pred(label_ix: usize, p: f32) -> (Classification, Timing) {
        let labels = Labels::new(["led", "diode", "resistor"].iter().map(|s| s.to_string()).collect()).unwrap();
        let task = Task::Classify(ClassifyTask {
            readout: Readout::Probs { output: "probs".into() },
            labels: Arc::new(labels),
            calibration: Calibration::Temperature { t: 1.0, in_graph: true },
            decision: Decision::Abstain { below: Prob::new(0.5).unwrap() },
        });
        let mut probs = vec![(1.0 - p) / 2.0; 3];
        probs[label_ix] = p;
        let HeadOutput::Classify(c) = read_head(&task, &probs).unwrap();
        (c, Timing { preprocess_ms: 2.0, run_ms: 8.0, shared_by: 1, ..Default::default() })
    }

    #[test]
    fn stats_and_disagreement() {
        let stack = DemoStack::parse(
            Path::new("/d"),
            r#"{"images":[{"file":"a","section":"s","true_label":"led"},{"file":"b","section":"s","true_label":null}]}"#,
        )
        .unwrap();
        let (ka, kb) = (HeadId::new("a"), HeadId::new("b"));
        let mut r = HashMap::new();
        r.insert((0, ka.clone()), Ok(pred(0, 0.9)));
        r.insert((0, kb.clone()), Ok(pred(1, 0.8)));
        r.insert((1, ka.clone()), Ok(pred(2, 0.4)));
        r.insert((1, kb.clone()), Err(RunError::Run("x".into())));
        let a = stats(&stack, &r, &ka);
        assert_eq!((a.correct, a.judged, a.unsure, a.errors), (2, 2, 0, 0), "out-of-scope + unsure counts as correct");
        assert!((a.mean_conf - 0.65).abs() < 1e-6);
        let b = stats(&stack, &r, &kb);
        assert_eq!((b.correct, b.judged, b.errors), (0, 1, 1));
        assert!(disagree(&r, 0, &[ka.clone(), kb.clone()]));
        assert!(!disagree(&r, 1, &[ka, kb]), "an error is not an answer");
    }
}
