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
    AnyElement, App, Context, FontWeight, ObjectFit, ScrollHandle, SharedString, TestSupportExt as _, div, img,
    prelude::*, px,
};

use super::widgets::Lucide;
use super::{DemoApp, ModelState, theme, widgets};
use crate::classification::{Classification, HeadOutput};
use crate::demo::Verdict;
use crate::inference::{Finish, ImageInput, RequestHandle, RunError, Ticket, Timing, Update};
use crate::manifest::HeadId;

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

pub struct BenchState {
    run: Option<BenchRun>,
    next_id: u64,
    /// columns of the last run
    models: Vec<HeadId>,
    pub(super) results: HashMap<(usize, HeadId), Result<(Classification, Timing), RunError>>,
    only_disagree: bool,
    wall_s: Option<f64>,
    cancelled: bool,
    scroll: ScrollHandle,
}

impl Default for BenchState {
    fn default() -> Self {
        Self {
            run: None,
            next_id: 1,
            models: Vec::new(),
            results: HashMap::new(),
            only_disagree: false,
            wall_s: None,
            cancelled: false,
            scroll: ScrollHandle::new(),
        }
    }
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

impl BenchState {
    pub fn running(&self) -> bool {
        self.run.is_some()
    }
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

impl DemoApp {
    pub(super) fn bench_start(&mut self, cx: &mut Context<Self>) {
        if self.bench.run.is_some() {
            return self.bench_cancel(cx);
        }
        let Some(n) = self.demo.stack().map(|s| s.images.len()) else { return };
        let models: Vec<HeadId> = self.head_ids().into_iter().filter(|k| self.model_usable(k)).collect();
        if models.is_empty() {
            return;
        }
        self.bench.models = models.clone();
        self.bench.results.clear();
        self.bench.wall_s = None;
        self.bench.cancelled = false;
        let id = self.bench.next_id;
        self.bench.next_id += 1;
        let mut tickets = Vec::new();
        for i in 0..n {
            let input = match (self.demo.items.get(i).and_then(|it| it.rgb.clone()), self.demo.stack()) {
                (Some(rgb), _) => ImageInput::Decoded(rgb),
                (None, Some(s)) => ImageInput::Source(crate::fetch::Source::File(s.path(i))),
                (None, None) => return,
            };
            let RequestHandle { updates: mut rx, ticket } = self.bulk.submit(input, &models, None);
            tickets.push(ticket);
            cx.spawn(async move |this, cx| {
                while let Some(u) = rx.next().await {
                    let alive = this
                        .update(cx, |this, cx| {
                            if this.bench.run.as_ref().is_none_or(|r| r.id != id) {
                                return;
                            }
                            match u {
                                Update::Head { head, result, timing } => {
                                    let r = result.map(|HeadOutput::Classify(c)| (c, timing));
                                    // share with the Demo tab so its cards show the result too
                                    if let (Ok((c, timing)), Some(it)) = (&r, this.demo.items.get_mut(i)) {
                                        it.results.done(head.clone(), c.clone(), *timing);
                                    }
                                    this.bench.results.insert((i, head), r);
                                }
                                Update::Done(Finish::Failed(e)) => {
                                    for k in this.bench.models.clone() {
                                        this.bench.results.insert((i, k), Err(e.clone()));
                                    }
                                    this.bench_progress(i, cx);
                                }
                                Update::Done(_) => this.bench_progress(i, cx),
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
        self.bench.run = Some(BenchRun { id, total: n, finished: HashSet::new(), started: Instant::now(), _tickets: tickets });
        cx.notify();
    }

    pub(super) fn bench_cancel(&mut self, cx: &mut Context<Self>) {
        if let Some(run) = self.bench.run.take() {
            self.bench.wall_s = Some(run.started.elapsed().as_secs_f64());
            self.bench.cancelled = true;
            cx.notify();
        }
    }

    /// Image `i` has all its results; follow the run in the matrix and close it after the last image.
    fn bench_progress(&mut self, i: usize, cx: &mut Context<Self>) {
        let Some(run) = self.bench.run.as_mut() else { return };
        run.finished.insert(i);
        let (done, next) = (run.finished.len() >= run.total, run.current());
        if done {
            self.bench.wall_s = Some(run.started.elapsed().as_secs_f64());
            self.bench.run = None;
        }
        if let Some(next) = next {
            let row = self.bench_row_index(next);
            self.bench.scroll.scroll_to_item(row);
        }
        cx.notify();
    }

    /// "one pass with X" when other benchmarked heads read the same graph as `k`.
    fn shared_pass(&self, k: &HeadId) -> Option<String> {
        let others: Vec<String> = self
            .registry()
            .siblings(k)
            .iter()
            .filter(|s| *s != k && self.bench.models.contains(s))
            .map(|s| self.short(s))
            .collect();
        (!others.is_empty()).then(|| format!("one pass with {}", others.join(" and ")))
    }

    fn bench_rows(&self) -> Vec<usize> {
        let n = self.demo.items.len();
        (0..n).filter(|&i| !self.bench.only_disagree || disagree(&self.bench.results, i, &self.bench.models)).collect()
    }

    fn bench_row_index(&self, i: usize) -> usize {
        self.bench_rows().iter().position(|&r| r == i).unwrap_or(0)
    }

    // ------------------------------------------------------------------------------------------ render
    pub(super) fn render_bench(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let t = cx.theme();
        let Some(stack) = self.demo.stack() else {
            return div()
                .p_6()
                .child(
                    Alert::warning("bench-missing", "The benchmark runs over the demo stack, which was not found.")
                        .title("No demo stack"),
                )
                .into_any_element();
        };
        let n = stack.images.len();
        let running = self.bench.run.as_ref();
        let available: Vec<HeadId> = self.head_ids().into_iter().filter(|k| self.model_usable(k)).collect();
        let unavailable: Vec<String> = self
            .head_ids()
            .into_iter()
            .filter_map(|k| match self.state(&k) {
                    ModelState::Unavailable(_) => Some(format!("{} (not exported)", self.short(&k))),
                ModelState::Failed(_) => Some(format!("{} (failed to load)", self.short(&k))),
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
                .when(running.is_none() && self.bench.wall_s.is_some(), |d| {
                    d.child(
                        h_flex()
                            .gap_1()
                            .items_center()
                            .text_sm()
                            .text_color(t.muted_foreground)
                            .child(
                                Icon::new(if self.bench.cancelled { Lucide::CircleStop } else { Lucide::Timer })
                                    .small(),
                            )
                            .child(format!(
                                "{} in {:.0} s",
                                if self.bench.cancelled { "stopped" } else { "finished" },
                                self.bench.wall_s.unwrap_or(0.0)
                            )),
                    )
                })
                .child(
                    Switch::new("bench-disagree")
                        .checked(self.bench.only_disagree)
                        .label("Only where models disagree")
                        .on_click(cx.listener(|this, checked: &bool, _, cx| {
                            this.bench.only_disagree = *checked;
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
                        .on_click(cx.listener(|this, _, _, cx| this.bench_cancel(cx)))
                } else {
                    Button::new("bench-run")
                        .primary()
                        .icon(IconName::Play)
                        .label(if self.bench.results.is_empty() { "Run benchmark" } else { "Run again" })
                        .disabled(available.is_empty())
                        .tooltip("Classify every demo image with every available model (Ctrl+R)")
                        .on_click(cx.listener(|this, _, _, cx| this.bench_start(cx)))
                }),
            cx,
        );
        let models = if self.bench.models.is_empty() { available.clone() } else { self.bench.models.clone() };
        // mid-run the counts favour whichever model got through more images
        let best = if running.is_some() {
            None
        } else {
            models.iter().map(|k| stats(stack, &self.bench.results, k)).filter(|s| s.judged > 0).map(|s| s.correct).max()
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
        let stack = self.demo.stack().expect("stack");
        let s = stats(stack, &self.bench.results, k);
        let big = if s.judged == 0 { "–".to_string() } else { format!("{}/{}", s.correct, s.judged) };
        let acc = if s.judged == 0 { 0.0 } else { s.correct as f32 / s.judged as f32 };
        let leading = s.judged > 0 && best == Some(s.correct);
        let meta =
            |icon: Lucide, text: String| h_flex().gap_1().items_center().child(Icon::new(icon).xsmall()).child(text);
        let title = self.title(k);
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
                    .child(div().flex_1().min_w_0().text_sm().font_weight(FontWeight::SEMIBOLD).truncate().child(self.short(k)))
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
                    .when_some(self.shared_pass(k), |d, note| d.child(widgets::icon_line(Icon::new(Lucide::Layers).xsmall(), note).w_full()))
                    .when(s.unsure > 0, |d| d.child(meta(Lucide::CircleQuestionMark, format!("{} unsure", s.unsure))))
                    .when(s.errors > 0, |d| d.child(meta(Lucide::CircleAlert, format!("{} errors", s.errors)))),
            )
    }

    fn render_matrix(&self, models: &[HeadId], cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let r = theme::roles(cx);
        let stack = self.demo.stack().expect("stack");
        let current = self.bench.run.as_ref().and_then(BenchRun::current);
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
            .children(models.iter().map(|k| div().flex_1().min_w_0().truncate().child(self.short(k))));
        let rows: Vec<AnyElement> = self
            .bench_rows()
            .into_iter()
            .map(|i| {
                let im = &stack.images[i];
                let thumb = self.demo.items.get(i).and_then(|it| it.thumb.clone());
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
                    match self.bench.results.get(&(i, k.clone())) {
                        Some(Ok((out, _))) => {
                            let v = im.judge(out.best_label(), out.abstained());
                            let c = r.verdict(v).0;
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
                                            .child(Icon::new(widgets::verdict_icon(v)).small().text_color(c))
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
            div().p_6().text_sm().text_color(t.muted_foreground).child(if self.bench.only_disagree {
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
                        .track_scroll(&self.bench.scroll)
                        .pb_4()
                        .children(rows)
                        .children(empty),
                )
                .vertical_scrollbar(&self.bench.scroll),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classification::read_head;
    use crate::demo::DemoStack;
    use crate::manifest::{Calibration, ClassifyTask, Decision, Labels, Prob, Readout, Task};
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
