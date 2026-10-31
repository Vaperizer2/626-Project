//! Benchmark tab: every available model over the demo stack, live (one image at a time, all models per image,
//! cancellable), then per-model accuracy / mean confidence / latency and a per-image matrix.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use futures::StreamExt;
use futures::channel::mpsc as fmpsc;
use gpui_kit::component::{
    ActiveTheme as _, Disableable as _, Icon, IconName, Sizable as _,
    alert::Alert,
    button::{Button, ButtonVariants as _},
    h_flex,
    scroll::ScrollableElement as _,
    spinner::Spinner,
    switch::Switch,
    v_flex,
};
use gpui_kit::{
    AnyElement, App, Context, FontWeight, ObjectFit, ScrollHandle, TestSupportExt as _, div, img, prelude::*, px,
};

use super::demo_tab::ItemResult;
use super::widgets::Lucide;
use super::worker::{Cmd, Job, JobInput, ModelState, Update};
use super::{DemoApp, theme, widgets};
use crate::contract::Prediction;
use crate::demo::Verdict;
use crate::manifest::HeadId;

struct BenchRun {
    queue: VecDeque<usize>,
    total: usize,
    current: Option<usize>,
    cancel: Arc<AtomicBool>,
    started: Instant,
}

pub struct BenchState {
    run: Option<BenchRun>,
    /// columns of the last run
    models: Vec<HeadId>,
    results: HashMap<(usize, HeadId), Result<Prediction, String>>,
    only_disagree: bool,
    wall_s: Option<f64>,
    cancelled: bool,
    scroll: ScrollHandle,
}

impl Default for BenchState {
    fn default() -> Self {
        Self {
            run: None,
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
    results: &HashMap<(usize, HeadId), Result<Prediction, String>>,
    model: &HeadId,
) -> Stats {
    let mut s = Stats::default();
    let (mut conf, mut ms) = (0f32, 0f64);
    for (i, im) in stack.images.iter().enumerate() {
        match results.get(&(i, model.clone())) {
            Some(Ok(p)) => {
                s.judged += 1;
                conf += p.top3[0].1;
                ms += p.ms;
                match im.judge(&p.top3[0].0, p.unsure) {
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

/// Do the models' answers (top-1 label, or UNSURE) differ on image `i`?
pub fn disagree(
    results: &HashMap<(usize, HeadId), Result<Prediction, String>>,
    i: usize,
    models: &[HeadId],
) -> bool {
    let answers: HashSet<&str> = models
        .iter()
        .filter_map(|k| results.get(&(i, k.clone())).and_then(|r| r.as_ref().ok()))
        .map(|p| if p.unsure { "UNSURE" } else { p.top3[0].0.as_str() })
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
        self.bench.models = models;
        self.bench.results.clear();
        self.bench.wall_s = None;
        self.bench.cancelled = false;
        self.bench.run = Some(BenchRun {
            queue: (0..n).collect(),
            total: n,
            current: None,
            cancel: Arc::new(AtomicBool::new(false)),
            started: Instant::now(),
        });
        self.bench_next(cx);
    }

    pub(super) fn bench_cancel(&mut self, cx: &mut Context<Self>) {
        if let Some(run) = self.bench.run.as_mut() {
            run.cancel.store(true, Ordering::Relaxed);
            self.bench.cancelled = true;
            cx.notify();
        }
    }

    fn bench_next(&mut self, cx: &mut Context<Self>) {
        let Some(run) = self.bench.run.as_mut() else { return };
        let next = if run.cancel.load(Ordering::Relaxed) { None } else { run.queue.pop_front() };
        let Some(i) = next else {
            self.bench.wall_s = Some(run.started.elapsed().as_secs_f64());
            self.bench.run = None;
            return cx.notify();
        };
        run.current = Some(i);
        let cancel = run.cancel.clone();
        let input = match (self.demo.items.get(i).and_then(|it| it.rgb.clone()), self.demo.stack()) {
            (Some(rgb), _) => JobInput::Decoded(rgb),
            (None, Some(s)) => JobInput::Source(crate::fetch::Source::File(s.path(i))),
            (None, None) => return,
        };
        let (tx, mut rx) = fmpsc::unbounded();
        let job = Job { input, models: self.bench.models.clone(), display: false, cancel, updates: tx };
        if self.jobs.send(Cmd::Job(job)).is_err() {
            self.bench.run = None;
            return;
        }
        let row = self.bench_row_index(i);
        self.bench.scroll.scroll_to_item(row);
        cx.spawn(async move |this, cx| {
            while let Some(u) = rx.next().await {
                let alive = this
                    .update(cx, |this, cx| {
                        match u {
                            Update::Result(k, r) => {
                                // share with the Demo tab so its cards show the result too
                                if let (Ok(p), Some(it)) = (&r, this.demo.items.get_mut(i)) {
                                    it.results.insert(k.clone(), ItemResult::Done(p.clone()));
                                }
                                this.bench.results.insert((i, k), r);
                            }
                            Update::Failed(e) => {
                                for k in this.bench.models.clone() {
                                    this.bench.results.insert((i, k), Err(e.clone()));
                                }
                            }
                            Update::Image(..) => {}
                            Update::Done(_) => this.bench_next(cx),
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

    /// "one pass shared by C and Deployed" when other benchmarked heads read the same graph as `k`.
    fn shared_pass(&self, k: &HeadId) -> Option<String> {
        let sib = self.registry.as_ref()?.siblings(k);
        let names: Vec<String> = sib.iter().filter(|s| self.bench.models.contains(s)).map(|s| self.short(s)).collect();
        (names.len() > 1).then(|| format!("one pass shared by {}", names.join(" and ")))
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
                Some(ModelState::Unavailable(_)) => Some(format!("{} (not exported)", self.short(&k))),
                Some(ModelState::Failed(_)) => Some(format!("{} (failed to load)", self.short(&k))),
                _ => None,
            })
            .collect();
        let roles = theme::roles(cx);
        let progress = running.map(|r| {
            let done = r.total - r.queue.len() - usize::from(r.current.is_some());
            h_flex()
                .gap_2()
                .items_center()
                .child(Spinner::new().small().color(roles.accent))
                .child(div().w(px(240.)).child(widgets::bar(
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
        let header = h_flex()
            .justify_between()
            .items_center()
            .gap_4()
            .px_4()
            .py_3()
            .bg(roles.chrome)
            .border_b_1()
            .border_color(t.border)
            .child(
                v_flex()
                    .child(div().text_lg().font_weight(FontWeight::SEMIBOLD).child("Benchmark on the demo stack"))
                    .child(div().text_xs().text_color(t.muted_foreground).child(format!(
                        "{n} images × {} models{}",
                        available.len(),
                        if unavailable.is_empty() {
                            String::new()
                        } else {
                            format!(" · skipped: {}", unavailable.join(", "))
                        }
                    ))),
            )
            .child(
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
                            .tooltip("Stop after the current image (Esc)")
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
            );
        let models = if self.bench.models.is_empty() { available.clone() } else { self.bench.models.clone() };
        let best = models
            .iter()
            .map(|k| stats(stack, &self.bench.results, k))
            .filter(|s| s.judged > 0)
            .map(|s| s.correct)
            .max();
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
        v_flex()
            .flex_1()
            .min_w_0()
            .gap_2()
            .p_4()
            .rounded(t.radius_lg)
            .border_1()
            .border_color(if leading { r.accent.opacity(0.6) } else { r.border })
            .bg(r.surface)
            .child(
                h_flex()
                    .justify_between()
                    .items_center()
                    .gap_2()
                    .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).truncate().child(self.title(k)))
                    .when(leading, |d| {
                        d.child(
                            h_flex()
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
                    .gap_2()
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
                    .when_some(self.shared_pass(k), |d, note| d.child(note))
                    .when(s.unsure > 0, |d| d.child(meta(Lucide::CircleQuestionMark, format!("{} unsure", s.unsure))))
                    .when(s.errors > 0, |d| d.child(meta(Lucide::CircleAlert, format!("{} errors", s.errors)))),
            )
    }

    fn render_matrix(&self, models: &[HeadId], cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let r = theme::roles(cx);
        let stack = self.demo.stack().expect("stack");
        let current = self.bench.run.as_ref().and_then(|r| r.current);
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
                        Some(Ok(p)) => {
                            let v = im.judge(&p.top3[0].0, p.unsure);
                            let c = r.verdict(v).0;
                            let label =
                                if p.unsure { format!("unsure · {}", p.top3[0].0) } else { p.top3[0].0.clone() };
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
                                                    .when(p.unsure, |d| d.text_color(r.muted_fg))
                                                    .child(label),
                                            )
                                            .child(
                                                div()
                                                    .flex_none()
                                                    .text_xs()
                                                    .text_color(r.muted_fg)
                                                    .child(widgets::pct(p.top3[0].1)),
                                            ),
                                    )
                                    .child(widgets::bar(
                                        p.top3[0].1,
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
                                        .child(div().truncate().child(format!("error: {e}"))),
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
    use crate::contract::decide;
    use crate::demo::DemoStack;
    use std::path::Path;

    fn pred(label_ix: usize, p: f32) -> Prediction {
        let labels: Vec<String> = ["led", "diode", "resistor"].iter().map(|s| s.to_string()).collect();
        let mut probs = vec![(1.0 - p) / 2.0; 3];
        probs[label_ix] = p;
        decide(&probs, &labels, 0.5, 10.0)
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
        r.insert((1, kb.clone()), Err("x".into()));
        let a = stats(&stack, &r, &ka);
        assert_eq!((a.correct, a.judged, a.unsure, a.errors), (2, 2, 0, 0), "out-of-scope + unsure counts as correct");
        assert!((a.mean_conf - 0.65).abs() < 1e-6);
        let b = stats(&stack, &r, &kb);
        assert_eq!((b.correct, b.judged, b.errors), (0, 1, 1));
        assert!(disagree(&r, 0, &[ka.clone(), kb.clone()]));
        assert!(!disagree(&r, 1, &[ka, kb]), "an error is not an answer");
    }
}
