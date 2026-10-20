//! Small presentational pieces shared by the tabs: probability bars, the prediction block, verdict and model
//! state tags, keyboard hints. Colours come from the Kit theme by semantic role only.

use gpui_kit::component::{
    ActiveTheme as _, Icon, IconName, Sizable as _, h_flex, kbd::Kbd, progress::Progress, spinner::Spinner, tag::Tag,
    v_flex,
};
use gpui_kit::{
    AnyElement, App, FontWeight, Hsla, IntoElement, Keystroke, ParentElement, SharedString, Styled, div, prelude::*,
};

use super::worker::ModelState;
use crate::contract::Prediction;
use crate::demo::Verdict;
use crate::models::ModelKey;

pub fn pct(p: f32) -> String {
    if p >= 0.995 {
        format!("{:.1}%", p * 100.0)
    } else if p < 0.005 {
        "<1%".to_string()
    } else {
        format!("{:.0}%", p * 100.0)
    }
}

/// A keyboard shortcut chip, e.g. `kbd("ctrl-o")`.
pub fn kbd(keys: &str) -> Kbd {
    Kbd::new(Keystroke::parse(keys).unwrap_or_else(|_| Keystroke::parse("escape").expect("valid keystroke")))
}

/// "Ctrl O  Open" style hint.
pub fn hint(keys: &[&str], label: &'static str, cx: &App) -> impl IntoElement {
    h_flex()
        .gap_1()
        .items_center()
        .children(keys.iter().map(|k| kbd(k)))
        .child(div().text_xs().text_color(cx.theme().muted_foreground).child(label))
}

/// Label + percentage + bar. `strong` = the committed top-1 (primary colour), others are muted.
pub fn prob_row(id: impl Into<SharedString>, label: &str, p: f32, strong: bool, large: bool, cx: &App) -> AnyElement {
    let t = cx.theme();
    let color = if strong { t.primary } else { t.muted_foreground.opacity(0.55) };
    let bar = Progress::new(id.into()).value((p * 100.0).clamp(0.0, 100.0)).color(color);
    v_flex()
        .gap_1()
        .child(
            h_flex()
                .justify_between()
                .when(large, |d| d.text_base())
                .when(!large, |d| d.text_sm())
                .child(
                    div()
                        .text_color(t.foreground)
                        .when(strong, |d| d.font_weight(FontWeight::SEMIBOLD))
                        .child(label.to_string()),
                )
                .child(div().text_color(t.muted_foreground).child(pct(p))),
        )
        .child(if large { bar.into_any_element() } else { bar.small().into_any_element() })
        .into_any_element()
}

/// Headline + top-3 + all classes. Used by the Classify result panel and the Demo detail pane.
pub fn prediction_block(
    id: &str,
    p: &Prediction,
    key: ModelKey,
    threshold: f32,
    verdict: Option<(Verdict, &str)>,
    labels: &[String],
    cx: &App,
) -> AnyElement {
    let t = cx.theme();
    let top = &p.top3[0];
    let headline = if p.unsure {
        v_flex()
            .gap_1()
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(Icon::new(IconName::TriangleAlert).text_color(t.warning))
                    .child(div().text_3xl().font_weight(FontWeight::BOLD).text_color(t.warning).child("Unsure")),
            )
            .child(div().text_sm().text_color(t.muted_foreground).child(format!(
                "Top probability {} is below the {} threshold, so the model abstains. Best guess: {}.",
                pct(top.1),
                pct(threshold),
                top.0
            )))
    } else {
        v_flex()
            .gap_0p5()
            .child(div().text_xs().text_color(t.muted_foreground).child("Prediction"))
            .child(div().text_3xl().font_weight(FontWeight::BOLD).text_color(t.foreground).child(top.0.clone()))
            .child(div().text_sm().text_color(t.muted_foreground).child(if key.calibrated() {
                format!("{} confidence (calibrated)", pct(top.1))
            } else {
                format!("{} confidence (softmax, not calibrated)", pct(top.1))
            }))
            .when(!key.calibrated() && top.1 < threshold, |d| {
                d.child(
                    h_flex()
                        .gap_1()
                        .items_center()
                        .text_xs()
                        .text_color(t.muted_foreground)
                        .child(Icon::new(IconName::Info).xsmall())
                        .child("Low confidence. Only the deployed model is calibrated and abstains; this one always answers."),
                )
            })
    };
    let verdict_row = verdict.map(|(v, truth)| {
        h_flex()
            .gap_2()
            .items_center()
            .child(verdict_tag(v, false))
            .child(div().text_sm().text_color(t.muted_foreground).child(format!("True label: {truth}")))
    });
    let mut order: Vec<usize> = (0..p.probs.len()).collect();
    order.sort_by(|&a, &b| p.probs[b].partial_cmp(&p.probs[a]).unwrap_or(std::cmp::Ordering::Equal));
    let rest = order.iter().skip(3).map(|&i| {
        h_flex()
            .justify_between()
            .text_xs()
            .child(
                div()
                    .text_color(t.muted_foreground)
                    .child(labels.get(i).cloned().unwrap_or_else(|| format!("class {i}"))),
            )
            .child(div().text_color(t.muted_foreground).child(format!("{:.1}%", p.probs[i] * 100.0)))
    });
    v_flex()
        .gap_4()
        .child(headline)
        .children(verdict_row)
        .child(
            v_flex().gap_3().children(
                p.top3
                    .iter()
                    .enumerate()
                    .map(|(i, (l, v))| prob_row(format!("{id}-top{i}"), l, *v, i == 0 && !p.unsure, true, cx)),
            ),
        )
        .child(
            v_flex()
                .gap_1()
                .pt_2()
                .border_t_1()
                .border_color(t.border)
                .child(
                    div()
                        .text_xs()
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(t.muted_foreground)
                        .child("Other classes"),
                )
                .children(rest),
        )
        .child(div().text_xs().text_color(t.muted_foreground).child(format!(
            "{} · preprocess + inference {:.0} ms",
            key.name(),
            p.ms
        )))
        .into_any_element()
}

pub fn verdict_tag(v: Verdict, small: bool) -> Tag {
    let tag = match v {
        Verdict::Correct => Tag::success()
            .child(h_flex().gap_1().items_center().child(Icon::new(IconName::CircleCheck).xsmall()).child("Correct")),
        Verdict::Wrong => Tag::danger()
            .child(h_flex().gap_1().items_center().child(Icon::new(IconName::CircleX).xsmall()).child("Wrong")),
        Verdict::Unsure => Tag::warning()
            .child(h_flex().gap_1().items_center().child(Icon::new(IconName::TriangleAlert).xsmall()).child("Unsure")),
    };
    if small { tag.small() } else { tag }
}

/// Semantic colour for a verdict (cells, card accents).
pub fn verdict_color(v: Verdict, cx: &App) -> Hsla {
    match v {
        Verdict::Correct => cx.theme().success,
        Verdict::Wrong => cx.theme().danger,
        Verdict::Unsure => cx.theme().warning,
    }
}

/// Compact model state: spinner while loading, otherwise a short tag.
pub fn model_state_chip(state: Option<&ModelState>, cx: &App) -> AnyElement {
    let t = cx.theme();
    match state {
        None | Some(ModelState::Idle) => Tag::secondary().small().child("not loaded").into_any_element(),
        Some(ModelState::Loading(t0)) => h_flex()
            .gap_1()
            .items_center()
            .child(Spinner::new().small().color(t.muted_foreground))
            .child(
                div()
                    .text_xs()
                    .text_color(t.muted_foreground)
                    .child(format!("loading… {:.0} s", t0.elapsed().as_secs_f64())),
            )
            .into_any_element(),
        Some(ModelState::Ready(ms)) => {
            Tag::secondary().small().child(format!("ready · loaded in {:.1} s", ms / 1000.0)).into_any_element()
        }
        Some(ModelState::Unavailable(_)) => Tag::secondary().outline().small().child("not exported").into_any_element(),
        Some(ModelState::Failed(_)) => Tag::danger().outline().small().child("failed to load").into_any_element(),
    }
}
