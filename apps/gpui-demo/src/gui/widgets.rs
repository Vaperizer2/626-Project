//! Small presentational pieces shared by the tabs: bars, probability rows, the prediction block, pills, model
//! state chips, keyboard hints, section captions. Colours come from the Kit theme or `theme::roles` only.

use gpui_kit::component::{
    ActiveTheme as _, Icon, IconName, Sizable as _, alert::Alert, h_flex, kbd::Kbd, spinner::Spinner, tag::Tag, v_flex,
};
use gpui_kit::{
    AnyElement, App, Div, ElementId, FontWeight, Hsla, IntoElement, Keystroke, ParentElement, Pixels, SharedString,
    Styled, div, prelude::*, px, relative,
};

/// The complete bundled Lucide catalog (the component crate's `IconName` is a subset).
pub use gpui_kit::assets::IconName as Lucide;

// The Kit's `Assets` embeds only the ~100 component icons; every other Lucide icon the app uses must be
// listed here (the `every_lucide_icon_used_is_bundled` test fails otherwise).
gpui_kit::assets::icon_assets!(
    ExtraIcons,
    [
        ChartNoAxesColumn,
        Circle,
        CircleOff,
        CircleQuestionMark,
        CircleStop,
        Gauge,
        ImageOff,
        ImagePlus,
        Images,
        Layers,
        LayoutGrid,
        Microchip,
        ScanSearch,
        Timer,
        Trophy,
    ]
);

/// Asset source for the window: the selected extra icons, then the Kit's component assets.
pub struct AppAssets;

impl gpui_kit::AssetSource for AppAssets {
    fn load(&self, path: &str) -> gpui_kit::Result<Option<std::borrow::Cow<'static, [u8]>>> {
        match ExtraIcons.load(path)? {
            Some(bytes) => Ok(Some(bytes)),
            None => gpui_kit::assets::Assets.load(path),
        }
    }

    fn list(&self, path: &str) -> gpui_kit::Result<Vec<SharedString>> {
        let mut all = ExtraIcons.list(path)?;
        all.extend(gpui_kit::assets::Assets.list(path)?);
        Ok(all)
    }
}

use super::models::ModelState;
use super::theme::{self, Roles};
use inference::classification::{Answer, Classification};
use inference::service::{LoadStage, RunError, Timing};

pub fn pct(p: f32) -> String {
    if p >= 0.995 {
        format!("{:.1}%", p * 100.0)
    } else if p < 0.005 {
        "<1%".to_string()
    } else {
        format!("{:.0}%", p * 100.0)
    }
}

/// A load or run time: "16 ms" below a second, "2.1 s" above.
pub fn duration(ms: f64) -> String {
    if ms < 1000.0 { format!("{ms:.0} ms") } else { format!("{:.1} s", ms / 1000.0) }
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

/// An icon before text that wraps under itself instead of running past its container.
pub fn icon_line(icon: Icon, text: impl Into<SharedString>) -> Div {
    h_flex().gap_1().items_start().child(icon.flex_none()).child(div().flex_1().min_w_0().child(text.into()))
}

/// Small semibold muted caption above a group ("Prediction", "Other classes", "Photo credit").
pub fn caption(text: impl Into<SharedString>, cx: &App) -> Div {
    div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(cx.theme().muted_foreground).child(text.into())
}

/// Rounded bar: `value` in 0..=1 of `fill` over the theme track.
pub fn bar(value: f32, fill: Hsla, height: Pixels, r: &Roles) -> Div {
    let v = value.clamp(0.0, 1.0);
    div()
        .w_full()
        .h(height)
        .rounded_full()
        .bg(r.track)
        .overflow_hidden()
        .child(div().h_full().w(relative(v)).rounded_full().bg(fill))
}

/// Pill: icon + word on a soft tint of `color`.
pub fn tag(icon: Lucide, text: &'static str, (color, soft): (Hsla, Hsla), small: bool) -> Tag {
    let tag = Tag::custom(soft, color, color.opacity(0.35)).rounded_full().child(
        h_flex()
            .gap_1()
            .items_center()
            .child(Icon::new(icon).xsmall())
            .child(div().font_weight(FontWeight::SEMIBOLD).child(text)),
    );
    if small { tag.small() } else { tag }
}

/// The small pill of an answer whose head abstained.
pub fn unsure_tag(cx: &App) -> Tag {
    let r = theme::roles(cx);
    tag(Lucide::CircleQuestionMark, "Unsure", (r.unsure, r.unsure_soft), true)
}

/// Label + percentage + bar. `strong` = the committed top-1 (accent bar), others are muted.
pub fn prob_row(label: &str, p: f32, strong: bool, large: bool, cx: &App) -> AnyElement {
    let r = theme::roles(cx);
    let fill = if strong { r.accent } else { r.bar_muted };
    v_flex()
        .gap_1()
        .child(
            h_flex()
                .justify_between()
                .items_baseline()
                .when(large, |d| d.text_base())
                .when(!large, |d| d.text_sm())
                .child(
                    div()
                        .text_color(r.fg)
                        .truncate()
                        .when(strong, |d| d.font_weight(FontWeight::SEMIBOLD))
                        .child(label.to_string()),
                )
                .child(div().text_sm().text_color(if strong { r.fg } else { r.muted_fg }).child(pct(p))),
        )
        .child(bar(p, fill, px(if large { 8. } else { 5. }), &r))
        .into_any_element()
}

/// Headline + top-3 + all classes, with `note` under the headline (the Demo detail pane's verdict). Used by the
/// Classify result panel and the Demo detail pane.
pub fn prediction_block(
    c: &Classification,
    timing: &Timing,
    title: &str,
    note: Option<AnyElement>,
    cx: &App,
) -> AnyElement {
    let r = theme::roles(cx);
    let headline = match c.answer {
        Answer::Abstain { best, p, below } => v_flex()
            .gap_1()
            .child(caption("Prediction", cx))
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .text_color(r.unsure)
                    .child(Icon::new(Lucide::CircleQuestionMark).large())
                    .child(div().text_3xl().font_weight(FontWeight::BOLD).child("Unsure")),
            )
            .child(div().text_sm().text_color(r.muted_fg).child(format!(
                "Top probability {} is below the {} threshold, so the model abstains. Best guess: {}.",
                pct(p),
                pct(below),
                c.labels.name(best)
            ))),
        Answer::Class { idx, p, low_confidence } => v_flex()
            .gap_1()
            .child(caption("Prediction", cx))
            .child(
                div().text_3xl().font_weight(FontWeight::BOLD).text_color(r.fg).child(c.labels.name(idx).to_string()),
            )
            .child(
                h_flex()
                    .gap_2()
                    .items_baseline()
                    .child(div().text_2xl().font_weight(FontWeight::SEMIBOLD).text_color(r.accent).child(pct(p)))
                    .child(div().text_sm().text_color(r.muted_fg).child(if c.calibrated {
                        "confidence · calibrated"
                    } else {
                        "confidence · softmax, not calibrated"
                    })),
            )
            .when(low_confidence, |d| {
                d.child(
                    icon_line(
                        Icon::new(IconName::Info).xsmall(),
                        "Low confidence. This model is not calibrated, so it never abstains; it always answers.",
                    )
                    .text_xs()
                    .text_color(r.muted_fg),
                )
            }),
    };
    let ranked = c.top(c.labels.len());
    let strong = c.committed();
    let rest = ranked.iter().skip(3).map(|&(i, p)| {
        h_flex()
            .justify_between()
            .text_xs()
            .text_color(r.muted_fg)
            .child(div().truncate().child(c.labels.name(i).to_string()))
            .child(div().child(format!("{:.1}%", p * 100.0)))
    });
    v_flex()
        .gap_5()
        .child(headline)
        .children(note)
        .child(
            v_flex().gap_3().child(caption("Top 3", cx)).children(
                ranked.iter().take(3).map(|&(i, p)| prob_row(c.labels.name(i), p, strong == Some(i), true, cx)),
            ),
        )
        .child(
            v_flex()
                .gap_1()
                .pt_3()
                .border_t_1()
                .border_color(r.border)
                .child(caption("Other classes", cx).pb_1())
                .children(rest),
        )
        .child({
            let shared = if timing.shared_by > 1 {
                format!(" · one pass for {} heads", timing.shared_by)
            } else {
                String::new()
            };
            icon_line(
                Icon::new(Lucide::Timer).xsmall(),
                format!("{title} · preprocess + inference {:.0} ms{shared}", timing.ms()),
            )
            .text_xs()
            .text_color(r.muted_fg)
        })
        .into_any_element()
}

/// Short status of a failed answer: the picture could not be read, or the model failed.
pub fn failure_label(e: &RunError) -> &'static str {
    match e {
        RunError::Read(_) | RunError::Decode(_) => "could not read",
        RunError::Preprocess(_) => "could not prepare",
        RunError::Unavailable(_) => "model unavailable",
        RunError::Load(_) => "model failed to load",
        RunError::Run(_) => "model failed",
    }
}

/// Error alert for a failed answer from the model titled `model`, saying what failed.
pub fn failure_alert(id: impl Into<ElementId>, e: &RunError, model: &str) -> Alert {
    let (title, detail) = match e {
        RunError::Read(d) | RunError::Decode(d) => ("Could not read this picture".to_string(), d),
        RunError::Preprocess(d) => (format!("Could not prepare this picture for {model}"), d),
        RunError::Unavailable(d) => (format!("{model} is unavailable"), d),
        RunError::Load(d) => (format!("{model} failed to load"), d),
        RunError::Run(d) => (format!("{model} failed on this picture"), d),
    };
    Alert::error(id, SharedString::from(detail.clone())).title(title)
}

/// Compact model state: spinner while loading, otherwise a short pill.
pub fn model_state_chip(state: &ModelState, cx: &App) -> AnyElement {
    let r = theme::roles(cx);
    let pill = |icon: Lucide, text: String, color: Hsla| {
        h_flex()
            .gap_1()
            .items_center()
            .px_2()
            .py_0p5()
            .rounded_full()
            .border_1()
            .border_color(r.border)
            .bg(r.surface)
            .text_xs()
            .text_color(color)
            .child(Icon::new(icon).xsmall())
            .child(text)
            .into_any_element()
    };
    match state {
        ModelState::Idle => pill(Lucide::Circle, "loads on first use".into(), r.muted_fg),
        ModelState::Loading { since, stage } => h_flex()
            .gap_1()
            .items_center()
            .px_2()
            .py_0p5()
            .rounded_full()
            .border_1()
            .border_color(r.accent.opacity(0.5))
            .bg(r.accent_soft)
            .child(Spinner::new().xsmall().color(r.accent))
            .child(div().text_xs().text_color(r.fg).child(match stage {
                LoadStage::BuildingEngine => {
                    format!("building TensorRT engine, first run only · {:.0} s", since.elapsed().as_secs_f64())
                }
                LoadStage::Verifying | LoadStage::Loading => format!("loading… {:.0} s", since.elapsed().as_secs_f64()),
            }))
            .into_any_element(),
        ModelState::Ready(l) => pill(
            Lucide::CircleCheck,
            format!("ready on {} · loaded in {}", l.placement.primary().name(), duration(l.ms)),
            r.muted_fg,
        ),
        ModelState::Unavailable(_) => pill(Lucide::CircleOff, "not exported".into(), r.muted_fg),
        ModelState::Failed(_) => pill(Lucide::CircleAlert, "failed to load".into(), r.wrong),
    }
}

#[cfg(test)]
mod unit {
    use super::*;
    use gpui_kit::AssetSource as _;

    #[test]
    fn durations_keep_their_digits() {
        assert_eq!(duration(16.4), "16 ms", "a fast load does not read as 0.0 s");
        assert_eq!(duration(2137.0), "2.1 s");
    }

    /// Every Lucide icon named in the GUI sources must resolve through `AppAssets`, or it renders blank.
    #[test]
    fn every_lucide_icon_used_is_bundled() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = vec![dir.join("gui.rs"), dir.join("app.rs")];
        files.extend(std::fs::read_dir(dir.join("gui")).unwrap().map(|e| e.unwrap().path()));
        let mut used = std::collections::BTreeSet::new();
        for f in files.iter().filter(|f| f.extension().is_some_and(|e| e == "rs")) {
            let src = std::fs::read_to_string(f).unwrap();
            for part in src.split("Lucide::").skip(1) {
                let name: String = part.chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
                // variants are CamelCase; skips `Lucide::ALL` and format strings
                if name.chars().next().is_some_and(|c| c.is_ascii_uppercase())
                    && name.chars().any(|c| c.is_ascii_lowercase())
                {
                    used.insert(name);
                }
            }
        }
        assert!(used.len() > 5, "found {used:?}");
        for name in &used {
            let icon =
                Lucide::ALL.iter().find(|i| format!("{i:?}") == *name).unwrap_or_else(|| panic!("no Lucide::{name}"));
            let bytes = AppAssets.load(&icon.path()).unwrap();
            assert!(bytes.is_some(), "Lucide::{name} ({}) is not bundled; add it to ExtraIcons", icon.path());
        }
    }
}
