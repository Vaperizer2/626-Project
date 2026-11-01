//! The app's two GPUI Kit themes, "Bench Light" and "Bench Dark" (`themes/bench.json`, rationale in
//! `docs/VISUAL_DIRECTION.md`), and the few app-level roles derived from them.
//!
//! Views never hard-code colours: they read Kit roles through `cx.theme()` or, for meanings the Kit has no
//! name for (a failure, an abstention, bar tracks, the canvas-chrome-surface layering), through [`roles`].

use std::rc::Rc;

use gpui_kit::component::{ActiveTheme as _, Theme, ThemeConfig, ThemeMode, ThemeRegistry};
use gpui_kit::{App, Hsla, Window};

/// ThemeSet JSON in the Kit's theme-file format (same keys as the Kit's `default-theme.json`).
pub const THEME_SET: &str = include_str!("themes/bench.json");
pub const LIGHT: &str = "Bench Light";
pub const DARK: &str = "Bench Dark";

/// UI families tried in order. GPUI's Linux text system (cosmic-text) renders only the default instance of a
/// variable font: with a variable system font (GNOME's Adwaita Sans) the matcher reports a 600-weight face for
/// SEMIBOLD/BOLD, but the glyphs are drawn at the regular weight, so the type hierarchy disappears. These families ship static Bold faces; Noto Sans is also embedded (`register_bundled_fonts`). If none is found the
/// themes keep the system UI font.
pub const UI_FONTS: &[&str] = &["Noto Sans", "DejaVu Sans", "Liberation Sans", "Ubuntu"];

/// The embedded theme set with `font.family` set to `family` in every theme (or unchanged for `None`).
pub fn theme_set_with_font(family: Option<&str>) -> String {
    let Some(family) = family else { return THEME_SET.to_string() };
    let mut v: serde_json::Value = match serde_json::from_str(THEME_SET) {
        Ok(v) => v,
        Err(_) => return THEME_SET.to_string(),
    };
    if let Some(themes) = v["themes"].as_array_mut() {
        for t in themes {
            t["font.family"] = serde_json::Value::String(family.to_string());
        }
    }
    v.to_string()
}

/// Noto Sans Regular + Bold (SIL OFL 1.1, `assets/fonts/OFL.txt`), embedded so the type looks the same on
/// machines without it (+1.2 MB). Registered before the themes pick `UI_FONTS`, so "Noto Sans" is always found.
const BUNDLED_FONTS: [&[u8]; 2] =
    [include_bytes!("../../assets/fonts/NotoSans-Regular.ttf"), include_bytes!("../../assets/fonts/NotoSans-Bold.ttf")];

/// Add the embedded fonts to GPUI's text system. On failure the app keeps the system fonts.
pub fn register_bundled_fonts(cx: &mut App) {
    let fonts = BUNDLED_FONTS.iter().map(|b| std::borrow::Cow::Borrowed(*b)).collect();
    if let Err(e) = cx.text_system().add_fonts(fonts) {
        eprintln!("{}: could not register the bundled Noto Sans ({e:#}); using system fonts", crate::app::BIN);
    }
}

/// Register the Bench themes and make them the Kit's light / dark themes. Call once after `gpui_kit::init`.
/// Falls back to the Kit defaults (with a message) if the embedded file ever fails to parse.
pub fn install(cx: &mut App) {
    let installed = cx.text_system().all_font_names();
    let family = UI_FONTS.iter().copied().find(|f| installed.iter().any(|n| n == f));
    if let Err(e) = ThemeRegistry::global_mut(cx).load_themes_from_str(&theme_set_with_font(family)) {
        eprintln!("{}: built-in theme failed to load ({e:#}); using the GPUI Kit defaults", crate::app::BIN);
        return;
    }
    let (light, dark) = {
        let themes = ThemeRegistry::global(cx).themes();
        (themes.get(LIGHT).cloned(), themes.get(DARK).cloned())
    };
    if let (Some(light), Some(dark)) = (light, dark) {
        set_configs(light, dark, cx);
    }
}

fn set_configs(light: Rc<ThemeConfig>, dark: Rc<ThemeConfig>, cx: &mut App) {
    let theme = Theme::global_mut(cx);
    theme.light_theme = light;
    theme.dark_theme = dark;
}

/// Apply the start-up appearance: `Some(true)` dark, `Some(false)` light, `None` follow the system.
pub fn apply_initial(dark: Option<bool>, window: &mut Window, cx: &mut App) {
    match dark {
        Some(dark) => Theme::change(if dark { ThemeMode::Dark } else { ThemeMode::Light }, Some(window), cx),
        None => Theme::sync_system_appearance(Some(window), cx),
    }
}

/// App-level roles, resolved from the active Kit theme.
#[derive(Clone, Copy)]
pub struct Roles {
    /// Window background behind content (darkest in dark mode, light grey in light mode).
    pub canvas: Hsla,
    /// Title bar, tab row, toolbar, status bar.
    pub chrome: Hsla,
    /// Cards and panels that sit on the canvas.
    pub surface: Hsla,
    /// Hairline between regions.
    pub border: Hsla,
    /// Control outlines and the selected card ring.
    pub border_strong: Hsla,
    pub fg: Hsla,
    pub muted_fg: Hsla,
    /// The one accent: primary action, committed top-1 bar, selection.
    pub accent: Hsla,
    pub accent_soft: Hsla,
    /// Empty part of a bar.
    pub track: Hsla,
    /// Non-top-1 bars.
    pub bar_muted: Hsla,
    /// A failure (vermillion).
    pub wrong: Hsla,
    /// An abstention (violet), and its soft tint.
    pub unsure: Hsla,
    pub unsure_soft: Hsla,
}

pub fn roles(cx: &App) -> Roles {
    let t = cx.theme();
    Roles {
        canvas: t.background,
        chrome: t.title_bar,
        surface: t.group_box,
        border: t.border,
        border_strong: t.input,
        fg: t.foreground,
        muted_fg: t.muted_foreground,
        accent: t.primary,
        accent_soft: t.list_active,
        track: t.skeleton,
        bar_muted: t.muted_foreground.opacity(0.45),
        wrong: t.danger,
        unsure: t.magenta,
        unsure_soft: t.magenta_light,
    }
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn embedded_theme_set_parses_with_both_modes() {
        let v: serde_json::Value = serde_json::from_str(THEME_SET).expect("bench.json is valid JSON");
        let names: Vec<&str> = v["themes"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(names, [LIGHT, DARK]);
        let with: serde_json::Value = serde_json::from_str(&theme_set_with_font(Some("Noto Sans"))).unwrap();
        assert!(with["themes"].as_array().unwrap().iter().all(|t| t["font.family"] == "Noto Sans"));
    }
}
