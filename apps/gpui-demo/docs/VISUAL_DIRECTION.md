# Visual direction: gpui-demo

This note covers how the desktop demo should look, why it looks that way, and how the look is built.
The app is used live: on a classroom projector, by a presenter who explains a classifier to people
seeing it for the first time. The four reference apps were chosen by the author. They are all built
with GPUI (Zed's UI framework) or GPUI Component.

Sources studied on 2026-10-20:

- **Delta** (delta.dev, from the Zed team): the site's HTML/CSS app mock, which uses the Zed One
  theme variables, and the OG image. No downloadable build or app screenshots exist.
- **GitComet** (gitcomet.dev, github.com/Auto-Explore/GitComet): the press PNGs (merge, release,
  workspaces, stage, diff) and the README.
- **Cat** (getcat.io, github.com/finch-xu/GetCat): the site screenshot and the OG image.
- **Probe** (github.com/crizant/probe): `theme.rs`, `docs/DESIGN.md`, the README screenshot. Probe
  is built on gpui-base semantic tokens, the same layer GPUI Kit uses.

Hex values below were read from the CSS or theme source where one exists (Delta, Probe). For
GitComet and Cat they were sampled from the screenshots, so treat them as ±2 per channel.

## 1. What each reference does

### Delta: quiet editor chrome (Zed One)

| Trait | Observation |
|---|---|
| Palette, light | background `#dcdcdd`, panels `#ebebec`, editor `#fafafa`, hover `#dfdfe0`, selected `#dadbdd`; text `#242529` / muted `#58585a`; accent `#5c78e2` |
| Palette, dark | background `#3b414d`, surface `#22252a`, panel `#202327`, editor `#191c1f`, hover `#272a2f`, selected `#2c2f35`; text `#dce0e5` / muted `#a9afbc`; accent `#74ade8` |
| Status | success `#669f59` / bg `#dfeadb`, warning `#a48819` / `#faf2e6`, error `#d36151` / `#fbdfd9`; in dark the same hues at 10% alpha for backgrounds |
| Layering | 3–4 flat grey steps (background → panel → editor). Hairline borders (`#c9c9ca` / `#32363e`), almost no shadows |
| Radii | Small, 4–6 px. Panels meet edge to edge |
| Type | System UI at 13–14 px. The marketing site uses a serif headline on warm cream; the app does not |
| Density | Compact editor density, ~24–28 px rows |
| Chrome | Title bar, panel and status bar are the same grey; the editor is the lightest (or darkest) region |
| Takeaway | Separate regions with **tone**, not boxes. One cool accent. Muted status colours with soft tints |

### GitComet: dark, dense, data-first

| Trait | Observation |
|---|---|
| Palette | content `#0b0d11`; chrome (toolbar, sidebar, status bar) `#1b2128`; title bar `#262a2f`; hover `#1b1e21`; border `#13181f` |
| Selection | sidebar `#25354d`, list row `#16223a`: a dark navy, not a bright fill |
| Accent | cyan `#5ac1fe` (site background `#0d0f13` / `#00131d`); badges are fully rounded pills, blue `#4d76b2` |
| Layering | Chrome is **lighter** than content in dark mode: the content well recedes and the frame carries the controls |
| Type | IBM Plex Sans; small (12–13 px) labels, semibold section headers with icons |
| Density | ~28 px rows, tight lists, counts right-aligned |
| Empty states | One calm muted line ("Nothing staged yet."), no illustration |
| Dark / light | Dark-first; a lavender-grey light theme exists |
| Takeaway | Dark navy selection, pill badges, section headers with an icon and a count, quiet empty states |

### Cat: light, rounded, friendly API client

| Trait | Observation |
|---|---|
| Palette | window `#f2f4f5`, panels white with a `#e2e7e5` border; primary blue `#1d63d2` (the Send button) |
| Layering | **Inset cards**: white panels with 10–12 px radius on a light grey window |
| Controls | Segmented controls: `#e3e7eb` track with a white selected segment. Tab underline in the primary blue. Left icon rail |
| Status | An outlined green pill "200 OK" (`#367969`) followed by muted metrics "742 ms 705 B JSON" |
| Type | System sans, 13–14 px; generous padding |
| Takeaway | A light theme that reads well from a distance: white surfaces on grey, one strong blue action, the status as a pill followed by quiet numbers |

### Probe: warm, token-driven, documented

| Trait | Observation |
|---|---|
| Light ("Porcelain Honey") | window `#f8f6f2`, sidebar `#f1eee9`, editor `#faf8f5`, raised `#fcf9f5`; borders `#e7e1d9` / `#dcd4cb` / `#c5b9ad`; text `#4b4641` / `#6c655e` / `#756e66`; accent `#e7821b`; success `#2d8a5b`, error `#c43d3d`, info `#227c8f` |
| Dark ("Graphite Honey") | window `#1d1e22`, sidebar `#1a1b1f`, editor `#222327`, overlay `#292a2e`; borders as alpha (`#33353aa6`…); text `#e9e6e0` / `#cac5bd` / `#a7a19a`; accent `#d98e26` |
| Metrics | spacing 4/8/12/16/20/24, radius 6/8/10, control height 28, tab bar 32, icons 12/16; body 13, caption 12, title 24, line height 1.45; JetBrains Mono for code |
| Principles | "quiet, legible, dense without cramping", semantic tokens only, never colour alone |
| Takeaway | The method: a small named token set, three border strengths, three text strengths, and views that never use raw colours |

## 2. Synthesis for a classroom demo

The demo has different constraints from the references. Those are tools used at arm's length for
hours; this app is seen from 3–10 m for a few minutes, through a projector that washes out dark
tones and low contrast. So:

1. **Layer with tone, like Delta and GitComet, plus one inset-card level, like Cat.** Three
   surfaces: *chrome* (title bar, tab row, toolbars, status bar), *canvas* (the work area behind the
   pictures) and *surface* (cards and result panels). Projectors flatten 1–2% luminance steps, so
   the steps are 4–8% L*, and every region also gets a hairline border.
2. **One accent, and it means "the answer".** It is used for the primary action, the committed
   top-1 bar, the selected card and the best model. Cool blue: Cat's `#1d63d2` direction in light
   mode, GitComet's cyan-blue in dark. Nothing else uses it, so the eye goes to the prediction.
3. **Verdicts are colour-blind safe and never colour alone.** Correct / wrong / unsure use
   bluish-green / vermillion / violet (Okabe–Ito-inspired). Each always comes with a distinct icon
   shape (✓ circle, ✕ circle, ? circle) and a word. The first draft used amber for *unsure*, as Delta
   and Probe use for warnings. A deuteranopia simulation (Machado 2009) put amber at ΔE 7.7 from
   vermillion, which is indistinguishable, so violet replaced it (minimum pairwise ΔE across normal,
   deutan, protan and tritan vision: 36 light, 29 dark). All verdict colours are ≥ 4.5 : 1 against
   their card.
4. **Large numerals for the numbers the room should read**: the predicted class (30 px bold),
   its confidence (24 px in the accent), score lines such as `21/28` (30 px). Numbers that sit in
   columns (percentages, counts, timings) use the monospace family so the digits align.
5. **Soft, low-saturation fills; strong colour only in small marks.** Benchmark cells were solid
   red/green tiles. Now they are neutral cards with a 4 px verdict edge, a verdict icon and a thin
   confidence bar, like GitComet's dense lists and Cat's "pill + muted metrics" status line. A
   28 × 4 matrix of mostly-wrong A-model cells no longer floods the screen red.
6. **Pills for status, à la GitComet and Cat**: verdict tags, the model-state chip, "best".
7. **Calm empty states, à la GitComet**: one icon, one short title, one line of help, the
   shortcuts. Errors get their own empty state rather than a spinner that never stops.
8. **Consistent Lucide icons**: tabs (scan-search, layout-grid, chart-column), the app mark
   (microchip), verdicts, metrics (gauge = mean confidence, timer = latency). They come from the
   full catalog in `gpui_kit::assets::IconName`, re-exported as `widgets::Lucide`; the component
   crate's `IconName` is a subset.
9. **Radii:** 6 px for controls and cells, 10 px for cards (Probe / Cat). Flat, no shadows (Delta).

### Light or dark by default?

Three of the four references are dark-first, and the dark theme here is the more striking one.
But in a lit classroom a projector cannot make black darker than the screen under room light. Dark
UIs lose their layering and look muddy grey, while dark text on white survives. So the default stays
**follow the system** (the author's desktop is light). Use `--dark` in a dimmed room, `--light` to
force light, and Ctrl+T toggles live. Both themes are first-class and screenshot-tested.

### Fonts

GitComet's IBM Plex and Probe's Inter-like stack are not installed on the target machine, and bundling
fonts would add MBs and licence text for little gain, so nothing is bundled. The system UI font on the
author's GNOME desktop is **Adwaita Sans**, a *variable* font. GPUI's Linux text system (cosmic-text)
loads every face with `Weight::NORMAL`. The font matcher does report a 600-weight face for SEMIBOLD
and BOLD, but the glyphs are drawn from the default (regular) instance. In the screenshots, SEMIBOLD and
BOLD text looked the same as regular (zoomed crops of the "before" and pass-1 shots); switching to a
static family fixed it. `theme::install` therefore sets
`font.family` to the first installed family from `theme::UI_FONTS` (Noto Sans, DejaVu Sans,
Liberation Sans, Ubuntu); all of them ship static Bold faces. If none is installed, the system font is
kept.

## 3. Tokens

The two themes are GPUI Kit theme files: `src/gui/themes/bench.json` (ThemeSet format, the same keys
as the Kit's `default-theme.json`). They are loaded at start-up by `gui/theme.rs` through
`ThemeRegistry::load_themes_from_str`, installed as `Theme::light_theme` / `dark_theme`, and switched
with `Theme::change`. Views read Kit roles through `cx.theme()`. The few app meanings the Kit has no
name for come from `theme::roles(cx)`, which is derived from the Kit theme (no hex in views).

| App role (`theme::Roles`) | Kit theme key | Bench Light | Bench Dark | Used for |
|---|---|---|---|---|
| canvas | `background` | `#f4f5f7` | `#0f1216` | Work area behind pictures, grid, matrix |
| chrome | `title_bar.background` (also `tab_bar`, `status_bar`) | `#eceef1` | `#161a20` | Title bar, tab row, toolbars, status bar |
| surface | `group_box.background` | `#ffffff` | `#161a20` | Cards, result and detail panels |
| sidebar | `sidebar.background` | `#f9fafb` | `#12151a` | Recent list |
| border | `border` | `#dde1e6` | `#262c35` | Hairlines between regions |
| border_strong | `input.border` | `#d3d8de` | `#303742` | Control outlines, drop zone |
| fg | `foreground` | `#1b1e23` | `#e3e7ed` | Text |
| muted_fg | `muted.foreground` | `#5b6270` | `#98a2b0` | Secondary text, captions |
| accent | `primary.background` | `#2160d4` | `#5aa8f2` | Primary action, top-1 bar, selection ring, best model |
| accent_soft | `list.active.background` | `#e4ecfb` | `#172a45` | Selected rows, the running row, the "best" pill |
| track | `skeleton.background` | `#e6e9ee` | `#1f252e` | Empty part of every bar |
| bar_muted | `muted.foreground` @ 45% | — | — | Non-top-1 bars, confidence bars in cells |
| correct | `success.background` / `base.green.light` | `#007a5e` / `#e0f2ec` | `#4cd3a4` / `#123129` | ✓ correct |
| wrong | `danger.background` / `base.red.light` | `#c4480a` / `#fbe8de` | `#ff6b4a` / `#3a1a14` | ✕ wrong, errors |
| unsure | `base.magenta` / `base.magenta.light` | `#7b5ea7` / `#ede7f6` | `#b9a3ff` / `#2a2340` | ? abstained |
| (warning) | `warning.background` | `#a86f00` | `#f0c24b` | Real warnings only (missing demo stack) |
| radius / radius_lg | `radius`, `radius.lg` | 6 / 10 px | 6 / 10 px | Controls and cells / cards |

In dark mode, chrome and surface share `#161a20`: the frame and the cards sit one step above the
`#0f1216` canvas, as in GitComet.

**Type scale** (GPUI rem steps, 16 px root): caption `text_xs` 12 · body `text_sm` 14 · emphasis
`text_base` 16 · section `text_lg` 18 · numerals `text_2xl` 24 / `text_3xl` 30. Weights: regular,
medium for list labels, semibold for headers and captions, bold for the headline class and scores.

**Figures**: Noto Sans digits are tabular, so percentages and counts line up in columns without a monospace
face. Pass 1 used the system monospace font for them, and it looked out of place next to the sans text.

**Spacing**: the 4 px grid (Probe): 4/8/12/16/20/24/32. Panels pad 16–20, cards 8–16.

**Bars**: 8 px (top-3 in the result panel), 6 px (progress, accuracy), 5 px (compact rows), 3 px (cell
confidence). All are fully rounded, on `track`.

## 4. Passes

Every pass was checked with screenshots of each tab and state, light and dark, in a virtual X server
(Xvfb + Mesa software GL, 1360 × 860 window). The states: Classify empty / model loading / result /
unsure / error, Demo grid + detail during and after Run all, Benchmark before / during / after.

**Before (GPUI Kit default theme).** Monochrome. Title bar, tabs, toolbar and work area were all white
(or all black), with a black primary button. The benchmark matrix was a wall of solid red and green
tiles, and the amber "unsure" was confusable with red for deutan viewers. Bold text rendered regular.
The Classify error state left a "Reading picture…" spinner running in the picture pane.

**Pass 1: tokens and layout.**
- `bench.json` themes and `theme.rs` roles.
- Chrome / canvas / surface layering; one blue accent.
- Verdict trio: bluish green / vermillion / violet, each with its own icon shape, as pills.
- Large class name and confidence; own rounded bars with a visible track.
- Benchmark cells: neutral, with a verdict edge, icon and confidence bar. Accuracy bars and a "best"
  mark on the summary cards.
- Demo cards: a verdict strip under the thumbnail.
- Calm empty states for the empty result panel, the empty recent list and the error picture pane.
- Lucide icons on tabs, metrics and the app mark.

Critique of pass 1:
1. Every icon outside the Kit's 101 default icons rendered blank: the tab icons, the app mark, the "?"
   unsure icon and the empty-state icons. The Kit's `Assets` embeds only the component icons.
2. Tab labels vanished: `Tab::icon()` makes an icon-only tab.
3. Bold still rendered regular (the variable-font issue above).
4. The monospace figures looked out of place.
5. The selected model segment was a faint grey.

**Pass 2: fixes.**
1. `widgets::AppAssets` serves `icon_assets!(ExtraIcons, [...])` in front of the Kit assets (~20 KB),
   and a unit test fails if any `Lucide::X` used in `src/gui` is not bundled.
2. Tabs use `prefix(icon)` + label.
3. `theme::UI_FONTS` picks a static family with real bold.
4. Figures use the UI font.
5. The selected model is an outline-primary segment (accent tint, accent text).

The dark theme needed no separate fixes: the chrome and cards sit one step above the canvas, as in
GitComet.

## 5. What could not be matched, and why

- **Delta's actual app**: it is not publicly available. Only the HTML mock (Zed One tokens) and the
  OG image were studied, so its density and motion were not observed.
- **Typefaces**: IBM Plex (GitComet) and Inter (Probe, Delta's marketing site) are not installed and
  were deliberately not bundled. Noto Sans is the closest neutral sans that is installed and has a
  real bold.
- **SemiBold / Medium**: Noto Sans here has only Regular and Bold. Under CSS font matching, SEMIBOLD
  renders as Bold and MEDIUM as Regular, so headers are a little heavier and list labels a little
  lighter than designed.
- **Cat's left icon rail and Probe's sidebar trees**: these are navigation patterns for many
  documents. This app has three tabs, so the tab row stays.
- **Shadows and blur** (Cat's soft card shadow): left out on purpose. They read as smudges on a
  projector, and the software-rendered screenshots cannot verify them.
- **Real hardware**: only Xvfb + Mesa software rendering was available. Colours and anti-aliasing on
  the GPU (wgpu/Vulkan) renderer and on a real projector are untested. Check `--light` and `--dark`
  on the room's projector before the talk.
