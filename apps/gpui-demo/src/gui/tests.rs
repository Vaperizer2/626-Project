//! The harness of the headless window tests (GPUI Kit's test platform, feature `test-support`: no display, no GPU).
//! A test opens the real window, `Shell` with the tabs it names, over the real inference service with the fake
//! loader of `inference::testing` behind it. The shell's tests are in `tests/shell.rs`. Each tab's tests sit beside
//! the tab in `<tab>/tests.rs`, so deleting a tab deletes its tests.

mod shell;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use gpui_kit::{
    AnyWindowHandle, App, AppContext as _, Bounds, Context, Entity, FocusHandle, Pixels, Point, Size, TestAppContext,
    Window, WindowBounds, WindowOptions, div, prelude::*, px, size,
};

use super::classify::ClassifyTab;
use super::models::Models;
use super::shell::Shell;
use super::{Cancel, GuiOptions, Hint, Lucide, RunAll, Tab, TabSpec};
use crate::app;
use inference::fetch::Source;
use inference::manifest::HeadId;
use inference::registry::Registry;
use inference::runtime::Settings;
use inference::service::{ImageInput, Inference};
use inference::testing::FakeLoader;

/// The shell's tests open the input tab and a tab that takes no pictures.
pub const SHELL_TABS: &[TabSpec] = &[TabSpec::of::<ClassifyTab>(), TabSpec::of::<ProbeTab>()];

/// A tab that takes no pictures and records what the shell asks of it.
pub struct ProbeTab {
    models: Entity<Models>,
    focus: FocusHandle,
    /// the model to run, each time the shell asked the tab to follow it
    pub followed: Vec<Option<HeadId>>,
    /// from its run (Ctrl+R) to Esc
    pub running: bool,
}

impl Tab for ProbeTab {
    const TITLE: &'static str = "Probe";
    const ICON: Lucide = Lucide::Circle;
    const HINTS: &'static [Hint] = &[(&["ctrl-r"], "Run"), (&["escape"], "Stop")];

    fn new(models: Entity<Models>, _: &app::Shared, _: &mut Window, cx: &mut Context<Self>) -> Self {
        Self { models, focus: cx.focus_handle(), followed: Vec::new(), running: false }
    }

    fn focus_handle(&self) -> FocusHandle {
        self.focus.clone()
    }

    fn follow_model(&mut self, cx: &mut Context<Self>) {
        self.followed.push(self.models.read(cx).run_model());
    }

    fn run(&mut self, _: &mut Context<Self>) {
        self.running = true;
    }
}

impl Render for ProbeTab {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("probe-tab")
            .track_focus(&self.focus)
            .key_context("Probe")
            .on_action(cx.listener(|this, _: &RunAll, _, cx| this.run(cx)))
            .on_action(cx.listener(|this, _: &Cancel, _, _| this.running = false))
            .size_full()
    }
}

/// The open tab of type `T`.
pub fn tab<T: Tab>(v: &Shell) -> Entity<T> {
    v.tabs.iter().find_map(|t| t.view.clone().downcast::<T>().ok()).expect("the tab is open")
}

/// The index of tab `T` in `tabs`.
pub fn ix<T: Tab>(tabs: &[TabSpec]) -> usize {
    tabs.iter().position(|s| s.title == T::TITLE).expect("the tab is listed")
}

/// A small picture file of its own for each `tag`.
pub fn picture(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("{}-picture-{tag}-{}.png", app::BIN, std::process::id()));
    let shade = tag.bytes().fold(0u8, u8::wrapping_add);
    image::RgbImage::from_pixel(64, 48, image::Rgb([shade, 128, 255 - shade])).save(&p).expect("write a test picture");
    p
}

/// An index with model A's manifest and labels but not its graph.
pub fn index_without_graphs(tag: &str) -> PathBuf {
    let tmp = std::env::temp_dir().join(format!("{}-gui-{tag}-{}", app::BIN, std::process::id()));
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../exports/itri626/A_scratch_cnn");
    std::fs::create_dir_all(tmp.join("A")).unwrap();
    for f in ["model.json", "labels.json"] {
        std::fs::copy(src.join(f), tmp.join("A").join(f)).unwrap();
    }
    std::fs::write(tmp.join("models.json"), r#"{"schema":"model-index/1","default":["a"],"manifests":["A/model.json"]}"#)
        .unwrap();
    tmp
}

/// The registry of the index in `dir`.
pub fn registry_in(dir: &Path) -> Arc<Registry> {
    Arc::new(Registry::open(&Settings { models: Some(dir.to_path_buf()), ..Default::default() }).unwrap())
}

pub fn open(cx: &mut TestAppContext, registry: Arc<Registry>, tabs: &'static [TabSpec]) -> (AnyWindowHandle, Entity<Shell>) {
    open_with(cx, registry, FakeLoader::default(), tabs)
}

pub fn open_with(
    cx: &mut TestAppContext,
    registry: Arc<Registry>,
    loader: FakeLoader,
    tabs: &'static [TabSpec],
) -> (AnyWindowHandle, Entity<Shell>) {
    open_sized(cx, registry, loader, tabs, size(px(1360.), px(860.)), Vec::new(), GuiOptions::default())
}

/// The window with `tabs` over `registry` and `loader`, `window` big, given `initial` pictures and `gui` as on the
/// command line.
pub fn open_sized(
    cx: &mut TestAppContext,
    registry: Arc<Registry>,
    loader: FakeLoader,
    tabs: &'static [TabSpec],
    window: Size<Pixels>,
    initial: Vec<String>,
    gui: GuiOptions,
) -> (AnyWindowHandle, Entity<Shell>) {
    // The app's inference and thumbnail threads are real OS threads that wake GPUI tasks; tell the
    // deterministic test scheduler that this is expected.
    cx.executor().allow_parking();
    cx.update(|cx| {
        gpui_kit::init(cx);
        super::theme::install(cx);
    });
    cx.update(|cx| cx.bind_keys(super::keymap()));
    let service = Inference::spawn(registry, loader);
    let (handle, view) = cx.update(|cx| {
        let bounds = Bounds { origin: Point::default(), size: window };
        gpui_kit::open_window(
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() },
            cx,
            |window, cx| cx.new(|cx| Shell::new(service, &gui, initial, tabs, window, cx)),
        )
        .expect("open test window")
    });
    cx.run_until_parked();
    (handle, view)
}

/// Poll `done` while the service threads work, or panic after about 5 s.
pub fn wait(cx: &mut TestAppContext, view: &Entity<Shell>, what: &str, done: &dyn Fn(&Shell, &App) -> bool) {
    for _ in 0..200 {
        cx.run_until_parked();
        if cx.update(|cx| done(view.read(cx), cx)) {
            return;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!("timed out waiting for {what}");
}

/// The picture at `path` reaches the window, as from Ctrl+O.
pub fn submit(cx: &mut TestAppContext, handle: AnyWindowHandle, view: &Entity<Shell>, path: PathBuf) {
    cx.update_window(handle, |_, window, cx| {
        view.update(cx, |v, cx| v.send_inputs(vec![ImageInput::Source(Source::File(path))], window, cx))
    })
    .unwrap()
}

/// Show tab `ix`, then run `f` on the shell, as a user does on that tab.
pub fn on(
    cx: &mut TestAppContext,
    handle: AnyWindowHandle,
    view: &Entity<Shell>,
    ix: usize,
    f: &dyn Fn(&mut Shell, &mut Window, &mut Context<Shell>),
) {
    cx.update_window(handle, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.set_tab(ix, window, cx);
            f(v, window, cx);
        })
    })
    .unwrap()
}

pub fn selected(v: &Shell, cx: &App) -> Option<HeadId> {
    v.models.read(cx).selected().cloned()
}

pub fn run_model(v: &Shell, cx: &App) -> Option<HeadId> {
    v.models.read(cx).run_model()
}

/// The user picks `k` in the model picker.
pub fn select(cx: &mut TestAppContext, handle: AnyWindowHandle, view: &Entity<Shell>, k: &HeadId) {
    cx.update_window(handle, |_, window, cx| {
        let models = view.read(cx).models.clone();
        models.update(cx, |m, cx| m.select_model(k.clone(), window, cx))
    })
    .unwrap()
}
