//! Headless tests with GPUI Kit's test harness (`gpui-kit` feature `test-support`, a test platform: no display, no
//! GPU). They build the real window (`Shell` and the tabs of `app::TABS`) over the real inference service, with the
//! fake loader from `inference::testing` behind it: one renders every tab over an index whose graph file is absent,
//! one runs the benchmark and a Classify submission on fake models, one opens with no index at all.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use gpui_kit::test::TestWindowExt as _;
use gpui_kit::{
    AnyWindowHandle, App, AppContext as _, Bounds, Context, Entity, ExternalPaths, FileDropEvent, Pixels, Point,
    InputEvent as _, MouseMoveEvent, SharedString, Size, TestAppContext, Window, WindowBounds, WindowOptions, point, px, size,
};

use super::bench::BenchTab;
use super::classify::ClassifyTab;
use super::demo_tab::DemoTab;
use super::shell::Shell;
use super::stack::Stack;
use super::{GuiOptions, Tab};
use inference::fetch::Source;
use inference::manifest::HeadId;
use inference::service::{ImageInput, Inference};
use inference::testing::{FakeLoader, Seen, registry, shared_registry};
use inference::registry::Registry;
use inference::runtime::Settings;

fn stack_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("demo_stack")
}

/// The open tab of type `T`.
fn tab<T: Tab>(v: &Shell) -> Entity<T> {
    v.tabs.iter().find_map(|t| t.view.clone().downcast::<T>().ok()).expect("the tab is open")
}

/// The index of tab `T` in `app::TABS`.
fn ix<T: Tab>() -> usize {
    crate::app::TABS.iter().position(|s| s.title == T::TITLE).expect("the tab is listed")
}

fn classify<'a>(v: &Shell, cx: &'a App) -> &'a ClassifyTab {
    tab::<ClassifyTab>(v).read(cx)
}

fn demo<'a>(v: &Shell, cx: &'a App) -> &'a DemoTab {
    tab::<DemoTab>(v).read(cx)
}

fn bench<'a>(v: &Shell, cx: &'a App) -> &'a BenchTab {
    tab::<BenchTab>(v).read(cx)
}

/// An index with model A's manifest and labels but not its graph.
fn index_without_graphs(tag: &str) -> PathBuf {
    let tmp = std::env::temp_dir().join(format!("{}-gui-{tag}-{}", crate::app::BIN, std::process::id()));
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../exports/itri626/A_scratch_cnn");
    std::fs::create_dir_all(tmp.join("A")).unwrap();
    for f in ["model.json", "labels.json"] {
        std::fs::copy(src.join(f), tmp.join("A").join(f)).unwrap();
    }
    std::fs::write(tmp.join("models.json"), r#"{"schema":"model-index/1","default":["a"],"manifests":["A/model.json"]}"#)
        .unwrap();
    tmp
}

fn open(cx: &mut TestAppContext, registry: Arc<Registry>) -> (AnyWindowHandle, Entity<Shell>) {
    open_with(cx, registry, FakeLoader::default())
}

fn open_with(cx: &mut TestAppContext, registry: Arc<Registry>, loader: FakeLoader) -> (AnyWindowHandle, Entity<Shell>) {
    open_sized(cx, registry, loader, size(px(1360.), px(860.)), Vec::new(), GuiOptions::default())
}

/// The window over `registry` and `loader`, `window` big, given `initial` pictures and `gui` as on the command line.
fn open_sized(
    cx: &mut TestAppContext,
    registry: Arc<Registry>,
    loader: FakeLoader,
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
            |window, cx| cx.new(|cx| Shell::new(service, &gui, initial, window, cx)),
        )
        .expect("open test window")
    });
    // let the demo thumbnails arrive
    for _ in 0..50 {
        cx.run_until_parked();
        let loaded = cx.update(|cx| {
            view.read(cx).stack(cx).items.iter().all(|i| i.thumb.is_some() || i.load_error.is_some())
        });
        if loaded {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    (handle, view)
}

#[gpui_kit::test]
async fn window_renders_every_tab(cx: &mut TestAppContext) {
    let index = index_without_graphs("tabs");
    let (handle, view) = open(cx, Arc::new(Registry::open(&Settings { models: Some(index.clone()), ..Default::default() }).unwrap()));
    cx.update(|cx| {
        let v = view.read(cx);
        let m = v.models.read(cx);
        assert!(m.registry_error().is_none());
        assert_eq!(m.selected().map(|m| m.as_str()), Some("a"), "the index's default head is selected");
        assert!(!m.model_usable(m.selected().unwrap()), "a model without its graph file is unavailable");
        let stack = v.stack(cx);
        let n = stack.stack().map(|s| s.images.len()).unwrap_or(0);
        if stack_dir().join(crate::demo::STACK_FILE).is_file() {
            assert!(n >= 20, "demo stack loaded");
            assert!(stack.items.iter().all(|i| i.thumb.is_some()), "every thumbnail decoded");
        }
    });
    cx.update_window(handle, |_, window, cx| {
        window.render_frame(cx);
        assert!(window.try_find("theme").is_some(), "Kit TitleBar with the theme button rendered");
        assert!(window.try_find("open").is_some(), "Classify toolbar rendered");
        assert!(window.try_find("model-a").is_some(), "model picker rendered from the index");
    })
    .unwrap();
    for (tab, probe) in [
        (ix::<DemoTab>(), "demo-run"),
        (ix::<DemoTab>(), "demo-grid"),
        (ix::<BenchTab>(), "bench-run"),
        (ix::<BenchTab>(), "bench-matrix"),
        (ix::<ClassifyTab>(), "fetch"),
    ] {
        cx.update_window(handle, |_, window, cx| {
            view.update(cx, |v, cx| v.set_tab(tab, window, cx));
            window.render_frame(cx);
            assert!(window.try_find(probe).is_some(), "{tab:?} tab shows #{probe}");
        })
        .unwrap();
        cx.run_until_parked();
    }
    // keyboard navigation in the Demo tab moves the selection (wraps around)
    cx.update_window(handle, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.set_tab(ix::<DemoTab>(), window, cx);
            tab::<DemoTab>(v).update(cx, |d, cx| d.step(-1, cx));
        });
        window.render_frame(cx);
    })
    .unwrap();
    cx.update(|cx| {
        let v = view.read(cx);
        let n = v.stack(cx).items.len();
        if n > 0 {
            assert_eq!(demo(v, cx).selected, n - 1);
        }
    });
    let _ = std::fs::remove_dir_all(&index); // test scratch dir under the OS temp dir only
}

#[gpui_kit::test]
async fn benchmark_and_classify_run_through_the_service(cx: &mut TestAppContext) {
    if !stack_dir().join(crate::demo::STACK_FILE).is_file() {
        return;
    }
    let (handle, view) = open(cx, registry(&[("x", 1), ("y", 4)], &["y", "x"]));
    wait(cx, &view, "the default head", &|v, cx| selected(v, cx).is_some_and(|m| m.as_str() == "y"));
    cx.update(|cx| view.update(cx, |v, cx| start_bench(v, cx)));
    wait(cx, &view, "the benchmark to finish", &|v, cx| !bench(v, cx).running());
    cx.update(|cx| {
        let v = view.read(cx);
        let (bench, stack) = (bench(v, cx), v.stack(cx));
        let n = stack.items.len();
        assert_eq!(bench.results.len(), 2 * n, "every image × every head");
        assert!(bench.results.values().all(|r| r.is_ok()));
        let stack = stack.stack().unwrap();
        for h in ["x", "y"] {
            let s = super::bench::stats(stack, &bench.results, &HeadId::new(h));
            assert_eq!(s.judged, n, "{h} judged every image");
        }
    });
    let picture = stack_dir().join("01_battery_holder.jpg");
    submit(cx, handle, &view, picture);
    wait(cx, &view, "the Classify result", &|v, cx| !classify(v, cx).busy());
    let y = HeadId::new("y");
    cx.update(|cx| assert!(classify(view.read(cx), cx).answered(&y), "the entry has an answer and its picture"));
}

/// Poll `done` while the service threads work, or panic after about 5 s.
fn wait(cx: &mut TestAppContext, view: &Entity<Shell>, what: &str, done: &dyn Fn(&Shell, &App) -> bool) {
    for _ in 0..200 {
        cx.run_until_parked();
        if cx.update(|cx| done(view.read(cx), cx)) {
            return;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!("timed out waiting for {what}");
}

impl Shell {
    /// The demo stack the Demo and Benchmark tabs share.
    fn stack<'a>(&self, cx: &'a App) -> &'a Stack {
        demo(self, cx).stack.read(cx)
    }
}

/// The Benchmark tab's Run button.
fn start_bench(v: &mut Shell, cx: &mut Context<Shell>) {
    tab::<BenchTab>(v).update(cx, |b, cx| b.run(cx));
}

/// The picture at `path` reaches the window, as from Ctrl+O.
fn submit(cx: &mut TestAppContext, handle: AnyWindowHandle, view: &Entity<Shell>, path: PathBuf) {
    cx.update_window(handle, |_, window, cx| {
        view.update(cx, |v, cx| v.send_inputs(vec![ImageInput::Source(Source::File(path))], window, cx))
    })
    .unwrap()
}

fn selected(v: &Shell, cx: &App) -> Option<HeadId> {
    v.models.read(cx).selected().cloned()
}

fn run_model(v: &Shell, cx: &App) -> Option<HeadId> {
    v.models.read(cx).run_model()
}

/// The user picks `k` in the model picker.
fn select(cx: &mut TestAppContext, handle: AnyWindowHandle, view: &Entity<Shell>, k: &HeadId) {
    cx.update_window(handle, |_, window, cx| {
        let models = view.read(cx).models.clone();
        models.update(cx, |m, cx| m.select_model(k.clone(), window, cx))
    })
    .unwrap()
}

#[test]
fn tabs_are_named_by_title_and_listed_in_the_help() {
    assert_eq!(super::tab_index("Benchmark"), Some(ix::<BenchTab>()));
    assert_eq!(super::tab_index("bench"), Some(ix::<BenchTab>()), "the start of a title will do, in any case");
    assert_eq!(super::tab_index(""), None);
    assert_eq!(super::tab_index("settings"), None);
    let help = super::keys_help();
    assert!(help.contains(&format!("Ctrl+1..{}", crate::app::TABS.len())), "{help}");
    for t in crate::app::TABS {
        assert!(help.contains(t.title), "--help names the {} tab", t.title);
        for (_, label) in t.hints {
            assert!(help.contains(label), "--help lists the {} tab's {label}", t.title);
        }
    }
}

#[gpui_kit::test]
async fn tab_keys_and_pictures_reach_their_tabs(cx: &mut TestAppContext) {
    if !stack_dir().join(crate::demo::STACK_FILE).is_file() {
        return;
    }
    let (handle, view) = open(cx, registry(&[("x", 1)], &["x"]));
    wait(cx, &view, "the default head", &|v, cx| run_model(v, cx).is_some());
    let has_keyboard = |cx: &mut TestAppContext, tab: usize| {
        cx.update_window(handle, |_, window, cx| view.read(cx).tabs[tab].focus.is_focused(window)).unwrap()
    };
    cx.update_window(handle, |_, window, cx| window.focus(&view.read(cx).active_focus(), cx)).unwrap();
    for (keys, tab) in [("ctrl-3", ix::<BenchTab>()), ("ctrl-2", ix::<DemoTab>()), ("ctrl-1", ix::<ClassifyTab>())] {
        cx.simulate_keystrokes(handle, keys);
        assert!(has_keyboard(cx, tab), "{keys} shows tab {tab} and gives it the keyboard");
    }
    cx.simulate_keystrokes(handle, "ctrl-9");
    assert!(has_keyboard(cx, ix::<ClassifyTab>()), "Ctrl+9 without a ninth tab changes nothing");

    cx.simulate_keystrokes(handle, "ctrl-3");
    // a pointer move first: hover hit tests ignore the pointer while the last input was a key
    let position = point(px(600.), px(500.));
    let paths = ExternalPaths(smallvec::smallvec![stack_dir().join("01_battery_holder.jpg")]);
    cx.update_window(handle, |_, window, cx| {
        let pointer = MouseMoveEvent { position, pressed_button: None, modifiers: Default::default() };
        let events = [
            pointer.to_platform_input(),
            FileDropEvent::Entered { position, paths }.to_platform_input(),
            FileDropEvent::Submit { position }.to_platform_input(),
        ];
        for event in events {
            window.render_frame(cx);
            window.dispatch_event(event, cx);
        }
    })
    .unwrap();
    assert!(
        has_keyboard(cx, ix::<ClassifyTab>()),
        "a picture dropped on the Benchmark tab opens the tab that takes it"
    );
    wait(cx, &view, "x's answer for the dropped picture", &|v, cx| classify(v, cx).answered(&HeadId::new("x")));
}

#[gpui_kit::test]
async fn autorun_starts_the_start_tabs_run_once_the_default_is_known(cx: &mut TestAppContext) {
    if !stack_dir().join(crate::demo::STACK_FILE).is_file() {
        return;
    }
    let gui = GuiOptions { tab: ix::<BenchTab>(), autorun: true, ..Default::default() };
    let window = size(px(1360.), px(860.));
    let (_handle, view) = open_sized(cx, registry(&[("x", 1)], &["x"]), FakeLoader::default(), window, Vec::new(), gui);
    wait(cx, &view, "the benchmark --autorun starts", &|v, cx| !bench(v, cx).results.is_empty());
}

#[gpui_kit::test]
async fn escape_leaves_the_url_field_and_stops_only_the_visible_run(cx: &mut TestAppContext) {
    if !stack_dir().join(crate::demo::STACK_FILE).is_file() {
        return;
    }
    let loader = FakeLoader { run: Duration::from_millis(40), ..Default::default() };
    let (handle, view) = open_with(cx, registry(&[("x", 1)], &["x"]), loader);
    let app_focused = |cx: &mut TestAppContext| {
        cx.update_window(handle, |_, window, cx| view.read(cx).tabs[ix::<ClassifyTab>()].focus.is_focused(window))
            .unwrap()
    };
    cx.update_window(handle, |_, window, cx| window.focus(&view.read(cx).active_focus(), cx)).unwrap();
    cx.simulate_keystrokes(handle, "ctrl-l");
    assert!(!app_focused(cx), "Ctrl+L puts the cursor in the URL field");
    cx.simulate_keystrokes(handle, "escape");
    assert!(app_focused(cx), "Esc hands the keyboard back to the app");

    wait(cx, &view, "the default head", &|v, cx| run_model(v, cx).is_some());
    cx.update(|cx| view.update(cx, |v, cx| start_bench(v, cx)));
    cx.update_window(handle, |_, window, cx| view.update(cx, |v, cx| v.set_tab(ix::<ClassifyTab>(), window, cx)))
        .unwrap();
    cx.simulate_keystrokes(handle, "escape");
    assert!(cx.update(|cx| bench(view.read(cx), cx).running()), "Esc on Classify leaves the benchmark running");
    cx.update_window(handle, |_, window, cx| view.update(cx, |v, cx| v.set_tab(ix::<BenchTab>(), window, cx))).unwrap();
    cx.simulate_keystrokes(handle, "escape");
    assert!(cx.update(|cx| !bench(view.read(cx), cx).running()), "Esc on the Benchmark tab stops it");
}

#[gpui_kit::test]
async fn ctrl_l_on_another_tab_opens_the_url_field(cx: &mut TestAppContext) {
    if !stack_dir().join(crate::demo::STACK_FILE).is_file() {
        return;
    }
    let (handle, view) = open_with(cx, registry(&[("x", 1)], &["x"]), FakeLoader::default());
    cx.update_window(handle, |_, window, cx| view.update(cx, |v, cx| v.set_tab(ix::<BenchTab>(), window, cx))).unwrap();
    cx.simulate_keystrokes(handle, "ctrl-l");
    cx.update_window(handle, |_, window, cx| {
        let v = view.read(cx);
        assert_eq!(v.active_tab(), ix::<ClassifyTab>(), "Ctrl+L shows the tab with the URL field");
        assert!(!v.tabs[ix::<ClassifyTab>()].focus.is_focused(window), "and puts the cursor in the field");
    })
    .unwrap();
}

#[gpui_kit::test]
async fn switching_model_answers_the_picture_on_screen_with_it(cx: &mut TestAppContext) {
    if !stack_dir().join(crate::demo::STACK_FILE).is_file() {
        return;
    }
    let (handle, view) = open(cx, registry(&[("x", 1), ("y", 1)], &["y", "x"]));
    let (x, y) = (HeadId::new("x"), HeadId::new("y"));
    wait(cx, &view, "the default head", &|v, cx| selected(v, cx).as_ref() == Some(&y));
    let select = |cx: &mut TestAppContext, k: &HeadId| select(cx, handle, &view, k);

    let picture = stack_dir().join("01_battery_holder.jpg");
    submit(cx, handle, &view, picture);
    wait(cx, &view, "y's answer", &|v, cx| classify(v, cx).answered(&y));
    select(cx, &x);
    wait(cx, &view, "x's answer for the same picture", &|v, cx| classify(v, cx).answered(&x));
    cx.update(|cx| {
        let v = view.read(cx);
        assert_eq!(classify(v, cx).len(), 1, "the switch answers the picture in place, not as a new entry");
        assert!(classify(v, cx).answered(&y), "y's answer is kept for switching back");
    });

    cx.update_window(handle, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.set_tab(ix::<DemoTab>(), window, cx);
            tab::<DemoTab>(v).update(cx, |d, cx| d.classify_selected(cx));
        })
    })
    .unwrap();
    wait(cx, &view, "x's answer on the selected card", &|v, cx| demo(v, cx).answered(&x, cx));
    select(cx, &y);
    wait(cx, &view, "y's answer on the same card", &|v, cx| demo(v, cx).answered(&y, cx));
}

#[gpui_kit::test]
async fn switching_tab_answers_its_input_with_the_model_picked_elsewhere(cx: &mut TestAppContext) {
    if !stack_dir().join(crate::demo::STACK_FILE).is_file() {
        return;
    }
    let (handle, view) = open(cx, registry(&[("x", 1), ("y", 1), ("z", 1)], &["y", "x"]));
    let [x, y, z] = ["x", "y", "z"].map(HeadId::new);
    wait(cx, &view, "the default head", &|v, cx| run_model(v, cx).as_ref() == Some(&y));
    let on = |cx: &mut TestAppContext, tab: usize, f: &dyn Fn(&mut Shell, &mut Window, &mut Context<Shell>)| {
        cx.update_window(handle, |_, window, cx| {
            view.update(cx, |v, cx| {
                v.set_tab(tab, window, cx);
                f(v, window, cx);
            })
        })
        .unwrap()
    };
    let picture = stack_dir().join("01_battery_holder.jpg");
    on(cx, ix::<ClassifyTab>(), &|v, w, cx| v.send_inputs(vec![ImageInput::Source(Source::File(picture.clone()))], w, cx));
    wait(cx, &view, "y's answer for the picture", &|v, cx| classify(v, cx).answered(&y));
    on(cx, ix::<DemoTab>(), &|v, _, cx| tab::<DemoTab>(v).update(cx, |d, cx| d.classify_selected(cx)));
    wait(cx, &view, "y's answer on the selected card", &|v, cx| demo(v, cx).answered(&y, cx));

    on(cx, ix::<ClassifyTab>(), &|v, w, cx| v.models.update(cx, |m, cx| m.select_model(x.clone(), w, cx)));
    wait(cx, &view, "x's answer for the picture", &|v, cx| classify(v, cx).answered(&x));
    on(cx, ix::<DemoTab>(), &|_, _, _| {});
    wait(cx, &view, "x's answer on the card shown when the tab opens", &|v, cx| demo(v, cx).answered(&x, cx));

    on(cx, ix::<DemoTab>(), &|v, w, cx| v.models.update(cx, |m, cx| m.select_model(z.clone(), w, cx)));
    wait(cx, &view, "z's answer on the card", &|v, cx| demo(v, cx).answered(&z, cx));
    on(cx, ix::<ClassifyTab>(), &|_, _, _| {});
    wait(cx, &view, "z's answer for the picture shown when the tab opens", &|v, cx| classify(v, cx).answered(&z));
}

#[gpui_kit::test]
async fn a_model_that_fails_to_load_is_not_called_an_unreadable_picture(cx: &mut TestAppContext) {
    if !stack_dir().join(crate::demo::STACK_FILE).is_file() {
        return;
    }
    let loader = FakeLoader::breaking(&["x"]);
    let (handle, view) = open_with(cx, registry(&[("x", 1), ("y", 1)], &["x", "y"]), loader);
    let x = HeadId::new("x");
    wait(cx, &view, "the default head", &|v, cx| selected(v, cx).as_ref() == Some(&x));

    submit(cx, handle, &view, stack_dir().join("01_battery_holder.jpg"));
    wait(cx, &view, "x's failure", &|v, cx| classify(v, cx).failure(&x).is_some());
    cx.update(|cx| assert_eq!(classify(view.read(cx), cx).failure(&x), Some("model failed to load")));

    submit(cx, handle, &view, stack_dir().join("no_such_picture.jpg"));
    wait(cx, &view, "the read failure", &|v, cx| classify(v, cx).failure(&x).is_some());
    cx.update(|cx| assert_eq!(classify(view.read(cx), cx).failure(&x), Some("could not read")));
}

#[gpui_kit::test]
async fn picking_a_model_that_failed_to_load_tries_it_again(cx: &mut TestAppContext) {
    if !stack_dir().join(crate::demo::STACK_FILE).is_file() {
        return;
    }
    let loader = FakeLoader::breaking(&["x"]);
    let (handle, view) = open_with(cx, registry(&[("x", 1), ("y", 1)], &["x", "y"]), loader.clone());
    let x = HeadId::new("x");
    wait(cx, &view, "the default head", &|v, cx| run_model(v, cx).as_ref() == Some(&x));
    let picture = stack_dir().join("01_battery_holder.jpg");
    submit(cx, handle, &view, picture);
    wait(cx, &view, "x's failure", &|v, cx| classify(v, cx).failure(&x).is_some());
    loader.repair("x"); // say, the user freed GPU memory
    select(cx, handle, &view, &x);
    wait(cx, &view, "x's answer after the retry", &|v, cx| classify(v, cx).answered(&x));
}

#[gpui_kit::test]
async fn a_retry_clears_every_head_of_the_recovered_graph(cx: &mut TestAppContext) {
    if !stack_dir().join(crate::demo::STACK_FILE).is_file() {
        return;
    }
    let loader = FakeLoader::breaking(&["g"]);
    let (handle, view) = open_with(cx, shared_registry("g", &["x", "y"], &["x"]), loader.clone());
    let (x, y) = (HeadId::new("x"), HeadId::new("y"));
    let select = |cx: &mut TestAppContext, k: &HeadId| select(cx, handle, &view, k);
    wait(cx, &view, "the default head", &|v, cx| run_model(v, cx).as_ref() == Some(&x));
    let picture = stack_dir().join("01_battery_holder.jpg");
    submit(cx, handle, &view, picture);
    wait(cx, &view, "x's failure", &|v, cx| classify(v, cx).failure(&x).is_some());
    select(cx, &y); // the graph is still broken: y fails too
    wait(cx, &view, "y's failure", &|v, cx| classify(v, cx).failure(&y).is_some());
    loader.repair("g");
    select(cx, &x);
    wait(cx, &view, "x's answer after the retry", &|v, cx| classify(v, cx).answered(&x));
    select(cx, &y); // g is Ready now; y's old failure must not block it
    wait(cx, &view, "y's answer from the recovered graph", &|v, cx| classify(v, cx).answered(&y));
}

#[gpui_kit::test]
async fn controls_stay_inside_the_smallest_window(cx: &mut TestAppContext) {
    if !stack_dir().join(crate::demo::STACK_FILE).is_file() {
        return;
    }
    let loader = FakeLoader { run: Duration::from_millis(40), ..Default::default() };
    let small = size(px(1024.), px(640.));
    let (handle, view) = open_sized(cx, registry(&[("x", 1), ("y", 1)], &["x", "y"]), loader, small, Vec::new(), GuiOptions::default());
    wait(cx, &view, "the default head", &|v, cx| run_model(v, cx).is_some());
    let inside = |cx: &mut TestAppContext, tab: usize, start: &dyn Fn(&mut Shell, &mut Context<Shell>)| {
        cx.update_window(handle, |_, window, cx| {
            view.update(cx, |v, cx| {
                v.set_tab(tab, window, cx);
                start(v, cx);
            });
            window.render_frame(cx);
            for id in ["demo-run", "bench-run", "status-hints"] {
                if let Some(el) = window.try_find(id) {
                    let right = el.bounds().right();
                    assert!(right <= px(1024.), "{tab:?}: #{id} ends at {right:?}, past the 1024 px window");
                }
            }
        })
        .unwrap()
    };
    inside(cx, ix::<ClassifyTab>(), &|_, _| {});
    inside(cx, ix::<DemoTab>(), &|v, cx| tab::<DemoTab>(v).update(cx, |d, cx| d.run(cx)));
    assert!(cx.update(|cx| demo(view.read(cx), cx).running()), "the header was checked during Run all");
    inside(cx, ix::<BenchTab>(), &|v, cx| start_bench(v, cx));
    assert!(cx.update(|cx| bench(view.read(cx), cx).running()), "the header was checked during the benchmark");
}

#[gpui_kit::test]
async fn pictures_given_at_start_wait_for_the_default_model(cx: &mut TestAppContext) {
    if !stack_dir().join(crate::demo::STACK_FILE).is_file() {
        return;
    }
    let loader = FakeLoader::default();
    let ran = loader.seen.clone();
    let pictures: Vec<String> = ["01_battery_holder.jpg", "02_breadboard.jpg", "03_capacitor.jpg"]
        .iter()
        .map(|f| stack_dir().join(f).display().to_string())
        .collect();
    // y is the default once the start-up probe passes it; x, the last default, is only the fallback
    let reg = registry(&[("x", 1), ("y", 1)], &["y", "x"]);
    let (_handle, view) = open_sized(cx, reg, loader, size(px(1360.), px(860.)), pictures, GuiOptions::default());
    let y = HeadId::new("y");
    wait(cx, &view, "y's answer for every picture", &|v, cx| classify(v, cx).answered_entries(&y) == 3);
    let fallback_runs = ran.lock().unwrap().iter().filter(|s| matches!(s, Seen::Started { graph, .. } if graph == "x")).count();
    assert_eq!(fallback_runs, 0, "the fallback model never ran");
}

#[gpui_kit::test]
async fn the_fallback_notice_goes_once_the_user_picks_a_model(cx: &mut TestAppContext) {
    let loader = FakeLoader { no_provider: ["y".to_string()].into(), ..Default::default() };
    let (handle, view) = open_with(cx, registry(&[("x", 1), ("y", 1)], &["y", "x"]), loader);
    wait(cx, &view, "the fallback default", &|v, cx| run_model(v, cx).is_some_and(|m| m.as_str() == "x"));
    let notice = |cx: &mut TestAppContext| {
        cx.update_window(handle, |_, window, cx| {
            window.render_frame(cx);
            window.try_find("fallback-notice").is_some()
        })
        .unwrap()
    };
    assert!(notice(cx), "the start-up default says why y was passed over");
    select(cx, handle, &view, &HeadId::new("y"));
    assert!(!notice(cx), "after the user picks, the start-up notice no longer describes the model in use");
}

#[gpui_kit::test]
async fn the_best_model_is_named_when_the_benchmark_ends(cx: &mut TestAppContext) {
    if !stack_dir().join(crate::demo::STACK_FILE).is_file() {
        return;
    }
    // x answers every image before slow y has answered many, so mid-run x has the most correct
    let loader = FakeLoader { run: Duration::from_millis(30), ..Default::default() };
    let (handle, view) = open_with(cx, registry(&[("x", 8), ("y", 1)], &["x"]), loader);
    let badges = |cx: &mut TestAppContext| {
        cx.update_window(handle, |_, window, cx| {
            view.update(cx, |v, cx| v.set_tab(ix::<BenchTab>(), window, cx));
            window.render_frame(cx);
            ["x", "y"].iter().filter(|k| window.try_find(SharedString::from(format!("bench-best-{k}"))).is_some()).count()
        })
        .unwrap()
    };
    cx.update(|cx| view.update(cx, |v, cx| start_bench(v, cx)));
    wait(cx, &view, "the benchmark to be under way", &|v, cx| !bench(v, cx).results.is_empty());
    assert!(cx.update(|cx| bench(view.read(cx), cx).running()));
    assert_eq!(badges(cx), 0, "no model is called best while the counts are partial");
    wait(cx, &view, "the benchmark to finish", &|v, cx| !bench(v, cx).running());
    assert!(badges(cx) >= 1, "the finished benchmark names the best model");
}

#[gpui_kit::test]
async fn missing_index_is_reported(cx: &mut TestAppContext) {
    let (handle, view) = open(cx, Arc::new(Registry::empty("no exports/models.json found".into())));
    cx.update(|cx| assert!(view.read(cx).models.read(cx).registry_error().is_some(), "a missing index must be reported"));
    cx.update_window(handle, |_, window, cx| {
        window.render_frame(cx);
        assert!(window.try_find("open").is_some(), "the window still renders without models");
    })
    .unwrap();
}

/// The Benchmark tab over the shipped exports with onnxruntime on the CPU EP matches the golden classify rows
/// (top-1, abstain flag, probabilities within 1e-6). Needs the exports, ORT_DYLIB_PATH and the turbojpeg feature,
/// because the golden rows were decoded with libjpeg-turbo; about 90 s in a debug build.
///     ORT_DYLIB_PATH=... cargo test -p gpui-demo --features turbojpeg -- --ignored real_benchmark
#[gpui_kit::test]
#[ignore]
async fn real_benchmark_matches_the_golden_rows(cx: &mut TestAppContext) {
    let exports = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../exports");
    let opts = Settings { models: Some(exports), eps: Some(vec![inference::manifest::EpKind::Cpu]), ..Default::default() };
    let reg = Arc::new(Registry::open(&opts).unwrap());
    let service = Inference::spawn(reg.clone(), inference::runtime::OrtLoader::new(&opts, &reg));
    cx.executor().allow_parking();
    cx.update(|cx| {
        gpui_kit::init(cx);
        super::theme::install(cx);
    });
    let (_handle, view) = cx.update(|cx| {
        let bounds = Bounds { origin: Point::default(), size: size(px(1360.), px(860.)) };
        gpui_kit::open_window(
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() },
            cx,
            |window, cx| cx.new(|cx| Shell::new(service, &GuiOptions::default(), Vec::new(), window, cx)),
        )
        .unwrap()
    });
    let wait = |cx: &mut TestAppContext, secs: u64, done: &dyn Fn(&Shell, &App) -> bool| {
        let t0 = std::time::Instant::now();
        while t0.elapsed() < Duration::from_secs(secs) {
            cx.run_until_parked();
            if cx.update(|cx| done(view.read(cx), cx)) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("timed out");
    };
    wait(cx, 30, &|v, cx| v.stack(cx).items.iter().all(|i| i.rgb.is_some()));
    cx.update(|cx| view.update(cx, |v, cx| start_bench(v, cx)));
    wait(cx, 600, &|v, cx| !bench(v, cx).running());
    let golden: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/demo/golden/classify_all.json"))
            .unwrap(),
    )
    .unwrap();
    cx.update(|cx| {
        let v = view.read(cx);
        let stack = v.stack(cx).stack().unwrap();
        let results = &bench(v, cx).results;
        let mut compared = 0;
        for row in &golden {
            let file = Path::new(row["source"].as_str().unwrap()).file_name().unwrap().to_str().unwrap();
            let i = stack.images.iter().position(|im| im.file == file).unwrap();
            let head = HeadId::new(row["model"].as_str().unwrap());
            let (c, _) = results[&(i, head.clone())].as_ref().unwrap();
            let label = c.committed().map_or(crate::app::UNSURE, |k| c.labels.name(k));
            assert_eq!(label, row["label"].as_str().unwrap(), "{file} {head}");
            assert_eq!(c.abstained(), row["unsure"].as_bool().unwrap(), "{file} {head}");
            let probs: Vec<f64> = row["probs"].as_array().unwrap().iter().map(|p| p.as_f64().unwrap()).collect();
            let worst = c.probs.iter().zip(&probs).map(|(a, b)| (f64::from(*a) - b).abs()).fold(0.0, f64::max);
            assert!(worst <= 1e-6, "{file} {head}: max |dp| {worst}");
            compared += 1;
        }
        assert_eq!(compared, 112);
        eprintln!("benchmark tab: {compared} cells match the golden rows");
    });
}

#[gpui_kit::test]
async fn registry_problems_are_shown_beside_working_models(cx: &mut TestAppContext) {
    // "fast" names no head: a registry problem, and the start-up default passes it over
    let (handle, view) = open(cx, registry(&[("x", 1)], &["fast", "x"]));
    wait(cx, &view, "the default head", &|v, cx| run_model(v, cx).is_some());
    let shown = |cx: &mut TestAppContext, id: &'static str| {
        cx.update_window(handle, |_, window, cx| {
            window.render_frame(cx);
            window.try_find(id).is_some()
        })
        .unwrap()
    };
    assert!(shown(cx, "registry-problems"), "the index names a default no manifest provides");
    assert!(shown(cx, "fallback-notice"), "the start-up notice says why fast was passed over");
    cx.update(|cx| view.update(cx, |v, cx| v.dismiss_problems(cx)));
    assert!(!shown(cx, "registry-problems"), "the notice can be dismissed");
}
