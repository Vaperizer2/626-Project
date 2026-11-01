//! Headless tests with GPUI Kit's test harness (`gpui-kit` feature `test-support`, a test platform: no display, no
//! GPU). They build the real window view over the real inference service, with the fake loader from
//! `inference::fake` behind it: one renders every tab over an index whose graph file is absent, one runs the
//! benchmark and a Classify submission on fake models, one opens with no index at all.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use gpui_kit::test::TestWindowExt as _;
use gpui_kit::{
    AnyWindowHandle, AppContext as _, Bounds, Entity, Pixels, Point, SharedString, Size, TestAppContext, WindowBounds,
    WindowOptions, px, size,
};

use super::{DemoApp, TabKind};
use crate::inference::Inference;
use crate::inference::fake::{FakeLoader, Seen, registry};
use crate::registry::Registry;

fn stack_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("demo_stack")
}

/// An index with model A's manifest and labels but not its graph.
fn index_without_graphs(tag: &str) -> PathBuf {
    let tmp = std::env::temp_dir().join(format!("gpui-demo-gui-{tag}-{}", std::process::id()));
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../exports/itri626/A_scratch_cnn");
    std::fs::create_dir_all(tmp.join("A")).unwrap();
    for f in ["model.json", "labels.json"] {
        std::fs::copy(src.join(f), tmp.join("A").join(f)).unwrap();
    }
    std::fs::write(tmp.join("models.json"), r#"{"schema":"model-index/1","default":["a"],"manifests":["A/model.json"]}"#)
        .unwrap();
    tmp
}

fn open(cx: &mut TestAppContext, registry: Arc<Registry>) -> (AnyWindowHandle, Entity<DemoApp>) {
    open_with(cx, registry, FakeLoader::default())
}

fn open_with(cx: &mut TestAppContext, registry: Arc<Registry>, loader: FakeLoader) -> (AnyWindowHandle, Entity<DemoApp>) {
    open_sized(cx, registry, loader, size(px(1360.), px(860.)), Vec::new())
}

/// The window over `registry` and `loader`, `window` big, given `initial` pictures as on the command line.
fn open_sized(
    cx: &mut TestAppContext,
    registry: Arc<Registry>,
    loader: FakeLoader,
    window: Size<Pixels>,
    initial: Vec<String>,
) -> (AnyWindowHandle, Entity<DemoApp>) {
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
            |window, cx| cx.new(|cx| DemoApp::new(service, Some(stack_dir()), initial, window, cx)),
        )
        .expect("open test window")
    });
    // let the demo thumbnails arrive
    for _ in 0..50 {
        cx.run_until_parked();
        let loaded = cx.update(|cx| {
            view.read(cx).demo.items.iter().all(|i| i.thumb.is_some() || i.load_error.is_some())
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
    let (handle, view) = open(cx, Arc::new(Registry::load(&index.join("models.json")).unwrap()));
    cx.update(|cx| {
        let v = view.read(cx);
        assert!(v.registry_error.is_none());
        assert_eq!(v.model.as_ref().map(|m| m.as_str()), Some("a"), "the index's default head is selected");
        assert!(!v.model_usable(v.model.as_ref().unwrap()), "a model without its graph file is unavailable");
        let n = v.demo.stack().map(|s| s.images.len()).unwrap_or(0);
        if stack_dir().join(crate::demo::STACK_FILE).is_file() {
            assert!(n >= 20, "demo stack loaded");
            assert!(v.demo.items.iter().all(|i| i.thumb.is_some()), "every thumbnail decoded");
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
        (TabKind::Demo, "demo-run"),
        (TabKind::Demo, "demo-grid"),
        (TabKind::Bench, "bench-run"),
        (TabKind::Bench, "bench-matrix"),
        (TabKind::Classify, "fetch"),
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
            v.set_tab(TabKind::Demo, window, cx);
            v.demo_step(-1, window, cx);
        });
        window.render_frame(cx);
    })
    .unwrap();
    cx.update(|cx| {
        let v = view.read(cx);
        if !v.demo.items.is_empty() {
            assert_eq!(v.demo.selected, v.demo.items.len() - 1);
        }
    });
    let _ = std::fs::remove_dir_all(&index); // test scratch dir under the OS temp dir only
}

#[gpui_kit::test]
async fn benchmark_and_classify_run_through_the_service(cx: &mut TestAppContext) {
    if !stack_dir().join(crate::demo::STACK_FILE).is_file() {
        return;
    }
    let (_handle, view) = open(cx, registry(&[("x", 1), ("y", 4)], &["y", "x"]));
    let wait = |cx: &mut TestAppContext, what: &str, done: &dyn Fn(&DemoApp) -> bool| {
        for _ in 0..200 {
            cx.run_until_parked();
            if cx.update(|cx| done(view.read(cx))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        panic!("timed out waiting for {what}");
    };
    wait(cx, "the default head", &|v| v.model.as_ref().is_some_and(|m| m.as_str() == "y"));
    cx.update(|cx| view.update(cx, |v, cx| v.bench_start(cx)));
    wait(cx, "the benchmark to finish", &|v| !v.bench.running());
    cx.update(|cx| {
        let v = view.read(cx);
        let n = v.demo.items.len();
        assert_eq!(v.bench.results.len(), 2 * n, "every image × every head");
        assert!(v.bench.results.values().all(|r| r.is_ok()));
        let stack = v.demo.stack().unwrap();
        for h in ["x", "y"] {
            let s = super::bench::stats(stack, &v.bench.results, &crate::manifest::HeadId::new(h));
            assert_eq!(s.judged, n, "{h} judged every image");
        }
    });
    let picture = stack_dir().join("01_battery_holder.jpg");
    cx.update(|cx| view.update(cx, |v, cx| v.submit_source(crate::fetch::Source::File(picture), cx)));
    wait(cx, "the Classify result", &|v| !v.classify.busy());
    let y = crate::manifest::HeadId::new("y");
    cx.update(|cx| assert!(view.read(cx).classify.answered(&y), "the entry has an answer and its picture"));
}

/// Poll `done` while the service threads work, or panic after about 5 s.
fn wait(cx: &mut TestAppContext, view: &Entity<DemoApp>, what: &str, done: &dyn Fn(&DemoApp) -> bool) {
    for _ in 0..200 {
        cx.run_until_parked();
        if cx.update(|cx| done(view.read(cx))) {
            return;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!("timed out waiting for {what}");
}

#[gpui_kit::test]
async fn escape_leaves_the_url_field_and_stops_only_the_visible_run(cx: &mut TestAppContext) {
    if !stack_dir().join(crate::demo::STACK_FILE).is_file() {
        return;
    }
    let loader = FakeLoader { run: Duration::from_millis(40), ..Default::default() };
    let (handle, view) = open_with(cx, registry(&[("x", 1)], &["x"]), loader);
    let app_focused = |cx: &mut TestAppContext| {
        cx.update_window(handle, |_, window, cx| view.read(cx).focus.is_focused(window)).unwrap()
    };
    cx.update_window(handle, |_, window, cx| window.focus(&view.read(cx).focus.clone(), cx)).unwrap();
    cx.simulate_keystrokes(handle, "ctrl-l");
    assert!(!app_focused(cx), "Ctrl+L puts the cursor in the URL field");
    cx.simulate_keystrokes(handle, "escape");
    assert!(app_focused(cx), "Esc hands the keyboard back to the app");

    wait(cx, &view, "the default head", &|v| v.run_model().is_some());
    cx.update(|cx| view.update(cx, |v, cx| v.bench_start(cx)));
    cx.update_window(handle, |_, window, cx| view.update(cx, |v, cx| v.set_tab(TabKind::Classify, window, cx))).unwrap();
    cx.simulate_keystrokes(handle, "escape");
    assert!(cx.update(|cx| view.read(cx).bench.running()), "Esc on Classify leaves the benchmark running");
    cx.update_window(handle, |_, window, cx| view.update(cx, |v, cx| v.set_tab(TabKind::Bench, window, cx))).unwrap();
    cx.simulate_keystrokes(handle, "escape");
    assert!(cx.update(|cx| !view.read(cx).bench.running()), "Esc on the Benchmark tab stops it");
}

#[gpui_kit::test]
async fn switching_model_answers_the_picture_on_screen_with_it(cx: &mut TestAppContext) {
    if !stack_dir().join(crate::demo::STACK_FILE).is_file() {
        return;
    }
    let (handle, view) = open(cx, registry(&[("x", 1), ("y", 1)], &["y", "x"]));
    let (x, y) = (crate::manifest::HeadId::new("x"), crate::manifest::HeadId::new("y"));
    wait(cx, &view, "the default head", &|v| v.model.as_ref() == Some(&y));
    let select = |cx: &mut TestAppContext, k: &crate::manifest::HeadId| {
        cx.update_window(handle, |_, window, cx| view.update(cx, |v, cx| v.select_model(k.clone(), window, cx))).unwrap()
    };

    let picture = stack_dir().join("01_battery_holder.jpg");
    cx.update(|cx| view.update(cx, |v, cx| v.submit_source(crate::fetch::Source::File(picture), cx)));
    wait(cx, &view, "y's answer", &|v| v.classify.answered(&y));
    select(cx, &x);
    wait(cx, &view, "x's answer for the same picture", &|v| v.classify.answered(&x));
    cx.update(|cx| {
        let v = view.read(cx);
        assert_eq!(v.classify.len(), 1, "the switch answers the picture in place, not as a new entry");
        assert!(v.classify.answered(&y), "y's answer is kept for switching back");
    });

    cx.update_window(handle, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.set_tab(TabKind::Demo, window, cx);
            v.demo_classify_selected(cx);
        })
    })
    .unwrap();
    wait(cx, &view, "x's answer on the selected card", &|v| v.demo_answered(&x));
    select(cx, &y);
    wait(cx, &view, "y's answer on the same card", &|v| v.demo_answered(&y));
}

#[gpui_kit::test]
async fn a_model_that_fails_to_load_is_not_called_an_unreadable_picture(cx: &mut TestAppContext) {
    if !stack_dir().join(crate::demo::STACK_FILE).is_file() {
        return;
    }
    let loader = FakeLoader { broken: ["x".to_string()].into(), ..Default::default() };
    let (_handle, view) = open_with(cx, registry(&[("x", 1), ("y", 1)], &["x", "y"]), loader);
    let x = crate::manifest::HeadId::new("x");
    wait(cx, &view, "the default head", &|v| v.model.as_ref() == Some(&x));
    let submit = |cx: &mut TestAppContext, path: PathBuf| {
        cx.update(|cx| view.update(cx, |v, cx| v.submit_source(crate::fetch::Source::File(path), cx)))
    };

    submit(cx, stack_dir().join("01_battery_holder.jpg"));
    wait(cx, &view, "x's failure", &|v| v.classify.failure(&x).is_some());
    cx.update(|cx| assert_eq!(view.read(cx).classify.failure(&x), Some("model failed to load")));

    submit(cx, stack_dir().join("no_such_picture.jpg"));
    wait(cx, &view, "the read failure", &|v| v.classify.failure(&x).is_some());
    cx.update(|cx| assert_eq!(view.read(cx).classify.failure(&x), Some("could not read")));
}

#[gpui_kit::test]
async fn controls_stay_inside_the_smallest_window(cx: &mut TestAppContext) {
    if !stack_dir().join(crate::demo::STACK_FILE).is_file() {
        return;
    }
    let loader = FakeLoader { run: Duration::from_millis(40), ..Default::default() };
    let small = size(px(1024.), px(640.));
    let (handle, view) = open_sized(cx, registry(&[("x", 1), ("y", 1)], &["x", "y"]), loader, small, Vec::new());
    wait(cx, &view, "the default head", &|v| v.run_model().is_some());
    let inside = |cx: &mut TestAppContext, tab: TabKind, start: &dyn Fn(&mut DemoApp, &mut gpui_kit::Context<DemoApp>)| {
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
    inside(cx, TabKind::Classify, &|_, _| {});
    inside(cx, TabKind::Demo, &|v, cx| v.demo_run_all(cx));
    assert!(cx.update(|cx| view.read(cx).demo.running()), "the header was checked during Run all");
    inside(cx, TabKind::Bench, &|v, cx| v.bench_start(cx));
    assert!(cx.update(|cx| view.read(cx).bench.running()), "the header was checked during the benchmark");
}

#[gpui_kit::test]
async fn pictures_given_at_start_wait_for_the_default_model(cx: &mut TestAppContext) {
    if !stack_dir().join(crate::demo::STACK_FILE).is_file() {
        return;
    }
    let loader = FakeLoader::default();
    let ran = loader.seen.clone();
    let picture = stack_dir().join("01_battery_holder.jpg").display().to_string();
    // y is the default once the start-up probe passes it; x, the last default, is only the fallback
    let reg = registry(&[("x", 1), ("y", 1)], &["y", "x"]);
    let (_handle, view) = open_sized(cx, reg, loader, size(px(1360.), px(860.)), vec![picture]);
    let y = crate::manifest::HeadId::new("y");
    wait(cx, &view, "y's answer for the picture", &|v| v.classify.answered(&y));
    let fallback_runs = ran.lock().unwrap().iter().filter(|s| matches!(s, Seen::Started { graph, .. } if graph == "x")).count();
    assert_eq!(fallback_runs, 0, "the fallback model never ran");
}

#[gpui_kit::test]
async fn the_fallback_notice_goes_once_the_user_picks_a_model(cx: &mut TestAppContext) {
    let loader = FakeLoader { no_provider: ["y".to_string()].into(), ..Default::default() };
    let (handle, view) = open_with(cx, registry(&[("x", 1), ("y", 1)], &["y", "x"]), loader);
    wait(cx, &view, "the fallback default", &|v| v.run_model().is_some_and(|m| m.as_str() == "x"));
    let notice = |cx: &mut TestAppContext| {
        cx.update_window(handle, |_, window, cx| {
            window.render_frame(cx);
            window.try_find("fallback-notice").is_some()
        })
        .unwrap()
    };
    assert!(notice(cx), "the start-up default says why y was passed over");
    cx.update_window(handle, |_, window, cx| {
        view.update(cx, |v, cx| v.select_model(crate::manifest::HeadId::new("y"), window, cx))
    })
    .unwrap();
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
            view.update(cx, |v, cx| v.set_tab(TabKind::Bench, window, cx));
            window.render_frame(cx);
            ["x", "y"].iter().filter(|k| window.try_find(SharedString::from(format!("bench-best-{k}"))).is_some()).count()
        })
        .unwrap()
    };
    cx.update(|cx| view.update(cx, |v, cx| v.bench_start(cx)));
    wait(cx, &view, "the benchmark to be under way", &|v| !v.bench.results.is_empty());
    assert!(cx.update(|cx| view.read(cx).bench.running()));
    assert_eq!(badges(cx), 0, "no model is called best while the counts are partial");
    wait(cx, &view, "the benchmark to finish", &|v| !v.bench.running());
    assert!(badges(cx) >= 1, "the finished benchmark names the best model");
}

#[gpui_kit::test]
async fn missing_index_is_reported(cx: &mut TestAppContext) {
    let (handle, view) = open(cx, Arc::new(Registry::empty("no exports/models.json found".into())));
    cx.update(|cx| assert!(view.read(cx).registry_error.is_some(), "a missing index must be reported"));
    cx.update_window(handle, |_, window, cx| {
        window.render_frame(cx);
        assert!(window.try_find("open").is_some(), "the window still renders without models");
    })
    .unwrap();
}

/// The Benchmark tab over the shipped exports with onnxruntime on the CPU EP matches the golden classify rows
/// (top-1, abstain flag, probabilities within 1e-6). Needs the exports and ORT_DYLIB_PATH; about 30 s.
///     ORT_DYLIB_PATH=... cargo test --release -- --ignored real_benchmark
#[gpui_kit::test]
#[ignore]
async fn real_benchmark_matches_the_golden_rows(cx: &mut TestAppContext) {
    let exports = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../exports");
    let reg = Arc::new(Registry::load(&exports.join("models.json")).unwrap());
    let opts = crate::runtime::ModelOptions { eps: Some(vec![crate::manifest::EpKind::Cpu]), ..Default::default() };
    let service = Inference::spawn(reg, crate::runtime::OrtLoader::new(&opts, Some(&exports)));
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
            |window, cx| cx.new(|cx| DemoApp::new(service, Some(stack_dir()), Vec::new(), window, cx)),
        )
        .unwrap()
    });
    let wait = |cx: &mut TestAppContext, secs: u64, done: &dyn Fn(&DemoApp) -> bool| {
        let t0 = std::time::Instant::now();
        while t0.elapsed() < Duration::from_secs(secs) {
            cx.run_until_parked();
            if cx.update(|cx| done(view.read(cx))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("timed out");
    };
    wait(cx, 30, &|v| v.demo.items.iter().all(|i| i.rgb.is_some()));
    cx.update(|cx| view.update(cx, |v, cx| v.bench_start(cx)));
    wait(cx, 600, &|v| !v.bench.running());
    let golden: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/demo/golden/classify_all.json"))
            .unwrap(),
    )
    .unwrap();
    cx.update(|cx| {
        let v = view.read(cx);
        let stack = v.demo.stack().unwrap();
        let mut compared = 0;
        for row in &golden {
            let file = Path::new(row["source"].as_str().unwrap()).file_name().unwrap().to_str().unwrap();
            let i = stack.images.iter().position(|im| im.file == file).unwrap();
            let head = crate::manifest::HeadId::new(row["model"].as_str().unwrap());
            let (c, _) = v.bench.results[&(i, head.clone())].as_ref().unwrap();
            let label = c.committed().map_or("UNSURE", |k| c.labels.name(k));
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
