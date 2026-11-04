//! The Benchmark tab over the committed demo stack: every available model on every picture, its stats, Esc, the best
//! model, the header at the smallest window size, and the real models against the golden rows.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use gpui_kit::test::TestWindowExt as _;
use gpui_kit::{
    AnyWindowHandle, Bounds, Entity, Pixels, Point, Size, TestAppContext, WindowBounds, WindowOptions, px, size,
};

use super::*;
use crate::demo::DemoStack;
use crate::gui::GuiOptions;
use crate::gui::classify::ClassifyTab;
use crate::gui::shell::Shell;
use crate::gui::tests::{ix, on, open_sized, run_model, tab, wait};
use crate::gui::{TabSpec, theme};
use inference::classification::read_head;
use inference::manifest::{Calibration, ClassifyTask, Decision, Labels, Prob, Readout, Task};
use inference::registry::Registry;
use inference::runtime::{OrtLoader, Settings};
use inference::service::Inference;
use inference::testing::{FakeLoader, registry};

const TABS: &[TabSpec] = &[TabSpec::of::<ClassifyTab>(), TabSpec::of::<BenchTab>()];

fn bench<'a>(v: &Shell, cx: &'a App) -> &'a BenchTab {
    tab::<BenchTab>(v).read(cx)
}

fn stack<'a>(v: &Shell, cx: &'a App) -> &'a Stack {
    bench(v, cx).stack.read(cx)
}

/// The window with the Benchmark tab, once the demo stack's pictures are decoded.
fn open_bench(
    cx: &mut TestAppContext,
    registry: Arc<Registry>,
    loader: FakeLoader,
    window: Size<Pixels>,
    gui: GuiOptions,
) -> (AnyWindowHandle, Entity<Shell>) {
    let (handle, view) = open_sized(cx, registry, loader, TABS, window, Vec::new(), gui);
    cx.update(|cx| assert!(stack(view.read(cx), cx).stack().is_some(), "the demo stack is committed in demo_stack/"));
    wait(cx, &view, "the demo stack's pictures", &|v, cx| {
        stack(v, cx).items.iter().all(|i| i.thumb.is_some() || i.load_error.is_some())
    });
    (handle, view)
}

fn window() -> Size<Pixels> {
    size(px(1360.), px(860.))
}

/// The Benchmark tab's Run button.
fn start(v: &mut Shell, cx: &mut gpui_kit::Context<Shell>) {
    tab::<BenchTab>(v).update(cx, |b, cx| b.run(cx));
}

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

#[gpui_kit::test]
async fn autorun_benchmarks_every_model_on_every_picture(cx: &mut TestAppContext) {
    let gui = GuiOptions { tab: ix::<BenchTab>(TABS), autorun: true, ..Default::default() };
    let (handle, view) =
        open_bench(cx, registry(&[("x", 1), ("y", 4)], &["y", "x"]), FakeLoader::default(), window(), gui);
    wait(cx, &view, "the --autorun benchmark to finish", &|v, cx| {
        !bench(v, cx).results.is_empty() && !bench(v, cx).running()
    });
    cx.update(|cx| {
        let v = view.read(cx);
        let (bench, stack) = (bench(v, cx), stack(v, cx));
        let n = stack.items.len();
        assert_eq!(bench.results.len(), 2 * n, "every image × every head");
        assert!(bench.results.values().all(|r| r.is_ok()));
        for h in ["x", "y"] {
            assert_eq!(
                stats(stack.stack().unwrap(), &bench.results, &HeadId::new(h)).judged,
                n,
                "{h} judged every image"
            );
        }
    });
    cx.update_window(handle, |_, window, cx| {
        window.render_frame(cx);
        for probe in ["bench-run", "bench-matrix"] {
            assert!(window.try_find(probe).is_some(), "the Benchmark tab shows #{probe}");
        }
    })
    .unwrap();
}

#[gpui_kit::test]
async fn escape_stops_the_benchmark_only_on_its_tab(cx: &mut TestAppContext) {
    let loader = FakeLoader { run: Duration::from_millis(40), ..Default::default() };
    let (handle, view) = open_bench(cx, registry(&[("x", 1)], &["x"]), loader, window(), GuiOptions::default());
    wait(cx, &view, "the default head", &|v, cx| run_model(v, cx).is_some());
    cx.update(|cx| view.update(cx, |v, cx| start(v, cx)));
    on(cx, handle, &view, ix::<ClassifyTab>(TABS), &|_, _, _| {});
    cx.simulate_keystrokes(handle, "escape");
    assert!(cx.update(|cx| bench(view.read(cx), cx).running()), "Esc on Classify leaves the benchmark running");
    on(cx, handle, &view, ix::<BenchTab>(TABS), &|_, _, _| {});
    cx.simulate_keystrokes(handle, "escape");
    assert!(cx.update(|cx| !bench(view.read(cx), cx).running()), "Esc on the Benchmark tab stops it");
}

#[gpui_kit::test]
async fn the_best_model_is_named_when_the_benchmark_ends(cx: &mut TestAppContext) {
    // x answers every image before slow y has answered many, so mid-run x has the most correct
    let loader = FakeLoader { run: Duration::from_millis(30), ..Default::default() };
    let (handle, view) =
        open_bench(cx, registry(&[("x", 8), ("y", 1)], &["x"]), loader, window(), GuiOptions::default());
    let badges = |cx: &mut TestAppContext| {
        cx.update_window(handle, |_, window, cx| {
            view.update(cx, |v, cx| v.set_tab(ix::<BenchTab>(TABS), window, cx));
            window.render_frame(cx);
            ["x", "y"]
                .iter()
                .filter(|k| window.try_find(SharedString::from(format!("bench-best-{k}"))).is_some())
                .count()
        })
        .unwrap()
    };
    cx.update(|cx| view.update(cx, |v, cx| start(v, cx)));
    wait(cx, &view, "the benchmark to be under way", &|v, cx| !bench(v, cx).results.is_empty());
    assert!(cx.update(|cx| bench(view.read(cx), cx).running()));
    assert_eq!(badges(cx), 0, "no model is called best while the counts are partial");
    wait(cx, &view, "the benchmark to finish", &|v, cx| !bench(v, cx).running());
    assert!(badges(cx) >= 1, "the finished benchmark names the best model");
}

#[gpui_kit::test]
async fn the_header_stays_inside_the_smallest_window_while_it_runs(cx: &mut TestAppContext) {
    let loader = FakeLoader { run: Duration::from_millis(40), ..Default::default() };
    let small = size(px(1024.), px(640.));
    let (handle, view) =
        open_bench(cx, registry(&[("x", 1), ("y", 1)], &["x", "y"]), loader, small, GuiOptions::default());
    wait(cx, &view, "the default head", &|v, cx| run_model(v, cx).is_some());
    cx.update_window(handle, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.set_tab(ix::<BenchTab>(TABS), window, cx);
            start(v, cx);
        });
        window.render_frame(cx);
        for id in ["bench-run", "status-hints"] {
            let right = window.try_find(id).unwrap_or_else(|| panic!("#{id}")).bounds().right();
            assert!(right <= px(1024.), "#{id} ends at {right:?}, past the 1024 px window");
        }
    })
    .unwrap();
    assert!(cx.update(|cx| bench(view.read(cx), cx).running()), "the header was checked during the benchmark");
}

/// The Benchmark tab over the shipped exports with onnxruntime on the CPU EP matches the golden classify rows
/// (top-1, abstain flag, probabilities within 1e-6). Needs the exports, ORT_DYLIB_PATH and the turbojpeg feature,
/// because the golden rows were decoded with libjpeg-turbo; about 90 s in a debug build.
///     ORT_DYLIB_PATH=... cargo test -p gpui-demo --features turbojpeg -- --ignored real_benchmark
#[gpui_kit::test]
#[ignore]
async fn real_benchmark_matches_the_golden_rows(cx: &mut TestAppContext) {
    let exports = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../exports");
    let opts =
        Settings { models: Some(exports), eps: Some(vec![inference::manifest::EpKind::Cpu]), ..Default::default() };
    let reg = Arc::new(Registry::open(&opts).unwrap());
    let service = Inference::spawn(reg.clone(), OrtLoader::new(&opts, &reg));
    cx.executor().allow_parking();
    cx.update(|cx| {
        gpui_kit::init(cx);
        theme::install(cx);
    });
    let (_handle, view) = cx.update(|cx| {
        let bounds = Bounds { origin: Point::default(), size: window() };
        gpui_kit::open_window(
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() },
            cx,
            |window, cx| cx.new(|cx| Shell::new(service, &GuiOptions::default(), Vec::new(), TABS, window, cx)),
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
    wait(cx, 30, &|v, cx| stack(v, cx).items.iter().all(|i| i.rgb.is_some()));
    cx.update(|cx| view.update(cx, |v, cx| start(v, cx)));
    wait(cx, 600, &|v, cx| !bench(v, cx).running());
    let golden: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/demo/golden/classify_all.json"),
        )
        .unwrap(),
    )
    .unwrap();
    cx.update(|cx| {
        let v = view.read(cx);
        let stack = stack(v, cx).stack().unwrap();
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
