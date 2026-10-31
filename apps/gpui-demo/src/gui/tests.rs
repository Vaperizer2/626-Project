//! Headless smoke tests with GPUI Kit's test harness (`gpui-kit` feature `test-support`, a test platform: no
//! display, no GPU). They build the real window view, render every tab, and switch tabs by clicking the Kit TabBar.
//! No model is loaded: the index lists model A, whose ONNX file is absent, so it shows as unavailable instead of
//! loading weights; a second test points `--models` at nothing and expects the error alert.

use std::path::{Path, PathBuf};
use std::time::Duration;

use gpui_kit::test::TestWindowExt as _;
use gpui_kit::{
    AnyWindowHandle, AppContext as _, Bounds, Entity, Point, TestAppContext, WindowBounds, WindowOptions, px, size,
};

use super::{DemoApp, TabKind};
use crate::model::ModelOptions;

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

fn open(cx: &mut TestAppContext, models: PathBuf) -> (AnyWindowHandle, Entity<DemoApp>) {
    // The app's inference and thumbnail threads are real OS threads that wake GPUI tasks; tell the
    // deterministic test scheduler that this is expected.
    cx.executor().allow_parking();
    cx.update(|cx| {
        gpui_kit::init(cx);
        super::theme::install(cx);
    });
    cx.update(|cx| cx.bind_keys(super::keymap()));
    let opts = ModelOptions { models: Some(models), ..Default::default() };
    let (handle, view) = cx.update(|cx| {
        let bounds = Bounds { origin: Point::default(), size: size(px(1360.), px(860.)) };
        gpui_kit::open_window(
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() },
            cx,
            |window, cx| cx.new(|cx| DemoApp::new(opts, Some(stack_dir()), Vec::new(), window, cx)),
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
    let (handle, view) = open(cx, index.clone());
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
async fn missing_index_is_reported(cx: &mut TestAppContext) {
    let (handle, view) = open(cx, std::env::temp_dir().join("gpui-demo-no-model-here"));
    cx.update(|cx| assert!(view.read(cx).registry_error.is_some(), "a missing index must be reported"));
    cx.update_window(handle, |_, window, cx| {
        window.render_frame(cx);
        assert!(window.try_find("open").is_some(), "the window still renders without models");
    })
    .unwrap();
}
