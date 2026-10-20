//! Headless smoke test with GPUI Kit's test harness (`gpui-kit` feature `test-support`, a test platform: no
//! display, no GPU). It builds the real window view, renders every tab, and switches tabs by clicking the
//! Kit TabBar. No model is loaded: `--model-dir` points at a folder without one, so the registry reports an
//! error (rendered as an Alert) instead of loading 1.7 GB of weights.

use std::path::PathBuf;
use std::time::Duration;

use gpui_kit::test::TestWindowExt as _;
use gpui_kit::{AppContext as _, Bounds, Point, TestAppContext, WindowBounds, WindowOptions, px, size};

use super::{DemoApp, TabKind};
use crate::model::ModelOptions;

fn opts() -> ModelOptions {
    ModelOptions { model_dir: Some(std::env::temp_dir().join("gpui-demo-no-model-here")), ..Default::default() }
}

fn stack_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("demo_stack")
}

#[gpui_kit::test]
async fn window_renders_every_tab(cx: &mut TestAppContext) {
    // The app's inference and thumbnail threads are real OS threads that wake GPUI tasks; tell the
    // deterministic test scheduler that this is expected.
    cx.executor().allow_parking();
    cx.update(gpui_kit::init);
    cx.update(|cx| cx.bind_keys(super::keymap()));
    let (handle, view) = cx.update(|cx| {
        let bounds = Bounds { origin: Point::default(), size: size(px(1360.), px(860.)) };
        gpui_kit::open_window(
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() },
            cx,
            |window, cx| cx.new(|cx| DemoApp::new(opts(), Some(stack_dir()), Vec::new(), window, cx)),
        )
        .expect("open test window")
    });
    // let the registry event (an error here) and the demo thumbnails arrive
    for _ in 0..50 {
        cx.run_until_parked();
        let loaded = cx.update(|cx| {
            let v = view.read(cx);
            v.registry_error.is_some() && v.demo.items.iter().all(|i| i.thumb.is_some() || i.load_error.is_some())
        });
        if loaded {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    cx.update(|cx| {
        let v = view.read(cx);
        assert!(v.registry_error.is_some(), "missing model folder must be reported, not loaded");
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
        assert!(window.try_find("model-deployed").is_some(), "model picker rendered");
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
}
