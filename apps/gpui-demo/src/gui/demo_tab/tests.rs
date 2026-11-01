//! The Demo tab over the committed demo stack: its pictures and grid, stepping, a card's answer following the model,
//! and Run all at the smallest window size.

use std::time::Duration;

use gpui_kit::test::TestWindowExt as _;
use gpui_kit::{AnyWindowHandle, App, AppContext as _, Entity, Pixels, Size, TestAppContext, px, size};

use super::DemoTab;
use crate::gui::classify::ClassifyTab;
use crate::gui::shell::Shell;
use crate::gui::stack::Stack;
use crate::gui::tests::{ix, on, open_sized, run_model, select, tab, wait};
use crate::gui::{GuiOptions, Tab, TabSpec};
use inference::manifest::HeadId;
use inference::registry::Registry;
use inference::testing::{FakeLoader, registry};

const TABS: &[TabSpec] = &[TabSpec::of::<ClassifyTab>(), TabSpec::of::<DemoTab>()];

fn demo<'a>(v: &Shell, cx: &'a App) -> &'a DemoTab {
    tab::<DemoTab>(v).read(cx)
}

fn stack<'a>(v: &Shell, cx: &'a App) -> &'a Stack {
    demo(v, cx).stack.read(cx)
}

/// The window with the Demo tab, once the demo stack's pictures are decoded.
fn open_demo(
    cx: &mut TestAppContext,
    registry: std::sync::Arc<Registry>,
    loader: FakeLoader,
    window: Size<Pixels>,
) -> (AnyWindowHandle, Entity<Shell>) {
    let (handle, view) = open_sized(cx, registry, loader, TABS, window, Vec::new(), GuiOptions::default());
    cx.update(|cx| assert!(stack(view.read(cx), cx).stack().is_some(), "the demo stack is committed in demo_stack/"));
    wait(cx, &view, "the demo stack's pictures", &|v, cx| {
        stack(v, cx).items.iter().all(|i| i.thumb.is_some() || i.load_error.is_some())
    });
    (handle, view)
}

fn window() -> Size<Pixels> {
    size(px(1360.), px(860.))
}

#[gpui_kit::test]
async fn the_demo_stack_loads_and_the_grid_steps_through_it(cx: &mut TestAppContext) {
    let (handle, view) = open_demo(cx, registry(&[("x", 1)], &["x"]), FakeLoader::default(), window());
    cx.update(|cx| {
        let s = stack(view.read(cx), cx);
        assert!(s.items.len() >= 20, "demo stack loaded");
        assert!(s.items.iter().all(|i| i.thumb.is_some()), "every thumbnail decoded");
    });
    let demo_ix = ix::<DemoTab>(TABS);
    for probe in ["demo-run", "demo-grid"] {
        cx.update_window(handle, |_, window, cx| {
            view.update(cx, |v, cx| v.set_tab(demo_ix, window, cx));
            window.render_frame(cx);
            assert!(window.try_find(probe).is_some(), "the Demo tab shows #{probe}");
        })
        .unwrap();
    }
    // keyboard navigation moves the selection (wraps around)
    on(cx, handle, &view, demo_ix, &|v, _, cx| tab::<DemoTab>(v).update(cx, |d, cx| d.step(-1, cx)));
    cx.update(|cx| {
        let v = view.read(cx);
        assert_eq!(demo(v, cx).selected, stack(v, cx).items.len() - 1);
    });
}

#[gpui_kit::test]
async fn switching_model_answers_the_card_on_screen_with_it(cx: &mut TestAppContext) {
    let (handle, view) = open_demo(cx, registry(&[("x", 1), ("y", 1)], &["y", "x"]), FakeLoader::default(), window());
    let (x, y) = (HeadId::new("x"), HeadId::new("y"));
    let demo_ix = ix::<DemoTab>(TABS);
    wait(cx, &view, "the default head", &|v, cx| run_model(v, cx).as_ref() == Some(&y));
    on(cx, handle, &view, demo_ix, &|_, _, _| {});
    select(cx, handle, &view, &x);
    on(cx, handle, &view, demo_ix, &|v, _, cx| tab::<DemoTab>(v).update(cx, |d, cx| d.classify_selected(cx)));
    wait(cx, &view, "x's answer on the selected card", &|v, cx| demo(v, cx).answered(&x, cx));
    select(cx, handle, &view, &y);
    wait(cx, &view, "y's answer on the same card", &|v, cx| demo(v, cx).answered(&y, cx));
}

#[gpui_kit::test]
async fn the_card_shown_when_the_tab_opens_gets_the_model_picked_elsewhere(cx: &mut TestAppContext) {
    let reg = registry(&[("x", 1), ("y", 1), ("z", 1)], &["y", "x"]);
    let (handle, view) = open_demo(cx, reg, FakeLoader::default(), window());
    let [x, y, z] = ["x", "y", "z"].map(HeadId::new);
    let (classify_ix, demo_ix) = (ix::<ClassifyTab>(TABS), ix::<DemoTab>(TABS));
    wait(cx, &view, "the default head", &|v, cx| run_model(v, cx).as_ref() == Some(&y));
    on(cx, handle, &view, demo_ix, &|v, _, cx| tab::<DemoTab>(v).update(cx, |d, cx| d.classify_selected(cx)));
    wait(cx, &view, "y's answer on the selected card", &|v, cx| demo(v, cx).answered(&y, cx));

    on(cx, handle, &view, classify_ix, &|v, w, cx| v.models.update(cx, |m, cx| m.select_model(x.clone(), w, cx)));
    on(cx, handle, &view, demo_ix, &|_, _, _| {});
    wait(cx, &view, "x's answer on the card shown when the tab opens", &|v, cx| demo(v, cx).answered(&x, cx));

    on(cx, handle, &view, demo_ix, &|v, w, cx| v.models.update(cx, |m, cx| m.select_model(z.clone(), w, cx)));
    wait(cx, &view, "z's answer on the card", &|v, cx| demo(v, cx).answered(&z, cx));
}

#[gpui_kit::test]
async fn the_header_stays_inside_the_smallest_window_during_run_all(cx: &mut TestAppContext) {
    let loader = FakeLoader { run: Duration::from_millis(40), ..Default::default() };
    let small = size(px(1024.), px(640.));
    let (handle, view) = open_demo(cx, registry(&[("x", 1), ("y", 1)], &["x", "y"]), loader, small);
    wait(cx, &view, "the default head", &|v, cx| run_model(v, cx).is_some());
    let demo_ix = ix::<DemoTab>(TABS);
    cx.update_window(handle, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.set_tab(demo_ix, window, cx);
            tab::<DemoTab>(v).update(cx, |d, cx| d.run(cx));
        });
        window.render_frame(cx);
        for id in ["demo-run", "status-hints"] {
            let right = window.try_find(id).unwrap_or_else(|| panic!("#{id}")).bounds().right();
            assert!(right <= px(1024.), "#{id} ends at {right:?}, past the 1024 px window");
        }
    })
    .unwrap();
    assert!(cx.update(|cx| demo(view.read(cx), cx).running()), "the header was checked during Run all");
}
