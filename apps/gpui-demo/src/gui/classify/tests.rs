//! The Classify tab: a picture keeps its answers as the model changes, and a model that fails to load says so and
//! can be tried again.

use std::path::PathBuf;

use gpui_kit::{App, TestAppContext};

use super::ClassifyTab;
use crate::gui::TabSpec;
use crate::gui::shell::Shell;
use crate::gui::tests::{open, open_with, picture, run_model, select, selected, submit, tab, wait};
use inference::manifest::HeadId;
use inference::testing::{FakeLoader, registry, shared_registry};

const TABS: &[TabSpec] = &[TabSpec::of::<ClassifyTab>()];

fn classify<'a>(v: &Shell, cx: &'a App) -> &'a ClassifyTab {
    tab::<ClassifyTab>(v).read(cx)
}

#[gpui_kit::test]
async fn switching_model_answers_the_picture_on_screen_with_it(cx: &mut TestAppContext) {
    let (handle, view) = open(cx, registry(&[("x", 1), ("y", 1)], &["y", "x"]), TABS);
    let (x, y) = (HeadId::new("x"), HeadId::new("y"));
    wait(cx, &view, "the default head", &|v, cx| selected(v, cx).as_ref() == Some(&y));
    submit(cx, handle, &view, picture("switching-model"));
    wait(cx, &view, "y's answer", &|v, cx| classify(v, cx).answered(&y));
    select(cx, handle, &view, &x);
    wait(cx, &view, "x's answer for the same picture", &|v, cx| classify(v, cx).answered(&x));
    cx.update(|cx| {
        let v = view.read(cx);
        assert_eq!(classify(v, cx).len(), 1, "the switch answers the picture in place, not as a new entry");
        assert!(classify(v, cx).answered(&y), "y's answer is kept for switching back");
    });
}

#[gpui_kit::test]
async fn a_model_that_fails_to_load_is_not_called_an_unreadable_picture(cx: &mut TestAppContext) {
    let loader = FakeLoader::breaking(&["x"]);
    let (handle, view) = open_with(cx, registry(&[("x", 1), ("y", 1)], &["x", "y"]), loader, TABS);
    let x = HeadId::new("x");
    wait(cx, &view, "the default head", &|v, cx| selected(v, cx).as_ref() == Some(&x));

    submit(cx, handle, &view, picture("fails-to-load"));
    wait(cx, &view, "x's failure", &|v, cx| classify(v, cx).failure(&x).is_some());
    cx.update(|cx| assert_eq!(classify(view.read(cx), cx).failure(&x), Some("model failed to load")));

    submit(cx, handle, &view, PathBuf::from("/nonexistent/no_such_picture.jpg"));
    wait(cx, &view, "the read failure", &|v, cx| classify(v, cx).failure(&x).is_some());
    cx.update(|cx| assert_eq!(classify(view.read(cx), cx).failure(&x), Some("could not read")));
}

#[gpui_kit::test]
async fn picking_a_model_that_failed_to_load_tries_it_again(cx: &mut TestAppContext) {
    let loader = FakeLoader::breaking(&["x"]);
    let (handle, view) = open_with(cx, registry(&[("x", 1), ("y", 1)], &["x", "y"]), loader.clone(), TABS);
    let x = HeadId::new("x");
    wait(cx, &view, "the default head", &|v, cx| run_model(v, cx).as_ref() == Some(&x));
    submit(cx, handle, &view, picture("tried-again"));
    wait(cx, &view, "x's failure", &|v, cx| classify(v, cx).failure(&x).is_some());
    loader.repair("x"); // say, the user freed GPU memory
    select(cx, handle, &view, &x);
    wait(cx, &view, "x's answer after the retry", &|v, cx| classify(v, cx).answered(&x));
}

#[gpui_kit::test]
async fn a_retry_clears_every_head_of_the_recovered_graph(cx: &mut TestAppContext) {
    let loader = FakeLoader::breaking(&["g"]);
    let (handle, view) = open_with(cx, shared_registry("g", &["x", "y"], &["x"]), loader.clone(), TABS);
    let (x, y) = (HeadId::new("x"), HeadId::new("y"));
    let select = |cx: &mut TestAppContext, k: &HeadId| select(cx, handle, &view, k);
    wait(cx, &view, "the default head", &|v, cx| run_model(v, cx).as_ref() == Some(&x));
    submit(cx, handle, &view, picture("recovered-graph"));
    wait(cx, &view, "x's failure", &|v, cx| classify(v, cx).failure(&x).is_some());
    select(cx, &y); // the graph is still broken: y fails too
    wait(cx, &view, "y's failure", &|v, cx| classify(v, cx).failure(&y).is_some());
    loader.repair("g");
    select(cx, &x);
    wait(cx, &view, "x's answer after the retry", &|v, cx| classify(v, cx).answered(&x));
    select(cx, &y); // g is Ready now; y's old failure must not block it
    wait(cx, &view, "y's answer from the recovered graph", &|v, cx| classify(v, cx).answered(&y));
}
