//! The shell: the tab row and the keys that pick a tab, where pictures go, Esc, `--autorun`, the model picker's
//! notices and alerts. These run with the input tab and `ProbeTab`, or with whatever tabs `app::TABS` lists, and need
//! no demo stack.

use std::sync::Arc;

use gpui_kit::test::TestWindowExt as _;
use gpui_kit::{ExternalPaths, FileDropEvent, InputEvent as _, MouseMoveEvent, TestAppContext, point, px, size};

use super::*;
use inference::testing::{Seen, registry};

fn classify<'a>(v: &Shell, cx: &'a App) -> &'a ClassifyTab {
    tab::<ClassifyTab>(v).read(cx)
}

fn probe<'a>(v: &Shell, cx: &'a App) -> &'a ProbeTab {
    tab::<ProbeTab>(v).read(cx)
}

#[gpui_kit::test]
async fn window_renders_every_tab(cx: &mut TestAppContext) {
    let index = index_without_graphs("tabs");
    let (handle, view) = open(cx, registry_in(&index), app::TABS);
    cx.update(|cx| {
        let m = view.read(cx).models.read(cx);
        assert!(m.registry_error().is_none());
        assert_eq!(m.selected().map(|m| m.as_str()), Some("a"), "the index's default head is selected");
        assert!(!m.model_usable(m.selected().unwrap()), "a model without its graph file is unavailable");
    });
    cx.update_window(handle, |_, window, cx| {
        window.render_frame(cx);
        assert!(window.try_find("theme").is_some(), "Kit TitleBar with the theme button rendered");
        assert!(window.try_find("open").is_some(), "Classify toolbar rendered");
        assert!(window.try_find("model-a").is_some(), "model picker rendered from the index");
    })
    .unwrap();
    for (ix, spec) in app::TABS.iter().enumerate() {
        cx.update_window(handle, |_, window, cx| {
            view.update(cx, |v, cx| v.set_tab(ix, window, cx));
            window.render_frame(cx);
            assert!(
                view.read(cx).tabs[ix].focus.is_focused(window),
                "the {} tab renders with the keyboard",
                spec.title
            );
            assert!(window.try_find("status-hints").is_some(), "the status bar renders under the {} tab", spec.title);
        })
        .unwrap();
        cx.run_until_parked();
    }
    let _ = std::fs::remove_dir_all(&index); // test scratch dir under the OS temp dir only
}

#[test]
fn tabs_are_named_by_title_and_listed_in_the_help() {
    assert_eq!(super::super::tab_index(SHELL_TABS, "Probe"), Some(ix::<ProbeTab>(SHELL_TABS)));
    assert_eq!(super::super::tab_index(SHELL_TABS, "pro"), Some(1), "the start of a title will do, in any case");
    assert_eq!(super::super::tab_index(SHELL_TABS, ""), None);
    assert_eq!(super::super::tab_index(SHELL_TABS, "settings"), None);
    let help = super::super::keys_help(SHELL_TABS);
    assert!(help.starts_with("Ctrl+1..2 Classify / Probe tab,"), "{help}");
    let one = super::super::keys_help(&SHELL_TABS[..1]);
    assert!(one.starts_with("Ctrl+1 Classify tab,"), "one tab has one key: {one}");
    for tabs in [SHELL_TABS, app::TABS] {
        let help = super::super::keys_help(tabs);
        for t in tabs {
            assert!(help.contains(t.title), "--help names the {} tab", t.title);
            for (_, label) in t.hints {
                assert!(help.contains(label), "--help lists the {} tab's {label}", t.title);
            }
        }
    }
}

#[gpui_kit::test]
async fn tab_keys_and_pictures_reach_their_tabs(cx: &mut TestAppContext) {
    let (handle, view) = open(cx, registry(&[("x", 1)], &["x"]), SHELL_TABS);
    wait(cx, &view, "the default head", &|v, cx| run_model(v, cx).is_some());
    let has_keyboard = |cx: &mut TestAppContext, tab: usize| {
        cx.update_window(handle, |_, window, cx| view.read(cx).tabs[tab].focus.is_focused(window)).unwrap()
    };
    let (classify_ix, probe_ix) = (ix::<ClassifyTab>(SHELL_TABS), ix::<ProbeTab>(SHELL_TABS));
    cx.update_window(handle, |_, window, cx| window.focus(&view.read(cx).active_focus(), cx)).unwrap();
    for (keys, tab) in [("ctrl-2", probe_ix), ("ctrl-1", classify_ix)] {
        cx.simulate_keystrokes(handle, keys);
        assert!(has_keyboard(cx, tab), "{keys} shows tab {tab} and gives it the keyboard");
    }
    cx.simulate_keystrokes(handle, "ctrl-9");
    assert!(has_keyboard(cx, classify_ix), "Ctrl+9 without a ninth tab changes nothing");

    cx.simulate_keystrokes(handle, "ctrl-2");
    // a pointer move first: hover hit tests ignore the pointer while the last input was a key
    let position = point(px(600.), px(500.));
    let paths = ExternalPaths(smallvec::smallvec![picture("dropped")]);
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
    assert!(has_keyboard(cx, classify_ix), "a picture dropped on a tab that takes none opens the tab that takes it");
    wait(cx, &view, "x's answer for the dropped picture", &|v, cx| classify(v, cx).answered(&HeadId::new("x")));
}

#[gpui_kit::test]
async fn autorun_starts_the_start_tabs_run_once_the_default_is_known(cx: &mut TestAppContext) {
    let gui = GuiOptions { tab: ix::<ProbeTab>(SHELL_TABS), autorun: true, ..Default::default() };
    let window = size(px(1360.), px(860.));
    let reg = registry(&[("x", 1)], &["x"]);
    let (_handle, view) = open_sized(cx, reg, FakeLoader::default(), SHELL_TABS, window, Vec::new(), gui);
    wait(cx, &view, "the probe's --autorun run", &|v, cx| probe(v, cx).running);
    cx.update(|cx| {
        let followed = &probe(view.read(cx), cx).followed;
        assert_eq!(followed.last(), Some(&Some(HeadId::new("x"))), "the run starts once the default is known");
    });
}

#[gpui_kit::test]
async fn escape_leaves_the_url_field_and_stops_only_the_visible_run(cx: &mut TestAppContext) {
    let (handle, view) = open(cx, registry(&[("x", 1)], &["x"]), SHELL_TABS);
    let (classify_ix, probe_ix) = (ix::<ClassifyTab>(SHELL_TABS), ix::<ProbeTab>(SHELL_TABS));
    let classify_focused = |cx: &mut TestAppContext| {
        cx.update_window(handle, |_, window, cx| view.read(cx).tabs[classify_ix].focus.is_focused(window)).unwrap()
    };
    cx.update_window(handle, |_, window, cx| window.focus(&view.read(cx).active_focus(), cx)).unwrap();
    cx.simulate_keystrokes(handle, "ctrl-l");
    assert!(!classify_focused(cx), "Ctrl+L puts the cursor in the URL field");
    cx.simulate_keystrokes(handle, "escape");
    assert!(classify_focused(cx), "Esc hands the keyboard back to the tab");

    on(cx, handle, &view, probe_ix, &|v, _, cx| tab::<ProbeTab>(v).update(cx, |p, cx| p.run(cx)));
    on(cx, handle, &view, classify_ix, &|_, _, _| {});
    cx.simulate_keystrokes(handle, "escape");
    assert!(cx.update(|cx| probe(view.read(cx), cx).running), "Esc on Classify leaves another tab's run going");
    on(cx, handle, &view, probe_ix, &|_, _, _| {});
    cx.simulate_keystrokes(handle, "escape");
    assert!(cx.update(|cx| !probe(view.read(cx), cx).running), "Esc on the tab stops its run");
}

#[gpui_kit::test]
async fn ctrl_l_on_another_tab_opens_the_url_field(cx: &mut TestAppContext) {
    let (handle, view) = open(cx, registry(&[("x", 1)], &["x"]), SHELL_TABS);
    on(cx, handle, &view, ix::<ProbeTab>(SHELL_TABS), &|_, _, _| {});
    cx.simulate_keystrokes(handle, "ctrl-l");
    cx.update_window(handle, |_, window, cx| {
        let v = view.read(cx);
        let classify_ix = ix::<ClassifyTab>(SHELL_TABS);
        assert_eq!(v.active_tab(), classify_ix, "Ctrl+L shows the tab with the URL field");
        assert!(!v.tabs[classify_ix].focus.is_focused(window), "and puts the cursor in the field");
    })
    .unwrap();
}

#[gpui_kit::test]
async fn switching_tab_answers_its_input_with_the_model_picked_elsewhere(cx: &mut TestAppContext) {
    let (handle, view) = open(cx, registry(&[("x", 1), ("y", 1), ("z", 1)], &["y", "x"]), SHELL_TABS);
    let [x, y, z] = ["x", "y", "z"].map(HeadId::new);
    let (classify_ix, probe_ix) = (ix::<ClassifyTab>(SHELL_TABS), ix::<ProbeTab>(SHELL_TABS));
    let followed =
        |cx: &mut TestAppContext| cx.update(|cx| probe(view.read(cx), cx).followed.last().cloned().flatten());
    wait(cx, &view, "the default head", &|v, cx| run_model(v, cx).as_ref() == Some(&y));
    let shown = picture("switching-tab");
    on(cx, handle, &view, classify_ix, &|v, w, cx| {
        v.send_inputs(vec![ImageInput::Source(Source::File(shown.clone()))], w, cx)
    });
    wait(cx, &view, "y's answer for the picture", &|v, cx| classify(v, cx).answered(&y));
    on(cx, handle, &view, probe_ix, &|_, _, _| {});
    assert_eq!(followed(cx).as_ref(), Some(&y), "the tab follows the model when it opens");

    on(cx, handle, &view, classify_ix, &|v, w, cx| v.models.update(cx, |m, cx| m.select_model(x.clone(), w, cx)));
    wait(cx, &view, "x's answer for the picture", &|v, cx| classify(v, cx).answered(&x));
    on(cx, handle, &view, probe_ix, &|_, _, _| {});
    assert_eq!(followed(cx).as_ref(), Some(&x), "the tab shown gets the model picked on another tab");

    on(cx, handle, &view, probe_ix, &|v, w, cx| v.models.update(cx, |m, cx| m.select_model(z.clone(), w, cx)));
    assert_eq!(followed(cx).as_ref(), Some(&z), "the visible tab follows a model picked while it is shown");
    on(cx, handle, &view, classify_ix, &|_, _, _| {});
    wait(cx, &view, "z's answer for the picture shown when the tab opens", &|v, cx| classify(v, cx).answered(&z));
}

#[gpui_kit::test]
async fn pictures_given_at_start_wait_for_the_default_model(cx: &mut TestAppContext) {
    let loader = FakeLoader::default();
    let ran = loader.seen.clone();
    let pictures: Vec<String> = ["start-1", "start-2", "start-3"].map(|t| picture(t).display().to_string()).to_vec();
    // y is the default once the start-up probe passes it; x, the last default, is only the fallback
    let reg = registry(&[("x", 1), ("y", 1)], &["y", "x"]);
    let window = size(px(1360.), px(860.));
    let (_handle, view) = open_sized(cx, reg, loader, SHELL_TABS, window, pictures, GuiOptions::default());
    let y = HeadId::new("y");
    wait(cx, &view, "y's answer for every picture", &|v, cx| classify(v, cx).answered_entries(&y) == 3);
    let fallback_runs =
        ran.lock().unwrap().iter().filter(|s| matches!(s, Seen::Started { graph, .. } if graph == "x")).count();
    assert_eq!(fallback_runs, 0, "the fallback model never ran");
}

#[gpui_kit::test]
async fn controls_stay_inside_the_smallest_window(cx: &mut TestAppContext) {
    let small = size(px(1024.), px(640.));
    let reg = registry(&[("x", 1), ("y", 1)], &["x", "y"]);
    let (handle, view) =
        open_sized(cx, reg, FakeLoader::default(), app::TABS, small, Vec::new(), GuiOptions::default());
    wait(cx, &view, "the default head", &|v, cx| run_model(v, cx).is_some());
    for (ix, spec) in app::TABS.iter().enumerate() {
        cx.update_window(handle, |_, window, cx| {
            view.update(cx, |v, cx| v.set_tab(ix, window, cx));
            window.render_frame(cx);
            let right = window.try_find("status-hints").expect("the status bar's shortcuts").bounds().right();
            assert!(right <= px(1024.), "{}: the shortcuts end at {right:?}, past the 1024 px window", spec.title);
        })
        .unwrap();
    }
}

#[gpui_kit::test]
async fn the_fallback_notice_goes_once_the_user_picks_a_model(cx: &mut TestAppContext) {
    let loader = FakeLoader { no_provider: ["y".to_string()].into(), ..Default::default() };
    let (handle, view) = open_with(cx, registry(&[("x", 1), ("y", 1)], &["y", "x"]), loader, SHELL_TABS);
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
async fn missing_index_is_reported(cx: &mut TestAppContext) {
    let (handle, view) = open(cx, Arc::new(Registry::empty("no exports/models.json found".into())), SHELL_TABS);
    cx.update(|cx| {
        assert!(view.read(cx).models.read(cx).registry_error().is_some(), "a missing index must be reported")
    });
    cx.update_window(handle, |_, window, cx| {
        window.render_frame(cx);
        assert!(window.try_find("open").is_some(), "the window still renders without models");
    })
    .unwrap();
}

#[gpui_kit::test]
async fn registry_problems_are_shown_beside_working_models(cx: &mut TestAppContext) {
    // "fast" names no head: a registry problem, and the start-up default passes it over
    let (handle, view) = open(cx, registry(&[("x", 1)], &["fast", "x"]), SHELL_TABS);
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
