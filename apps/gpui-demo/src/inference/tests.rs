use super::fake::*;
use super::*;
use futures::channel::mpsc::TryRecvError;

const WAIT: Duration = Duration::from_secs(5);

/// Next update, or panic after `WAIT`.
fn next(h: &mut RequestHandle) -> Update {
    let t0 = Instant::now();
    loop {
        match h.updates.try_recv() {
            Ok(u) => return u,
            Err(TryRecvError::Closed) => panic!("request stream ended without Done"),
            Err(TryRecvError::Empty) if t0.elapsed() < WAIT => std::thread::sleep(Duration::from_millis(1)),
            Err(TryRecvError::Empty) => panic!("no update within {WAIT:?}"),
        }
    }
}

/// Head results and the finish of one request.
fn finish(h: &mut RequestHandle) -> (Vec<(HeadId, Result<HeadOutput, RunError>)>, Finish) {
    let mut heads = Vec::new();
    loop {
        match next(h) {
            Update::Decoded { .. } => {}
            Update::Head { head, result, .. } => heads.push((head, result)),
            Update::Done(f) => return (heads, f),
        }
    }
}

fn wait_until(what: &str, cond: impl Fn() -> bool) {
    let t0 = Instant::now();
    while !cond() {
        assert!(t0.elapsed() < WAIT, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn ids(v: &[&str]) -> Vec<HeadId> {
    v.iter().map(|s| HeadId::new(s)).collect()
}

#[test]
fn a_request_streams_decoded_then_heads_then_done() {
    let (inf, _ev) = Inference::spawn(registry(&[("x", 1)], &["x"]), FakeLoader::default());
    let mut h = inf.lane(Priority::Interactive, Policy::Keep).submit(image(200), &ids(&["x", "nope"]), Some(2));
    assert!(matches!(next(&mut h), Update::Decoded { size: (4, 4), display: Some(d) } if d.width() == 2));
    let (heads, f) = finish(&mut h);
    assert!(matches!(f, Finish::Complete { .. }), "{f:?}");
    assert!(matches!(&heads[0], (id, Err(RunError::Unavailable(_))) if id.as_str() == "nope"));
    let Some((_, Ok(HeadOutput::Classify(c)))) = heads.iter().find(|h| h.0.as_str() == "x") else { panic!() };
    assert_eq!(c.best_label(), "light");
}

#[test]
fn superseded_requests_never_complete() {
    let fake = FakeLoader { run: Duration::from_millis(80), ..Default::default() };
    let (inf, _ev) = Inference::spawn(registry(&[("x", 1)], &["x"]), fake.clone());
    let lane = inf.lane(Priority::Interactive, Policy::Latest);
    let mut a = lane.submit(image(10), &ids(&["x"]), None);
    wait_until("the first run to start", || fake.seen().iter().any(|s| matches!(s, Seen::Started { .. })));
    let mut b = lane.submit(image(20), &ids(&["x"]), None);
    let mut c = lane.submit(image(30), &ids(&["x"]), None);
    assert!(matches!(finish(&mut a).1, Finish::Superseded));
    assert!(matches!(finish(&mut b).1, Finish::Superseded));
    let (heads, f) = finish(&mut c);
    assert!(matches!(f, Finish::Complete { .. }) && heads.len() == 1);
    let ran: Vec<Seen> = fake.seen().into_iter().filter(|s| matches!(s, Seen::Ran { .. })).collect();
    assert_eq!(ran, [Seen::Ran { graph: "x".into(), rows: vec![30] }], "only the latest submit ran to the end");
    assert!(fake.seen().contains(&Seen::Stopped { graph: "x".into() }), "the running one was stopped");
}

#[test]
fn a_slow_load_does_not_delay_other_graphs() {
    let gate = Arc::new(Gate::default());
    let fake = FakeLoader { gates: [("slow".to_string(), gate.clone())].into(), ..Default::default() };
    let (inf, _ev) = Inference::spawn(registry(&[("slow", 1), ("fast", 1)], &["fast"]), fake);
    let lane = inf.lane(Priority::Interactive, Policy::Keep);
    let mut slow = lane.submit(image(1), &ids(&["slow"]), None);
    let mut fast = lane.submit(image(2), &ids(&["fast"]), None);
    assert!(matches!(finish(&mut fast).1, Finish::Complete { .. }), "fast finished while slow is still loading");
    assert!(matches!(next(&mut slow), Update::Decoded { .. }));
    assert!(matches!(slow.updates.try_recv(), Err(TryRecvError::Empty)), "slow has no result yet");
    gate.open();
    assert!(matches!(finish(&mut slow).1, Finish::Complete { .. }));
}

#[test]
fn queued_bulk_jobs_form_one_batch() {
    let gate = Arc::new(Gate::default());
    let fake = FakeLoader { gates: [("z".to_string(), gate.clone())].into(), ..Default::default() };
    let (inf, _ev) = Inference::spawn(registry(&[("z", 8)], &["z"]), fake.clone());
    let lane = inf.lane(Priority::Bulk, Policy::Keep);
    let mut hs: Vec<RequestHandle> = (0..8).map(|v| lane.submit(image(v * 30), &ids(&["z"]), None)).collect();
    // the first job is held by the loading actor, the other seven wait in its queue
    wait_until("one held job and seven queued", || gate.holds() && inf.queued(&GraphId::new("z")) == 7);
    gate.open();
    for h in &mut hs {
        assert!(matches!(finish(h).1, Finish::Complete { .. }));
    }
    let mut runs: Vec<Vec<u8>> =
        fake.seen().into_iter().filter_map(|s| if let Seen::Ran { rows, .. } = s { Some(rows) } else { None }).collect();
    assert_eq!(runs.len(), 1, "one run for all eight: {runs:?}");
    runs[0].sort();
    assert_eq!(runs[0], (0..8).map(|v| v * 30).collect::<Vec<u8>>(), "two io workers may reorder, none is lost");
}

#[test]
fn a_run_stops_only_when_its_whole_batch_is_dead() {
    let gate = Arc::new(Gate::default());
    let fake = FakeLoader { run: Duration::from_millis(150), gates: [("z".to_string(), gate.clone())].into(), ..Default::default() };
    let (inf, _ev) = Inference::spawn(registry(&[("z", 8)], &["z"]), fake.clone());
    let lane = inf.lane(Priority::Bulk, Policy::Keep);
    let started = |k: usize| fake.seen().iter().filter(|s| matches!(s, Seen::Started { n: 2, .. })).count() >= k;

    let a = lane.submit(image(1), &ids(&["z"]), None);
    let mut b = lane.submit(image(2), &ids(&["z"]), None);
    wait_until("one held job and one queued", || gate.holds() && inf.queued(&GraphId::new("z")) == 1);
    gate.open();
    wait_until("a batch of two to start", || started(1));
    let RequestHandle { updates: mut a_updates, ticket } = a;
    drop(ticket);
    assert!(matches!(finish(&mut b).1, Finish::Complete { .. }), "b survives a's cancel");
    // two io workers may hand b to the graph before a, so the rows come in either order
    let seen = fake.seen();
    let ran_both = |rows: &[u8]| rows.len() == 2 && rows.contains(&1) && rows.contains(&2);
    assert!(seen.iter().any(|s| matches!(s, Seen::Ran { rows, .. } if ran_both(rows))), "the run was not stopped: {seen:?}");
    let a_done = loop {
        match a_updates.try_recv() {
            Ok(Update::Done(f)) => break f,
            Ok(_) => {}
            Err(e) => panic!("a has no Done: {e:?}"),
        }
    };
    assert!(matches!(a_done, Finish::Cancelled));

    // while one job runs, two more queue up and then run together; cancelling both stops that run
    let mut blocker = lane.submit(image(3), &ids(&["z"]), None);
    wait_until("the blocker to start", || fake.seen().iter().any(|s| matches!(s, Seen::Started { n: 1, .. })));
    let c = lane.submit(image(4), &ids(&["z"]), None);
    let d = lane.submit(image(5), &ids(&["z"]), None);
    wait_until("c and d to queue", || inf.queued(&GraphId::new("z")) == 2);
    assert!(matches!(finish(&mut blocker).1, Finish::Complete { .. }));
    wait_until("c and d to run together", || started(2));
    assert!(!fake.seen().iter().any(|s| matches!(s, Seen::Stopped { .. })));
    drop(c);
    drop(d);
    wait_until("the run to stop", || fake.seen().iter().any(|s| matches!(s, Seen::Stopped { .. })));
}

#[test]
fn shutdown_waits_for_every_graph_to_let_go() {
    let fake = FakeLoader { run: Duration::from_millis(50), ..Default::default() };
    let (inf, _ev) = Inference::spawn(registry(&[("x", 1), ("y", 1)], &["x"]), fake.clone());
    let lane = inf.lane(Priority::Bulk, Policy::Keep);
    let _a = lane.submit(image(1), &ids(&["x"]), None);
    let _b = lane.submit(image(2), &ids(&["y"]), None);
    wait_until("both graphs to run", || fake.seen().iter().filter(|s| matches!(s, Seen::Started { .. })).count() == 2);
    assert!(inf.shutdown(WAIT), "every worker ended");
    assert_eq!(fake.dropped(), 2, "both graphs were released before shutdown returned");
    let mut late = lane.submit(image(3), &ids(&["x"]), None);
    assert!(matches!(finish(&mut late).1, Finish::Failed(_)), "a stopped service refuses work");
}

/// The service's start-up default, or panic after `WAIT`.
fn default_of(ev: &mut fmpsc::UnboundedReceiver<ServiceEvent>) -> (Option<HeadId>, Vec<(HeadId, String)>) {
    let t0 = Instant::now();
    loop {
        match ev.try_recv() {
            Ok(ServiceEvent::Default { head, passed_over }) => return (head, passed_over),
            Ok(_) => {}
            Err(_) if t0.elapsed() < WAIT => std::thread::sleep(Duration::from_millis(1)),
            Err(e) => panic!("no Default event: {e:?}"),
        }
    }
}

#[test]
fn the_default_skips_a_head_whose_provider_is_missing() {
    let fake = FakeLoader { no_provider: ["fast".to_string()].into(), ..Default::default() };
    let (_inf, mut ev) = Inference::spawn(registry(&[("fast", 1), ("deployed", 1)], &["fast", "deployed"]), fake);
    let (head, passed_over) = default_of(&mut ev);
    assert_eq!(head, Some(HeadId::new("deployed")));
    assert_eq!(passed_over.len(), 1);
    assert!(passed_over[0].1.contains("TensorRT"), "{passed_over:?}");
}

#[test]
fn the_default_says_so_when_no_head_can_run() {
    let missing = || Err(crate::registry::Unavailable::Missing(vec!["/fake/x/model.onnx".into()]));
    let graphs = vec![crate::registry::Graph { manifest: manifest("x", 1), availability: missing() }];
    let reg = Registry::from_graphs("/fake/models.json".into(), graphs, ids(&["x"]), vec![]).unwrap();
    let (_inf, mut ev) = Inference::spawn(Arc::new(reg), FakeLoader::default());
    let (head, passed_over) = default_of(&mut ev);
    assert_eq!(head, None, "nothing can run, and the service says so instead of staying silent");
    assert!(matches!(&passed_over[..], [(x, why)] if x.as_str() == "x" && why.contains("missing")), "{passed_over:?}");
}

/// A 4x4 PNG whose every channel is `v`.
fn png(v: u8) -> Vec<u8> {
    let mut bytes = Vec::new();
    RgbImage::from_pixel(4, 4, image::Rgb([v, v, v]))
        .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)
        .unwrap();
    bytes
}

#[cfg(target_os = "linux")]
#[test]
fn a_picture_read_during_shutdown_starts_no_graph() {
    use std::io::Write as _;
    use std::os::unix::fs::OpenOptionsExt as _;
    const O_NONBLOCK: i32 = 0o4000;
    let dir = std::env::temp_dir().join(format!("gpui-demo-shutdown-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let fifo = dir.join("picture.png");
    assert!(std::process::Command::new("mkfifo").arg(&fifo).status().unwrap().success());
    let fake = FakeLoader::default();
    let (inf, _ev) = Inference::spawn(registry(&[("x", 1)], &["x"]), fake.clone());
    let lane = inf.lane(Priority::Interactive, Policy::Keep);
    let mut h = lane.submit(ImageInput::Source(Source::File(fifo.clone())), &ids(&["x"]), None);
    // a non-blocking open for writing succeeds once a reader has the fifo open: an io worker is mid-prepare
    let t0 = Instant::now();
    let mut writer = loop {
        match std::fs::OpenOptions::new().write(true).custom_flags(O_NONBLOCK).open(&fifo) {
            Ok(w) => break w,
            Err(_) if t0.elapsed() < WAIT => std::thread::sleep(Duration::from_millis(1)),
            Err(e) => panic!("no io worker opened the picture: {e}"),
        }
    };
    assert!(!inf.shutdown(Duration::ZERO), "the io worker is still reading the picture");
    writer.write_all(&png(200)).unwrap();
    drop(writer);
    let (heads, _) = finish(&mut h);
    assert!(heads.iter().all(|(_, r)| r.is_err()), "a stopped service answers nothing");
    inf.preload(&HeadId::new("x"));
    std::thread::sleep(Duration::from_millis(50));
    assert!(fake.seen().is_empty(), "no graph loaded or ran after shutdown: {:?}", fake.seen());
    let _ = std::fs::remove_dir_all(&dir); // test scratch dir under the OS temp dir only
}

#[test]
fn a_default_no_manifest_provides_is_passed_over() {
    let (_inf, mut ev) = Inference::spawn(registry(&[("deployed", 1)], &["fast", "deployed"]), FakeLoader::default());
    let (head, passed_over) = default_of(&mut ev);
    assert_eq!(head, Some(HeadId::new("deployed")));
    assert!(matches!(&passed_over[..], [(f, _)] if f.as_str() == "fast"), "the notice can say why: {passed_over:?}");
}
