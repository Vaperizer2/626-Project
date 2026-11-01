//! gpui-demo: desktop demo for the 626-Project electronic-component classifier.
//!
//!   gpui-demo [OPTIONS] [IMAGE|URL ...]               open the window (optionally classifying the given images)
//!   gpui-demo classify [OPTIONS] [--json] IMAGE|URL ...  headless: print predictions, no window
//!   gpui-demo preprocess IMAGE OUT.f32 [--model HEAD]    dump the [1,3,H,W] float32 input tensor (parity tests)
//!   gpui-demo models [--check]                           list the model registry and what is available
//!   gpui-demo bench [--model HEAD] IMAGE ...             latency by stage and streamed FPS through the service

mod app;
mod demo;
mod gui;

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use futures::StreamExt;
use inference::{integrity, preprocess, registry, runtime};

use inference::classification::HeadOutput;
use inference::fetch::Source;
use inference::service::{
    Finish, GraphState, ImageInput, Inference, LoadStage, Loaded, Policy, Priority, RequestHandle, RunError,
    ServiceEvent, Timing, Update,
};
use inference::manifest::{GraphId, HeadId, Precision, Task};
use inference::manifest::EpKind;
use inference::service::Loader as _;
use inference::runtime::{Settings, OrtLoader};
use inference::registry::Registry;

/// `--help`.
fn help() -> String {
    let (title, env) = (app::TITLE, app::ENV_PREFIX);
    format!(
        "\
gpui-demo - {title}: classify electronic components in pictures (GPUI + onnxruntime)

USAGE:
    gpui-demo [OPTIONS] [IMAGE|URL ...]                 open the window
    gpui-demo classify [OPTIONS] [--json] IMAGE|URL ... headless; one line (or JSON object) per image
    gpui-demo preprocess IMAGE OUT.f32 [--model HEAD]  write the model input tensor (little-endian f32)
    gpui-demo bench [--model HEAD] [--rounds N] [--warmup N] IMAGE ...
                                                        time one model through the inference service:
                                                        per-image latency by stage, then streamed FPS
    gpui-demo models [--check]                          list the models, whether each is available and which
                                                        execution providers register; --check also verifies
                                                        every file's sha256 and loads each graph

OPTIONS:
    --models PATH     the model index exports/models.json, or its folder (default: ${env}_MODELS,
                      else the first exports/models.json found walking up from the current directory,
                      then from the executable)
    --ort-lib PATH    libonnxruntime.so to load (default: $ORT_DYLIB_PATH, else
                      <repo>/.venv/.../onnxruntime/capi/libonnxruntime.so*, else next to the
                      executable, else the system library path)
    --threads N       onnxruntime intra-op threads (default: onnxruntime's choice)
    --ep LIST         execution providers for every model, in order of preference: tensorrt, cuda
                      and/or cpu, comma-separated (default: each model's manifest; usually
                      cuda,cpu, and tensorrt,cuda,cpu for the fast model)
    --json            classify: print a JSON array instead of text lines
    --model HEADS     classify with these models (comma-separated head ids, or all). Default: the
                      first of the index's defaults whose preferred execution provider registers here
                      and is not displaced by --ep (fast with TensorRT, else deployed). See
                      `gpui-demo models` for the ids.
    --demo-dir DIR    window: folder with demo_stack.json (default: ${env}_DEMO_DIR, else the
                      first demo_stack/ found walking up from the current directory, then the executable)
    --tab TAB         window: start on classify (default), demo or benchmark
    --autorun         window: start Run all (Demo tab) or the benchmark (Benchmark tab) right away
    --dark, --light   window: initial theme (default: follow the system)
    -h, --help        this help
    -V, --version     version

KEYS (window): Ctrl+1/2/3 Classify / Demo / Benchmark tab, Ctrl+M next model, Ctrl+T light/dark theme,
               Ctrl+Q quit. Classify: Ctrl+O open, Ctrl+V paste URL / path / image, Ctrl+L URL field,
               Enter fetch, Up/Down recent images. Demo: Left/Right or J/K step, Enter classify,
               Ctrl+R run all, Esc cancel. Benchmark: Ctrl+R run, Esc cancel"
    )
}

#[derive(Debug, Default)]
struct Args {
    mode: String,
    opts: Settings,
    json: bool,
    check: bool,
    rounds: usize,
    warmup: usize,
    models: Vec<String>,
    demo_dir: Option<PathBuf>,
    gui: gui::GuiOptions,
    rest: Vec<String>,
}

fn parse_args() -> Result<Args> {
    let mut it = std::env::args().skip(1).peekable();
    let mut a = Args { mode: "gui".into(), rounds: 5, warmup: 5, ..Default::default() };
    if let Some(first) = it.peek()
        && (first == "classify" || first == "preprocess" || first == "models" || first == "bench")
    {
        a.mode = it.next().unwrap();
    }
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().with_context(|| format!("{name} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{}", help());
                std::process::exit(0);
            }
            "-V" | "--version" => {
                println!("gpui-demo {} (jpeg decoder: {})", env!("CARGO_PKG_VERSION"), preprocess::JPEG_DECODER);
                std::process::exit(0);
            }
            "--models" => a.opts.models = Some(PathBuf::from(value("--models")?)),
            // the export folder (exports/final) of the app before the model index; not in --help
            "--model-dir" => {
                let dir = PathBuf::from(value("--model-dir")?);
                let index = match dir.parent() {
                    Some(p) if !dir.join(registry::INDEX_FILE).is_file() => p.to_path_buf(),
                    _ => dir,
                };
                eprintln!("[gpui-demo] --model-dir is deprecated: use --models {}", index.display());
                a.opts.models = Some(index);
            }
            "--ort-lib" => a.opts.ort_lib = Some(PathBuf::from(value("--ort-lib")?)),
            "--threads" => a.opts.threads = Some(value("--threads")?.parse().context("--threads")?),
            "--ep" => {
                let list = value("--ep")?
                    .split(',')
                    .filter(|e| !e.trim().is_empty())
                    .map(|e| EpKind::parse(e).with_context(|| format!("--ep {e:?}: expected tensorrt, cuda or cpu")))
                    .collect::<Result<Vec<_>>>()?;
                a.opts.eps = Some(list);
            }
            "--json" => a.json = true,
            "--check" => a.check = true,
            "--rounds" => a.rounds = value("--rounds")?.parse().context("--rounds")?,
            "--warmup" => a.warmup = value("--warmup")?.parse().context("--warmup")?,
            "--demo-dir" => a.demo_dir = Some(PathBuf::from(value("--demo-dir")?)),
            "--tab" => {
                a.gui.tab = match value("--tab")?.to_ascii_lowercase().as_str() {
                    "classify" => gui::TabKind::Classify,
                    "demo" => gui::TabKind::Demo,
                    "bench" | "benchmark" => gui::TabKind::Bench,
                    t => bail!("--tab {t}: expected classify, demo or benchmark"),
                }
            }
            "--autorun" => a.gui.autorun = true,
            "--dark" => a.gui.theme = Some(true),
            "--light" => a.gui.theme = Some(false),
            "--model" => {
                a.models.extend(value("--model")?.split(',').map(str::trim).filter(|k| !k.is_empty()).map(String::from))
            }
            s if s.starts_with("--") => bail!("unknown option {s} (see --help)"),
            _ => a.rest.push(arg),
        }
    }
    a.opts.models = a.opts.models.or_else(|| env_path("MODELS"));
    a.opts.cache_dir = cache_dir();
    Ok(a)
}

/// `$<ENV_PREFIX>_<name>`, when it is set and not empty.
fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(format!("{}_{name}", app::ENV_PREFIX)).filter(|s| !s.is_empty()).map(PathBuf::from)
}

/// `$XDG_CACHE_HOME/<NAME>`, else `~/.cache/<NAME>`.
fn cache_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join(app::NAME))
}

fn registry(a: &Args) -> Result<Arc<Registry>> {
    let index = Registry::locate(a.opts.models.as_deref())?;
    let reg = Registry::load(&index)?;
    for p in reg.problems() {
        eprintln!("[gpui-demo] {p}");
    }
    Ok(Arc::new(reg))
}

/// `--model` names resolved against the registry (`all` = every head); none = the index's default head.
fn heads(a: &Args, reg: &Registry) -> Result<Vec<HeadId>> {
    let known = || reg.ids().iter().map(|h| h.to_string()).collect::<Vec<_>>().join(", ");
    let mut out: Vec<HeadId> = Vec::new();
    for name in &a.models {
        let ids = if name == "all" {
            reg.ids()
        } else {
            vec![reg.find(name).with_context(|| format!("unknown model {name:?} (known: {}; or all)", known()))?]
        };
        for id in ids {
            if !out.contains(&id) {
                out.push(id);
            }
        }
    }
    if out.is_empty() {
        out.push(reg.first_default().context("the model index has no heads")?);
    }
    Ok(out)
}

/// The `label` of an abstaining head in classify output (the Python demo's spelling).
const UNSURE: &str = "UNSURE";

/// One `classify --json` row; an error row has the same keys.
#[derive(serde::Serialize)]
struct JsonRow<'a> {
    source: &'a str,
    model: &'a str,
    label: Option<&'a str>,
    top3: Vec<(&'a str, f32)>,
    unsure: bool,
    /// a head that always answers is below its confidence bound (heads that abstain say `unsure` instead)
    low_confidence: bool,
    ms: f64,
    probs: Vec<f32>,
    error: Option<&'a str>,
    timing: Option<Timing>,
}

impl<'a> JsonRow<'a> {
    fn new(source: &'a str, head: &'a HeadId, row: &'a Result<(HeadOutput, Timing), String>) -> Self {
        match row {
            Ok((HeadOutput::Classify(c), timing)) => JsonRow {
                source,
                model: head.as_str(),
                label: Some(c.committed().map_or(UNSURE, |i| c.labels.name(i))),
                top3: c.top(3).into_iter().map(|(i, p)| (c.labels.name(i), p)).collect(),
                unsure: c.abstained(),
                low_confidence: c.low_confidence(),
                ms: (timing.ms() * 10.0).round() / 10.0,
                probs: c.probs.to_vec(),
                error: None,
                timing: Some(*timing),
            },
            Err(e) => JsonRow {
                source,
                model: head.as_str(),
                label: None,
                top3: vec![],
                unsure: false,
                low_confidence: false,
                ms: 0.0,
                probs: vec![],
                error: Some(e),
                timing: None,
            },
        }
    }
}

/// The start-up default (see `ServiceEvent::Default`) and the defaults passed over on the way.
type DefaultHead = (Option<HeadId>, Vec<(HeadId, String)>);

/// The service with onnxruntime behind it, the start-up default head, and each graph's load once it is ready.
struct Service {
    inf: Inference,
    default: std::sync::mpsc::Receiver<DefaultHead>,
    loaded: Arc<std::sync::Mutex<Vec<(GraphId, Loaded)>>>,
}

impl Service {
    /// The head to use without `--model`; an error saying why when no head can run here.
    fn default_head(&self) -> Result<HeadId> {
        let (head, passed_over) = self.default.recv().context("the inference service stopped before picking a model")?;
        let Some(head) = head else {
            let why: Vec<String> = passed_over.iter().map(|(h, why)| format!("{h}: {why}")).collect();
            let why = if why.is_empty() { String::new() } else { format!(" ({})", why.join("; ")) };
            bail!("no model can run here{why}; see `gpui-demo models`");
        };
        for (h, why) in &passed_over {
            eprintln!("[gpui-demo] {h} is not the default here: {why}; using {head}");
        }
        Ok(head)
    }
}

/// Start the service; graph load events also go to stderr.
fn service(a: &Args, reg: &Arc<Registry>) -> Service {
    let (inf, mut events) = Inference::spawn(reg.clone(), OrtLoader::new(&a.opts, reg.index_path().parent()));
    let reg = reg.clone();
    let (default_tx, default) = std::sync::mpsc::channel();
    let loaded = Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen = loaded.clone();
    std::thread::spawn(move || {
        while let Some(ev) = futures::executor::block_on(events.next()) {
            let (g, state) = match ev {
                ServiceEvent::Graph(g, state) => (g, state),
                ServiceEvent::Default { head, passed_over } => {
                    let _ = default_tx.send((head, passed_over));
                    continue;
                }
            };
            let heads: Vec<String> = reg.graph(&g).map_or(vec![], |g| g.manifest.heads.iter().map(|h| h.id.to_string()).collect());
            match state {
                GraphState::Loading { stage: LoadStage::BuildingEngine, .. } => {
                    eprintln!("[gpui-demo] {g} ({}): building the TensorRT engine (first run only)", heads.join(", "))
                }
                GraphState::Ready(l) => {
                    seen.lock().expect("loaded lock").push((g.clone(), l.clone()));
                    let checked = if l.verified { "checksums verified" } else { "checksums not verified: models --check" };
                    eprintln!(
                        "[gpui-demo] {g} ({}) ready in {:.0} ms on {} ({}; {checked})",
                        heads.join(", "),
                        l.ms,
                        l.placement,
                        l.runtime
                    );
                    for (ep, why) in &l.placement.skipped {
                        eprintln!("[gpui-demo]   {} skipped: {why}", ep.name());
                    }
                }
                GraphState::Failed(e) => eprintln!("[gpui-demo] {g} ({}) failed to load: {e}", heads.join(", ")),
                GraphState::Loading { .. } => {}
            }
        }
    });
    Service { inf, default, loaded }
}

/// One request's outcome: every head's result (or the input's failure).
fn collect(h: &mut RequestHandle) -> (Vec<(HeadId, Result<HeadOutput, RunError>, Timing)>, Option<RunError>) {
    let mut heads = Vec::new();
    while let Some(u) = futures::executor::block_on(h.updates.next()) {
        match u {
            Update::Decoded { .. } => {}
            Update::Head { head, result, timing } => heads.push((head, result, timing)),
            Update::Done(Finish::Failed(e)) => return (heads, Some(e)),
            Update::Done(_) => break,
        }
    }
    (heads, None)
}

/// One row per requested head, in `keys` order; a head the request ended without is a failed row.
fn rows_of(
    keys: &[HeadId],
    mut results: Vec<(HeadId, Result<HeadOutput, RunError>, Timing)>,
    failed: Option<RunError>,
) -> Vec<(HeadId, Result<(HeadOutput, Timing), String>)> {
    keys.iter()
        .map(|k| {
            let row = match (&failed, results.iter().position(|r| r.0 == *k)) {
                (Some(e), _) => Err(e.to_string()),
                (None, Some(i)) => {
                    let (_, r, t) = results.swap_remove(i);
                    r.map(|o| (o, t)).map_err(|e| e.to_string())
                }
                (None, None) => Err("no answer: the inference service ended the request without one".into()),
            };
            (k.clone(), row)
        })
        .collect()
}

/// `classify`: every image is submitted at once to one bulk lane, so decoding overlaps inference; rows come out in
/// input order, heads in `--model` order (default: the index's default head).
fn run_classify(a: &Args) -> Result<u8> {
    if a.rest.is_empty() {
        bail!("classify needs at least one image path or URL");
    }
    let reg = registry(a)?;
    let service = service(a, &reg);
    let keys = if a.models.is_empty() {
        // the first default whose preferred provider registers here, as in the window
        vec![service.default_head()?]
    } else {
        heads(a, &reg)?
    };
    let inf = &service.inf;
    let lane = inf.lane(Priority::Bulk, Policy::Keep);
    let mut requests: Vec<(&String, Option<RequestHandle>)> = a
        .rest
        .iter()
        .map(|s| (s, Source::parse(s).map(|src| lane.submit(ImageInput::Source(src), &keys, None))))
        .collect();
    let mut failures = 0u8;
    let mut rows = Vec::new();
    let stdout = std::io::stdout();
    for (s, handle) in &mut requests {
        let (results, failed) = match handle {
            Some(h) => collect(h),
            None => (vec![], Some(RunError::Read("empty source".into()))),
        };
        for (head, r) in rows_of(&keys, results, failed) {
            if r.is_err() {
                failures = failures.saturating_add(1);
            }
            let row = JsonRow::new(s, &head, &r);
            if a.json {
                rows.push(serde_json::to_value(row)?);
                continue;
            }
            match (&r, row.label) {
                (Ok((HeadOutput::Classify(c), timing)), Some(label)) => {
                    let top: Vec<String> = row.top3.iter().map(|(l, v)| format!("{l} {v:.3}")).collect();
                    writeln!(
                        stdout.lock(),
                        "{s}\t{head}\t{label}\t{:.3}\ttop3: {}\t{:.0} ms",
                        c.best().1,
                        top.join(", "),
                        timing.ms()
                    )?;
                }
                (_, _) => writeln!(stdout.lock(), "{s}\t{head}\tERROR\t{}", row.error.unwrap_or_default())?,
            }
        }
    }
    if a.json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
    }
    Ok(end(inf, if failures > 0 { 2 } else { 0 }))
}

/// Median, 10th and 90th percentile (nearest rank), min and max.
fn spread(mut v: Vec<f64>) -> serde_json::Value {
    if v.is_empty() {
        return serde_json::Value::Null;
    }
    v.sort_by(f64::total_cmp);
    let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    let r = |x: f64| (x * 1000.0).round() / 1000.0;
    serde_json::json!({"median": r(at(0.5)), "p10": r(at(0.1)), "p90": r(at(0.9)), "min": r(v[0]), "max": r(v[v.len() - 1]), "n": v.len()})
}

/// `bench`: one head through the service. Latency: every image submitted alone and awaited, read and decoded from
/// its file each time; `--warmup` runs first, unrecorded, then `--rounds` passes over the images. Stream: rounds x
/// images submitted at once to a bulk lane, FPS = images / wall time from the first submit to the last result.
fn run_bench(a: &Args) -> Result<u8> {
    let reg = registry(a)?;
    let service = service(a, &reg);
    let head = if a.models.is_empty() {
        service.default_head()?
    } else {
        match heads(a, &reg)?.as_slice() {
            [h] => h.clone(),
            _ => bail!("bench takes one --model"),
        }
    };
    let sources: Vec<Source> =
        a.rest.iter().map(|s| Source::parse(s).context("empty source")).collect::<Result<_>>()?;
    if sources.is_empty() {
        bail!("bench needs at least one image");
    }
    let Service { inf, loaded, .. } = service;
    let keys = [head.clone()];
    let interactive = inf.lane(Priority::Interactive, Policy::Keep);
    let mut errors = 0usize;
    let mut answers: Vec<String> = Vec::new();
    let mut one = |src: &Source, record: bool, answers: &mut Vec<String>| -> Option<(f64, Timing)> {
        let t0 = Instant::now();
        let mut h = interactive.submit(ImageInput::Source(src.clone()), &keys, None);
        let (results, failed) = collect(&mut h);
        let wall = t0.elapsed().as_secs_f64() * 1000.0;
        match (failed, results.into_iter().next()) {
            (None, Some((_, Ok(HeadOutput::Classify(c)), timing))) => {
                if record {
                    answers.push(c.committed().map_or(UNSURE, |i| c.labels.name(i)).to_string());
                }
                Some((wall, timing))
            }
            _ => {
                errors += 1;
                None
            }
        }
    };
    let mut scratch = Vec::new();
    for i in 0..a.warmup {
        one(&sources[i % sources.len()], false, &mut scratch);
    }
    let mut stages: [Vec<f64>; 6] = Default::default();
    for round in 0..a.rounds {
        for src in &sources {
            if let Some((wall, t)) = one(src, round == 0, &mut answers) {
                for (v, x) in stages.iter_mut().zip([t.decode_ms, t.preprocess_ms, t.queue_ms, t.run_ms, t.post_ms, wall]) {
                    v.push(x);
                }
            }
        }
    }
    let bulk = inf.lane(Priority::Bulk, Policy::Keep);
    let t0 = Instant::now();
    let mut handles: Vec<RequestHandle> = (0..a.rounds)
        .flat_map(|_| sources.iter())
        .map(|src| bulk.submit(ImageInput::Source(src.clone()), &keys, None))
        .collect();
    let mut streamed = 0usize;
    for h in &mut handles {
        match collect(h) {
            (r, None) if r.iter().all(|x| x.1.is_ok()) => streamed += 1,
            _ => errors += 1,
        }
    }
    let stream_s = t0.elapsed().as_secs_f64();
    let graph = reg.head(&head).expect("resolved head").manifest().id.clone();
    let load = loaded.lock().expect("loaded lock").iter().find(|(g, _)| *g == graph).map(|(_, l)| l.clone());
    let [decode, preprocess, queue, run, post, e2e] = stages;
    let report = serde_json::json!({
        "head": head.as_str(),
        "graph": graph.as_str(),
        "providers": load.as_ref().map(|l| l.placement.to_string()),
        "skipped": load.as_ref().map(|l| l.placement.skipped.iter().map(|(k, w)| format!("{}: {w}", k.name())).collect::<Vec<_>>()),
        "runtime": load.as_ref().map(|l| l.runtime.clone()),
        "load_ms": load.as_ref().map(|l| l.ms),
        "images": sources.len(),
        "warmup": a.warmup,
        "rounds": a.rounds,
        "errors": errors,
        "latency_ms": {"read_decode": spread(decode), "preprocess": spread(preprocess), "queue": spread(queue),
                       "run": spread(run), "readout": spread(post), "end_to_end": spread(e2e)},
        "stream": {"images": streamed, "wall_s": (stream_s * 1000.0).round() / 1000.0, "fps": streamed as f64 / stream_s},
        "answers": answers,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(end(&inf, if errors > 0 { 2 } else { 0 }))
}

/// Release every session before the process exits with `code` (see `Inference::shutdown`); if a load is still
/// running, leave with `code` without exit handlers rather than race its provider's teardown.
fn end(inf: &Inference, code: u8) -> u8 {
    if !inf.shutdown(std::time::Duration::from_secs(30)) {
        eprintln!("[gpui-demo] a model is still loading; exiting without waiting for it");
        let _ = std::io::stdout().flush();
        runtime::exit_now(code.into());
    }
    code
}

/// The heads and their files, then per graph which of its execution providers register here (a probe: the
/// provider library is loaded, no session is built).
fn run_models(a: &Args) -> Result<u8> {
    let reg = registry(a)?;
    for h in reg.heads() {
        let Task::Classify(t) = &h.spec.task;
        match h.available() {
            Ok(()) => println!(
                "{:9} {:30} available  {} [output {}]",
                h.id(),
                h.spec.title,
                h.manifest().graph.display(),
                t.readout.output()
            ),
            Err(e) => println!("{:9} {:30} unavailable  {e}", h.id(), h.spec.title),
        }
    }
    // --check: a manifest the index lists but that cannot be used is a failure too (registry() printed why)
    let mut bad = if a.check { reg.problems().len() } else { 0 };
    if bad > 0 {
        println!("index     {bad} problem(s) with {}", reg.index_path().display());
    }
    let loader = OrtLoader::new(&a.opts, reg.index_path().parent());
    let rt = match loader.runtime() {
        Ok(rt) => rt,
        Err(e) => {
            println!("execution providers: unknown, {e}");
            return Ok(if a.check { 1 } else { 0 });
        }
    };
    println!("onnxruntime {} ({})", rt.version, rt.ort_lib.display());
    for g in reg.graphs() {
        let m = &g.manifest;
        let probes: Vec<String> = rt
            .preferences(m)
            .iter()
            .map(|ep| match rt.probe(ep, m) {
                Ok(()) => format!("{} registers", ep.kind.name()),
                Err(e) => format!("{} does not ({e})", ep.kind.name()),
            })
            .collect();
        let precision = match m.precision {
            Precision::Fp32 => "fp32",
            Precision::Fp16 => "fp16",
            Precision::Int8Qdq => "int8 QDQ",
        };
        println!("{:9} {precision} graph; {}", m.id, probes.join("; "));
    }
    if !a.check {
        return Ok(0);
    }
    // --check: hash every file against its manifest, then load the graph the way the app does
    let mut stamps = integrity::Stamps::load(a.opts.cache_dir.as_deref());
    for g in reg.graphs() {
        let m = &g.manifest;
        if let Err(e) = &g.availability {
            bad += 1;
            println!("{:9} UNAVAILABLE  {e}", m.id);
            continue;
        }
        let t0 = Instant::now();
        match integrity::check(m, &mut stamps) {
            integrity::Check::Ok { bytes } => {
                let hashed = format!("sha256 ok ({:.1} MB in {:.1} s)", bytes as f64 / 1e6, t0.elapsed().as_secs_f64());
                match loader.load(m, &|_| {}) {
                    Ok((_, l)) => println!("{:9} READY  {hashed}; loaded in {:.0} ms on {}", m.id, l.ms, l.placement),
                    Err(e) => {
                        bad += 1;
                        println!("{:9} FAILED  {hashed}; load: {e}", m.id);
                    }
                }
            }
            integrity::Check::Mismatch { file, expected, found } => {
                bad += 1;
                println!("{:9} CHECKSUM  {}: sha256 {found}, the manifest says {expected}", m.id, file.display());
            }
            integrity::Check::Unreadable { file, error } => {
                bad += 1;
                println!("{:9} UNREADABLE  {}: {error}", m.id, file.display());
            }
        }
    }
    if let Err(e) = stamps.save() {
        eprintln!("[gpui-demo] could not save the checksum stamps: {e}");
    }
    Ok(if bad > 0 { 1 } else { 0 })
}

fn run_preprocess(a: &Args) -> Result<u8> {
    let [img, out] = a.rest.as_slice() else { bail!("usage: gpui-demo preprocess IMAGE OUT.f32 [--model HEAD]") };
    let reg = registry(a)?;
    let head = match heads(a, &reg)?.as_slice() {
        [h] => h.clone(),
        _ => bail!("preprocess takes one --model"),
    };
    let manifest = reg.head(&head).ok_or_else(|| anyhow!("unknown model {head}"))?.manifest().clone();
    let t0 = Instant::now();
    let bytes = Source::parse(img).context("empty source")?.read()?;
    let rgb = preprocess::decode(&bytes)?;
    let t = preprocess::to_tensor(&rgb, &manifest.input.pre)?;
    let mut buf = Vec::with_capacity(t.len() * 4);
    for v in &t {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(out, buf)?;
    eprintln!("[gpui-demo] {} -> {} ({} floats, {:.1} ms)", img, out, t.len(), t0.elapsed().as_secs_f64() * 1000.0);
    Ok(0)
}

fn main() -> ExitCode {
    // `--features hotpath`: timing report on exit (HOTPATH_OUTPUT_FORMAT / HOTPATH_OUTPUT_PATH configure it).
    // HOTPATH_SHUTDOWN_MS=N (e.g. for a GUI session): print the report and exit after N ms.
    #[cfg(feature = "hotpath")]
    let _hotpath = {
        let b = hotpath::HotpathGuardBuilder::new("gpui-demo").percentiles(&[50.0, 95.0]).limit(0);
        if std::env::var_os("HOTPATH_SHUTDOWN_MS").is_some() {
            b.build_with_shutdown(std::time::Duration::ZERO);
            None
        } else {
            Some(b.build())
        }
    };
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("gpui-demo: {e:#}");
            return ExitCode::from(64);
        }
    };
    let r = match args.mode.as_str() {
        "classify" => run_classify(&args),
        "bench" => run_bench(&args),
        "models" => run_models(&args),
        "preprocess" => run_preprocess(&args),
        _ => {
            gui::run(args.opts, args.demo_dir, args.gui, args.rest);
            Ok(0)
        }
    };
    match r {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            eprintln!("gpui-demo: {e:#}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference::{classification, manifest};

    #[test]
    fn a_requested_head_without_a_result_is_a_failed_row() {
        let (x, y) = (HeadId::new("x"), HeadId::new("y"));
        let task = &inference::testing::manifest("x", 1).heads[0].task;
        let answer = classification::read_head(task, &[0.9, 0.1]).unwrap();
        let rows = rows_of(&[x.clone(), y.clone()], vec![(x.clone(), Ok(answer), Timing::default())], None);
        let heads: Vec<&HeadId> = rows.iter().map(|r| &r.0).collect();
        assert_eq!(heads, [&x, &y], "every requested head has a row");
        assert!(rows[0].1.is_ok() && rows[1].1.is_err(), "the head that never answered counts as a failure");
    }

    /// packaging/gpu-env.sh, through scripts/test_gpu_env.sh (fake venvs under a path with spaces)
    #[cfg(unix)]
    #[test]
    fn gpu_env_passes_over_a_wrong_onnxruntime_and_keeps_paths_whole() {
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/test_gpu_env.sh");
        let out = std::process::Command::new("sh").arg(&script).output().unwrap();
        assert!(out.status.success(), "{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    }

    #[test]
    fn json_rows_have_one_shape_and_flag_low_confidence() {
        let head = HeadId::new("x");
        let mut m = Arc::try_unwrap(inference::testing::manifest("x", 1)).unwrap();
        let Task::Classify(t) = &mut m.heads[0].task;
        t.decision = manifest::Decision::Commit { low_confidence_below: manifest::Prob::new(0.95) };
        let answer = classification::read_head(&m.heads[0].task, &[0.9, 0.1]).unwrap();
        let rows = [Ok((answer, Timing::default())), Err("could not read".to_string())];
        let [ok, err] = rows.each_ref().map(|r| serde_json::to_value(JsonRow::new("p.jpg", &head, r)).unwrap());
        let keys = |v: &serde_json::Value| v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
        assert_eq!(keys(&ok), keys(&err), "an error row has every key");
        assert_eq!(ok["low_confidence"], true, "0.9 is below the head's 0.95 bound");
        assert!(err["timing"].is_null() && err["low_confidence"] == false);
    }

    #[test]
    fn models_check_fails_when_the_index_lists_a_manifest_it_cannot_read() {
        let dir = std::env::temp_dir().join(format!("gpui-demo-check-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(registry::INDEX_FILE), r#"{"schema":"model-index/1","default":[],"manifests":["gone/model.json"]}"#)
            .unwrap();
        let args = Args { check: true, opts: Settings { models: Some(dir.clone()), ..Default::default() }, ..Default::default() };
        assert_eq!(run_models(&args).unwrap(), 1, "a manifest that cannot be read is a failed check");
        let _ = std::fs::remove_dir_all(&dir); // test scratch dir under the OS temp dir only
    }
}
