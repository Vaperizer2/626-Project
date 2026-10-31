//! gpui-demo: desktop demo for the 626-Project electronic-component classifier.
//!
//!   gpui-demo [OPTIONS] [IMAGE|URL ...]               open the window (optionally classifying the given images)
//!   gpui-demo classify [OPTIONS] [--json] IMAGE|URL ...  headless: print predictions, no window
//!   gpui-demo preprocess IMAGE OUT.f32 [--model HEAD]    dump the [1,3,H,W] float32 input tensor (parity tests)
//!   gpui-demo models                                     list the model registry and what is available

/// Profile a block with hotpath when built with `--features hotpath`; otherwise expands to the block itself.
#[cfg(feature = "hotpath")]
macro_rules! prof {
    ($label:literal, $e:expr) => {
        hotpath::measure_block!($label, { $e })
    };
}
#[cfg(not(feature = "hotpath"))]
macro_rules! prof {
    ($label:literal, $e:expr) => {
        $e
    };
}

mod classification;
mod demo;
mod fetch;
mod gui;
mod inference;
mod integrity;
mod manifest;
mod preprocess;
mod registry;
mod runtime;

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use futures::StreamExt;

use crate::classification::HeadOutput;
use crate::fetch::Source;
use crate::inference::{
    Finish, GraphState, ImageInput, Inference, LoadStage, Policy, Priority, RequestHandle, RunError, ServiceEvent,
    Timing, Update,
};
use crate::manifest::{HeadId, Precision, Task};
use crate::manifest::EpKind;
use crate::inference::Loader as _;
use crate::runtime::{ModelOptions, OrtLoader};
use crate::registry::Registry;

const HELP: &str = "\
gpui-demo - classify electronic components in pictures (GPUI + onnxruntime)

USAGE:
    gpui-demo [OPTIONS] [IMAGE|URL ...]                 open the window
    gpui-demo classify [OPTIONS] [--json] IMAGE|URL ... headless; one line (or JSON object) per image
    gpui-demo preprocess IMAGE OUT.f32 [--model HEAD]  write the model input tensor (little-endian f32)
    gpui-demo models [--check]                          list the models, whether each is available and which
                                                        execution providers register; --check also verifies
                                                        every file's sha256 and loads each graph

OPTIONS:
    --models PATH     the model index exports/models.json, or its folder (default: $COMPCLASS_MODELS,
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
    --model HEADS     classify with these models (comma-separated head ids, or all; default: the
                      index's default head). See `gpui-demo models` for the ids.
    --demo-dir DIR    window: folder with demo_stack.json (default: $COMPCLASS_DEMO_DIR, else the
                      first demo_stack/ found walking up from the current directory, then the executable)
    --tab TAB         window: start on classify (default), demo or benchmark
    --autorun         window: start Run all (Demo tab) or the benchmark (Benchmark tab) right away
    --dark, --light   window: initial theme (default: follow the system)
    -h, --help        this help
    -V, --version     version

KEYS (window): Ctrl+1/2/3 Classify / Demo / Benchmark tab, Ctrl+M next model, Ctrl+T light/dark theme,
               Ctrl+Q quit. Classify: Ctrl+O open, Ctrl+V paste URL / path / image, Ctrl+L URL field,
               Enter fetch, Up/Down recent images. Demo: Left/Right or J/K step, Enter classify,
               Ctrl+R run all, Esc cancel. Benchmark: Ctrl+R run, Esc cancel";

#[derive(Debug, Default)]
struct Args {
    mode: String,
    opts: ModelOptions,
    json: bool,
    check: bool,
    models: Vec<String>,
    demo_dir: Option<PathBuf>,
    gui: gui::GuiOptions,
    rest: Vec<String>,
}

fn parse_args() -> Result<Args> {
    let mut it = std::env::args().skip(1).peekable();
    let mut a = Args { mode: "gui".into(), ..Default::default() };
    if let Some(first) = it.peek()
        && (first == "classify" || first == "preprocess" || first == "models")
    {
        a.mode = it.next().unwrap();
    }
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().with_context(|| format!("{name} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            "-V" | "--version" => {
                println!("gpui-demo {} (jpeg decoder: {})", env!("CARGO_PKG_VERSION"), preprocess::JPEG_DECODER);
                std::process::exit(0);
            }
            "--models" => a.opts.models = Some(PathBuf::from(value("--models")?)),
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
    Ok(a)
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

#[derive(serde::Serialize)]
struct JsonRow<'a> {
    source: &'a str,
    model: &'a str,
    label: Option<&'a str>,
    top3: Vec<(&'a str, f32)>,
    unsure: bool,
    ms: f64,
    probs: Vec<f32>,
    error: Option<String>,
    timing: Timing,
}

/// Start the service with onnxruntime behind it; graph load events go to stderr.
fn service(a: &Args, reg: &Arc<Registry>) -> Inference {
    let (inf, mut events) = Inference::spawn(reg.clone(), OrtLoader::new(&a.opts, reg.index_path().parent()));
    let reg = reg.clone();
    std::thread::spawn(move || {
        while let Some(ev) = futures::executor::block_on(events.next()) {
            let ServiceEvent::Graph(g, state) = ev else { continue };
            let heads: Vec<String> = reg.graph(&g).map_or(vec![], |g| g.manifest.heads.iter().map(|h| h.id.to_string()).collect());
            match state {
                GraphState::Loading { stage: LoadStage::BuildingEngine, .. } => {
                    eprintln!("[gpui-demo] {g} ({}): building the TensorRT engine (first run only)", heads.join(", "))
                }
                GraphState::Ready(l) => {
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
    inf
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

/// `classify`: every image is submitted at once to one bulk lane, so decoding overlaps inference; rows come out in
/// input order, heads in `--model` order (default: the index's default head).
fn run_classify(a: &Args) -> Result<u8> {
    if a.rest.is_empty() {
        bail!("classify needs at least one image path or URL");
    }
    let reg = registry(a)?;
    let keys = heads(a, &reg)?;
    let inf = service(a, &reg);
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
        let (mut results, failed) = match handle {
            Some(h) => collect(h),
            None => (vec![], Some(RunError::Read("empty source".into()))),
        };
        results.sort_by_key(|r| keys.iter().position(|k| *k == r.0));
        let outcome: Vec<(HeadId, Result<(HeadOutput, Timing), String>)> = match failed {
            Some(e) => keys.iter().map(|k| (k.clone(), Err(e.to_string()))).collect(),
            None => results.into_iter().map(|(k, r, t)| (k, r.map(|o| (o, t)).map_err(|e| e.to_string()))).collect(),
        };
        for (head, r) in outcome {
            match r {
                Ok((HeadOutput::Classify(c), timing)) => {
                    let label = c.committed().map_or(UNSURE, |i| c.labels.name(i));
                    let top3: Vec<(&str, f32)> = c.top(3).into_iter().map(|(i, p)| (c.labels.name(i), p)).collect();
                    if a.json {
                        rows.push(serde_json::to_value(JsonRow {
                            source: s,
                            model: head.as_str(),
                            label: Some(label),
                            top3,
                            unsure: c.abstained(),
                            ms: (timing.ms() * 10.0).round() / 10.0,
                            probs: c.probs.to_vec(),
                            error: None,
                            timing,
                        })?);
                    } else {
                        let top: Vec<String> = top3.iter().map(|(l, v)| format!("{l} {v:.3}")).collect();
                        writeln!(
                            stdout.lock(),
                            "{s}\t{head}\t{label}\t{:.3}\ttop3: {}\t{:.0} ms",
                            c.best().1,
                            top.join(", "),
                            timing.ms()
                        )?;
                    }
                }
                Err(e) => {
                    failures = failures.saturating_add(1);
                    if a.json {
                        rows.push(serde_json::json!({"source": s, "model": head.as_str(), "label": null, "top3": [],
                                                     "unsure": false, "ms": 0.0, "probs": [], "error": e}));
                    } else {
                        writeln!(stdout.lock(), "{s}\t{head}\tERROR\t{e}")?;
                    }
                }
            }
        }
    }
    if a.json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
    }
    Ok(if failures > 0 { 2 } else { 0 })
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
    let mut stamps = integrity::Stamps::load();
    let mut bad = 0;
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
