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

mod contract;
mod demo;
mod fetch;
mod gui;
#[allow(dead_code)] // runtime and integrity fields; read once the EP and checksum steps land
mod manifest;
mod model;
mod models;
mod preprocess;
#[allow(dead_code)]
mod registry;

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};

use crate::fetch::Source;
use crate::manifest::{HeadId, Task};
use crate::model::ModelOptions;
use crate::models::Models;
use crate::registry::Registry;

const HELP: &str = "\
gpui-demo - classify electronic components in pictures (GPUI + onnxruntime)

USAGE:
    gpui-demo [OPTIONS] [IMAGE|URL ...]                 open the window
    gpui-demo classify [OPTIONS] [--json] IMAGE|URL ... headless; one line (or JSON object) per image
    gpui-demo preprocess IMAGE OUT.f32 [--model HEAD]  write the model input tensor (little-endian f32)
    gpui-demo models                                    list the models and whether each is available

OPTIONS:
    --models PATH     the model index exports/models.json, or its folder (default: $COMPCLASS_MODELS,
                      else the first exports/models.json found walking up from the current directory,
                      then from the executable)
    --ort-lib PATH    libonnxruntime.so to load (default: $ORT_DYLIB_PATH, else
                      <repo>/.venv/.../onnxruntime/capi/libonnxruntime.so*, else next to the
                      executable, else the system library path)
    --threads N       onnxruntime intra-op threads (default: onnxruntime's choice)
    --cuda            request the CUDA execution provider (build with --features cuda)
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
            "--cuda" => a.opts.cuda = true,
            "--json" => a.json = true,
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
}

/// `classify`: lazy engines, one forward pass per graph and image; the deployed model unless `--model` says otherwise.
fn run_classify(a: &Args) -> Result<u8> {
    if a.rest.is_empty() {
        bail!("classify needs at least one image path or URL");
    }
    let reg = registry(a)?;
    let keys = heads(a, &reg)?;
    let mut models = Models::new(reg.clone(), &a.opts);
    for k in &keys {
        let title = &reg.head(k).expect("resolved head").spec.title;
        match models.ensure(k) {
            Ok(ms) => eprintln!("[gpui-demo] model {k} ({title}) ready in {ms:.0} ms"),
            Err(e) => eprintln!("[gpui-demo] model {k} ({title}) unavailable: {e}"),
        }
    }
    if let Some((lib, provider)) = models.runtime_info() {
        eprintln!("[gpui-demo] onnxruntime {lib}, {provider}; jpeg decoder: {}", preprocess::JPEG_DECODER);
    }
    let mut failures = 0u8;
    let mut rows = Vec::new();
    let stdout = std::io::stdout();
    for s in &a.rest {
        let img = Source::parse(s)
            .context("empty source")
            .and_then(|src| prof!("cli::read (registry)", src.read()))
            .and_then(|b| prof!("cli::decode (registry)", preprocess::decode(&b)));
        let runs = match img {
            Ok(img) => models.predict(&img, &keys, &|| false),
            Err(e) => keys.iter().map(|k| models::ModelRun { head: k.clone(), result: Err(format!("{e:#}")) }).collect(),
        };
        for r in runs {
            match r.result {
                Ok(p) => {
                    if a.json {
                        rows.push(serde_json::to_value(JsonRow {
                            source: s,
                            model: r.head.as_str(),
                            label: Some(p.label()),
                            top3: p.top3.iter().map(|(l, v)| (l.as_str(), *v)).collect(),
                            unsure: p.unsure,
                            ms: (p.ms * 10.0).round() / 10.0,
                            probs: p.probs.clone(),
                            error: None,
                        })?);
                    } else {
                        let top: Vec<String> = p.top3.iter().map(|(l, v)| format!("{l} {v:.3}")).collect();
                        writeln!(
                            stdout.lock(),
                            "{s}\t{}\t{}\t{:.3}\ttop3: {}\t{:.0} ms",
                            r.head,
                            p.label(),
                            p.top3[0].1,
                            top.join(", "),
                            p.ms
                        )?;
                    }
                }
                Err(e) => {
                    failures = failures.saturating_add(1);
                    if a.json {
                        rows.push(serde_json::json!({"source": s, "model": r.head.as_str(), "label": null, "top3": [],
                                                     "unsure": false, "ms": 0.0, "probs": [], "error": e}));
                    } else {
                        writeln!(stdout.lock(), "{s}\t{}\tERROR\t{e}", r.head)?;
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
    Ok(0)
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
