//! gpui-demo: desktop demo for the 626-Project electronic-component classifier.
//!
//!   gpui-demo [OPTIONS] [IMAGE|URL ...]               open the window (optionally classifying the given images)
//!   gpui-demo classify [OPTIONS] [--json] IMAGE|URL ...  headless: print predictions, no window
//!   gpui-demo preprocess IMAGE OUT.f32 [--model-dir D]   dump the [1,3,S,S] float32 input tensor (parity tests)
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
mod model;
mod models;
mod preprocess;

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result, bail};

use crate::contract::Contract;
use crate::fetch::Source;
use crate::model::{Classifier, ModelOptions, resolve_model_dir};
use crate::models::{ModelKey, Models};

const HELP: &str = "\
gpui-demo - classify electronic components in pictures (GPUI + onnxruntime)

USAGE:
    gpui-demo [OPTIONS] [IMAGE|URL ...]                 open the window
    gpui-demo classify [OPTIONS] [--json] IMAGE|URL ... headless; one line (or JSON object) per image
    gpui-demo preprocess IMAGE OUT.f32 [--model-dir DIR] [--model KEY]
                                                        write the model input tensor (little-endian f32)
    gpui-demo models [--model-dir DIR]                  list the four models and whether each is available

OPTIONS:
    --model-dir DIR   folder with model.onnx(+.data), preprocess.json, labels.json
                      (default: $COMPCLASS_MODEL_DIR, else the first exports/final/ found walking up
                      from the current directory, then from the executable)
    --ort-lib PATH    libonnxruntime.so to load (default: $ORT_DYLIB_PATH, else
                      <repo>/.venv/.../onnxruntime/capi/libonnxruntime.so*, else next to the
                      executable, else the system library path)
    --threads N       onnxruntime intra-op threads (default: onnxruntime's choice)
    --cuda            request the CUDA execution provider (build with --features cuda)
    --json            classify: print a JSON array instead of text lines
    --model KEYS      classify with these registry models instead of the deployed model.onnx:
                      a, b, c, deployed (comma-separated) or all. See `gpui-demo models`.
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
    models: Vec<ModelKey>,
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
            "--model-dir" => a.opts.model_dir = Some(PathBuf::from(value("--model-dir")?)),
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
                let v = value("--model")?;
                for k in v.split(',').filter(|k| !k.trim().is_empty()) {
                    if k.trim() == "all" {
                        a.models.extend(ModelKey::ALL);
                    } else {
                        a.models.push(
                            ModelKey::parse(k)
                                .with_context(|| format!("unknown model {k:?} (a, b, c, deployed, all)"))?,
                        );
                    }
                }
                a.models.dedup();
            }
            s if s.starts_with("--") => bail!("unknown option {s} (see --help)"),
            _ => a.rest.push(arg),
        }
    }
    Ok(a)
}

#[derive(serde::Serialize)]
struct JsonRow<'a> {
    source: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    label: Option<&'a str>,
    top3: Vec<(&'a str, f32)>,
    unsure: bool,
    ms: f64,
    probs: Vec<f32>,
    error: Option<String>,
}

fn run_classify(a: &Args) -> Result<u8> {
    if a.rest.is_empty() {
        bail!("classify needs at least one image path or URL");
    }
    let mut clf = Classifier::load(&a.opts)?;
    eprintln!(
        "[gpui-demo] model {} loaded in {:.0} ms (onnxruntime {}, {}; jpeg decoder: {})",
        clf.model_dir.join("model.onnx").display(),
        clf.load_ms,
        clf.ort_lib.display(),
        clf.provider,
        preprocess::JPEG_DECODER
    );
    let mut failures = 0u8;
    let mut rows = Vec::new();
    let stdout = std::io::stdout();
    for s in &a.rest {
        let result = Source::parse(s)
            .context("empty source")
            .and_then(|src| prof!("cli::read", src.read()))
            .and_then(|bytes| prof!("cli::decode", preprocess::decode(&bytes)))
            .and_then(|img| clf.predict(&img));
        match result {
            Ok(p) => {
                if a.json {
                    rows.push(serde_json::to_value(JsonRow {
                        source: s,
                        model: None,
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
                        "{s}\t{}\t{:.3}\ttop3: {}\t{:.0} ms",
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
                    rows.push(serde_json::json!({"source": s, "label": null, "top3": [], "unsure": false,
                                                 "ms": 0.0, "probs": [], "error": format!("{e:#}")}));
                } else {
                    writeln!(stdout.lock(), "{s}\tERROR\t{e:#}")?;
                }
            }
        }
    }
    if a.json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
    }
    Ok(if failures > 0 { 2 } else { 0 })
}

/// `classify --model ...`: the registry path (lazy engines; one forward pass per graph and image).
fn run_classify_models(a: &Args) -> Result<u8> {
    if a.rest.is_empty() {
        bail!("classify needs at least one image path or URL");
    }
    let mut models = Models::new(&a.opts)?;
    for &k in &a.models {
        match models.ensure(k) {
            Ok(ms) => eprintln!("[gpui-demo] model {} ({}) ready in {ms:.0} ms", k.id(), k.name()),
            Err(e) => eprintln!("[gpui-demo] model {} ({}) unavailable: {e}", k.id(), k.name()),
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
            Ok(img) => models.predict(&img, &a.models, &|| false),
            Err(e) => a.models.iter().map(|&key| models::ModelRun { key, result: Err(format!("{e:#}")) }).collect(),
        };
        for r in runs {
            match r.result {
                Ok(p) => {
                    if a.json {
                        rows.push(serde_json::to_value(JsonRow {
                            source: s,
                            model: Some(r.key.id()),
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
                            r.key.id(),
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
                        rows.push(serde_json::json!({"source": s, "model": r.key.id(), "label": null, "top3": [],
                                                     "unsure": false, "ms": 0.0, "probs": [], "error": e}));
                    } else {
                        writeln!(stdout.lock(), "{s}\t{}\tERROR\t{e}", r.key.id())?;
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
    let models = Models::new(&a.opts)?;
    for (k, spec) in &models.entries {
        match spec {
            Ok(s) => println!("{:9} {:30} available  {} [{}]", k.id(), k.name(), s.onnx.display(), s.readout_name()),
            Err(e) => println!("{:9} {:30} unavailable  {e}", k.id(), k.name()),
        }
    }
    Ok(0)
}

fn run_preprocess(a: &Args) -> Result<u8> {
    let [img, out] = a.rest.as_slice() else { bail!("usage: gpui-demo preprocess IMAGE OUT.f32 [--model KEY]") };
    let dir = match a.models.as_slice() {
        [] => resolve_model_dir(a.opts.model_dir.as_deref())?,
        [k] => Models::new(&a.opts)?.spec(*k).map_err(|e| anyhow::anyhow!("{e}"))?.contract_dir.clone(),
        _ => bail!("preprocess takes one --model"),
    };
    let c = Contract::load(&dir)?;
    let t0 = Instant::now();
    let bytes = Source::parse(img).context("empty source")?.read()?;
    let rgb = preprocess::decode(&bytes)?;
    let t = preprocess::to_tensor(&rgb, &c.pre)?;
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
        "classify" if !args.models.is_empty() => run_classify_models(&args),
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
