//! gpui-demo: desktop demo for the 626-Project electronic-component classifier.
//!
//!   gpui-demo [OPTIONS] [IMAGE|URL ...]               open the window (optionally classifying the given images)
//!   gpui-demo classify [OPTIONS] [--json] IMAGE|URL ...  headless: print predictions, no window
//!   gpui-demo preprocess IMAGE OUT.f32 [--model-dir D]   dump the [1,3,S,S] float32 input tensor (parity tests)

mod contract;
mod fetch;
mod gui;
mod model;
mod preprocess;

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result, bail};

use crate::contract::Contract;
use crate::fetch::Source;
use crate::model::{Classifier, ModelOptions, resolve_model_dir};

const HELP: &str = "\
gpui-demo - classify electronic components in pictures (GPUI + onnxruntime)

USAGE:
    gpui-demo [OPTIONS] [IMAGE|URL ...]                 open the window
    gpui-demo classify [OPTIONS] [--json] IMAGE|URL ... headless; one line (or JSON object) per image
    gpui-demo preprocess IMAGE OUT.f32 [--model-dir DIR] write the model input tensor (little-endian f32)

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
    -h, --help        this help
    -V, --version     version

KEYS (window): Ctrl+O open, Ctrl+V paste URL / path / image, Ctrl+L focus URL field,
               Enter fetch, Ctrl+T light/dark theme, Up/Down recent images, Ctrl+Q quit";

#[derive(Debug, Default)]
struct Args {
    mode: String,
    opts: ModelOptions,
    json: bool,
    rest: Vec<String>,
}

fn parse_args() -> Result<Args> {
    let mut it = std::env::args().skip(1).peekable();
    let mut a = Args { mode: "gui".into(), ..Default::default() };
    if let Some(first) = it.peek()
        && (first == "classify" || first == "preprocess")
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
            s if s.starts_with("--") => bail!("unknown option {s} (see --help)"),
            _ => a.rest.push(arg),
        }
    }
    Ok(a)
}

#[derive(serde::Serialize)]
struct JsonRow<'a> {
    source: &'a str,
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
            .and_then(|src| src.read())
            .and_then(|bytes| preprocess::decode(&bytes))
            .and_then(|img| clf.predict(&img));
        match result {
            Ok(p) => {
                if a.json {
                    rows.push(serde_json::to_value(JsonRow {
                        source: s,
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

fn run_preprocess(a: &Args) -> Result<u8> {
    let [img, out] = a.rest.as_slice() else { bail!("usage: gpui-demo preprocess IMAGE OUT.f32") };
    let dir = resolve_model_dir(a.opts.model_dir.as_deref())?;
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
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("gpui-demo: {e:#}");
            return ExitCode::from(64);
        }
    };
    let r = match args.mode.as_str() {
        "classify" => run_classify(&args),
        "preprocess" => run_preprocess(&args),
        _ => {
            gui::run(args.opts, args.rest);
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
