//! The command line: its options, `--help`, and what the subcommands share (the index, the service with its log
//! lines, the start-up default, one request's results). A subcommand is a module with a [`Command`], listed in
//! `app::COMMANDS`.

pub mod bench;
pub mod classify;
pub mod models;
pub mod preprocess;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use inference::manifest::{EpKind, GraphId, HeadId};
use inference::registry::Registry;
use inference::runtime::{OrtLoader, Settings};
use inference::service::{
    Finish, GraphState, HeadOutput, Inference, LoadStage, Loaded, RequestHandle, RunError, ServiceEvent, Timing, Update,
};

use crate::{app, gui};

/// A subcommand: `<bin> NAME ARGS`.
pub struct Command {
    pub name: &'static str,
    /// its arguments, for `--help`
    pub args: &'static str,
    /// what it does, for `--help`; each `\n` starts a line
    pub about: &'static str,
    pub run: fn(&Args) -> Result<u8>,
}

/// Where `--help` starts what each command does.
const ABOUT_COLUMN: usize = 56;

/// One `--help` usage entry: `left`, then `about` from `ABOUT_COLUMN` on (under `left` when it is too long).
fn usage(left: String, about: &str) -> String {
    let w = ABOUT_COLUMN;
    let mut lines = about.lines();
    let first = lines.next().unwrap_or_default();
    let mut out = if left.len() < w { format!("{left:<w$}{first}") } else { format!("{left}\n{:w$}{first}", "") };
    for line in lines {
        out += &format!("\n{:w$}{line}", "");
    }
    out
}

/// `--help`.
pub fn help() -> String {
    let (bin, title, about, env) = (app::BIN, app::TITLE, app::ABOUT, app::ENV_PREFIX);
    let titles: Vec<String> = app::TABS.iter().map(|t| t.title.to_lowercase()).collect();
    let tabs = match titles.split_last() {
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} or {last}", rest.join(", ")),
        None => String::new(),
    };
    let first = titles.first().cloned().unwrap_or_default();
    let keys = gui::keys_help();
    let mut lines = vec![usage(format!("    {bin} [OPTIONS] [IMAGE|URL ...]"), "open the window")];
    lines.extend(app::COMMANDS.iter().map(|c| usage(format!("    {bin} {} {}", c.name, c.args), c.about)));
    let usage = lines.join("\n");
    format!(
        "\
{bin} - {title}: {about} (GPUI + onnxruntime)

USAGE:
{usage}

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
                      `{bin} models` for the ids.
    --demo-dir DIR    window: folder with demo_stack.json (default: ${env}_DEMO_DIR, else the
                      first demo_stack/ found walking up from the current directory, then the executable)
    --tab TAB         window: start on tab TAB, one of {tabs} (default: {first})
    --autorun         window: start the tab's run (its Ctrl+R) once the default model is known
    --dark, --light   window: initial theme (default: follow the system)
    -h, --help        this help
    -V, --version     version

KEYS (window): {keys}"
    )
}

#[derive(Default)]
pub struct Args {
    /// None opens the window
    pub command: Option<&'static Command>,
    pub opts: Settings,
    pub json: bool,
    pub check: bool,
    pub rounds: usize,
    pub warmup: usize,
    pub models: Vec<String>,
    pub gui: gui::GuiOptions,
    pub rest: Vec<String>,
}

pub fn parse_args() -> Result<Args> {
    let mut it = std::env::args().skip(1).peekable();
    let mut a = Args { rounds: 5, warmup: 5, ..Default::default() };
    if let Some(command) = it.peek().and_then(|first| app::COMMANDS.iter().find(|c| c.name == first)) {
        a.command = Some(command);
        it.next();
    }
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().with_context(|| format!("{name} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{}", help());
                std::process::exit(0);
            }
            "-V" | "--version" => {
                let jpeg = inference::preprocess::JPEG_DECODER;
                println!("{} {} (jpeg decoder: {jpeg})", app::BIN, env!("CARGO_PKG_VERSION"));
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
            "--rounds" => a.rounds = value("--rounds")?.parse().context("--rounds")?,
            "--warmup" => a.warmup = value("--warmup")?.parse().context("--warmup")?,
            "--demo-dir" => a.gui.demo_dir = Some(PathBuf::from(value("--demo-dir")?)),
            "--tab" => {
                let name = value("--tab")?;
                let Some(ix) = gui::tab_index(&name) else {
                    let titles: Vec<&str> = app::TABS.iter().map(|t| t.title).collect();
                    bail!("--tab {name}: expected one of {}", titles.join(", "));
                };
                a.gui.tab = ix;
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
    let reg = Registry::open(&a.opts)?;
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
    let (inf, mut events) = Inference::spawn(reg.clone(), OrtLoader::new(&a.opts, reg));
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

/// Release every session before the process exits with `code` (see `Inference::shutdown`); if a load is still
/// running, leave with `code` without exit handlers rather than race its provider's teardown.
fn end(inf: &Inference, code: u8) -> u8 {
    inf.shutdown_or_exit(std::time::Duration::from_secs(30), code.into(), || {
        eprintln!("[gpui-demo] a model is still loading; exiting without waiting for it")
    });
    code
}
