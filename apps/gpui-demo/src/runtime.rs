//! onnxruntime: finding the library, the one environment per process, and one session per graph with that graph's
//! execution providers registered in the manifest's order of preference (or `--ep`'s). A provider that does not
//! register, or whose session cannot be built, is skipped with its reason and the next one is tried. What the app
//! reports is the `Placement` that resulted, never the request.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use ort::ep::ExecutionProviderDispatch;
use ort::ep::ArbitrarilyConfigurableExecutionProvider as _;
use ort::session::{Session, builder::GraphOptimizationLevel, builder::SessionBuilder};
use ort::value::TensorRef;

use crate::manifest::{Ep, EpKind, Manifest, OptLevel};
use crate::registry::search_bases;

/// Options shared by the GUI and the headless CLI.
#[derive(Debug, Clone, Default)]
pub struct ModelOptions {
    /// `--models`: the index (exports/models.json) or its folder
    pub models: Option<PathBuf>,
    pub ort_lib: Option<PathBuf>,
    pub threads: Option<usize>,
    /// `--ep`: these providers, in this order, for every graph instead of each manifest's list
    pub eps: Option<Vec<EpKind>>,
}

/// `<dir>/.venv/lib/python3.*/site-packages/onnxruntime/capi/libonnxruntime.so*`
fn venv_ort_lib(base: &Path) -> Option<PathBuf> {
    let lib = base.join(".venv").join("lib");
    let mut pys: Vec<PathBuf> = std::fs::read_dir(&lib).ok()?.flatten().map(|e| e.path()).collect();
    pys.sort();
    for py in pys.iter().rev() {
        let capi = py.join("site-packages").join("onnxruntime").join("capi");
        if let Some(p) = find_ort_in(&capi) {
            return Some(p);
        }
    }
    None
}

fn find_ort_in(dir: &Path) -> Option<PathBuf> {
    let mut libs: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("libonnxruntime.so") || n == "libonnxruntime.dylib" || n == "onnxruntime.dll")
                .unwrap_or(false)
        })
        .collect();
    libs.sort();
    libs.pop()
}

/// Resolve libonnxruntime: `--ort-lib`, `$ORT_DYLIB_PATH`, the project's uv venv
/// (`<repo>/.venv/.../onnxruntime/capi/`), a copy next to the executable, else the system loader path.
pub fn resolve_ort_lib(explicit: Option<&Path>, exports: Option<&Path>) -> PathBuf {
    if let Some(p) = explicit {
        return p.to_path_buf();
    }
    if let Some(p) = std::env::var_os("ORT_DYLIB_PATH").filter(|s| !s.is_empty()) {
        return PathBuf::from(p);
    }
    let mut bases: Vec<PathBuf> = Vec::new();
    if let Some(m) = exports {
        let m = m.canonicalize().unwrap_or_else(|_| m.to_path_buf());
        bases.extend(m.ancestors().map(Path::to_path_buf));
    }
    bases.extend(search_bases());
    for b in &bases {
        if let Some(p) = venv_ort_lib(b) {
            return p;
        }
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(p) = exe.parent().and_then(find_ort_in)
    {
        return p;
    }
    PathBuf::from(format!("{}onnxruntime{}", std::env::consts::DLL_PREFIX, std::env::consts::DLL_SUFFIX))
}

/// The execution providers a session ended up with, and the preferred ones that were skipped and why.
#[derive(Clone, Debug, PartialEq)]
pub struct Placement {
    /// registered in this order; the last is always CPU, which runs whatever the others do not take
    pub registered: Vec<EpKind>,
    pub skipped: Vec<(EpKind, String)>,
}

impl Placement {
    /// The provider the graph was meant for: the first registered one.
    pub fn primary(&self) -> EpKind {
        self.registered.first().copied().unwrap_or(EpKind::Cpu)
    }
}

impl fmt::Display for Placement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.registered.iter().map(|k| k.name()).collect();
        f.write_str(&names.join(" + "))
    }
}

/// The process-wide onnxruntime environment (ort allows exactly one per process) plus session options.
pub struct Runtime {
    env: ort::environment::Environment,
    pub ort_lib: PathBuf,
    /// e.g. "1.31.1"
    pub version: String,
    threads: Option<usize>,
    eps: Option<Vec<EpKind>>,
}

impl Runtime {
    /// dlopen libonnxruntime (see [`resolve_ort_lib`]) and create the environment. No provider is registered on
    /// the environment: each session gets its own list.
    pub fn init(opts: &ModelOptions, exports: Option<&Path>) -> Result<Self> {
        let ort_lib = resolve_ort_lib(opts.ort_lib.as_deref(), exports);
        let env = ort::init_from(&ort_lib)
            .map_err(|e| {
                anyhow!(
                    "cannot load the onnxruntime library {}: {e}\n  install one with `uv pip install onnxruntime` \
                     and pass --ort-lib .../onnxruntime/capi/libonnxruntime.so.<version> (or set ORT_DYLIB_PATH); \
                     see apps/gpui-demo/README.md",
                    ort_lib.display()
                )
            })?
            .with_name("gpui-demo")
            .build()
            .map_err(|e| anyhow!("creating onnxruntime environment: {e}"))?;
        // the build info string has no version; the library's file name does (libonnxruntime.so.1.31.1)
        let real = ort_lib.canonicalize().unwrap_or_else(|_| ort_lib.clone());
        let name = real.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let version = name.split_once(".so.").map_or(name.as_str(), |(_, v)| v).to_string();
        Ok(Self { env, ort_lib, version, threads: opts.threads, eps: opts.eps.clone() })
    }

    /// The providers to try for `m`, in order: the manifest's list, or `--ep`'s (taking the manifest's options
    /// for a provider it lists).
    pub fn preferences(&self, m: &Manifest) -> Vec<Ep> {
        match &self.eps {
            None => m.runtime.providers.clone(),
            Some(kinds) => kinds
                .iter()
                .map(|&kind| {
                    m.runtime.providers.iter().find(|e| e.kind == kind).cloned().unwrap_or(Ep { kind, options: vec![] })
                })
                .collect(),
        }
    }

    fn builder(&self, m: &Manifest) -> Result<SessionBuilder> {
        let level = match m.runtime.opt_level {
            OptLevel::Disable => GraphOptimizationLevel::Disable,
            OptLevel::Basic => GraphOptimizationLevel::Level1,
            OptLevel::Extended => GraphOptimizationLevel::Level2,
            OptLevel::Layout => GraphOptimizationLevel::Level3,
            OptLevel::All => GraphOptimizationLevel::All,
        };
        let mut b = Session::builder(&self.env)
            .map_err(|e| anyhow!("{e}"))?
            .with_optimization_level(level)
            .map_err(|e| anyhow!("{e}"))?;
        if let Some(n) = self.threads {
            b = b.with_intra_threads(n).map_err(|e| anyhow!("{e}"))?;
        }
        Ok(b)
    }

    /// Load `m`'s graph with the first providers that work. A session that fails with a GPU provider is retried
    /// without it, so a provider that registers but cannot build its part of the graph (an unsupported GPU, an
    /// engine that does not fit) still ends up as a visible skip rather than a dead model.
    pub fn session(&self, m: &Manifest) -> Result<(Session, Placement)> {
        let mut candidates = self.preferences(m);
        let mut skipped: Vec<(EpKind, String)> = Vec::new();
        loop {
            let mut b = self.builder(m)?;
            let mut registered = Vec::new();
            for ep in candidates.iter().filter(|e| e.kind != EpKind::Cpu) {
                match dispatch(ep).register(&mut b) {
                    Ok(()) => registered.push(ep.kind),
                    Err(e) => skipped.push((ep.kind, reason(&e.to_string()))),
                }
            }
            if registered.is_empty() && !candidates.iter().any(|e| e.kind == EpKind::Cpu) {
                let tried: Vec<String> = skipped.iter().map(|(k, e)| format!("{}: {e}", k.name())).collect();
                bail!("{}: none of its execution providers is available ({})", m.graph.display(), tried.join("; "));
            }
            match b.commit_from_file(&m.graph) {
                Ok(session) => {
                    registered.push(EpKind::Cpu);
                    return Ok((session, Placement { registered, skipped }));
                }
                Err(e) if !registered.is_empty() => {
                    let first = registered[0];
                    skipped.push((first, format!("session failed: {}", reason(&e.to_string()))));
                    candidates.retain(|c| c.kind != first);
                }
                Err(e) => {
                    bail!("loading {}: {e} (are its external weights next to it and complete?)", m.graph.display())
                }
            }
        }
    }

    /// Would `ep` register for a session? (Loads the provider library; builds no session.)
    pub fn probe(&self, ep: &Ep, m: &Manifest) -> Result<(), String> {
        if ep.kind == EpKind::Cpu {
            return Ok(());
        }
        let mut b = self.builder(m).map_err(|e| e.to_string())?;
        dispatch(ep).register(&mut b).map_err(|e| reason(&e.to_string()))
    }
}

/// The cause in an onnxruntime error, without its source location ("... with error: libnvinfer.so.10: cannot
/// open shared object file" becomes the part after "with error: ").
fn reason(e: &str) -> String {
    let e = e.trim();
    let tail = e.rsplit_once("with error: ").map_or(e, |(_, t)| t);
    let tail = tail.rsplit_once("] : ").map_or(tail, |(_, t)| t);
    tail.trim().to_string()
}

/// The ort provider for one preference, with the manifest's options passed through unchanged.
fn dispatch(ep: &Ep) -> ExecutionProviderDispatch {
    match ep.kind {
        EpKind::TensorRt => {
            let mut p = ort::ep::TensorRT::default();
            for (k, v) in &ep.options {
                p = p.with_arbitrary_config(k, v);
            }
            p.build()
        }
        EpKind::Cuda => {
            let mut p = ort::ep::CUDA::default();
            for (k, v) in &ep.options {
                p = p.with_arbitrary_config(k, v);
            }
            p.build()
        }
        EpKind::Cpu => ort::ep::CPU::default().build(),
    }
    .error_on_failure()
}

/// One loaded ONNX graph; serves every head of its manifest (one per output).
pub struct Engine {
    session: Session,
    pub manifest: Arc<Manifest>,
    pub placement: Placement,
}

impl Engine {
    pub fn load(rt: &Runtime, manifest: Arc<Manifest>) -> Result<Self> {
        let (session, placement) = rt.session(&manifest)?;
        let name = &manifest.input.name;
        if !session.inputs().iter().any(|i| i.name() == name) {
            bail!("{} has no input '{name}'", manifest.graph.display());
        }
        Ok(Self { session, manifest, placement })
    }

    pub fn output_names(&self) -> Vec<String> {
        self.session.outputs().iter().map(|o| o.name().to_string()).collect()
    }

    /// One forward pass over `n` images (`x` is `[n, 3, H, W]` float32); returns the requested outputs, each
    /// flattened `[n, ...]`, in the requested order.
    pub fn run(&mut self, x: &[f32], n: usize, outputs: &[&str]) -> Result<Vec<Vec<f32>>> {
        let (w, h) = self.manifest.input.size;
        let input = TensorRef::from_array_view(([n, 3, h as usize, w as usize], x)).map_err(|e| anyhow!("{e}"))?;
        let onnx = &self.manifest.graph;
        let res = self
            .session
            .run(ort::inputs![self.manifest.input.name.as_str() => input])
            .map_err(|e| anyhow!("inference ({}): {e}", onnx.display()))?;
        outputs
            .iter()
            .map(|name| {
                let v = res.get(name).with_context(|| format!("{} has no output '{name}'", onnx.display()))?;
                let (_, data) = v.try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
                Ok(data.to_vec())
            })
            .collect()
    }
}
