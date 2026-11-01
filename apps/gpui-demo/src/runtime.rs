//! onnxruntime: finding the library, the one environment per process, and one session per graph with that graph's
//! execution providers registered in the manifest's order of preference (or `--ep`'s). A provider that does not
//! register, or whose session cannot be built, is skipped with its reason and the next one is tried. What the app
//! reports is the `Placement` that resulted, never the request.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use ort::ep::ExecutionProviderDispatch;
use sha2::Digest as _;
use ort::ep::ArbitrarilyConfigurableExecutionProvider as _;
use ort::session::{RunOptions, Session, builder::GraphOptimizationLevel, builder::SessionBuilder};
use ort::value::TensorRef;

use crate::inference::{Graph, LoadStage, Loaded, Loader, RunFailure};
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
    pub fn session(&self, m: &Manifest, stage: &dyn Fn(LoadStage)) -> Result<(Session, Placement)> {
        let mut candidates = self.preferences(m);
        let mut skipped: Vec<(EpKind, String)> = Vec::new();
        loop {
            let mut b = self.builder(m)?;
            let mut registered = Vec::new();
            let mut failed = Vec::new();
            for ep in candidates.iter().filter(|e| e.kind != EpKind::Cpu) {
                match self.provider(ep, m).register(&mut b) {
                    Ok(()) => registered.push(ep.kind),
                    Err(e) => {
                        skipped.push((ep.kind, reason(&e.to_string())));
                        failed.push(ep.kind);
                    }
                }
            }
            candidates.retain(|c| !failed.contains(&c.kind));
            if registered.is_empty() && !candidates.iter().any(|e| e.kind == EpKind::Cpu) {
                let tried: Vec<String> = skipped.iter().map(|(k, e)| format!("{}: {e}", k.name())).collect();
                bail!("{}: none of its execution providers is available ({})", m.graph.display(), tried.join("; "));
            }
            let building = registered.first() == Some(&EpKind::TensorRt)
                && candidates.iter().find(|e| e.kind == EpKind::TensorRt).is_some_and(|ep| !self.engine_cached(ep, m));
            stage(if building { LoadStage::BuildingEngine } else { LoadStage::Loading });
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
        self.provider(ep, m).register(&mut b).map_err(|e| reason(&e.to_string()))
    }

    /// The ort provider for one preference: the manifest's options passed through unchanged, plus, for TensorRT
    /// with its engine cache on, the cache paths (the app's, see `engine_cache`).
    fn provider(&self, ep: &Ep, m: &Manifest) -> ExecutionProviderDispatch {
        match ep.kind {
            EpKind::TensorRt => {
                let mut p = ort::ep::TensorRT::default();
                for (k, v) in &ep.options {
                    p = p.with_arbitrary_config(k, v);
                }
                if ep.flag("trt_engine_cache_enable")
                    && let Some(dir) = self.engine_cache(m)
                    && std::fs::create_dir_all(&dir).is_ok()
                {
                    p = p.with_engine_cache_path(dir.display());
                    if ep.flag("trt_timing_cache_enable") {
                        p = p.with_timing_cache_path(dir.display());
                    }
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

    /// A graph's TensorRT engines live in a per-user directory named by what an engine depends on: the graph's
    /// files, the onnxruntime and TensorRT versions and the GPU. A new graph, library or GPU never meets a stale
    /// engine; it builds its own once. None when no libnvinfer loads, since the provider cannot load either.
    pub fn engine_cache(&self, m: &Manifest) -> Option<PathBuf> {
        let trt = tensorrt_version()?;
        let base = crate::integrity::cache_dir()?.join("tensorrt");
        let mut graph = sha2::Sha256::new();
        for f in m.files.iter().filter(|f| f.path == m.graph || m.external.contains(&f.path)) {
            graph.update(f.sha256.0);
        }
        let graph = crate::manifest::Sha256(graph.finalize().into()).short();
        Some(base.join(format!("{}-{graph}-ort{}-trt{trt}-{}", m.id.as_str(), self.version, gpu_name())))
    }

    /// An engine for `m` is already built (the next session loads it instead of building one).
    fn engine_cached(&self, ep: &Ep, m: &Manifest) -> bool {
        ep.flag("trt_engine_cache_enable")
            && self.engine_cache(m).and_then(|d| std::fs::read_dir(d).ok()).is_some_and(|mut it| {
                it.any(|e| e.is_ok_and(|e| e.path().extension().is_some_and(|x| x == "engine")))
            })
    }
}

/// End the process without running exit handlers: for when a provider is still busy in another thread (an engine
/// build cannot be interrupted) and its process-wide teardown would race it.
pub fn exit_now(code: i32) -> ! {
    #[cfg(unix)]
    {
        unsafe extern "C" {
            fn _exit(code: std::ffi::c_int) -> !;
        }
        // SAFETY: _exit only ends the process; stdout was flushed by the caller.
        unsafe { _exit(code) }
    }
    #[cfg(not(unix))]
    std::process::exit(code)
}

/// "10.16.1" from the libnvinfer.so.10 the dynamic linker finds (the one the TensorRT provider will load).
fn tensorrt_version() -> Option<String> {
    #[cfg(unix)]
    {
        use std::ffi::{c_char, c_int, c_void};
        unsafe extern "C" {
            fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
            fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        }
        const RTLD_NOW: c_int = 2;
        // SAFETY: dlopen/dlsym with NUL-terminated names; the library stays loaded (the provider loads it anyway),
        // and getInferLibVersion is `int32_t getInferLibVersion(void)` in NvInferRuntime.h.
        unsafe {
            let lib = dlopen(c"libnvinfer.so.10".as_ptr(), RTLD_NOW);
            if lib.is_null() {
                return None;
            }
            let f = dlsym(lib, c"getInferLibVersion".as_ptr());
            if f.is_null() {
                return None;
            }
            let f: extern "C" fn() -> i32 = std::mem::transmute(f);
            let v = f();
            Some(format!("{}.{}.{}", v / 10000, v / 100 % 100, v % 100))
        }
    }
    #[cfg(not(unix))]
    None
}

/// The first NVIDIA GPU's model as a file-name slug ("nvidia-geforce-rtx-3060-laptop-gpu"), from the driver's
/// /proc entry; "gpu" when there is none.
fn gpu_name() -> String {
    let model = std::fs::read_dir("/proc/driver/nvidia/gpus")
        .ok()
        .and_then(|mut d| d.next())
        .and_then(|e| e.ok())
        .and_then(|e| std::fs::read_to_string(e.path().join("information")).ok())
        .and_then(|t| t.lines().find_map(|l| l.strip_prefix("Model:").map(|m| m.trim().to_string())));
    let slug: String = model
        .unwrap_or_else(|| "gpu".into())
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    slug.split('-').filter(|s| !s.is_empty()).collect::<Vec<_>>().join("-")
}

/// The cause in an onnxruntime error, without its source location ("... with error: libnvinfer.so.10: cannot
/// open shared object file" becomes the part after "with error: "; "/onnxruntime_src/...cc:2465
/// onnxruntime::...GetProviderInfo_CUDA() CUDA Provider not available" the part after the function).
fn reason(e: &str) -> String {
    let e = e.trim();
    let tail = e.rsplit_once("with error: ").map_or(e, |(_, t)| t);
    let tail = tail.rsplit_once("] : ").map_or(tail, |(_, t)| t);
    let tail = match tail.rsplit_once("() ") {
        Some((location, cause)) if location.contains(".cc:") || location.contains(".h:") => cause,
        _ => tail,
    };
    tail.trim().trim_end_matches('.').to_string()
}

/// One loaded ONNX graph; serves every head of its manifest (one per output).
pub struct Engine {
    session: Session,
    pub manifest: Arc<Manifest>,
    pub placement: Placement,
}

impl Engine {
    pub fn load(rt: &Runtime, manifest: Arc<Manifest>, stage: &dyn Fn(LoadStage)) -> Result<Self> {
        let (session, placement) = rt.session(&manifest, stage)?;
        check_io(&session, &manifest)?;
        Ok(Self { session, manifest, placement })
    }
}

/// Load-time rules: the input has the manifest's name, float32 and [N, 3, H, W] (N dynamic or at most
/// batch.max); every readout output exists, is float32 and its last dimension (when fixed) is the label count.
/// After this nothing checks per prediction.
fn check_io(session: &Session, m: &Manifest) -> Result<()> {
    let graph = m.graph.display();
    let input = session
        .inputs()
        .iter()
        .find(|i| i.name() == m.input.name)
        .with_context(|| format!("{graph} has no input '{}'", m.input.name))?;
    if input.dtype().tensor_type() != Some(ort::value::TensorElementType::Float32) {
        bail!("{graph}: input '{}' is {:?}, the app feeds float32", m.input.name, input.dtype());
    }
    let (w, h) = m.input.size;
    if let Some(shape) = input.dtype().tensor_shape() {
        let want = [i64::from(m.input.max_batch), 3, i64::from(h), i64::from(w)];
        let ok = shape.len() == 4
            && (shape[0] < 0 || shape[0] >= want[0])
            && shape[1..].iter().zip(&want[1..]).all(|(&d, &x)| d < 0 || d == x);
        if !ok {
            bail!("{graph}: input shape {:?} does not take [{}, 3, {h}, {w}]", &shape[..], m.input.max_batch);
        }
    }
    for head in &m.heads {
        let crate::manifest::Task::Classify(t) = &head.task;
        let name = t.readout.output();
        let out = session
            .outputs()
            .iter()
            .find(|o| o.name() == name)
            .with_context(|| format!("{graph} has no output '{name}' (head {})", head.id))?;
        if out.dtype().tensor_type() != Some(ort::value::TensorElementType::Float32) {
            bail!("{graph}: output '{name}' is {:?}, the app reads float32", out.dtype());
        }
        if let Some(&last) = out.dtype().tensor_shape().and_then(|s| s.last())
            && last >= 0
            && last as usize != t.labels.len()
        {
            bail!("{graph}: output '{name}' has {last} values for {} labels", t.labels.len());
        }
    }
    Ok(())
}

impl Graph for Engine {
    fn run(
        &mut self,
        x: &[f32],
        n: usize,
        outputs: &[&str],
        alive: &(dyn Fn() -> bool + Sync),
    ) -> std::result::Result<Vec<Vec<f32>>, RunFailure> {
        let err = |e: String| RunFailure::Error(e);
        let (w, h) = self.manifest.input.size;
        let input = TensorRef::from_array_view(([n, 3, h as usize, w as usize], x)).map_err(|e| err(e.to_string()))?;
        let opts = RunOptions::new().map_err(|e| err(e.to_string()))?;
        let finished = AtomicBool::new(false);
        let stopped = AtomicBool::new(false);
        let onnx = &self.manifest.graph;
        let session = &mut self.session;
        let name = self.manifest.input.name.as_str();
        // a watcher stops the run once nobody waits for it (RunOptions::terminate); polled every 5 ms, and woken when
        // the run ends: the scope joins it, and a plain sleep made every run wait out the rest of its 5 ms
        std::thread::scope(|scope| {
            let watcher = scope.spawn(|| {
                while !finished.load(Ordering::Acquire) {
                    if !alive() {
                        stopped.store(true, Ordering::Release);
                        let _ = opts.terminate();
                        return;
                    }
                    std::thread::park_timeout(Duration::from_millis(5));
                }
            });
            let res = session.run_with_options(ort::inputs![name => input], &opts);
            finished.store(true, Ordering::Release);
            watcher.thread().unpark();
            if stopped.load(Ordering::Acquire) {
                return Err(RunFailure::Stopped);
            }
            let res = res.map_err(|e| err(format!("inference ({}): {e}", onnx.display())))?;
            outputs
                .iter()
                .map(|name| {
                    let v = res.get(name).ok_or_else(|| err(format!("{} has no output '{name}'", onnx.display())))?;
                    let (_, data) = v.try_extract_tensor::<f32>().map_err(|e| err(e.to_string()))?;
                    Ok(data.to_vec())
                })
                .collect()
        })
    }
}

/// The app's `Loader`: onnxruntime is loaded on first use (a load or a provider probe).
pub struct OrtLoader {
    opts: ModelOptions,
    exports: Option<PathBuf>,
    runtime: OnceLock<std::result::Result<Runtime, String>>,
}

impl OrtLoader {
    pub fn new(opts: &ModelOptions, exports: Option<&Path>) -> Self {
        Self { opts: opts.clone(), exports: exports.map(Path::to_path_buf), runtime: OnceLock::new() }
    }

    pub fn runtime(&self) -> std::result::Result<&Runtime, String> {
        self.runtime
            .get_or_init(|| Runtime::init(&self.opts, self.exports.as_deref()).map_err(|e| format!("{e:#}")))
            .as_ref()
            .map_err(Clone::clone)
    }
}

impl Loader for OrtLoader {
    fn load(&self, m: &Arc<Manifest>, stage: &dyn Fn(LoadStage)) -> std::result::Result<(Box<dyn Graph>, Loaded), String> {
        let t0 = Instant::now();
        let rt = self.runtime()?;
        let verified = crate::integrity::Stamps::load().all_verified(m);
        let engine = Engine::load(rt, m.clone(), stage).map_err(|e| format!("{e:#}"))?;
        let loaded = Loaded {
            ms: t0.elapsed().as_secs_f64() * 1000.0,
            placement: engine.placement.clone(),
            runtime: format!("onnxruntime {}", rt.version),
            verified,
        };
        Ok((Box::new(engine), loaded))
    }

    fn probe(&self, m: &Manifest) -> std::result::Result<(), String> {
        let rt = self.runtime()?;
        let ep = rt.preferences(m).into_iter().next().ok_or("the manifest lists no execution provider")?;
        rt.probe(&ep, m).map_err(|e| format!("{} does not register here: {e}", ep.kind.name()))
    }
}

#[cfg(test)]
mod tests {
    use super::reason;

    #[test]
    fn reasons_drop_onnxruntime_source_locations() {
        assert_eq!(
            reason(
                "/onnxruntime_src/onnxruntime/core/session/provider_bridge_ort.cc:2043 onnxruntime::Provider& \
                    onnxruntime::ProviderLibrary::Get() [ONNXRuntimeError] : 1 : FAIL : Failed to load library \
                    libonnxruntime_providers_cuda.so with error: libcuda.so.1: cannot open shared object file: No such \
                    file or directory"
            ),
            "libcuda.so.1: cannot open shared object file: No such file or directory"
        );
        assert_eq!(
            reason(
                "/onnxruntime_src/onnxruntime/core/session/provider_bridge_ort.cc:2465 \
                    onnxruntime::ProviderInfo_CUDA& onnxruntime::GetProviderInfo_CUDA() CUDA Provider not available, \
                    can't get interface for it"
            ),
            "CUDA Provider not available, can't get interface for it"
        );
        assert_eq!(
            reason("CUDA execution provider is not enabled in this build."),
            "CUDA execution provider is not enabled in this build"
        );
    }
}
