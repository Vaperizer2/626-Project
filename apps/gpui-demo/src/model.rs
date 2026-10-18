//! Model directory / onnxruntime library discovery and the ONNX classifier.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use image::RgbImage;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::TensorRef;

use crate::contract::{Contract, Prediction, decide};
use crate::preprocess::to_tensor;

/// Options shared by the GUI and the headless CLI.
#[derive(Debug, Clone, Default)]
pub struct ModelOptions {
    pub model_dir: Option<PathBuf>,
    pub ort_lib: Option<PathBuf>,
    pub threads: Option<usize>,
    pub cuda: bool,
}

/// Candidate base directories: the current directory and the executable's directory, plus their ancestors.
fn search_bases() -> Vec<PathBuf> {
    let mut starts = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        starts.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(d) = exe.canonicalize().unwrap_or(exe).parent()
    {
        starts.push(d.to_path_buf());
    }
    let mut out: Vec<PathBuf> = Vec::new();
    for s in starts {
        for a in s.ancestors() {
            if !out.iter().any(|p| p == a) {
                out.push(a.to_path_buf());
            }
        }
    }
    out
}

/// Resolve the export folder: `--model-dir`, then `$COMPCLASS_MODEL_DIR`, then the first `exports/final/`
/// containing `model.onnx` found walking up from the current directory and from the executable.
pub fn resolve_model_dir(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        if !p.join("model.onnx").is_file() {
            bail!("--model-dir {}: no model.onnx there", p.display());
        }
        return Ok(p.to_path_buf());
    }
    if let Some(p) = std::env::var_os("COMPCLASS_MODEL_DIR").map(PathBuf::from) {
        if p.join("model.onnx").is_file() {
            return Ok(p);
        }
        bail!("COMPCLASS_MODEL_DIR={}: no model.onnx there", p.display());
    }
    for b in search_bases() {
        let c = b.join("exports").join("final");
        if c.join("model.onnx").is_file() {
            return Ok(c);
        }
    }
    bail!(
        "cannot find exports/final/model.onnx above the current directory or the executable; \
         pass --model-dir <folder with model.onnx, model.onnx.data, preprocess.json, labels.json>"
    )
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
pub fn resolve_ort_lib(explicit: Option<&Path>, model_dir: Option<&Path>) -> PathBuf {
    if let Some(p) = explicit {
        return p.to_path_buf();
    }
    if let Some(p) = std::env::var_os("ORT_DYLIB_PATH").filter(|s| !s.is_empty()) {
        return PathBuf::from(p);
    }
    let mut bases: Vec<PathBuf> = Vec::new();
    if let Some(m) = model_dir {
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

/// The ONNX model plus its contract. `Session::run` needs `&mut self`, so share it behind a Mutex.
pub struct Classifier {
    session: Session,
    pub contract: Contract,
    pub model_dir: PathBuf,
    pub ort_lib: PathBuf,
    pub provider: String,
    pub load_ms: f64,
}

impl Classifier {
    pub fn load(opts: &ModelOptions) -> Result<Self> {
        let t0 = Instant::now();
        let model_dir = resolve_model_dir(opts.model_dir.as_deref())?;
        let contract = Contract::load(&model_dir)?;
        let ort_lib = resolve_ort_lib(opts.ort_lib.as_deref(), Some(&model_dir));
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut eps = Vec::new();
        #[cfg(feature = "cuda")]
        if opts.cuda {
            eps.push(ort::ep::CUDA::default().build());
        }
        if opts.cuda && !cfg!(feature = "cuda") {
            eprintln!("[gpui-demo] --cuda ignored: built without the `cuda` cargo feature");
        }
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
            .with_execution_providers(&eps)
            .build()
            .map_err(|e| anyhow!("creating onnxruntime environment: {e}"))?;
        let mut builder = Session::builder(&env)
            .map_err(|e| anyhow!("{e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("{e}"))?;
        if let Some(n) = opts.threads {
            builder = builder.with_intra_threads(n).map_err(|e| anyhow!("{e}"))?;
        }
        let onnx = model_dir.join("model.onnx");
        let session = builder
            .commit_from_file(&onnx)
            .map_err(|e| anyhow!("loading {}: {e} (is model.onnx.data next to it and complete?)", onnx.display()))?;
        let input_ok = session.inputs().iter().any(|i| i.name() == "input");
        let output_ok = session.outputs().iter().any(|o| o.name() == "probs");
        if !input_ok || !output_ok {
            bail!("{} does not have input 'input' and output 'probs'", onnx.display());
        }
        let provider = if opts.cuda && cfg!(feature = "cuda") { "CUDA (requested) + CPU" } else { "CPU" };
        Ok(Self {
            session,
            contract,
            model_dir,
            ort_lib,
            provider: provider.to_string(),
            load_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }

    pub fn threshold(&self) -> f32 {
        self.contract.pre.abstain_threshold
    }

    /// Run the model on one preprocessed tensor; returns the 20 calibrated probabilities.
    pub fn probs_from_tensor(&mut self, tensor: &[f32]) -> Result<Vec<f32>> {
        let s = self.contract.pre.input_size as usize;
        let input = TensorRef::from_array_view(([1usize, 3, s, s], tensor)).map_err(|e| anyhow!("{e}"))?;
        let outputs = self.session.run(ort::inputs!["input" => input]).map_err(|e| anyhow!("inference: {e}"))?;
        let probs = outputs.get("probs").context("model returned no 'probs' output")?;
        let (_, data) = probs.try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        Ok(data.to_vec())
    }

    /// Preprocess + infer + decision rule. `ms` covers preprocessing and inference, like the Python demo.
    pub fn predict(&mut self, img: &RgbImage) -> Result<Prediction> {
        let t0 = Instant::now();
        let tensor = to_tensor(img, &self.contract.pre)?;
        let probs = self.probs_from_tensor(&tensor)?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok(decide(&probs, &self.contract.labels, self.threshold(), ms))
    }
}
