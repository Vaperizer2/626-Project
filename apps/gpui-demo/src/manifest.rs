//! Model manifests (`<model dir>/model.json`) and their index (`exports/models.json`), written by
//! `tools/gen_manifests.py`. This is the only parser of model metadata: everything is validated here, and the rest
//! of the app trusts these types.
//!
//! A manifest describes one ONNX graph: its files (size + sha256), its input (name, size, max batch, preprocessing),
//! the heads it serves (one per readout output) and the execution providers it prefers, in order. The index lists
//! the manifests (UI order) and the default heads in order of preference.

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;

pub const SCHEMA: &str = "model-manifest/1";
pub const INDEX_SCHEMA: &str = "model-index/1";

/// A head's id: unique across the index, the `--model` name and the JSON `model` field.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HeadId(Arc<str>);

impl HeadId {
    pub fn new(s: &str) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for HeadId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(&self.0)
    }
}

/// A graph's id (the manifest `id`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GraphId(Arc<str>);

impl GraphId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for GraphId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(&self.0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sha256(pub [u8; 32]);

impl Sha256 {
    fn parse(hex: &str) -> Option<Self> {
        let hex = hex.strip_prefix("sha256:").unwrap_or(hex);
        if hex.len() != 64 {
            return None;
        }
        let mut out = [0u8; 32];
        for (i, b) in out.iter_mut().enumerate() {
            *b = u8::from_str_radix(hex.get(2 * i..2 * i + 2)?, 16).ok()?;
        }
        Some(Self(out))
    }

    pub fn short(&self) -> String {
        self.to_string()[..12].to_string()
    }
}

impl fmt::Display for Sha256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|b| write!(f, "{b:02x}"))
    }
}

impl fmt::Debug for Sha256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sha256:{self}")
    }
}

/// A probability in the open interval (0, 1), checked at parse.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Prob(f32);

impl Prob {
    pub fn new(p: f32) -> Option<Self> {
        (p > 0.0 && p < 1.0).then_some(Self(p))
    }
    pub fn get(self) -> f32 {
        self.0
    }
}

/// Class names in output order; never empty.
#[derive(Debug, PartialEq)]
pub struct Labels(Vec<String>);

impl Labels {
    pub fn new(names: Vec<String>) -> Option<Self> {
        (!names.is_empty()).then_some(Self(names))
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn name(&self, i: usize) -> &str {
        &self.0[i]
    }
    pub fn names(&self) -> &[String] {
        &self.0
    }
}

/// The two Pillow filters the exports use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Filter {
    PillowBicubic,
    PillowBilinear,
}

/// How an RGB image becomes the graph input. The spec is `Hash + Eq` (floats by bit pattern), so it is its own
/// tensor-cache key: two graphs share a tensor only when their transforms are identical.
#[derive(Clone, Debug)]
pub enum Preprocess {
    /// torchvision `Resize(short) -> CenterCrop(crop) -> ToTensor -> Normalize` on the Pillow image.
    ResizeShortCenterCrop {
        short: u32,
        /// (width, height)
        crop: (u32, u32),
        filter: Filter,
        scale: f64,
        mean: [f32; 3],
        std: [f32; 3],
    },
}

impl Preprocess {
    fn bits(&self) -> (u32, (u32, u32), Filter, u64, [u32; 3], [u32; 3]) {
        match self {
            Preprocess::ResizeShortCenterCrop { short, crop, filter, scale, mean, std } => {
                (*short, *crop, *filter, scale.to_bits(), mean.map(f32::to_bits), std.map(f32::to_bits))
            }
        }
    }
}

impl PartialEq for Preprocess {
    fn eq(&self, other: &Self) -> bool {
        self.bits() == other.bits()
    }
}
impl Eq for Preprocess {}
impl Hash for Preprocess {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.bits().hash(h)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Fp32,
    Fp16,
    Int8Qdq,
}

#[derive(Debug)]
pub struct FileEntry {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: Sha256,
}

#[derive(Debug)]
pub struct InputSpec {
    pub name: String,
    /// (width, height)
    pub size: (u32, u32),
    pub max_batch: u16,
    pub pre: Preprocess,
}

#[derive(Debug)]
pub enum Readout {
    /// the output already holds probabilities
    Probs { output: String },
    /// softmax is applied by the app
    Logits { output: String },
}

impl Readout {
    pub fn output(&self) -> &str {
        match self {
            Readout::Probs { output } | Readout::Logits { output } => output,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Calibration {
    None,
    /// `in_graph`: the readout already includes it; otherwise the app divides logits by `t`
    Temperature { t: f32, in_graph: bool },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Decision {
    /// answer "unsure" when the top probability is below `below`
    Abstain { below: Prob },
    /// always answer; flag a top probability below the bound as low confidence
    Commit { low_confidence_below: Option<Prob> },
}

#[derive(Debug)]
pub struct ClassifyTask {
    pub readout: Readout,
    pub labels: Arc<Labels>,
    pub calibration: Calibration,
    pub decision: Decision,
}

#[derive(Debug)]
pub enum Task {
    Classify(ClassifyTask),
}

#[derive(Debug)]
pub struct HeadSpec {
    pub id: HeadId,
    pub title: String,
    pub short: String,
    pub task: Task,
}

/// Execution providers the app knows, in the spelling of `--ep` and the manifests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EpKind {
    TensorRt,
    Cuda,
    Cpu,
}

impl EpKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "tensorrt" | "trt" => Some(EpKind::TensorRt),
            "cuda" => Some(EpKind::Cuda),
            "cpu" => Some(EpKind::Cpu),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            EpKind::TensorRt => "TensorRT",
            EpKind::Cuda => "CUDA",
            EpKind::Cpu => "CPU",
        }
    }
}

/// One execution-provider preference. `options` are onnxruntime provider options, passed through unchanged
/// (the TensorRT cache paths are the app's: see `runtime`).
#[derive(Clone, Debug, PartialEq)]
pub struct Ep {
    pub kind: EpKind,
    pub options: Vec<(String, String)>,
}

impl Ep {
    pub fn option(&self, key: &str) -> Option<&str> {
        self.options.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }
    pub fn flag(&self, key: &str) -> bool {
        matches!(self.option(key), Some("1" | "true" | "True"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptLevel {
    Disable,
    Basic,
    Extended,
    Layout,
    All,
}

#[derive(Debug)]
pub struct RuntimeSpec {
    pub opt_level: OptLevel,
    /// preference order, never empty
    pub providers: Vec<Ep>,
}

#[derive(Debug)]
pub struct Manifest {
    pub id: GraphId,
    /// the model.json this came from
    pub path: PathBuf,
    pub graph: PathBuf,
    pub external: Vec<PathBuf>,
    pub precision: Precision,
    /// every file the graph needs (graph, external data, labels)
    pub files: Vec<FileEntry>,
    pub input: InputSpec,
    /// never empty
    pub heads: Vec<HeadSpec>,
    pub runtime: RuntimeSpec,
    pub provenance: BTreeMap<String, serde_json::Value>,
}

impl Manifest {
    pub fn dir(&self) -> &Path {
        self.path.parent().unwrap_or(Path::new("."))
    }

    pub fn file(&self, path: &Path) -> Option<&FileEntry> {
        self.files.iter().find(|f| f.path == path)
    }
}

/// The index: manifests in UI order and the default heads in order of preference.
#[derive(Debug)]
pub struct Index {
    pub path: PathBuf,
    pub default: Vec<HeadId>,
    pub manifests: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ManifestError {
    pub path: PathBuf,
    pub msg: String,
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path.display(), self.msg)
    }
}

impl std::error::Error for ManifestError {}

// ----------------------------------------------------------------------------------------------- wire format
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireManifest {
    schema: String,
    id: String,
    graph: WireGraph,
    files: BTreeMap<String, WireFile>,
    input: WireInput,
    heads: Vec<WireHead>,
    runtime: WireRuntime,
    #[serde(default)]
    provenance: BTreeMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGraph {
    file: String,
    #[serde(default)]
    external_data: Vec<String>,
    precision: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFile {
    bytes: u64,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireInput {
    name: String,
    dtype: String,
    layout: String,
    /// [height, width]
    size: [u32; 2],
    batch: WireBatch,
    preprocess: WirePreprocess,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireBatch {
    max: u16,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum WirePreprocess {
    ResizeShortCenterCrop {
        resize_short_side: u32,
        filter: String,
        color: String,
        scale: f64,
        mean: [f32; 3],
        std: [f32; 3],
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireHead {
    id: String,
    title: String,
    short: String,
    task: WireTask,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum WireTask {
    Classify { readout: WireReadout, labels: String, calibration: WireCalibration, decision: WireDecision },
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum WireReadout {
    Probs { output: String },
    Logits { output: String },
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum WireCalibration {
    None,
    Temperature { t: f32, in_graph: bool },
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum WireDecision {
    Abstain { below: f32 },
    Commit { low_confidence_below: Option<f32> },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRuntime {
    #[serde(default = "default_opt_level")]
    graph_optimization_level: String,
    providers: Vec<WireEp>,
}

fn default_opt_level() -> String {
    "ORT_ENABLE_ALL".into()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireEp {
    ep: String,
    #[serde(default)]
    options: BTreeMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireIndex {
    schema: String,
    default: Vec<String>,
    manifests: Vec<String>,
}

// ---------------------------------------------------------------------------------------------- parsing
fn read(path: &Path) -> Result<String, ManifestError> {
    std::fs::read_to_string(path).map_err(|e| ManifestError { path: path.to_path_buf(), msg: e.to_string() })
}

/// Parse `<dir>/model.json` (and the labels files it names). Files the graph needs are not opened here; see
/// `registry` for their availability.
pub fn parse_manifest(path: &Path) -> Result<Manifest, ManifestError> {
    let text = read(path)?;
    let w: WireManifest =
        serde_json::from_str(&text).map_err(|e| ManifestError { path: path.to_path_buf(), msg: e.to_string() })?;
    let dir = path.parent().unwrap_or(Path::new("."));
    from_wire(w, path, dir).map_err(|msg| ManifestError { path: path.to_path_buf(), msg })
}

fn from_wire(w: WireManifest, path: &Path, dir: &Path) -> Result<Manifest, String> {
    if w.schema != SCHEMA {
        return Err(format!("schema {:?}, expected {SCHEMA:?}", w.schema));
    }
    if w.id.is_empty() {
        return Err("empty id".into());
    }
    let mut files = Vec::new();
    for (name, f) in &w.files {
        let sha256 = Sha256::parse(&f.sha256).ok_or_else(|| format!("files.{name}: bad sha256 {:?}", f.sha256))?;
        if f.bytes == 0 {
            return Err(format!("files.{name}: zero bytes"));
        }
        files.push(FileEntry { path: dir.join(name), bytes: f.bytes, sha256 });
    }
    let listed = |name: &str| -> Result<PathBuf, String> {
        if w.files.contains_key(name) { Ok(dir.join(name)) } else { Err(format!("{name} is not listed in files")) }
    };
    let graph = listed(&w.graph.file)?;
    let external = w.graph.external_data.iter().map(|e| listed(e)).collect::<Result<Vec<_>, _>>()?;
    let precision = match w.graph.precision.as_str() {
        "fp32" => Precision::Fp32,
        "fp16" => Precision::Fp16,
        "int8_qdq" => Precision::Int8Qdq,
        p => return Err(format!("graph.precision {p:?} (fp32, fp16 or int8_qdq)")),
    };

    let i = w.input;
    if i.dtype != "f32" {
        return Err(format!("input.dtype {:?}: only f32 input is implemented", i.dtype));
    }
    if i.layout != "NCHW" {
        return Err(format!("input.layout {:?}: only NCHW is implemented", i.layout));
    }
    let [h, wd] = i.size;
    if h == 0 || wd == 0 {
        return Err("input.size has a zero side".into());
    }
    if i.batch.max == 0 {
        return Err("input.batch.max is 0".into());
    }
    let pre = match i.preprocess {
        WirePreprocess::ResizeShortCenterCrop { resize_short_side, filter, color, scale, mean, std } => {
            let filter = match filter.as_str() {
                "pillow_bicubic" => Filter::PillowBicubic,
                "pillow_bilinear" => Filter::PillowBilinear,
                f => return Err(format!("preprocess.filter {f:?} (pillow_bicubic or pillow_bilinear)")),
            };
            if color != "RGB" {
                return Err(format!("preprocess.color {color:?}: only RGB is implemented"));
            }
            if resize_short_side < h.max(wd) {
                return Err(format!("resize_short_side {resize_short_side} is smaller than the crop {wd}x{h}"));
            }
            if std.contains(&0.0) || !(scale > 0.0) {
                return Err("preprocess std must be non-zero and scale positive".into());
            }
            Preprocess::ResizeShortCenterCrop { short: resize_short_side, crop: (wd, h), filter, scale, mean, std }
        }
    };
    let input = InputSpec { name: i.name, size: (wd, h), max_batch: i.batch.max, pre };

    if w.heads.is_empty() {
        return Err("no heads".into());
    }
    let mut seen = HashSet::new();
    let mut labels_cache: BTreeMap<String, Arc<Labels>> = BTreeMap::new();
    let mut heads = Vec::new();
    for h in w.heads {
        if h.id.is_empty() || !seen.insert(h.id.clone()) {
            return Err(format!("head id {:?} is empty or repeated", h.id));
        }
        let task = match h.task {
            WireTask::Classify { readout, labels, calibration, decision } => {
                let labels = match labels_cache.get(&labels) {
                    Some(l) => l.clone(),
                    None => {
                        let p = listed(&labels)?;
                        let names: Vec<String> = serde_json::from_str(&std::fs::read_to_string(&p).map_err(|e| {
                            format!("{}: {e}", p.display())
                        })?)
                        .map_err(|e| format!("{}: {e}", p.display()))?;
                        let l = Arc::new(Labels::new(names).ok_or_else(|| format!("{} is empty", p.display()))?);
                        labels_cache.insert(labels, l.clone());
                        l
                    }
                };
                let readout = match readout {
                    WireReadout::Probs { output } => Readout::Probs { output },
                    WireReadout::Logits { output } => Readout::Logits { output },
                };
                let calibration = match calibration {
                    WireCalibration::None => Calibration::None,
                    WireCalibration::Temperature { t, in_graph } if t > 0.0 => Calibration::Temperature { t, in_graph },
                    WireCalibration::Temperature { t, .. } => return Err(format!("head {}: temperature {t}", h.id)),
                };
                if matches!(readout, Readout::Probs { .. })
                    && matches!(calibration, Calibration::Temperature { in_graph: false, .. })
                {
                    return Err(format!("head {}: a probs readout cannot be recalibrated outside the graph", h.id));
                }
                let prob = |p: f32, what: &str| {
                    Prob::new(p).ok_or_else(|| format!("head {}: {what} {p} is not in (0, 1)", h.id))
                };
                let decision = match decision {
                    WireDecision::Abstain { below } => Decision::Abstain { below: prob(below, "abstain below")? },
                    WireDecision::Commit { low_confidence_below } => Decision::Commit {
                        low_confidence_below: low_confidence_below.map(|p| prob(p, "low_confidence_below")).transpose()?,
                    },
                };
                if matches!(decision, Decision::Abstain { .. }) && calibration == Calibration::None {
                    return Err(format!("head {}: abstaining needs calibrated probabilities", h.id));
                }
                Task::Classify(ClassifyTask { readout, labels, calibration, decision })
            }
        };
        heads.push(HeadSpec { id: HeadId::new(&h.id), title: h.title, short: h.short, task });
    }

    let opt_level = match w.runtime.graph_optimization_level.as_str() {
        "ORT_DISABLE_ALL" => OptLevel::Disable,
        "ORT_ENABLE_BASIC" => OptLevel::Basic,
        "ORT_ENABLE_EXTENDED" => OptLevel::Extended,
        "ORT_ENABLE_LAYOUT" => OptLevel::Layout,
        "ORT_ENABLE_ALL" => OptLevel::All,
        l => return Err(format!("runtime.graph_optimization_level {l:?}")),
    };
    if w.runtime.providers.is_empty() {
        return Err("runtime.providers is empty".into());
    }
    let mut providers = Vec::new();
    for p in w.runtime.providers {
        let kind = EpKind::parse(&p.ep).ok_or_else(|| format!("unknown execution provider {:?}", p.ep))?;
        let mut options = Vec::new();
        for (k, v) in p.options {
            let v = match v {
                serde_json::Value::Bool(b) => if b { "1" } else { "0" }.to_string(),
                serde_json::Value::Number(n) => n.to_string(),
                serde_json::Value::String(s) => s,
                v => return Err(format!("{} option {k}: {v} is not a scalar", p.ep)),
            };
            options.push((k, v));
        }
        providers.push(Ep { kind, options });
    }
    Ok(Manifest {
        id: GraphId(w.id.as_str().into()),
        path: path.to_path_buf(),
        graph,
        external,
        precision,
        files,
        input,
        heads,
        runtime: RuntimeSpec { opt_level, providers },
        provenance: w.provenance,
    })
}

/// Parse `exports/models.json`. Manifest paths are relative to it.
pub fn parse_index(path: &Path) -> Result<Index, ManifestError> {
    let err = |msg: String| ManifestError { path: path.to_path_buf(), msg };
    let w: WireIndex = serde_json::from_str(&read(path)?).map_err(|e| err(e.to_string()))?;
    if w.schema != INDEX_SCHEMA {
        return Err(err(format!("schema {:?}, expected {INDEX_SCHEMA:?}", w.schema)));
    }
    if w.manifests.is_empty() {
        return Err(err("no manifests".into()));
    }
    let dir = path.parent().unwrap_or(Path::new("."));
    Ok(Index {
        path: path.to_path_buf(),
        default: w.default.iter().map(|s| HeadId::new(s)).collect(),
        manifests: w.manifests.iter().map(|m| dir.join(m)).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exports() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../exports")
    }

    /// Every manifest in the shipped index parses; with REQUIRE_EXPORTS=1 the files it names must be present too.
    #[test]
    fn shipped_manifests_parse() {
        let index = parse_index(&exports().join("models.json")).unwrap();
        let require = std::env::var_os("REQUIRE_EXPORTS").is_some_and(|v| v == "1");
        let mut heads = HashSet::new();
        for p in &index.manifests {
            let m = parse_manifest(p).unwrap_or_else(|e| panic!("{e}"));
            for h in &m.heads {
                assert!(heads.insert(h.id.clone()), "head {} repeated", h.id);
            }
            for f in &m.files {
                let found = std::fs::metadata(&f.path).map(|md| md.len()).ok();
                if require {
                    assert_eq!(found, Some(f.bytes), "{} missing or wrong size", f.path.display());
                }
            }
        }
        for d in &index.default {
            assert!(heads.contains(d), "default head {d} is not in any manifest");
        }
    }

    const MANIFEST: &str = r#"{
      "schema": "model-manifest/1", "id": "g",
      "graph": {"file": "m.onnx", "precision": "fp32"},
      "files": {"m.onnx": {"bytes": 3, "sha256": "f90f49ba39895323551b872086aea0b760fe03f0be00c188e3eddc4a0a515b67"},
                "labels.json": {"bytes": 9, "sha256": "bb7bd956f6d7cd0645b381ee858006ef33bd16ede6107b926d9eb8a65eb2011e"}},
      "input": {"name": "input", "dtype": "f32", "layout": "NCHW", "size": [4, 6], "batch": {"max": 1},
                "preprocess": {"kind": "resize_short_center_crop", "resize_short_side": 7, "filter": "pillow_bilinear",
                               "color": "RGB", "scale": 0.00392156862745098, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
      "heads": [{"id": "h", "title": "H", "short": "H", "task": {"kind": "classify",
                 "readout": {"kind": "probs", "output": "probs"}, "labels": "labels.json",
                 "calibration": {"kind": "none"}, "decision": {"kind": "commit", "low_confidence_below": 0.5}}}],
      "runtime": {"providers": [{"ep": "tensorrt", "options": {"trt_fp16_enable": true, "trt_max_workspace_size": 1024}},
                                {"ep": "cpu"}]}
    }"#;

    fn parse_text(text: &str) -> Result<Manifest, String> {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("gpui-demo-manifest-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("labels.json"), r#"["a","b"]"#).unwrap();
        let w: WireManifest = serde_json::from_str(text).map_err(|e| e.to_string())?;
        from_wire(w, &dir.join("model.json"), &dir)
    }

    #[test]
    fn parses_and_types_a_manifest() {
        let m = parse_text(MANIFEST).unwrap();
        assert_eq!(m.input.size, (6, 4), "size is [h, w] on the wire, (w, h) inside");
        assert!(matches!(m.input.pre, Preprocess::ResizeShortCenterCrop { crop: (6, 4), filter: Filter::PillowBilinear, .. }));
        let Task::Classify(t) = &m.heads[0].task;
        assert_eq!(t.labels.len(), 2);
        assert_eq!(t.decision, Decision::Commit { low_confidence_below: Prob::new(0.5) });
        let trt = &m.runtime.providers[0];
        assert_eq!(trt.kind, EpKind::TensorRt);
        assert!(trt.flag("trt_fp16_enable"));
        assert_eq!(trt.option("trt_max_workspace_size"), Some("1024"));
        assert_eq!(m.runtime.opt_level, OptLevel::All, "ORT_ENABLE_ALL unless the manifest says otherwise");
    }

    #[test]
    fn rejects_what_the_app_cannot_honour() {
        let bad = [
            (r#""decision": {"kind": "commit", "low_confidence_below": 0.5}"#, r#""decision": {"kind": "abstain", "below": 0.5}"#, "abstaining needs calibrated"),
            (r#""resize_short_side": 7"#, r#""resize_short_side": 5"#, "smaller than the crop"),
            (r#""low_confidence_below": 0.5"#, r#""low_confidence_below": 1.5"#, "not in (0, 1)"),
            (r#""labels": "labels.json""#, r#""labels": "other.json""#, "not listed in files"),
            (r#""dtype": "f32""#, r#""dtype": "f16""#, "only f32"),
            (r#"{"ep": "cpu"}"#, r#"{"ep": "metal"}"#, "unknown execution provider"),
            (r#""std": [0.5, 0.5, 0.5]"#, r#""std": [0.5, 0.0, 0.5]"#, "non-zero"),
        ];
        for (from, to, why) in bad {
            assert!(MANIFEST.contains(from), "{from}");
            let e = parse_text(&MANIFEST.replace(from, to)).unwrap_err();
            assert!(e.contains(why), "{to}: {e}");
        }
        assert!(parse_text(&MANIFEST.replace(r#""id": "g","#, r#""id": "g", "extra": 1,"#)).is_err(), "unknown keys");
    }

    #[test]
    fn preprocess_is_its_own_cache_key() {
        let a = parse_text(MANIFEST).unwrap().input.pre;
        let b = parse_text(&MANIFEST.replace("pillow_bilinear", "pillow_bicubic")).unwrap().input.pre;
        let mut set = HashSet::new();
        set.insert(a.clone());
        assert!(set.contains(&a));
        assert!(!set.contains(&b), "same size, other filter: a different tensor");
    }
}
