//! Model registry: the report's three models plus the deployed one, each = ONNX graph + output name +
//! preprocessing contract, discovered on disk and loaded lazily.
//!
//! ```text
//! exports/final/model.onnx            Deployed (fused SigLIP), output "probs"          (always; the CLI default)
//! exports/final/model_multi.onnx      same weights (model.onnx.data), several outputs: fused + model C
//!   + model_multi.json                which output is which (see `MultiSpec`)
//! exports/itri626/A_scratch_cnn/      model.onnx + preprocess.json + labels.json       (A, 160 px)
//! exports/itri626/B_convnext_tiny/    model.onnx + preprocess.json + labels.json       (B, 224 px)
//! ```
//!
//! When `model_multi.onnx` exists, Deployed and C use that one graph, so the 1.7 GB SigLIP weights are loaded
//! once and one forward pass yields both predictions. A missing model is reported as unavailable, never a panic.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use image::RgbImage;
use serde::Deserialize;

use crate::contract::{Prediction, decide};
use crate::model::{Engine, Runtime};
use crate::preprocess::to_tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ModelKey {
    A,
    B,
    C,
    Deployed,
}

impl ModelKey {
    pub const ALL: [ModelKey; 4] = [ModelKey::A, ModelKey::B, ModelKey::C, ModelKey::Deployed];

    /// CLI / JSON key.
    pub fn id(self) -> &'static str {
        match self {
            ModelKey::A => "a",
            ModelKey::B => "b",
            ModelKey::C => "c",
            ModelKey::Deployed => "deployed",
        }
    }

    /// Full display name (the report's naming).
    pub fn name(self) -> &'static str {
        match self {
            ModelKey::A => "A · Custom CNN (scratch)",
            ModelKey::B => "B · ConvNeXt-T (fine-tuned)",
            ModelKey::C => "C · SigLIP ViT + linear head",
            ModelKey::Deployed => "Deployed (fused SigLIP)",
        }
    }

    /// Compact name for segmented controls and table headers.
    pub fn short(self) -> &'static str {
        match self {
            ModelKey::A => "A · CNN",
            ModelKey::B => "B · ConvNeXt-T",
            ModelKey::C => "C · SigLIP + linear",
            ModelKey::Deployed => "Deployed",
        }
    }

    /// Only the deployed model's probabilities are temperature-calibrated, so only it applies the UNSURE
    /// (abstain) rule. A, B and C always commit to their top-1; the UI marks a top-1 below 50% as low confidence.
    pub fn calibrated(self) -> bool {
        self == ModelKey::Deployed
    }

    pub fn parse(s: &str) -> Option<ModelKey> {
        match s.trim().to_ascii_lowercase().as_str() {
            "a" | "cnn" | "scratch" | "a_scratch_cnn" => Some(ModelKey::A),
            "b" | "convnext" | "convnext_tiny" | "b_convnext_tiny" => Some(ModelKey::B),
            "c" | "siglip_linear" | "c_siglip_linear" | "linear" => Some(ModelKey::C),
            "deployed" | "d" | "fused" | "final" => Some(ModelKey::Deployed),
            _ => None,
        }
    }
}

pub fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// How a registry entry turns a forward pass into probabilities.
#[derive(Debug, Clone)]
pub enum Readout {
    /// the named output already holds calibrated probabilities
    Probs(String),
    /// the named output holds logits
    Logits(String),
}

impl Readout {
    pub fn describe(&self) -> String {
        match self {
            Readout::Probs(o) => format!("output {o}"),
            Readout::Logits(o) => format!("softmax(output {o})"),
        }
    }

    fn output(&self) -> &str {
        match self {
            Readout::Probs(o) | Readout::Logits(o) => o,
        }
    }
}

impl ModelSpec {
    pub fn readout_name(&self) -> String {
        self.readout.describe()
    }
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub key: ModelKey,
    /// the ONNX graph (engines are shared by path)
    pub onnx: PathBuf,
    /// folder with preprocess.json + labels.json
    pub contract_dir: PathBuf,
    pub readout: Readout,
    /// overrides preprocess.json's abstain_threshold
    pub threshold: Option<f32>,
}

/// `model_multi.json`. Shape written by `tools/export_itri626_models.py`:
/// `{"file": "model_multi.onnx", "outputs": [{"name": "probs", "model": "deployed final model (...)"},
///  {"name": "probs_c", "model": "ITRI626 model C (...)"}, ...]}`. Also accepted: a map
/// `{"outputs": {"deployed": "probs", "c": "probs_c"}}`. `logits*` outputs are used only when a model has no
/// `probs*` output (then softmax is applied).
#[derive(Debug, Clone, Deserialize)]
pub struct MultiSpec {
    #[serde(default, alias = "file", alias = "graph")]
    pub onnx: Option<String>,
    pub outputs: MultiOutputs,
    #[serde(default)]
    pub thresholds: HashMap<String, f32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MultiOutputs {
    Map(HashMap<String, String>),
    List(Vec<MultiOutput>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct MultiOutput {
    #[serde(alias = "output")]
    pub name: String,
    #[serde(alias = "key", alias = "role")]
    pub model: String,
    /// "probs" (default) or "logits"
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub abstain_threshold: Option<f32>,
}

impl MultiSpec {
    /// (model key -> readout, threshold) for every output that names a known model.
    pub fn entries(&self) -> Vec<(ModelKey, Readout, Option<f32>)> {
        let mut out = Vec::new();
        match &self.outputs {
            MultiOutputs::Map(m) => {
                for (k, name) in m {
                    if let Some(key) = ModelKey::parse(k) {
                        out.push((key, Readout::Probs(name.clone()), self.thresholds.get(k).copied()));
                    }
                }
            }
            MultiOutputs::List(v) => {
                for o in v {
                    let Some(key) = key_for_output(&o.model, &o.name) else { continue };
                    let logits = o.kind.as_deref() == Some("logits") || o.name.starts_with("logits");
                    let r = if logits { Readout::Logits(o.name.clone()) } else { Readout::Probs(o.name.clone()) };
                    out.push((key, r, o.abstain_threshold.or_else(|| self.thresholds.get(&o.model).copied())));
                }
            }
        }
        // one readout per model: probabilities beat logits
        out.sort_by_key(|e| (e.0, matches!(e.1, Readout::Logits(_))));
        out.dedup_by_key(|e| e.0);
        out
    }
}

/// Which registry model a multi-graph output belongs to: an exact key ("c", "deployed"), else the description
/// ("deployed final model ...", "ITRI626 model C ..."), else the output name ("probs_c" -> C, "probs" -> Deployed).
fn key_for_output(model: &str, name: &str) -> Option<ModelKey> {
    if let Some(k) = ModelKey::parse(model) {
        return Some(k);
    }
    let m = model.to_ascii_lowercase();
    if m.contains("deployed") || m.contains("fused") {
        return Some(ModelKey::Deployed);
    }
    if m.contains("model c") || m.contains("siglip_linear") || m.contains("linear head") {
        return Some(ModelKey::C);
    }
    match name {
        "probs" | "logits" => Some(ModelKey::Deployed),
        n if n.ends_with("_c") => Some(ModelKey::C),
        _ => None,
    }
}

/// Registry entry: a spec, or why the model is unavailable.
pub type Entry = (ModelKey, Result<ModelSpec, String>);

/// Discover all four models relative to the deployed export folder (`exports/final`).
pub fn discover(final_dir: &Path) -> Vec<Entry> {
    let exports = final_dir.parent().map(Path::to_path_buf).unwrap_or_else(|| final_dir.to_path_buf());
    let itri = exports.join("itri626");
    let mut specs: HashMap<ModelKey, Result<ModelSpec, String>> = HashMap::new();
    let deployed = ModelSpec {
        key: ModelKey::Deployed,
        onnx: final_dir.join("model.onnx"),
        contract_dir: final_dir.to_path_buf(),
        readout: Readout::Probs("probs".into()),
        threshold: None,
    };
    specs.insert(ModelKey::Deployed, Ok(deployed));

    // Multi-output graph: Deployed + C from one session.
    let mj = final_dir.join("model_multi.json");
    let mut multi_err = None;
    if mj.is_file() {
        match read_json::<MultiSpec>(&mj) {
            Ok(m) => {
                let onnx = final_dir.join(m.onnx.as_deref().unwrap_or("model_multi.onnx"));
                if onnx.is_file() {
                    for (key, readout, threshold) in m.entries() {
                        let spec = ModelSpec {
                            key,
                            onnx: onnx.clone(),
                            contract_dir: final_dir.to_path_buf(),
                            readout,
                            threshold,
                        };
                        specs.insert(key, Ok(spec));
                    }
                } else {
                    multi_err = Some(format!("{} names {}, which does not exist", mj.display(), onnx.display()));
                }
            }
            Err(e) => multi_err = Some(format!("{e:#}")),
        }
    }
    if let Some(e) = &multi_err {
        eprintln!("[gpui-demo] ignoring model_multi.json: {e}");
    }
    if let std::collections::hash_map::Entry::Vacant(slot) = specs.entry(ModelKey::C) {
        slot.insert(Err(multi_err.clone().unwrap_or_else(|| {
            format!("not exported yet: needs {}", final_dir.join("model_multi.json").display())
        })));
    }
    for (key, sub) in [(ModelKey::A, "A_scratch_cnn"), (ModelKey::B, "B_convnext_tiny")] {
        let dir = itri.join(sub);
        let missing: Vec<&str> =
            ["model.onnx", "preprocess.json", "labels.json"].into_iter().filter(|f| !dir.join(f).is_file()).collect();
        let spec = if missing.is_empty() {
            Ok(ModelSpec {
                key,
                onnx: dir.join("model.onnx"),
                contract_dir: dir.clone(),
                readout: Readout::Probs("probs".into()),
                threshold: None,
            })
        } else {
            Err(format!("not exported yet: {} missing in {}", missing.join(", "), dir.display()))
        };
        specs.insert(key, spec);
    }
    ModelKey::ALL
        .iter()
        .map(|k| {
            let spec = specs.remove(k).unwrap_or_else(|| Err("unknown".into())).map(|mut s| {
                if !k.calibrated() {
                    s.threshold = Some(0.0); // never UNSURE: see ModelKey::calibrated
                }
                s
            });
            (*k, spec)
        })
        .collect()
}

fn read_json<T: serde::de::DeserializeOwned>(p: &Path) -> Result<T> {
    let text = std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", p.display()))
}

/// Lazily loaded engines for the registry. Owned by one thread (sessions need `&mut` to run).
pub struct Models {
    pub entries: Vec<Entry>,
    runtime: Option<Result<Runtime, String>>,
    opts: crate::model::ModelOptions,
    final_dir: PathBuf,
    engines: HashMap<PathBuf, Result<Engine, String>>,
}

/// Result of one model on one image.
pub struct ModelRun {
    pub key: ModelKey,
    pub result: Result<Prediction, String>,
}

impl Models {
    pub fn new(opts: &crate::model::ModelOptions) -> Result<Self> {
        let final_dir = crate::model::resolve_model_dir(opts.model_dir.as_deref())?;
        Ok(Self {
            entries: discover(&final_dir),
            runtime: None,
            opts: opts.clone(),
            final_dir,
            engines: HashMap::new(),
        })
    }

    /// Class names of the deployed export (all models share this order; checked per prediction).
    /// and the deployed backbone name from its preprocess.json (`model`).
    pub fn labels(&self) -> (Vec<String>, Option<String>) {
        crate::contract::Contract::load(&self.final_dir).map(|c| (c.labels, c.pre.model)).unwrap_or_default()
    }

    pub fn spec(&self, key: ModelKey) -> Result<&ModelSpec, String> {
        match self.entries.iter().find(|e| e.0 == key) {
            Some((_, Ok(s))) => Ok(s),
            Some((_, Err(e))) => Err(e.clone()),
            None => Err("unknown model".into()),
        }
    }

    pub fn runtime_info(&self) -> Option<(String, String)> {
        match &self.runtime {
            Some(Ok(rt)) => Some((rt.ort_lib.display().to_string(), rt.provider.clone())),
            _ => None,
        }
    }

    /// Registry entries served by the same graph as `key` (including `key`), e.g. C and Deployed.
    pub fn siblings(&self, key: ModelKey) -> Vec<ModelKey> {
        let Ok(onnx) = self.spec(key).map(|s| s.onnx.clone()) else { return vec![key] };
        self.entries.iter().filter(|(_, s)| s.as_ref().is_ok_and(|s| s.onnx == onnx)).map(|e| e.0).collect()
    }

    pub fn is_loaded(&self, key: ModelKey) -> bool {
        self.spec(key).ok().and_then(|s| self.engines.get(&s.onnx)).is_some_and(|e| e.is_ok())
    }

    /// Load (once) the engine behind `key`; returns its load time in ms (0 if it was already loaded).
    pub fn ensure(&mut self, key: ModelKey) -> Result<f64, String> {
        let spec = self.spec(key)?.clone();
        if let Some(e) = self.engines.get(&spec.onnx) {
            return e.as_ref().map(|_| 0.0).map_err(Clone::clone);
        }
        if self.runtime.is_none() {
            self.runtime = Some(Runtime::init(&self.opts, Some(&self.final_dir)).map_err(|e| format!("{e:#}")));
        }
        let rt = match self.runtime.as_ref() {
            Some(Ok(rt)) => rt,
            Some(Err(e)) => return Err(e.clone()),
            None => unreachable!(),
        };
        let t0 = Instant::now();
        let engine = prof!("models::load_engine", Engine::load(rt, &spec.onnx, &spec.contract_dir))
            .and_then(|e| {
                let outs = e.output_names();
                if !outs.iter().any(|o| o == spec.readout.output()) {
                    bail!(
                        "{} has no output '{}' (outputs: {})",
                        spec.onnx.display(),
                        spec.readout.output(),
                        outs.join(", ")
                    );
                }
                Ok(e)
            })
            .map_err(|e| format!("{e:#}"));
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let r = engine.as_ref().map(|_| ms).map_err(Clone::clone);
        self.engines.insert(spec.onnx.clone(), engine);
        r
    }

    /// Run `keys` on one image: one preprocessing per input contract and one forward pass per engine, however
    /// many registry entries read from it. `cancel` is checked between engines.
    pub fn predict(&mut self, img: &RgbImage, keys: &[ModelKey], cancel: &dyn Fn() -> bool) -> Vec<ModelRun> {
        let mut runs = Vec::new();
        // group keys by engine, keeping the requested order of first appearance
        let mut groups: Vec<(PathBuf, Vec<ModelKey>)> = Vec::new();
        for &k in keys {
            match self.spec(k) {
                Ok(s) => match groups.iter_mut().find(|g| g.0 == s.onnx) {
                    Some(g) => g.1.push(k),
                    None => groups.push((s.onnx.clone(), vec![k])),
                },
                Err(e) => runs.push(ModelRun { key: k, result: Err(e) }),
            }
        }
        let mut tensors: HashMap<(u32, u32, [u32; 6]), std::sync::Arc<Vec<f32>>> = HashMap::new();
        for (onnx, group) in groups {
            if cancel() {
                break;
            }
            if let Err(e) = self.ensure(group[0]) {
                runs.extend(group.into_iter().map(|key| ModelRun { key, result: Err(e.clone()) }));
                continue;
            }
            let specs: Vec<ModelSpec> = group.iter().filter_map(|k| self.spec(*k).ok().cloned()).collect();
            let Some(Ok(engine)) = self.engines.get_mut(&onnx) else { continue };
            let t0 = Instant::now();
            let pre = engine.contract.pre.clone();
            let tkey = (pre.input_size, pre.resize_short_side, pre_bits(&pre));
            let tensor = match tensors.get(&tkey) {
                Some(t) => Ok(t.clone()),
                None => prof!("preprocess::to_tensor (registry)", to_tensor(img, &pre)).map(std::sync::Arc::new),
            };
            let tensor = match tensor {
                Ok(t) => {
                    tensors.insert(tkey, t.clone());
                    t
                }
                Err(e) => {
                    let msg = format!("{e:#}");
                    runs.extend(group.into_iter().map(|key| ModelRun { key, result: Err(msg.clone()) }));
                    continue;
                }
            };
            let names: Vec<&str> = specs.iter().map(|s| s.readout.output()).collect();
            let outs = prof!("onnx::run (registry)", engine.run(&tensor, &names));
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            match outs {
                Ok(outs) => {
                    for (spec, out) in specs.iter().zip(outs) {
                        let probs = match &spec.readout {
                            Readout::Probs(_) => Ok(out),
                            Readout::Logits(_) => Ok(softmax(&out)),
                        };
                        let thr = spec.threshold.unwrap_or(engine.contract.pre.abstain_threshold);
                        let result = probs
                            .and_then(|p| {
                                if p.len() != engine.contract.labels.len() {
                                    Err(anyhow!(
                                        "{} probabilities for {} labels",
                                        p.len(),
                                        engine.contract.labels.len()
                                    ))
                                } else {
                                    Ok(decide(&p, &engine.contract.labels, thr, ms))
                                }
                            })
                            .map_err(|e| format!("{e:#}"));
                        runs.push(ModelRun { key: spec.key, result });
                    }
                }
                Err(e) => {
                    let msg = format!("{e:#}");
                    runs.extend(group.into_iter().map(|key| ModelRun { key, result: Err(msg.clone()) }));
                }
            }
        }
        runs.sort_by_key(|r| keys.iter().position(|k| *k == r.key).unwrap_or(usize::MAX));
        runs
    }
}

/// Bits of the normalisation constants, so tensors are shared only between identical contracts.
fn pre_bits(p: &crate::contract::Preprocess) -> [u32; 6] {
    [p.mean[0], p.mean[1], p.mean[2], p.std[0], p.std[1], p.std[2]].map(f32::to_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_parse_and_name() {
        for k in ModelKey::ALL {
            assert_eq!(ModelKey::parse(k.id()), Some(k));
        }
        assert_eq!(ModelKey::parse("Fused"), Some(ModelKey::Deployed));
        assert_eq!(ModelKey::parse("x"), None);
        assert!(ModelKey::A.name().starts_with("A · "));
    }

    #[test]
    fn multi_spec_both_shapes() {
        let m: MultiSpec =
            serde_json::from_str(r#"{"onnx":"m.onnx","outputs":{"deployed":"probs","c":"probs_c","junk":"x"}}"#)
                .unwrap();
        let e = m.entries();
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].0, ModelKey::C);
        assert!(matches!(&e[0].1, Readout::Probs(n) if n == "probs_c"));
        let m: MultiSpec = serde_json::from_str(
            r#"{"outputs":[{"name":"probs","model":"deployed"},{"name":"logits_c","model":"C","kind":"logits","abstain_threshold":0.4}]}"#,
        )
        .unwrap();
        let e = m.entries();
        assert!(matches!(&e[0].1, Readout::Logits(n) if n == "logits_c"));
        assert_eq!(e[0].2, Some(0.4));
    }

    #[test]
    fn multi_spec_as_exported() {
        // the shape written by tools/export_itri626_models.py (descriptive "model", logits + probs per model)
        let m: MultiSpec = serde_json::from_str(
            r#"{"file":"model_multi.onnx","outputs":[
                {"name":"logits","model":"deployed final model (R2_deployed_fused)","shape":[1,20]},
                {"name":"probs","model":"deployed final model (R2_deployed_fused)"},
                {"name":"logits_c","model":"ITRI626 model C (SigLIP So400m/14@384 frozen + nn.Linear(1152,20))"},
                {"name":"probs_c","model":"ITRI626 model C"}]}"#,
        )
        .unwrap();
        assert_eq!(m.onnx.as_deref(), Some("model_multi.onnx"));
        let e = m.entries();
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].0, ModelKey::C);
        assert!(matches!(&e[0].1, Readout::Probs(n) if n == "probs_c"));
        assert_eq!(e[1].0, ModelKey::Deployed);
        assert!(matches!(&e[1].1, Readout::Probs(n) if n == "probs"));
    }

    #[test]
    fn shipped_model_multi_json_parses() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../exports/final/model_multi.json");
        if !p.is_file() {
            return;
        }
        let e = read_json::<MultiSpec>(&p).unwrap().entries();
        assert!(e.iter().any(|x| x.0 == ModelKey::C && matches!(&x.1, Readout::Probs(n) if n == "probs_c")));
        assert!(e.iter().any(|x| x.0 == ModelKey::Deployed && matches!(&x.1, Readout::Probs(n) if n == "probs")));
    }

    #[test]
    fn discover_reports_missing_models() {
        let tmp = std::env::temp_dir().join(format!("gpui-demo-reg-{}", std::process::id()));
        let fin = tmp.join("exports").join("final");
        std::fs::create_dir_all(&fin).unwrap();
        let e = discover(&fin);
        assert_eq!(e.len(), 4);
        assert!(e.iter().find(|x| x.0 == ModelKey::Deployed).unwrap().1.is_ok());
        for k in [ModelKey::A, ModelKey::B, ModelKey::C] {
            let r = &e.iter().find(|x| x.0 == k).unwrap().1;
            assert!(r.as_ref().is_err_and(|m| m.contains("not exported")), "{k:?}");
        }
        // an itri626 model folder with all three files becomes available
        let a = tmp.join("exports").join("itri626").join("A_scratch_cnn");
        std::fs::create_dir_all(&a).unwrap();
        for f in ["model.onnx", "preprocess.json", "labels.json"] {
            std::fs::write(a.join(f), b"{}").unwrap();
        }
        std::fs::write(fin.join("model_multi.json"), r#"{"outputs":{"deployed":"probs","c":"probs_c"}}"#).unwrap();
        std::fs::write(fin.join("model_multi.onnx"), b"").unwrap();
        let e = discover(&fin);
        let get = |k| e.iter().find(|x: &&Entry| x.0 == k).unwrap().1.as_ref().unwrap().clone();
        assert_eq!(get(ModelKey::A).onnx, a.join("model.onnx"));
        assert_eq!(get(ModelKey::C).onnx, fin.join("model_multi.onnx"));
        assert_eq!(get(ModelKey::Deployed).onnx, get(ModelKey::C).onnx, "C and Deployed share one graph");
        let _ = std::fs::remove_dir_all(&tmp); // test scratch dir under the OS temp dir only
    }
}
