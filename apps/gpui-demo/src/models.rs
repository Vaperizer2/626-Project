//! Lazily loaded onnxruntime sessions for the registry's graphs: one session per graph, however many heads read
//! from it, so a forward pass of model_multi.onnx yields both C and Deployed. A missing model is reported as
//! unavailable, never a panic.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::bail;
use image::RgbImage;

use crate::classification::{HeadOutput, read_head};
use crate::manifest::{GraphId, HeadId, Preprocess, Task};
use crate::runtime::{Engine, ModelOptions, Placement, Runtime};
use crate::preprocess::to_tensor;
use crate::registry::Registry;

/// Owned by one thread (sessions need `&mut` to run).
pub struct Models {
    pub registry: Arc<Registry>,
    runtime: Option<Result<Runtime, String>>,
    opts: ModelOptions,
    engines: HashMap<GraphId, Result<Engine, String>>,
}

/// Where one head's time went on one image. Heads that share a forward pass share its preprocessing and run.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timing {
    pub preprocess_ms: f64,
    pub run_ms: f64,
    /// heads read from this forward pass
    pub shared_by: usize,
}

impl Timing {
    /// preprocessing + inference: the span the Python demo reports as `ms`
    pub fn ms(&self) -> f64 {
        self.preprocess_ms + self.run_ms
    }
}

/// Result of one head on one image.
pub struct ModelRun {
    pub head: HeadId,
    pub result: Result<(HeadOutput, Timing), String>,
}

impl Models {
    pub fn new(registry: Arc<Registry>, opts: &ModelOptions) -> Self {
        Self { registry, runtime: None, opts: opts.clone(), engines: HashMap::new() }
    }

    fn graph_of(&self, head: &HeadId) -> Result<GraphId, String> {
        let h = self.registry.head(head).ok_or_else(|| format!("unknown model {head}"))?;
        h.available().map_err(|e| e.to_string())?;
        Ok(h.manifest().id.clone())
    }

    /// (library path, version) once onnxruntime is loaded.
    pub fn runtime_info(&self) -> Option<(String, String)> {
        match &self.runtime {
            Some(Ok(rt)) => Some((rt.ort_lib.display().to_string(), rt.version.clone())),
            _ => None,
        }
    }

    pub fn is_loaded(&self, head: &HeadId) -> bool {
        self.graph_of(head).ok().and_then(|g| self.engines.get(&g)).is_some_and(|e| e.is_ok())
    }

    /// Load (once) the graph behind `head`; returns its load time in ms (0 if it was already loaded) and the
    /// providers its session registered.
    pub fn ensure(&mut self, head: &HeadId) -> Result<(f64, Placement), String> {
        let gid = self.graph_of(head)?;
        if let Some(e) = self.engines.get(&gid) {
            return e.as_ref().map(|e| (0.0, e.placement.clone())).map_err(Clone::clone);
        }
        if self.runtime.is_none() {
            let base = self.registry.index_path().parent().map(|p| p.to_path_buf());
            self.runtime = Some(Runtime::init(&self.opts, base.as_deref()).map_err(|e| format!("{e:#}")));
        }
        let rt = match self.runtime.as_ref() {
            Some(Ok(rt)) => rt,
            Some(Err(e)) => return Err(e.clone()),
            None => unreachable!(),
        };
        let manifest = self.registry.graph(&gid).expect("graph of a known head").manifest.clone();
        let t0 = Instant::now();
        let engine = prof!("models::load_engine", Engine::load(rt, manifest.clone()))
            .and_then(|e| {
                let outs = e.output_names();
                for h in &manifest.heads {
                    let Task::Classify(t) = &h.task;
                    if !outs.iter().any(|o| o == t.readout.output()) {
                        bail!(
                            "{} has no output '{}' (outputs: {})",
                            manifest.graph.display(),
                            t.readout.output(),
                            outs.join(", ")
                        );
                    }
                }
                Ok(e)
            })
            .map_err(|e| format!("{e:#}"));
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let r = engine.as_ref().map(|e| (ms, e.placement.clone())).map_err(Clone::clone);
        self.engines.insert(gid, engine);
        r
    }

    /// Run `heads` on one image: one preprocessing per distinct transform and one forward pass per graph, however
    /// many heads read from it. `cancel` is checked between graphs.
    pub fn predict(&mut self, img: &RgbImage, heads: &[HeadId], cancel: &dyn Fn() -> bool) -> Vec<ModelRun> {
        let mut runs = Vec::new();
        // group heads by graph, keeping the requested order of first appearance
        let mut groups: Vec<(GraphId, Vec<HeadId>)> = Vec::new();
        for h in heads {
            match self.graph_of(h) {
                Ok(g) => match groups.iter_mut().find(|x| x.0 == g) {
                    Some(x) => x.1.push(h.clone()),
                    None => groups.push((g, vec![h.clone()])),
                },
                Err(e) => runs.push(ModelRun { head: h.clone(), result: Err(e) }),
            }
        }
        let mut tensors: HashMap<Preprocess, Arc<Vec<f32>>> = HashMap::new();
        for (gid, group) in groups {
            if cancel() {
                break;
            }
            if let Err(e) = self.ensure(&group[0]) {
                runs.extend(group.into_iter().map(|head| ModelRun { head, result: Err(e.clone()) }));
                continue;
            }
            let Some(Ok(engine)) = self.engines.get_mut(&gid) else { continue };
            let manifest = engine.manifest.clone();
            let t0 = Instant::now();
            let pre = &manifest.input.pre;
            let tensor = match tensors.get(pre) {
                Some(t) => Ok(t.clone()),
                None => prof!("preprocess::to_tensor (registry)", to_tensor(img, pre)).map(Arc::new),
            };
            let tensor = match tensor {
                Ok(t) => {
                    tensors.insert(pre.clone(), t.clone());
                    t
                }
                Err(e) => {
                    let msg = format!("{e:#}");
                    runs.extend(group.into_iter().map(|head| ModelRun { head, result: Err(msg.clone()) }));
                    continue;
                }
            };
            let specs: Vec<_> = group
                .iter()
                .map(|id| manifest.heads.iter().find(|h| &h.id == id).expect("head of this graph"))
                .collect();
            let names: Vec<&str> = specs
                .iter()
                .map(|s| {
                    let Task::Classify(t) = &s.task;
                    t.readout.output()
                })
                .collect();
            let preprocess_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let outs = prof!("onnx::run (registry)", engine.run(&tensor, 1, &names));
            let timing = Timing {
                preprocess_ms,
                run_ms: t0.elapsed().as_secs_f64() * 1000.0 - preprocess_ms,
                shared_by: specs.len(),
            };
            match outs {
                Ok(outs) => {
                    for (spec, out) in specs.iter().zip(outs) {
                        let result = read_head(&spec.task, &out).map(|o| (o, timing));
                        runs.push(ModelRun { head: spec.id.clone(), result });
                    }
                }
                Err(e) => {
                    let msg = format!("{e:#}");
                    runs.extend(group.into_iter().map(|head| ModelRun { head, result: Err(msg.clone()) }));
                }
            }
        }
        runs.sort_by_key(|r| heads.iter().position(|k| *k == r.head).unwrap_or(usize::MAX));
        runs
    }
}
