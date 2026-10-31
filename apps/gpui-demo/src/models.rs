//! Lazily loaded onnxruntime sessions for the registry's graphs: one session per graph, however many heads read
//! from it, so a forward pass of model_multi.onnx yields both C and Deployed. A missing model is reported as
//! unavailable, never a panic.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::bail;
use image::RgbImage;

use crate::contract::{Prediction, decide};
use crate::manifest::{Calibration, Decision, GraphId, HeadId, Preprocess, Readout, Task};
use crate::model::{Engine, ModelOptions, Runtime};
use crate::preprocess::to_tensor;
use crate::registry::Registry;

pub fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// Owned by one thread (sessions need `&mut` to run).
pub struct Models {
    pub registry: Arc<Registry>,
    runtime: Option<Result<Runtime, String>>,
    opts: ModelOptions,
    engines: HashMap<GraphId, Result<Engine, String>>,
}

/// Result of one head on one image.
pub struct ModelRun {
    pub head: HeadId,
    pub result: Result<Prediction, String>,
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

    pub fn runtime_info(&self) -> Option<(String, String)> {
        match &self.runtime {
            Some(Ok(rt)) => Some((rt.ort_lib.display().to_string(), rt.provider.clone())),
            _ => None,
        }
    }

    pub fn is_loaded(&self, head: &HeadId) -> bool {
        self.graph_of(head).ok().and_then(|g| self.engines.get(&g)).is_some_and(|e| e.is_ok())
    }

    /// Load (once) the graph behind `head`; returns its load time in ms (0 if it was already loaded).
    pub fn ensure(&mut self, head: &HeadId) -> Result<f64, String> {
        let gid = self.graph_of(head)?;
        if let Some(e) = self.engines.get(&gid) {
            return e.as_ref().map(|_| 0.0).map_err(Clone::clone);
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
        let r = engine.as_ref().map(|_| ms).map_err(Clone::clone);
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
            let outs = prof!("onnx::run (registry)", engine.run(&tensor, &names));
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            match outs {
                Ok(outs) => {
                    for (spec, out) in specs.iter().zip(outs) {
                        let Task::Classify(t) = &spec.task;
                        let probs = match (&t.readout, t.calibration) {
                            (Readout::Probs { .. }, _) => out,
                            (Readout::Logits { .. }, Calibration::Temperature { t, in_graph: false }) => {
                                softmax(&out.iter().map(|z| z / t).collect::<Vec<_>>())
                            }
                            (Readout::Logits { .. }, _) => softmax(&out),
                        };
                        let thr = match t.decision {
                            Decision::Abstain { below } => below.get(),
                            Decision::Commit { .. } => 0.0,
                        };
                        let result = if probs.len() != t.labels.len() {
                            Err(format!("{} probabilities for {} labels", probs.len(), t.labels.len()))
                        } else {
                            Ok(decide(&probs, t.labels.names(), thr, ms))
                        };
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
