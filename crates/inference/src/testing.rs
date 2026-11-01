//! Test support (feature `test-support`): a loader whose graphs answer instantly or slowly, record what ran, and
//! can hold a load until released, plus small manifests, registries and images to run them on.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use image::RgbImage;

use crate::manifest::{
    Calibration, ClassifyTask, Decision, Ep, EpKind, Filter, GraphId, HeadId, HeadSpec, InputSpec, Labels, Manifest,
    OptLevel, Precision, Preprocess, Prob, Readout, RuntimeSpec, Task,
};
use crate::registry::{GraphEntry, Registry};
use crate::runtime::Placement;
use crate::service::{Graph, ImageInput, LoadStage, Loaded, Loader, RunFailure};

/// A graph `id` with one head `id`, a 4x4 input, two classes and `max_batch`.
pub fn manifest(id: &str, max_batch: u16) -> Arc<Manifest> {
    manifest_with_heads(id, max_batch, &[id])
}

/// A graph `id` whose one forward pass serves every head in `heads`, all reading the same output.
pub fn manifest_with_heads(id: &str, max_batch: u16, heads: &[&str]) -> Arc<Manifest> {
    let labels = Arc::new(Labels::new(vec!["dark".into(), "light".into()]).unwrap());
    let task = || {
        Task::Classify(ClassifyTask {
            readout: Readout::Probs { output: "probs".into() },
            labels: labels.clone(),
            calibration: Calibration::Temperature { t: 1.0, in_graph: true },
            decision: Decision::Abstain { below: Prob::new(0.5).unwrap() },
        })
    };
    Arc::new(Manifest {
        id: GraphId::new(id),
        path: PathBuf::from(format!("/fake/{id}/model.json")),
        graph: PathBuf::from(format!("/fake/{id}/model.onnx")),
        external: vec![],
        precision: Precision::Fp32,
        files: vec![],
        input: InputSpec {
            name: "input".into(),
            size: (4, 4),
            max_batch,
            pre: Preprocess::ResizeShortCenterCrop {
                short: 4,
                crop: (4, 4),
                filter: Filter::PillowBilinear,
                scale: 1.0 / 255.0,
                mean: [0.0; 3],
                std: [1.0; 3],
            },
        },
        heads: heads
            .iter()
            .map(|h| HeadSpec { id: HeadId::new(h), title: h.to_uppercase(), short: (*h).into(), task: task() })
            .collect(),
        runtime: RuntimeSpec {
            opt_level: OptLevel::All,
            session_config: vec![],
            providers: vec![Ep { kind: EpKind::Cpu, options: vec![] }],
        },
    })
}

pub fn registry(graphs: &[(&str, u16)], default: &[&str]) -> Arc<Registry> {
    let graphs = graphs.iter().map(|&(id, max)| GraphEntry { manifest: manifest(id, max), availability: Ok(()) });
    Arc::new(
        Registry::from_graphs(PathBuf::from("/fake/models.json"), graphs.collect(), default.iter().map(|d| HeadId::new(d)).collect(), vec![])
            .unwrap(),
    )
}

/// One graph `graph` whose single forward pass serves every head in `heads` (as Deployed and C share `final`).
pub fn shared_registry(graph: &str, heads: &[&str], default: &[&str]) -> Arc<Registry> {
    let graphs = vec![GraphEntry { manifest: manifest_with_heads(graph, 1, heads), availability: Ok(()) }];
    Arc::new(
        Registry::from_graphs(PathBuf::from("/fake/models.json"), graphs, default.iter().map(|d| HeadId::new(d)).collect(), vec![])
            .unwrap(),
    )
}

/// An image whose every channel is `v`; the fake graph sees `v / 255` as its first input value.
pub fn image(v: u8) -> ImageInput {
    ImageInput::Decoded(Arc::new(RgbImage::from_pixel(4, 4, image::Rgb([v, v, v]))))
}

#[derive(Debug, Clone, PartialEq)]
pub enum Seen {
    Loaded { graph: String },
    Started { graph: String, n: usize },
    /// first input value of each row
    Ran { graph: String, rows: Vec<u8> },
    Stopped { graph: String },
}

#[derive(Default)]
pub struct Gate(Mutex<bool>, Condvar, AtomicBool);

impl Gate {
    pub fn open(&self) {
        *self.0.lock().unwrap() = true;
        self.1.notify_all();
    }
    /// a load is waiting here, so its actor holds the job that asked for it
    pub fn holds(&self) -> bool {
        self.2.load(Ordering::SeqCst)
    }
    fn wait(&self) {
        let mut open = self.0.lock().unwrap();
        self.2.store(true, Ordering::SeqCst);
        while !*open {
            open = self.1.wait(open).unwrap();
        }
    }
}

#[derive(Default, Clone)]
pub struct FakeLoader {
    pub run: Duration,
    /// loads of these graphs wait until their gate opens
    pub gates: HashMap<String, Arc<Gate>>,
    /// graphs whose preferred provider does not register
    pub no_provider: HashSet<String>,
    /// graphs whose load fails (until `repair`)
    pub broken: Arc<Mutex<HashSet<String>>>,
    pub seen: Arc<Mutex<Vec<Seen>>>,
    /// graphs dropped so far
    pub drops: Arc<AtomicUsize>,
}

impl FakeLoader {
    /// Loads of `graphs` fail.
    pub fn breaking(graphs: &[&str]) -> Self {
        let broken = graphs.iter().map(|g| g.to_string()).collect();
        Self { broken: Arc::new(Mutex::new(broken)), ..Default::default() }
    }
    /// The next load of `graph` works.
    pub fn repair(&self, graph: &str) {
        self.broken.lock().unwrap().remove(graph);
    }
    pub fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
    pub fn dropped(&self) -> usize {
        self.drops.load(Ordering::SeqCst)
    }
}

impl Loader for FakeLoader {
    fn load(&self, m: &Arc<Manifest>, stage: &dyn Fn(LoadStage)) -> Result<(Box<dyn Graph>, Loaded), String> {
        stage(LoadStage::Loading);
        if let Some(g) = self.gates.get(m.id.as_str()) {
            g.wait();
        }
        self.seen.lock().unwrap().push(Seen::Loaded { graph: m.id.to_string() });
        if self.broken.lock().unwrap().contains(m.id.as_str()) {
            return Err(format!("{}: Protobuf parsing failed", m.graph.display()));
        }
        let graph = FakeGraph { id: m.id.to_string(), run: self.run, seen: self.seen.clone(), drops: self.drops.clone() };
        let placement = Placement { registered: vec![EpKind::Cpu], skipped: vec![] };
        Ok((Box::new(graph), Loaded { ms: 1.0, placement, runtime: "fake".into(), verified: true }))
    }

    fn probe(&self, m: &Manifest) -> Result<(), String> {
        if self.no_provider.contains(m.id.as_str()) { Err("TensorRT does not register: fake".into()) } else { Ok(()) }
    }
}

struct FakeGraph {
    id: String,
    run: Duration,
    seen: Arc<Mutex<Vec<Seen>>>,
    drops: Arc<AtomicUsize>,
}

impl Drop for FakeGraph {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

impl Graph for FakeGraph {
    fn run(
        &mut self,
        x: &[f32],
        n: usize,
        _outputs: &[&str],
        alive: &(dyn Fn() -> bool + Sync),
    ) -> Result<Vec<Vec<f32>>, RunFailure> {
        let graph = self.id.clone();
        self.seen.lock().unwrap().push(Seen::Started { graph: graph.clone(), n });
        let t0 = Instant::now();
        while t0.elapsed() < self.run {
            if !alive() {
                self.seen.lock().unwrap().push(Seen::Stopped { graph });
                return Err(RunFailure::Stopped);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        let per = x.len() / n;
        let rows: Vec<u8> = (0..n).map(|r| (x[r * per] * 255.0).round() as u8).collect();
        let probs = rows.iter().flat_map(|&v| if v < 128 { [0.9, 0.1] } else { [0.1, 0.9] }).collect();
        self.seen.lock().unwrap().push(Seen::Ran { graph, rows });
        Ok(vec![probs])
    }
}
