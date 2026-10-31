//! The inference service: the one way the tabs and the CLI run models. Nothing here blocks the caller; results
//! stream back over a channel per request.
//!
//! * io workers (2 threads) read or fetch and decode the image, make the display copy, and compute one tensor per
//!   distinct preprocessing spec of the request.
//! * one actor thread per graph owns its session. It loads lazily, so a 1.7 GB load or a TensorRT engine build
//!   blocks only that graph. It takes the next live job (interactive before bulk) and batches whatever else is
//!   already queued, up to the graph's `batch.max`, without waiting for more.
//! * a lane is a tab's stream of submissions with one priority and policy. `Latest` supersedes the lane's queued
//!   and running work on every submit. Dropping a request's `Ticket` cancels it. A run in flight is stopped only
//!   when every job in its batch is dead.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use futures::channel::mpsc as fmpsc;
use image::{RgbImage, RgbaImage};

use crate::classification::{HeadOutput, read_head};
use crate::fetch::Source;
use crate::manifest::{GraphId, HeadId, Manifest, Preprocess, Task};
use crate::preprocess;
use crate::registry::Registry;
use crate::runtime::Placement;

const IO_WORKERS: usize = 2;

// ------------------------------------------------------------------------------------------ backend seam
/// Why a run produced no outputs.
#[derive(Debug)]
pub enum RunFailure {
    /// every job in the batch died while it ran
    Stopped,
    Error(String),
}

/// A loaded graph. `run` gets `n` inputs as one `[n, 3, H, W]` float32 slice and returns each requested output
/// flattened `[n, ...]`. `alive` is polled while the run is in flight; once it is false the run may be abandoned.
pub trait Graph: Send {
    fn run(
        &mut self,
        x: &[f32],
        n: usize,
        outputs: &[&str],
        alive: &(dyn Fn() -> bool + Sync),
    ) -> Result<Vec<Vec<f32>>, RunFailure>;
}

/// Builds graphs from manifests (onnxruntime in the app, a fake in tests).
pub trait Loader: Send + Sync + 'static {
    /// Load `m`, reporting each stage it enters.
    fn load(&self, m: &Arc<Manifest>, stage: &dyn Fn(LoadStage)) -> Result<(Box<dyn Graph>, Loaded), String>;
    /// Does `m`'s preferred (first) execution provider register here?
    fn probe(&self, m: &Manifest) -> Result<(), String>;
}

#[derive(Clone, Debug)]
pub struct Loaded {
    pub ms: f64,
    pub placement: Placement,
    /// e.g. "onnxruntime 1.31.1"
    pub runtime: String,
    /// every file is the copy `models --check` last hashed and found matching its manifest
    pub verified: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadStage {
    Verifying,
    /// TensorRT is compiling an engine (first run only when its cache works)
    BuildingEngine,
    Loading,
}

#[derive(Clone, Debug)]
pub enum GraphState {
    Loading { since: Instant, stage: LoadStage },
    Ready(Loaded),
    Failed(String),
}

pub enum ServiceEvent {
    Graph(GraphId, GraphState),
    /// the head to start with: the first default whose preferred provider registers (the last default needs only
    /// its files), with the defaults passed over and why
    Default { head: HeadId, passed_over: Vec<(HeadId, String)> },
}

// ---------------------------------------------------------------------------------------------- requests
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Priority {
    Interactive,
    Bulk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// every submit runs (Classify history, Run all)
    Keep,
    /// a submit supersedes the lane's queued and running work (for a live camera; no tab streams yet)
    #[cfg_attr(not(test), allow(dead_code))]
    Latest,
}

pub enum ImageInput {
    Source(Source),
    Bytes(Vec<u8>),
    /// already decoded (the demo stack): no read, no decode
    Decoded(Arc<RgbImage>),
}

#[derive(Clone, Debug)]
pub enum RunError {
    Read(String),
    Decode(String),
    Preprocess(String),
    Unavailable(String),
    Load(String),
    Run(String),
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunError::Read(e) | RunError::Decode(e) | RunError::Preprocess(e) | RunError::Run(e) => f.write_str(e),
            RunError::Unavailable(e) => write!(f, "unavailable: {e}"),
            RunError::Load(e) => write!(f, "failed to load: {e}"),
        }
    }
}

/// Where one head's time went on one image (ms). Heads read from one pass share its preprocessing and run.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct Timing {
    /// read or fetch + decode (0 for an already decoded image)
    pub decode_ms: f64,
    /// this graph's tensor (0 when another graph of the request had the same spec)
    pub preprocess_ms: f64,
    /// waiting for the graph (includes its load on first use)
    pub queue_ms: f64,
    pub run_ms: f64,
    /// readout and decision
    pub post_ms: f64,
    /// images in the run
    pub batch: usize,
    /// heads read from this pass
    pub shared_by: usize,
}

impl Timing {
    /// preprocessing + inference: the span the Python demo reports as `ms`
    pub fn ms(&self) -> f64 {
        self.preprocess_ms + self.run_ms + self.post_ms
    }

}

pub enum Update {
    /// the image is decoded; `display` is the BGRA copy asked for (long side <= the requested size)
    Decoded { size: (u32, u32), display: Option<RgbaImage> },
    Head { head: HeadId, result: Result<HeadOutput, RunError>, timing: Timing },
    Done(Finish),
}

#[derive(Debug, Clone)]
pub enum Finish {
    Complete { wall_ms: f64 },
    /// the image could not be read or decoded
    Failed(RunError),
    Cancelled,
    Superseded,
}

/// Cancels its request when dropped (an evicted list entry, a closed tab, a stopped run).
pub struct Ticket(Arc<AtomicBool>);

impl Drop for Ticket {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

pub struct RequestHandle {
    pub updates: fmpsc::UnboundedReceiver<Update>,
    pub ticket: Ticket,
}

enum Status {
    Live,
    Cancelled,
    Superseded,
}

struct Request {
    updates: fmpsc::UnboundedSender<Update>,
    cancelled: Arc<AtomicBool>,
    lane: Arc<AtomicU64>,
    generation: u64,
    priority: Priority,
    t0: Instant,
    /// graph jobs not finished yet
    pending: AtomicUsize,
    done: AtomicBool,
}

impl Request {
    fn status(&self) -> Status {
        if self.cancelled.load(Ordering::Relaxed) {
            Status::Cancelled
        } else if self.lane.load(Ordering::Relaxed) != self.generation {
            Status::Superseded
        } else {
            Status::Live
        }
    }

    fn alive(&self) -> bool {
        matches!(self.status(), Status::Live)
    }

    fn send(&self, u: Update) {
        let _ = self.updates.unbounded_send(u);
    }

    fn finish(&self) {
        if self.done.swap(true, Ordering::AcqRel) {
            return;
        }
        self.send(Update::Done(match self.status() {
            Status::Live => Finish::Complete { wall_ms: self.t0.elapsed().as_secs_f64() * 1000.0 },
            Status::Cancelled => Finish::Cancelled,
            Status::Superseded => Finish::Superseded,
        }));
    }

    fn fail(&self, e: RunError) {
        if !self.done.swap(true, Ordering::AcqRel) {
            self.send(Update::Done(Finish::Failed(e)));
        }
    }

    fn graph_done(&self) {
        if self.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.finish();
        }
    }
}

// ------------------------------------------------------------------------------------------------ queues
/// Two-class FIFO: interactive items before bulk ones. `pop` blocks; it returns None once closed.
struct Queue<T> {
    state: Mutex<(VecDeque<T>, VecDeque<T>, bool)>,
    ready: Condvar,
}

impl<T> Queue<T> {
    fn new() -> Self {
        Self { state: Mutex::new((VecDeque::new(), VecDeque::new(), false)), ready: Condvar::new() }
    }

    fn push(&self, p: Priority, item: T) -> Result<(), T> {
        let mut s = self.state.lock().expect("queue lock");
        if s.2 {
            return Err(item);
        }
        match p {
            Priority::Interactive => s.0.push_back(item),
            Priority::Bulk => s.1.push_back(item),
        }
        self.ready.notify_one();
        Ok(())
    }

    fn pop(&self) -> Option<T> {
        let mut s = self.state.lock().expect("queue lock");
        loop {
            if s.2 {
                return None;
            }
            if let Some(x) = s.0.pop_front().or_else(|| s.1.pop_front()) {
                return Some(x);
            }
            s = self.ready.wait(s).expect("queue lock");
        }
    }

    fn try_pop(&self) -> Option<T> {
        let mut s = self.state.lock().expect("queue lock");
        s.0.pop_front().or_else(|| s.1.pop_front())
    }

    fn close(&self) {
        self.state.lock().expect("queue lock").2 = true;
        self.ready.notify_all();
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        let s = self.state.lock().expect("queue lock");
        s.0.len() + s.1.len()
    }
}

struct IoJob {
    req: Arc<Request>,
    input: ImageInput,
    heads: Vec<HeadId>,
    display: Option<u32>,
}

struct GraphJob {
    req: Arc<Request>,
    /// indices into the manifest's heads
    heads: Vec<usize>,
    tensor: Arc<Vec<f32>>,
    decode_ms: f64,
    preprocess_ms: f64,
    enqueued: Instant,
}

enum Msg {
    Preload,
    Job(GraphJob),
}

// ---------------------------------------------------------------------------------------------- service
struct Core {
    registry: Arc<Registry>,
    loader: Arc<dyn Loader>,
    events: fmpsc::UnboundedSender<ServiceEvent>,
    io: Queue<IoJob>,
    actors: Mutex<HashMap<GraphId, Arc<Queue<Msg>>>>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl Core {
    fn close(&self) {
        self.io.close();
        for q in self.actors.lock().expect("actors lock").values() {
            q.close();
        }
    }
}

/// Closes every queue when the last `Inference` handle goes, so the worker threads end.
struct Owner(Arc<Core>);

impl Drop for Owner {
    fn drop(&mut self) {
        self.0.close();
    }
}

#[derive(Clone)]
pub struct Inference {
    owner: Arc<Owner>,
}

pub struct Lane {
    inference: Inference,
    priority: Priority,
    policy: Policy,
    generation: Arc<AtomicU64>,
}

impl Inference {
    pub fn spawn(registry: Arc<Registry>, loader: impl Loader) -> (Inference, fmpsc::UnboundedReceiver<ServiceEvent>) {
        let (events, rx) = fmpsc::unbounded();
        let core = Arc::new(Core {
            registry,
            loader: Arc::new(loader),
            events,
            io: Queue::new(),
            actors: Mutex::new(HashMap::new()),
            threads: Mutex::new(Vec::new()),
        });
        let mut threads = Vec::new();
        for i in 0..IO_WORKERS {
            let core = core.clone();
            let t = std::thread::Builder::new().name(format!("io-{i}")).spawn(move || {
                while let Some(job) = core.io.pop() {
                    core.prepare(job);
                }
            });
            threads.push(t.expect("spawn io worker"));
        }
        let c = core.clone();
        let t = std::thread::Builder::new().name("default-head".into()).spawn(move || c.resolve_default());
        threads.push(t.expect("spawn default-head probe"));
        core.threads.lock().expect("threads lock").extend(threads);
        (Inference { owner: Arc::new(Owner(core)) }, rx)
    }

    /// Stop taking work and wait, up to `grace`, until every worker has ended and every graph has released its
    /// session. Call it before the process exits: onnxruntime's GPU providers tear down process-wide state in exit
    /// handlers, and a TensorRT session still being destroyed on another thread then crashes the process. Returns
    /// false when a worker is still busy (a load or engine build cannot be interrupted).
    pub fn shutdown(&self, grace: Duration) -> bool {
        let core = &self.owner.0;
        core.close();
        let deadline = Instant::now() + grace;
        let threads = std::mem::take(&mut *core.threads.lock().expect("threads lock"));
        let mut all = true;
        for t in threads {
            while !t.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            if t.is_finished() {
                let _ = t.join();
            } else {
                all = false;
            }
        }
        all
    }

    pub fn registry(&self) -> &Arc<Registry> {
        &self.owner.0.registry
    }

    pub fn lane(&self, priority: Priority, policy: Policy) -> Lane {
        Lane { inference: self.clone(), priority, policy, generation: Arc::new(AtomicU64::new(0)) }
    }

    /// Load `head`'s graph now (again, if its last load failed).
    pub fn preload(&self, head: &HeadId) {
        let core = &self.owner.0;
        if let Some(h) = core.registry.head(head)
            && h.available().is_ok()
            && let Some(q) = core.actor(h.manifest())
        {
            let _ = q.push(Priority::Interactive, Msg::Preload);
        }
    }

    #[cfg(test)]
    fn queued(&self, graph: &GraphId) -> usize {
        self.owner.0.actors.lock().unwrap().get(graph).map_or(0, |q| q.len())
    }
}

impl Lane {
    /// Run `heads` on `input`. With `display`, the first update carries a BGRA copy whose long side is at most that.
    pub fn submit(&self, input: ImageInput, heads: &[HeadId], display: Option<u32>) -> RequestHandle {
        let generation = match self.policy {
            Policy::Latest => self.generation.fetch_add(1, Ordering::Relaxed) + 1,
            Policy::Keep => self.generation.load(Ordering::Relaxed),
        };
        let (tx, updates) = fmpsc::unbounded();
        let cancelled = Arc::new(AtomicBool::new(false));
        let req = Arc::new(Request {
            updates: tx,
            cancelled: cancelled.clone(),
            lane: self.generation.clone(),
            generation,
            priority: self.priority,
            t0: Instant::now(),
            pending: AtomicUsize::new(0),
            done: AtomicBool::new(false),
        });
        let job = IoJob { req: req.clone(), input, heads: heads.to_vec(), display };
        if self.inference.owner.0.io.push(self.priority, job).is_err() {
            req.fail(RunError::Run("the inference service has stopped".into()));
        }
        RequestHandle { updates, ticket: Ticket(cancelled) }
    }

}

impl Core {
    /// The queue of `m`'s actor, spawning it on first use.
    fn actor(&self, m: &Arc<Manifest>) -> Option<Arc<Queue<Msg>>> {
        let mut actors = self.actors.lock().expect("actors lock");
        if let Some(q) = actors.get(&m.id) {
            return Some(q.clone());
        }
        let q = Arc::new(Queue::new());
        let (m2, q2, loader, events) = (m.clone(), q.clone(), self.loader.clone(), self.events.clone());
        let t = std::thread::Builder::new()
            .name(format!("graph-{}", m.id))
            .spawn(move || actor(m2, q2, loader, events))
            .ok()?;
        self.threads.lock().expect("threads lock").push(t);
        actors.insert(m.id.clone(), q.clone());
        Some(q)
    }

    fn resolve_default(&self) {
        let order = self.registry.default_order();
        let mut passed_over = Vec::new();
        for (i, id) in order.iter().enumerate() {
            let Some(h) = self.registry.head(id) else { continue };
            if let Err(e) = h.available() {
                passed_over.push((id.clone(), e.to_string()));
                continue;
            }
            if i + 1 < order.len()
                && let Err(e) = self.loader.probe(h.manifest())
            {
                passed_over.push((id.clone(), e));
                continue;
            }
            let _ = self.events.unbounded_send(ServiceEvent::Default { head: id.clone(), passed_over });
            return;
        }
        if let Some(h) = self.registry.heads().find(|h| h.available().is_ok()) {
            let _ = self.events.unbounded_send(ServiceEvent::Default { head: h.id().clone(), passed_over });
        }
    }

    /// Decode, then one tensor per distinct preprocessing spec, each handed to its graph as soon as it is ready.
    fn prepare(&self, job: IoJob) {
        let IoJob { req, input, heads, display } = job;
        if !req.alive() {
            return req.finish();
        }
        let t0 = Instant::now();
        let rgb = match input {
            ImageInput::Decoded(rgb) => Ok(rgb),
            ImageInput::Bytes(b) => prof!("job::decode_bytes", preprocess::decode(&b))
                .map(Arc::new)
                .map_err(|e| RunError::Decode(format!("{e:#}"))),
            ImageInput::Source(s) => prof!("job::read", s.read())
                .map_err(|e| RunError::Read(format!("{e:#}")))
                .and_then(|b| {
                    prof!("job::decode", preprocess::decode(&b))
                        .map(Arc::new)
                        .map_err(|e| RunError::Decode(format!("{e:#}")))
                }),
        };
        let rgb = match rgb {
            Ok(rgb) => rgb,
            Err(e) => return req.fail(e),
        };
        let decode_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let display = display.map(|side| prof!("job::display_copy", preprocess::display_bgra(&rgb, side)));
        req.send(Update::Decoded { size: rgb.dimensions(), display });

        let mut groups: Vec<(Arc<Manifest>, Vec<usize>)> = Vec::new();
        for id in &heads {
            let unavailable = |why: String| Update::Head {
                head: id.clone(),
                result: Err(RunError::Unavailable(why)),
                timing: Timing::default(),
            };
            let Some(h) = self.registry.head(id) else {
                req.send(unavailable("unknown model".into()));
                continue;
            };
            if let Err(e) = h.available() {
                req.send(unavailable(e.to_string()));
                continue;
            }
            let m = h.manifest();
            let ix = m.heads.iter().position(|s| &s.id == id).expect("head of its manifest");
            match groups.iter_mut().find(|g| g.0.id == m.id) {
                Some(g) => g.1.push(ix),
                None => groups.push((m.clone(), vec![ix])),
            }
        }
        req.pending.store(groups.len(), Ordering::Release);
        if groups.is_empty() || !req.alive() {
            return req.finish();
        }
        let mut tensors: HashMap<&Preprocess, Arc<Vec<f32>>> = HashMap::new();
        for (m, ix) in &groups {
            let t = Instant::now();
            let tensor = match tensors.get(&m.input.pre) {
                Some(t) => Ok(t.clone()),
                None => prof!("preprocess::to_tensor", preprocess::to_tensor(&rgb, &m.input.pre)).map(Arc::new),
            };
            let preprocess_ms = t.elapsed().as_secs_f64() * 1000.0;
            let tensor = match tensor {
                Ok(tensor) => tensor,
                Err(e) => {
                    for &i in ix {
                        req.send(Update::Head {
                            head: m.heads[i].id.clone(),
                            result: Err(RunError::Preprocess(format!("{e:#}"))),
                            timing: Timing { decode_ms, ..Default::default() },
                        });
                    }
                    req.graph_done();
                    continue;
                }
            };
            tensors.insert(&m.input.pre, tensor.clone());
            let job = GraphJob {
                req: req.clone(),
                heads: ix.clone(),
                tensor,
                decode_ms,
                preprocess_ms,
                enqueued: Instant::now(),
            };
            match self.actor(m) {
                Some(q) => {
                    if q.push(req.priority, Msg::Job(job)).is_err() {
                        req.graph_done();
                    }
                }
                None => req.graph_done(),
            }
        }
    }
}

/// One graph's thread: owns the loaded graph, serves its queue.
fn actor(m: Arc<Manifest>, queue: Arc<Queue<Msg>>, loader: Arc<dyn Loader>, events: fmpsc::UnboundedSender<ServiceEvent>) {
    let emit = |st: GraphState| {
        let _ = events.unbounded_send(ServiceEvent::Graph(m.id.clone(), st));
    };
    let load = || -> Result<Box<dyn Graph>, String> {
        let since = Instant::now();
        emit(GraphState::Loading { since, stage: LoadStage::Verifying });
        match prof!("graph::load", loader.load(&m, &|stage| emit(GraphState::Loading { since, stage }))) {
            Ok((g, loaded)) => {
                emit(GraphState::Ready(loaded));
                Ok(g)
            }
            Err(e) => {
                emit(GraphState::Failed(e.clone()));
                Err(e)
            }
        }
    };
    let mut graph: Option<Result<Box<dyn Graph>, String>> = None;
    let max = usize::from(m.input.max_batch);
    while let Some(msg) = queue.pop() {
        let first = match msg {
            Msg::Preload => {
                if !matches!(graph, Some(Ok(_))) {
                    graph = Some(load());
                }
                continue;
            }
            Msg::Job(j) => j,
        };
        if !first.req.alive() {
            first.req.graph_done();
            continue;
        }
        let g = match graph.get_or_insert_with(load) {
            Ok(g) => g,
            Err(e) => {
                let e = e.clone();
                for &i in &first.heads {
                    first.req.send(Update::Head {
                        head: m.heads[i].id.clone(),
                        result: Err(RunError::Load(e.clone())),
                        timing: Timing { decode_ms: first.decode_ms, preprocess_ms: first.preprocess_ms, ..Default::default() },
                    });
                }
                first.req.graph_done();
                continue;
            }
        };
        let mut batch = vec![first];
        while batch.len() < max {
            match queue.try_pop() {
                Some(Msg::Job(j)) if j.req.alive() => batch.push(j),
                Some(Msg::Job(j)) => j.req.graph_done(),
                Some(Msg::Preload) => {}
                None => break,
            }
        }
        run_batch(&m, g.as_mut(), batch);
    }
}

fn run_batch(m: &Manifest, g: &mut dyn Graph, batch: Vec<GraphJob>) {
    let n = batch.len();
    let output_of = |i: usize| {
        let Task::Classify(t) = &m.heads[i].task;
        t.readout.output()
    };
    let mut outputs: Vec<&str> = Vec::new();
    for j in &batch {
        for &i in &j.heads {
            if !outputs.contains(&output_of(i)) {
                outputs.push(output_of(i));
            }
        }
    }
    let joined;
    let x: &[f32] = if n == 1 {
        &batch[0].tensor
    } else {
        joined = batch.iter().flat_map(|j| j.tensor.iter().copied()).collect::<Vec<f32>>();
        &joined
    };
    let started = Instant::now();
    let alive = || batch.iter().any(|j| j.req.alive());
    let result = prof!("graph::run", g.run(x, n, &outputs, &alive));
    let run_ms = started.elapsed().as_secs_f64() * 1000.0;
    for (row, j) in batch.iter().enumerate() {
        let base = Timing {
            decode_ms: j.decode_ms,
            preprocess_ms: j.preprocess_ms,
            queue_ms: started.duration_since(j.enqueued).as_secs_f64() * 1000.0,
            run_ms,
            post_ms: 0.0,
            batch: n,
            shared_by: j.heads.len(),
        };
        match &result {
            Ok(outs) if j.req.alive() => {
                for &i in &j.heads {
                    let t = Instant::now();
                    let out = &outs[outputs.iter().position(|o| *o == output_of(i)).expect("requested output")];
                    let per = out.len() / n;
                    let res = read_head(&m.heads[i].task, &out[row * per..(row + 1) * per]).map_err(RunError::Run);
                    let timing = Timing { post_ms: t.elapsed().as_secs_f64() * 1000.0, ..base };
                    j.req.send(Update::Head { head: m.heads[i].id.clone(), result: res, timing });
                }
            }
            Err(RunFailure::Error(e)) => {
                for &i in &j.heads {
                    j.req.send(Update::Head {
                        head: m.heads[i].id.clone(),
                        result: Err(RunError::Run(e.clone())),
                        timing: base,
                    });
                }
            }
            Ok(_) | Err(RunFailure::Stopped) => {}
        }
        j.req.graph_done();
    }
}

#[cfg(test)]
pub(crate) mod fake {
    //! A loader whose graphs answer instantly or slowly, record what ran, and can hold a load until released.

    use super::*;
    use crate::manifest::{
        Calibration, ClassifyTask, Decision, Ep, EpKind, Filter, HeadSpec, InputSpec, Labels, OptLevel, Precision,
        Prob, Readout, RuntimeSpec,
    };
    use crate::registry::Graph as RegGraph;
    use std::collections::HashSet;
    use std::path::PathBuf;

    /// A graph `id` with one head `id`, a 4x4 input, two classes and `max_batch`.
    pub fn manifest(id: &str, max_batch: u16) -> Arc<Manifest> {
        let labels = Arc::new(Labels::new(vec!["dark".into(), "light".into()]).unwrap());
        let task = Task::Classify(ClassifyTask {
            readout: Readout::Probs { output: "probs".into() },
            labels,
            calibration: Calibration::Temperature { t: 1.0, in_graph: true },
            decision: Decision::Abstain { below: Prob::new(0.5).unwrap() },
        });
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
            heads: vec![HeadSpec { id: HeadId::new(id), title: id.to_uppercase(), short: id.into(), task }],
            runtime: RuntimeSpec { opt_level: OptLevel::All, providers: vec![Ep { kind: EpKind::Cpu, options: vec![] }] },
        })
    }

    pub fn registry(graphs: &[(&str, u16)], default: &[&str]) -> Arc<Registry> {
        let graphs = graphs.iter().map(|&(id, max)| RegGraph { manifest: manifest(id, max), availability: Ok(()) });
        Arc::new(
            Registry::from_graphs(PathBuf::from("/fake/models.json"), graphs.collect(), default.iter().map(|d| HeadId::new(d)).collect(), vec![])
                .unwrap(),
        )
    }

    /// An image whose every channel is `v`; the fake graph sees `v / 255` as its first input value.
    pub fn image(v: u8) -> ImageInput {
        ImageInput::Decoded(Arc::new(RgbImage::from_pixel(4, 4, image::Rgb([v, v, v]))))
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum Seen {
        Started { graph: String, n: usize },
        /// first input value of each row
        Ran { graph: String, rows: Vec<u8> },
        Stopped { graph: String },
    }

    #[derive(Default)]
    pub struct Gate(Mutex<bool>, Condvar);

    impl Gate {
        pub fn open(&self) {
            *self.0.lock().unwrap() = true;
            self.1.notify_all();
        }
        fn wait(&self) {
            let mut open = self.0.lock().unwrap();
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
        pub seen: Arc<Mutex<Vec<Seen>>>,
        /// graphs dropped so far
        pub drops: Arc<AtomicUsize>,
    }

    impl FakeLoader {
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
}

#[cfg(test)]
mod tests {
    use super::fake::*;
    use super::*;
    use futures::channel::mpsc::TryRecvError;

    const WAIT: Duration = Duration::from_secs(5);

    /// Next update, or panic after `WAIT`.
    fn next(h: &mut RequestHandle) -> Update {
        let t0 = Instant::now();
        loop {
            match h.updates.try_recv() {
                Ok(u) => return u,
                Err(TryRecvError::Closed) => panic!("request stream ended without Done"),
                Err(TryRecvError::Empty) if t0.elapsed() < WAIT => std::thread::sleep(Duration::from_millis(1)),
                Err(TryRecvError::Empty) => panic!("no update within {WAIT:?}"),
            }
        }
    }

    /// Head results and the finish of one request.
    fn finish(h: &mut RequestHandle) -> (Vec<(HeadId, Result<HeadOutput, RunError>)>, Finish) {
        let mut heads = Vec::new();
        loop {
            match next(h) {
                Update::Decoded { .. } => {}
                Update::Head { head, result, .. } => heads.push((head, result)),
                Update::Done(f) => return (heads, f),
            }
        }
    }

    fn wait_until(what: &str, cond: impl Fn() -> bool) {
        let t0 = Instant::now();
        while !cond() {
            assert!(t0.elapsed() < WAIT, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn ids(v: &[&str]) -> Vec<HeadId> {
        v.iter().map(|s| HeadId::new(s)).collect()
    }

    #[test]
    fn a_request_streams_decoded_then_heads_then_done() {
        let (inf, _ev) = Inference::spawn(registry(&[("x", 1)], &["x"]), FakeLoader::default());
        let mut h = inf.lane(Priority::Interactive, Policy::Keep).submit(image(200), &ids(&["x", "nope"]), Some(2));
        assert!(matches!(next(&mut h), Update::Decoded { size: (4, 4), display: Some(d) } if d.width() == 2));
        let (heads, f) = finish(&mut h);
        assert!(matches!(f, Finish::Complete { .. }), "{f:?}");
        assert!(matches!(&heads[0], (id, Err(RunError::Unavailable(_))) if id.as_str() == "nope"));
        let Some((_, Ok(HeadOutput::Classify(c)))) = heads.iter().find(|h| h.0.as_str() == "x") else { panic!() };
        assert_eq!(c.best_label(), "light");
    }

    #[test]
    fn superseded_requests_never_complete() {
        let fake = FakeLoader { run: Duration::from_millis(80), ..Default::default() };
        let (inf, _ev) = Inference::spawn(registry(&[("x", 1)], &["x"]), fake.clone());
        let lane = inf.lane(Priority::Interactive, Policy::Latest);
        let mut a = lane.submit(image(10), &ids(&["x"]), None);
        wait_until("the first run to start", || fake.seen().iter().any(|s| matches!(s, Seen::Started { .. })));
        let mut b = lane.submit(image(20), &ids(&["x"]), None);
        let mut c = lane.submit(image(30), &ids(&["x"]), None);
        assert!(matches!(finish(&mut a).1, Finish::Superseded));
        assert!(matches!(finish(&mut b).1, Finish::Superseded));
        let (heads, f) = finish(&mut c);
        assert!(matches!(f, Finish::Complete { .. }) && heads.len() == 1);
        let ran: Vec<Seen> = fake.seen().into_iter().filter(|s| matches!(s, Seen::Ran { .. })).collect();
        assert_eq!(ran, [Seen::Ran { graph: "x".into(), rows: vec![30] }], "only the latest submit ran to the end");
        assert!(fake.seen().contains(&Seen::Stopped { graph: "x".into() }), "the running one was stopped");
    }

    #[test]
    fn a_slow_load_does_not_delay_other_graphs() {
        let gate = Arc::new(Gate::default());
        let fake = FakeLoader { gates: [("slow".to_string(), gate.clone())].into(), ..Default::default() };
        let (inf, _ev) = Inference::spawn(registry(&[("slow", 1), ("fast", 1)], &["fast"]), fake);
        let lane = inf.lane(Priority::Interactive, Policy::Keep);
        let mut slow = lane.submit(image(1), &ids(&["slow"]), None);
        let mut fast = lane.submit(image(2), &ids(&["fast"]), None);
        assert!(matches!(finish(&mut fast).1, Finish::Complete { .. }), "fast finished while slow is still loading");
        assert!(matches!(next(&mut slow), Update::Decoded { .. }));
        assert!(matches!(slow.updates.try_recv(), Err(TryRecvError::Empty)), "slow has no result yet");
        gate.open();
        assert!(matches!(finish(&mut slow).1, Finish::Complete { .. }));
    }

    #[test]
    fn queued_bulk_jobs_form_one_batch() {
        let gate = Arc::new(Gate::default());
        let fake = FakeLoader { gates: [("z".to_string(), gate.clone())].into(), ..Default::default() };
        let (inf, _ev) = Inference::spawn(registry(&[("z", 8)], &["z"]), fake.clone());
        let lane = inf.lane(Priority::Bulk, Policy::Keep);
        let mut hs: Vec<RequestHandle> = (0..8).map(|v| lane.submit(image(v * 30), &ids(&["z"]), None)).collect();
        // the first job is held by the loading actor, the other seven wait in its queue
        wait_until("seven queued jobs", || inf.queued(&GraphId::new("z")) == 7);
        gate.open();
        for h in &mut hs {
            assert!(matches!(finish(h).1, Finish::Complete { .. }));
        }
        let mut runs: Vec<Vec<u8>> =
            fake.seen().into_iter().filter_map(|s| if let Seen::Ran { rows, .. } = s { Some(rows) } else { None }).collect();
        assert_eq!(runs.len(), 1, "one run for all eight: {runs:?}");
        runs[0].sort();
        assert_eq!(runs[0], (0..8).map(|v| v * 30).collect::<Vec<u8>>(), "two io workers may reorder, none is lost");
    }

    #[test]
    fn a_run_stops_only_when_its_whole_batch_is_dead() {
        let gate = Arc::new(Gate::default());
        let fake = FakeLoader { run: Duration::from_millis(150), gates: [("z".to_string(), gate.clone())].into(), ..Default::default() };
        let (inf, _ev) = Inference::spawn(registry(&[("z", 8)], &["z"]), fake.clone());
        let lane = inf.lane(Priority::Bulk, Policy::Keep);
        let started = |k: usize| fake.seen().iter().filter(|s| matches!(s, Seen::Started { n: 2, .. })).count() >= k;

        let a = lane.submit(image(1), &ids(&["z"]), None);
        let mut b = lane.submit(image(2), &ids(&["z"]), None);
        wait_until("one queued job", || inf.queued(&GraphId::new("z")) == 1);
        gate.open();
        wait_until("a batch of two to start", || started(1));
        let RequestHandle { updates: mut a_updates, ticket } = a;
        drop(ticket);
        assert!(matches!(finish(&mut b).1, Finish::Complete { .. }), "b survives a's cancel");
        assert!(fake.seen().contains(&Seen::Ran { graph: "z".into(), rows: vec![1, 2] }), "the run was not stopped");
        let a_done = loop {
            match a_updates.try_recv() {
                Ok(Update::Done(f)) => break f,
                Ok(_) => {}
                Err(e) => panic!("a has no Done: {e:?}"),
            }
        };
        assert!(matches!(a_done, Finish::Cancelled));

        // while one job runs, two more queue up and then run together; cancelling both stops that run
        let mut blocker = lane.submit(image(3), &ids(&["z"]), None);
        wait_until("the blocker to start", || fake.seen().iter().any(|s| matches!(s, Seen::Started { n: 1, .. })));
        let c = lane.submit(image(4), &ids(&["z"]), None);
        let d = lane.submit(image(5), &ids(&["z"]), None);
        wait_until("c and d to queue", || inf.queued(&GraphId::new("z")) == 2);
        assert!(matches!(finish(&mut blocker).1, Finish::Complete { .. }));
        wait_until("c and d to run together", || started(2));
        assert!(!fake.seen().iter().any(|s| matches!(s, Seen::Stopped { .. })));
        drop(c);
        drop(d);
        wait_until("the run to stop", || fake.seen().iter().any(|s| matches!(s, Seen::Stopped { .. })));
    }

    #[test]
    fn shutdown_waits_for_every_graph_to_let_go() {
        let fake = FakeLoader { run: Duration::from_millis(50), ..Default::default() };
        let (inf, _ev) = Inference::spawn(registry(&[("x", 1), ("y", 1)], &["x"]), fake.clone());
        let lane = inf.lane(Priority::Bulk, Policy::Keep);
        let _a = lane.submit(image(1), &ids(&["x"]), None);
        let _b = lane.submit(image(2), &ids(&["y"]), None);
        wait_until("both graphs to run", || fake.seen().iter().filter(|s| matches!(s, Seen::Started { .. })).count() == 2);
        assert!(inf.shutdown(WAIT), "every worker ended");
        assert_eq!(fake.dropped(), 2, "both graphs were released before shutdown returned");
        let mut late = lane.submit(image(3), &ids(&["x"]), None);
        assert!(matches!(finish(&mut late).1, Finish::Failed(_)), "a stopped service refuses work");
    }

    #[test]
    fn the_default_skips_a_head_whose_provider_is_missing() {
        let fake = FakeLoader { no_provider: ["fast".to_string()].into(), ..Default::default() };
        let (_inf, mut ev) = Inference::spawn(registry(&[("fast", 1), ("deployed", 1)], &["fast", "deployed"]), fake);
        let t0 = Instant::now();
        let (head, passed_over) = loop {
            match ev.try_recv() {
                Ok(ServiceEvent::Default { head, passed_over }) => break (head, passed_over),
                Ok(_) => {}
                Err(_) if t0.elapsed() < WAIT => std::thread::sleep(Duration::from_millis(1)),
                Err(e) => panic!("no Default event: {e:?}"),
            }
        };
        assert_eq!(head.as_str(), "deployed");
        assert_eq!(passed_over.len(), 1);
        assert!(passed_over[0].1.contains("TensorRT"), "{passed_over:?}");
    }
}
