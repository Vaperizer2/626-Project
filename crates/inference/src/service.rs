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
/// Bulk requests between the start of their preparation and the end of their last run. Their tensors wait in the
/// graphs' queues, so without a bound a `classify` of thousands of pictures prepares them all at decode speed
/// (deployed's tensor is 1.8 MB). Enough to fill a batch of 8 twice over.
const BULK_IN_FLIGHT: usize = 16;

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
    /// its files), else the first available head; None when no head can run here. Sent once, always.
    Default { head: Option<HeadId>, passed_over: Vec<(HeadId, String)> },
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

#[derive(Clone)]
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
/// Two-class FIFO: interactive items before bulk ones. `pop` blocks; it returns None once closed. With a bulk
/// limit, `pop` hands out a bulk item only while fewer than that many are out; each is given back with
/// `release_bulk`. Interactive items never wait for it.
struct Queue<T> {
    state: Mutex<Lanes<T>>,
    ready: Condvar,
    bulk_limit: Option<usize>,
}

struct Lanes<T> {
    interactive: VecDeque<T>,
    bulk: VecDeque<T>,
    closed: bool,
    bulk_out: usize,
}

impl<T> Queue<T> {
    fn new() -> Self {
        Self::bounded(None)
    }

    fn bounded(bulk_limit: Option<usize>) -> Self {
        let lanes = Lanes { interactive: VecDeque::new(), bulk: VecDeque::new(), closed: false, bulk_out: 0 };
        Self { state: Mutex::new(lanes), ready: Condvar::new(), bulk_limit }
    }

    fn push(&self, p: Priority, item: T) -> Result<(), T> {
        let mut s = self.state.lock().expect("queue lock");
        if s.closed {
            return Err(item);
        }
        match p {
            Priority::Interactive => s.interactive.push_back(item),
            Priority::Bulk => s.bulk.push_back(item),
        }
        self.ready.notify_one();
        Ok(())
    }

    fn pop(&self) -> Option<T> {
        let mut s = self.state.lock().expect("queue lock");
        loop {
            if s.closed {
                return None;
            }
            if let Some(x) = s.interactive.pop_front() {
                return Some(x);
            }
            if self.bulk_limit.is_none_or(|l| s.bulk_out < l)
                && let Some(x) = s.bulk.pop_front()
            {
                s.bulk_out += usize::from(self.bulk_limit.is_some());
                return Some(x);
            }
            s = self.ready.wait(s).expect("queue lock");
        }
    }

    fn try_pop(&self) -> Option<T> {
        let mut s = self.state.lock().expect("queue lock");
        s.interactive.pop_front().or_else(|| s.bulk.pop_front())
    }

    fn release_bulk(&self) {
        self.state.lock().expect("queue lock").bulk_out -= 1;
        self.ready.notify_one();
    }

    fn close(&self) {
        self.state.lock().expect("queue lock").closed = true;
        self.ready.notify_all();
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        let s = self.state.lock().expect("queue lock");
        s.interactive.len() + s.bulk.len()
    }
}

/// A bulk request's place in the io queue's budget, from its preparation to the end of its last graph job.
struct BulkSlot(Arc<Queue<IoJob>>);

impl Drop for BulkSlot {
    fn drop(&mut self) {
        self.0.release_bulk();
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
    _slot: Option<Arc<BulkSlot>>,
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
    io: Arc<Queue<IoJob>>,
    /// each graph's actor queue; None once the service is closed, so no actor starts after `shutdown` took the
    /// thread list
    actors: Mutex<Option<HashMap<GraphId, Arc<Queue<Msg>>>>>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

const STOPPED: &str = "the inference service has stopped";

impl Core {
    fn close(&self) {
        self.io.close();
        for q in self.actors.lock().expect("actors lock").take().into_iter().flat_map(HashMap::into_values) {
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
            io: Arc::new(Queue::bounded(Some(BULK_IN_FLIGHT))),
            actors: Mutex::new(Some(HashMap::new())),
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
            && let Ok(q) = core.actor(h.manifest())
        {
            let _ = q.push(Priority::Interactive, Msg::Preload);
        }
    }

    #[cfg(test)]
    fn queued(&self, graph: &GraphId) -> usize {
        self.owner.0.actors.lock().unwrap().as_ref().and_then(|a| a.get(graph)).map_or(0, |q| q.len())
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
            req.fail(RunError::Run(STOPPED.into()));
        }
        RequestHandle { updates, ticket: Ticket(cancelled) }
    }

}

impl Core {
    /// The queue of `m`'s actor, spawning it on first use; an error once the service is closed.
    fn actor(&self, m: &Arc<Manifest>) -> Result<Arc<Queue<Msg>>, String> {
        let mut actors = self.actors.lock().expect("actors lock");
        let actors = actors.as_mut().ok_or(STOPPED)?;
        if let Some(q) = actors.get(&m.id) {
            return Ok(q.clone());
        }
        let q = Arc::new(Queue::new());
        let (m2, q2, loader, events) = (m.clone(), q.clone(), self.loader.clone(), self.events.clone());
        let t = std::thread::Builder::new()
            .name(format!("graph-{}", m.id))
            .spawn(move || actor(m2, q2, loader, events))
            .map_err(|e| format!("cannot start a thread for {}: {e}", m.id))?;
        // still under the actors lock: `close` waits for this, so `shutdown` sees the thread
        self.threads.lock().expect("threads lock").push(t);
        actors.insert(m.id.clone(), q.clone());
        Ok(q)
    }

    fn resolve_default(&self) {
        let fallback = self.registry.last_default();
        let mut passed_over = Vec::new();
        let mut head = None;
        for id in self.registry.default_order() {
            let Some(h) = self.registry.head(id) else {
                passed_over.push((id.clone(), "no manifest in the model index provides it".into()));
                continue;
            };
            if let Err(e) = h.available() {
                passed_over.push((id.clone(), e.to_string()));
                continue;
            }
            if Some(id) != fallback.as_ref()
                && let Err(e) = self.loader.probe(h.manifest())
            {
                passed_over.push((id.clone(), e));
                continue;
            }
            head = Some(id.clone());
            break;
        }
        // no listed default can run: any head with its files, except a default just passed over
        let head = head.or_else(|| {
            self.registry
                .heads()
                .find(|h| h.available().is_ok() && !passed_over.iter().any(|(id, _)| id == h.id()))
                .map(|h| h.id().clone())
        });
        let _ = self.events.unbounded_send(ServiceEvent::Default { head, passed_over });
    }

    /// Decode, then one tensor per distinct preprocessing spec, each handed to its graph as soon as it is ready.
    fn prepare(&self, job: IoJob) {
        let IoJob { req, input, heads, display } = job;
        // `pop` counted a bulk job against the budget; its graph jobs give the place back when the last one ends
        let slot = (req.priority == Priority::Bulk).then(|| Arc::new(BulkSlot(self.io.clone())));
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
                _slot: slot.clone(),
            };
            let queued = self.actor(m).and_then(|q| q.push(req.priority, Msg::Job(job)).map_err(|_| STOPPED.to_string()));
            if let Err(e) = queued {
                for &i in ix {
                    req.send(Update::Head {
                        head: m.heads[i].id.clone(),
                        result: Err(RunError::Run(e.clone())),
                        timing: Timing { decode_ms, preprocess_ms, ..Default::default() },
                    });
                }
                req.graph_done();
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

#[cfg(any(test, feature = "test-support"))]
pub mod fake;

#[cfg(test)]
mod tests;
