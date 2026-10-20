//! Background threads. The UI thread never decodes, loads or runs a model.
//!
//! * `inference`: owns the model registry (lazy onnxruntime sessions). Serves commands one at a time: preload a
//!   model, or classify one image with one or more models (read/fetch -> decode -> display copy -> predict).
//! * `thumbnails`: decodes the demo stack once at startup (rayon), producing display-size textures and the
//!   decoded RGB that later classify jobs reuse.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use futures::channel::mpsc as fmpsc;
use gpui_kit::RenderImage;
use image::RgbImage;
use rayon::prelude::*;

use crate::contract::Prediction;
use crate::fetch::Source;
use crate::model::ModelOptions;
use crate::models::{ModelKey, Models};
use crate::preprocess;

pub const DISPLAY_MAX_SIDE: u32 = 1280;
/// Demo grid thumbnails: decoded once at 2x the card size (sharp on HiDPI, cheap to sample).
pub const THUMB_MAX_SIDE: u32 = 360;

pub enum JobInput {
    Source(Source),
    Bytes(Vec<u8>),
    /// already decoded (demo stack): skips read + decode
    Decoded(Arc<RgbImage>),
}

pub struct Job {
    pub input: JobInput,
    pub models: Vec<ModelKey>,
    /// send a display texture of the decoded picture first
    pub display: bool,
    pub cancel: Arc<AtomicBool>,
    pub updates: fmpsc::UnboundedSender<Update>,
}

pub enum Update {
    Image(Arc<RenderImage>, (u32, u32)),
    Result(ModelKey, Result<Prediction, String>),
    /// the input could not be read or decoded
    Failed(String),
    /// all results sent; total wall time of the job in ms
    Done(f64),
}

#[derive(Clone, Debug)]
pub enum ModelState {
    Unavailable(String),
    Idle,
    Loading(Instant),
    Ready(f64),
    Failed(String),
}

impl ModelState {
    pub fn usable(&self) -> bool {
        !matches!(self, ModelState::Unavailable(_) | ModelState::Failed(_))
    }
}

pub enum Event {
    /// model registry discovered (or the export folder is missing), with the class names
    Registry(Result<Registry, String>),
    Model(ModelKey, ModelState),
    /// onnxruntime library path and execution provider (after the first model load)
    Runtime(String, String),
}

pub struct Registry {
    pub models: Vec<(ModelKey, Result<String, String>)>,
    pub labels: Vec<String>,
    pub backbone: Option<String>,
}

pub enum Cmd {
    Preload(ModelKey),
    Job(Job),
}

pub fn spawn(opts: ModelOptions) -> (mpsc::Sender<Cmd>, fmpsc::UnboundedReceiver<Event>) {
    let (tx, rx) = mpsc::channel::<Cmd>();
    let (etx, erx) = fmpsc::unbounded();
    std::thread::Builder::new()
        .name("inference".into())
        .spawn(move || run(opts, rx, etx))
        .expect("spawn inference thread");
    (tx, erx)
}

fn run(opts: ModelOptions, rx: mpsc::Receiver<Cmd>, events: fmpsc::UnboundedSender<Event>) {
    let mut models = match Models::new(&opts) {
        Ok(m) => {
            let list = m.entries.iter().map(|(k, s)| (*k, s.as_ref().map(|s| s.readout_name()).map_err(Clone::clone)));
            let (labels, backbone) = m.labels();
            let _ = events.unbounded_send(Event::Registry(Ok(Registry { models: list.collect(), labels, backbone })));
            Some(m)
        }
        Err(e) => {
            let msg = format!("{e:#}");
            eprintln!("[gpui-demo] {msg}");
            let _ = events.unbounded_send(Event::Registry(Err(msg)));
            None
        }
    };
    let mut runtime_sent = false;
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Cmd::Preload(key) => {
                if let Some(m) = models.as_mut() {
                    ensure(m, key, &events);
                }
            }
            Cmd::Job(job) => run_job(models.as_mut(), job, &events),
        }
        if !runtime_sent && let Some((lib, provider)) = models.as_ref().and_then(Models::runtime_info) {
            runtime_sent = true;
            let _ = events.unbounded_send(Event::Runtime(lib, provider));
        }
    }
}

fn run_job(models: Option<&mut Models>, job: Job, events: &fmpsc::UnboundedSender<Event>) {
    let t0 = Instant::now();
    let done = |job: &Job| {
        let _ = job.updates.unbounded_send(Update::Done(t0.elapsed().as_secs_f64() * 1000.0));
    };
    let rgb = match &job.input {
        JobInput::Decoded(rgb) => Ok(rgb.clone()),
        JobInput::Source(s) => {
            prof!("job::read", s.read()).and_then(|b| prof!("job::decode", preprocess::decode(&b))).map(Arc::new)
        }
        JobInput::Bytes(b) => prof!("job::decode_bytes", preprocess::decode(b)).map(Arc::new),
    };
    let rgb = match rgb {
        Ok(r) => r,
        Err(e) => {
            let _ = job.updates.unbounded_send(Update::Failed(format!("{e:#}")));
            return done(&job);
        }
    };
    if job.display {
        let tex = prof!("job::display_texture", texture(&rgb, DISPLAY_MAX_SIDE));
        let _ = job.updates.unbounded_send(Update::Image(tex, rgb.dimensions()));
    }
    let Some(m) = models else {
        for &k in &job.models {
            let _ = job.updates.unbounded_send(Update::Result(k, Err("model folder not found".into())));
        }
        return done(&job);
    };
    for &k in &job.models {
        if job.cancel.load(Ordering::Relaxed) {
            break;
        }
        ensure(m, k, events);
    }
    let cancel = job.cancel.clone();
    let runs = prof!("job::predict", m.predict(&rgb, &job.models, &|| cancel.load(Ordering::Relaxed)));
    for r in runs {
        let _ = job.updates.unbounded_send(Update::Result(r.key, r.result));
    }
    done(&job)
}

/// Load `key` (and report every registry entry that shares its graph) unless it is already loaded.
fn ensure(m: &mut Models, key: ModelKey, events: &fmpsc::UnboundedSender<Event>) {
    if m.is_loaded(key) || m.spec(key).is_err() {
        return;
    }
    let siblings = m.siblings(key);
    for &k in &siblings {
        let _ = events.unbounded_send(Event::Model(k, ModelState::Loading(Instant::now())));
    }
    let state = match m.ensure(key) {
        Ok(ms) => {
            eprintln!("[gpui-demo] {} loaded in {ms:.0} ms", key.name());
            ModelState::Ready(ms)
        }
        Err(e) => {
            eprintln!("[gpui-demo] {} failed to load: {e}", key.name());
            ModelState::Failed(e)
        }
    };
    for k in siblings {
        let _ = events.unbounded_send(Event::Model(k, state.clone()));
    }
}

/// BGRA texture for GPUI, long side <= `max_side`.
pub fn texture(rgb: &RgbImage, max_side: u32) -> Arc<RenderImage> {
    let frame = image::Frame::new(preprocess::display_bgra(rgb, max_side));
    Arc::new(RenderImage::new(smallvec::smallvec![frame]))
}

/// One decoded demo image.
pub struct Decoded {
    pub rgb: Arc<RgbImage>,
    pub thumb: Arc<RenderImage>,
    pub full: Arc<RenderImage>,
}

/// Decode every demo image in parallel, once; results arrive in completion order as (index, result).
pub fn load_thumbnails(paths: Vec<std::path::PathBuf>) -> fmpsc::UnboundedReceiver<(usize, Result<Decoded, String>)> {
    let (tx, rx) = fmpsc::unbounded();
    std::thread::Builder::new()
        .name("thumbnails".into())
        .spawn(move || {
            let t0 = Instant::now();
            paths.into_par_iter().enumerate().for_each(|(i, p)| {
                let r = prof!("demo::decode_and_thumbnail", decode_one(&p));
                let _ = tx.unbounded_send((i, r));
            });
            eprintln!("[gpui-demo] demo stack decoded in {:.0} ms", t0.elapsed().as_secs_f64() * 1000.0);
        })
        .expect("spawn thumbnail thread");
    rx
}

fn decode_one(p: &std::path::Path) -> Result<Decoded, String> {
    let bytes = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let rgb = preprocess::decode(&bytes).map_err(|e| format!("{}: {e:#}", p.display()))?;
    let thumb = texture(&rgb, THUMB_MAX_SIDE);
    let full = texture(&rgb, DISPLAY_MAX_SIDE);
    Ok(Decoded { rgb: Arc::new(rgb), thumb, full })
}
