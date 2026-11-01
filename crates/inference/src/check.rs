//! What `models` reports about every graph of an index: which of its execution providers register here, and, when
//! asked, whether its files match their checksums and the graph loads. The app only prints the report.

use std::path::PathBuf;
use std::time::Instant;

use crate::integrity::{self, Stamps};
use crate::manifest::{EpKind, GraphId, Precision, Sha256};
use crate::registry::{GraphEntry, Registry, Unavailable};
use crate::runtime::{OrtLoader, Settings};
use crate::service::{Loaded, Loader as _};

pub struct Report {
    /// onnxruntime's version ("1.31.1") and library file, or why it could not be loaded; then `graphs` is empty
    pub runtime: Result<(String, PathBuf), String>,
    /// in the index's order
    pub graphs: Vec<GraphReport>,
    /// why the checksum stamps could not be saved
    pub stamps_error: Option<String>,
}

pub struct GraphReport {
    pub id: GraphId,
    pub precision: Precision,
    /// the providers the graph would try, in order, and whether each registers here (a probe loads the provider's
    /// library and builds no session)
    pub providers: Vec<(EpKind, Result<(), String>)>,
    /// only when asked to verify
    pub verified: Option<Verification>,
}

/// One graph's files hashed against its manifest, then the graph loaded the way the app loads it.
pub enum Verification {
    Unavailable(Unavailable),
    /// every file matched, in `secs`; then the load, or why it failed
    Hashed { bytes: u64, secs: f64, load: Result<Loaded, String> },
    Mismatch { file: PathBuf, expected: Sha256, found: Sha256 },
    Unreadable { file: PathBuf, error: String },
}

impl Verification {
    /// The files matched and the graph loaded.
    pub fn ok(&self) -> bool {
        matches!(self, Verification::Hashed { load: Ok(_), .. })
    }
}

/// Probe every graph's execution providers. With `verify`, then hash every graph's files and load the graph. Files
/// that match get a stamp in `settings.cache_dir`, so later loads know they are verified.
pub fn models(registry: &Registry, settings: &Settings, verify: bool) -> Report {
    let loader = OrtLoader::new(settings, registry);
    let rt = match loader.runtime() {
        Ok(rt) => rt,
        Err(e) => return Report { runtime: Err(e), graphs: Vec::new(), stamps_error: None },
    };
    let mut graphs: Vec<GraphReport> = registry
        .graphs()
        .iter()
        .map(|g| {
            let m = &g.manifest;
            let providers = rt.preferences(m).iter().map(|ep| (ep.kind, rt.probe(ep, m))).collect();
            GraphReport { id: m.id.clone(), precision: m.precision, providers, verified: None }
        })
        .collect();
    let mut stamps_error = None;
    if verify {
        let mut stamps = Stamps::load(settings.cache_dir.as_deref());
        for (report, g) in graphs.iter_mut().zip(registry.graphs()) {
            report.verified = Some(verification(g, &loader, &mut stamps));
        }
        stamps_error = stamps.save().err().map(|e| e.to_string());
    }
    Report { runtime: Ok((rt.version.clone(), rt.ort_lib.clone())), graphs, stamps_error }
}

fn verification(g: &GraphEntry, loader: &OrtLoader, stamps: &mut Stamps) -> Verification {
    if let Err(e) = &g.availability {
        return Verification::Unavailable(e.clone());
    }
    let t0 = Instant::now();
    match integrity::check(&g.manifest, stamps) {
        integrity::Check::Ok { bytes } => {
            let secs = t0.elapsed().as_secs_f64();
            let load = loader.load(&g.manifest, &|_| {}).map(|(_, loaded)| loaded);
            Verification::Hashed { bytes, secs, load }
        }
        integrity::Check::Mismatch { file, expected, found } => Verification::Mismatch { file, expected, found },
        integrity::Check::Unreadable { file, error } => Verification::Unreadable { file, error },
    }
}
