//! The model registry: the index (`exports/models.json`) and the manifests it lists, as pure data. No onnxruntime
//! here: whether a graph's files are present is a filesystem fact checked at load; how it runs is `runtime`'s job.
//!
//! Found like this: `--models PATH` (the index or its folder), then `$COMPCLASS_MODELS`, then the first
//! `exports/models.json` walking up from the current directory and from the executable.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::manifest::{self, GraphId, HeadId, HeadSpec, Manifest, ManifestError};

pub const INDEX_FILE: &str = "models.json";

/// Why a graph cannot be loaded although its manifest parsed.
#[derive(Debug, Clone, PartialEq)]
pub enum Unavailable {
    Missing(Vec<PathBuf>),
    Size { file: PathBuf, expected: u64, found: u64 },
}

impl fmt::Display for Unavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Unavailable::Missing(files) => {
                let names: Vec<String> = files.iter().map(|p| p.display().to_string()).collect();
                write!(f, "not exported or not downloaded: {} missing", names.join(", "))
            }
            Unavailable::Size { file, expected, found } => {
                write!(f, "{} is {found} bytes, the manifest says {expected} (incomplete copy?)", file.display())
            }
        }
    }
}

#[derive(Debug)]
pub enum IndexError {
    NotFound { searched: Vec<PathBuf> },
    Manifest(ManifestError),
    DuplicateHead { head: HeadId, first: PathBuf, second: PathBuf },
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexError::NotFound { searched } => {
                write!(f, "no exports/{INDEX_FILE} found (looked in ")?;
                let shown: Vec<String> = searched.iter().take(4).map(|p| p.display().to_string()).collect();
                write!(f, "{}{}); pass --models <exports folder>", shown.join(", "), if searched.len() > 4 { ", ..." } else { "" })
            }
            IndexError::Manifest(e) => write!(f, "{e}"),
            IndexError::DuplicateHead { head, first, second } => {
                write!(f, "head id {head:?} is used by both {} and {}", first.display(), second.display())
            }
        }
    }
}

impl std::error::Error for IndexError {}

/// One graph: its manifest and whether its files are all there with the recorded sizes.
#[derive(Debug)]
pub struct Graph {
    pub manifest: Arc<Manifest>,
    pub availability: Result<(), Unavailable>,
}

/// A head together with the graph serving it.
#[derive(Clone, Copy)]
pub struct Head<'r> {
    pub spec: &'r HeadSpec,
    pub graph: &'r Graph,
}

impl<'r> Head<'r> {
    pub fn id(&self) -> &'r HeadId {
        &self.spec.id
    }
    pub fn manifest(&self) -> &'r Arc<Manifest> {
        &self.graph.manifest
    }
    pub fn available(&self) -> Result<(), &'r Unavailable> {
        self.graph.availability.as_ref().map(|_| ())
    }
}

#[derive(Debug)]
pub struct Registry {
    index: PathBuf,
    graphs: Vec<Graph>,
    /// (graph, head) in UI order: manifest order, then head order
    heads: Vec<(usize, usize)>,
    default: Vec<HeadId>,
    /// manifests that did not parse and default ids that name no head; shown, never fatal
    problems: Vec<String>,
}

impl Registry {
    /// `--models`, then `$COMPCLASS_MODELS`, then walk up for `exports/models.json`.
    pub fn locate(explicit: Option<&Path>) -> Result<PathBuf, IndexError> {
        let as_index = |p: &Path| if p.is_dir() { p.join(INDEX_FILE) } else { p.to_path_buf() };
        if let Some(p) = explicit {
            return Ok(as_index(p));
        }
        if let Some(p) = std::env::var_os("COMPCLASS_MODELS").filter(|s| !s.is_empty()) {
            return Ok(as_index(Path::new(&p)));
        }
        let searched: Vec<PathBuf> = search_bases().into_iter().map(|b| b.join("exports").join(INDEX_FILE)).collect();
        searched.iter().find(|p| p.is_file()).cloned().ok_or(IndexError::NotFound { searched })
    }

    pub fn load(index: &Path) -> Result<Self, IndexError> {
        let idx = manifest::parse_index(index).map_err(IndexError::Manifest)?;
        let mut problems = Vec::new();
        let mut graphs = Vec::new();
        for p in &idx.manifests {
            match manifest::parse_manifest(p) {
                Ok(m) => {
                    let availability = availability(&m);
                    graphs.push(Graph { manifest: Arc::new(m), availability });
                }
                Err(e) => problems.push(e.to_string()),
            }
        }
        Self::from_graphs(index.to_path_buf(), graphs, idx.default, problems)
    }

    pub fn from_graphs(
        index: PathBuf,
        graphs: Vec<Graph>,
        default: Vec<HeadId>,
        mut problems: Vec<String>,
    ) -> Result<Self, IndexError> {
        let mut heads: Vec<(usize, usize)> = Vec::new();
        for (gi, g) in graphs.iter().enumerate() {
            for (hi, h) in g.manifest.heads.iter().enumerate() {
                if let Some(&(og, _)) = heads.iter().find(|&&(og, oh)| graphs[og].manifest.heads[oh].id == h.id) {
                    return Err(IndexError::DuplicateHead {
                        head: h.id.clone(),
                        first: graphs[og].manifest.path.clone(),
                        second: g.manifest.path.clone(),
                    });
                }
                heads.push((gi, hi));
            }
        }
        let mut reg = Self { index, graphs, heads, default: Vec::new(), problems: Vec::new() };
        for d in default {
            if reg.head(&d).is_some() {
                reg.default.push(d);
            } else {
                problems.push(format!("{}: default head {d:?} is not in any manifest", reg.index.display()));
            }
        }
        reg.problems = problems;
        Ok(reg)
    }

    pub fn index_path(&self) -> &Path {
        &self.index
    }

    pub fn heads(&self) -> impl Iterator<Item = Head<'_>> + '_ {
        self.heads.iter().map(|&(g, h)| Head { spec: &self.graphs[g].manifest.heads[h], graph: &self.graphs[g] })
    }

    pub fn head(&self, id: &HeadId) -> Option<Head<'_>> {
        self.heads().find(|h| h.id() == id)
    }

    /// A head named on the command line (ids are matched case-insensitively).
    pub fn find(&self, name: &str) -> Option<HeadId> {
        let name = name.trim();
        self.heads().find(|h| h.id().as_str().eq_ignore_ascii_case(name)).map(|h| h.id().clone())
    }

    pub fn ids(&self) -> Vec<HeadId> {
        self.heads().map(|h| h.id().clone()).collect()
    }

    pub fn graphs(&self) -> &[Graph] {
        &self.graphs
    }

    pub fn graph(&self, id: &GraphId) -> Option<&Graph> {
        self.graphs.iter().find(|g| &g.manifest.id == id)
    }

    /// Heads served by one forward pass of the same graph as `id` (including `id`).
    pub fn siblings(&self, id: &HeadId) -> Vec<HeadId> {
        match self.head(id) {
            Some(h) => h.manifest().heads.iter().map(|s| s.id.clone()).collect(),
            None => vec![id.clone()],
        }
    }

    /// Default heads in order of preference (only ids that exist).
    pub fn default_order(&self) -> &[HeadId] {
        &self.default
    }

    /// The first default head, else the first head.
    pub fn first_default(&self) -> Option<HeadId> {
        self.default.first().cloned().or_else(|| self.heads().next().map(|h| h.id().clone()))
    }

    pub fn problems(&self) -> &[String] {
        &self.problems
    }
}

/// Every listed file exists with the recorded size (a cheap stat; checksums are `models --check`).
fn availability(m: &Manifest) -> Result<(), Unavailable> {
    let mut missing = Vec::new();
    for f in &m.files {
        match std::fs::metadata(&f.path) {
            Ok(md) if md.len() == f.bytes => {}
            Ok(md) => return Err(Unavailable::Size { file: f.path.clone(), expected: f.bytes, found: md.len() }),
            Err(_) => missing.push(f.path.clone()),
        }
    }
    if missing.is_empty() { Ok(()) } else { Err(Unavailable::Missing(missing)) }
}

/// The current directory and the executable's directory, plus their ancestors.
pub fn search_bases() -> Vec<PathBuf> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn shipped() -> Registry {
        Registry::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../exports").join(INDEX_FILE)).unwrap()
    }

    #[test]
    fn shipped_registry_orders_heads_and_groups_siblings() {
        let r = shipped();
        assert!(r.problems().is_empty(), "{:?}", r.problems());
        let ids: Vec<String> = r.ids().iter().map(|h| h.to_string()).collect();
        assert_eq!(&ids[..4], ["a", "b", "c", "deployed"], "manifest order, then head order");
        let c = r.find("C").unwrap();
        let sib: Vec<String> = r.siblings(&c).iter().map(|h| h.to_string()).collect();
        assert_eq!(sib, ["c", "deployed"], "C and Deployed share one graph");
        assert_eq!(r.siblings(&r.find("a").unwrap()).len(), 1);
        assert!(r.default_order().iter().any(|h| h.as_str() == "deployed"));
        assert_eq!(r.find("nope"), None);
    }

    #[test]
    fn missing_or_truncated_files_make_a_graph_unavailable() {
        let tmp = std::env::temp_dir().join(format!("gpui-demo-registry-{}", std::process::id()));
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../exports/itri626/A_scratch_cnn");
        let dir = tmp.join("A");
        std::fs::create_dir_all(&dir).unwrap();
        for f in ["model.json", "labels.json"] {
            std::fs::copy(src.join(f), dir.join(f)).unwrap();
        }
        std::fs::write(tmp.join(INDEX_FILE), r#"{"schema":"model-index/1","default":["a","x"],"manifests":["A/model.json","gone/model.json"]}"#)
            .unwrap();
        let r = Registry::load(&tmp.join(INDEX_FILE)).unwrap();
        assert_eq!(r.problems().len(), 2, "a manifest that does not parse and an unknown default: {:?}", r.problems());
        let a = r.head(&r.find("a").unwrap()).unwrap();
        assert!(matches!(a.available(), Err(Unavailable::Missing(f)) if f == &[dir.join("model.onnx")]));
        std::fs::write(dir.join("model.onnx"), b"truncated").unwrap();
        let r = Registry::load(&tmp.join(INDEX_FILE)).unwrap();
        let a = r.head(&r.find("a").unwrap()).unwrap();
        assert!(matches!(a.available(), Err(Unavailable::Size { found: 9, .. })));
        let _ = std::fs::remove_dir_all(&tmp); // test scratch dir under the OS temp dir only
    }

    #[test]
    fn duplicate_head_ids_are_an_error() {
        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../exports/itri626/A_scratch_cnn/model.json");
        let g = || Graph { manifest: Arc::new(manifest::parse_manifest(&base).unwrap()), availability: Ok(()) };
        let e = Registry::from_graphs(PathBuf::from("i"), vec![g(), g()], vec![], vec![]).unwrap_err();
        assert!(matches!(e, IndexError::DuplicateHead { .. }));
    }
}
