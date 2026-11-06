//! The demo stack (`demo_stack/demo_stack.json` + images) used by the Demo and Benchmark tabs.
//!
//! Found through `$<ENV_PREFIX>_DEMO_DIR`, else as the first `demo_stack/` (or `apps/<bin>/demo_stack/`) with a
//! `demo_stack.json` found walking up from the current directory and from the executable.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

pub const STACK_FILE: &str = "demo_stack.json";

#[derive(Debug, Clone, Deserialize)]
pub struct DemoImage {
    pub file: String,
    pub section: String,
    /// None = not one of the classes (the correct answer is UNSURE)
    pub true_label: Option<String>,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub author: String,
    #[serde(default)]
    pub licence: String,
    #[serde(default)]
    pub licence_url: String,
    #[serde(default)]
    pub source_url: String,
    #[serde(default)]
    pub note: String,
    /// deployed model's verified prediction (top-1 label) and confidence, for reference
    #[serde(default)]
    pub model_pred: Option<String>,
    #[serde(default)]
    pub model_conf: Option<f32>,
}

impl DemoImage {
    /// Is a prediction (top-1 label, unsure flag) correct for this image? Out-of-scope images are correct when
    /// the model abstains; in-scope images when it commits to the true label.
    pub fn judge(&self, top1: &str, unsure: bool) -> Verdict {
        match &self.true_label {
            None if unsure => Verdict::Correct,
            None => Verdict::Wrong,
            Some(_) if unsure => Verdict::Unsure,
            Some(t) if t == top1 => Verdict::Correct,
            Some(_) => Verdict::Wrong,
        }
    }

    pub fn true_text(&self) -> &str {
        self.true_label.as_deref().unwrap_or("not a component")
    }

    /// "author · licence" for card footers.
    pub fn credit(&self) -> String {
        match (self.author.is_empty(), self.licence.is_empty()) {
            (false, false) => format!("{} · {}", self.author, self.licence),
            (false, true) => self.author.clone(),
            (true, false) => self.licence.clone(),
            (true, true) => String::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Correct,
    Wrong,
    /// in-scope image, model abstained (counts as not correct)
    Unsure,
}

#[derive(Debug, Clone, Deserialize)]
struct StackFile {
    #[serde(default)]
    sections: Vec<String>,
    images: Vec<DemoImage>,
}

#[derive(Debug, Clone)]
pub struct DemoStack {
    pub dir: PathBuf,
    /// section names in display order
    pub sections: Vec<String>,
    pub images: Vec<DemoImage>,
}

impl DemoStack {
    pub fn load(dir: &Path) -> Result<Self> {
        let p = dir.join(STACK_FILE);
        let text = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        Self::parse(dir, &text).with_context(|| format!("parsing {}", p.display()))
    }

    pub fn parse(dir: &Path, json: &str) -> Result<Self> {
        let f: StackFile = serde_json::from_str(json)?;
        if f.images.is_empty() {
            bail!("no images");
        }
        let mut sections = f.sections;
        for im in &f.images {
            if !sections.contains(&im.section) {
                sections.push(im.section.clone());
            }
        }
        // images grouped by section order (stable inside a section)
        let mut images = f.images;
        images.sort_by_key(|im| sections.iter().position(|s| *s == im.section).unwrap_or(usize::MAX));
        Ok(Self { dir: dir.to_path_buf(), sections, images })
    }

    pub fn path(&self, i: usize) -> PathBuf {
        self.dir.join(&self.images[i].file)
    }

    /// Indices of the images in `section`.
    pub fn section_indices(&self, section: &str) -> Vec<usize> {
        (0..self.images.len()).filter(|&i| self.images[i].section == section).collect()
    }
}

fn is_stack(d: &Path) -> bool {
    d.join(STACK_FILE).is_file()
}

/// `$<ENV_PREFIX>_DEMO_DIR` (`env`), else walk up from `starts` looking for `demo_stack/` or `apps/<bin>/demo_stack/`.
pub fn locate_in(env: Option<PathBuf>, starts: &[PathBuf]) -> Result<PathBuf> {
    if let Some(p) = env.filter(|p| !p.as_os_str().is_empty()) {
        if is_stack(&p) {
            return Ok(p);
        }
        bail!("{}_DEMO_DIR={}: no {STACK_FILE} there", crate::app::ENV_PREFIX, p.display());
    }
    for s in starts {
        for a in s.ancestors() {
            for c in [a.join("demo_stack"), a.join("apps").join(crate::app::BIN).join("demo_stack")] {
                if is_stack(&c) {
                    return Ok(c);
                }
            }
        }
    }
    bail!(
        "no demo_stack/{STACK_FILE} found above the current directory or the executable; set {}_DEMO_DIR",
        crate::app::ENV_PREFIX
    )
}

pub fn locate() -> Result<PathBuf> {
    let mut starts = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        starts.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(d) = exe.canonicalize().unwrap_or(exe).parent()
    {
        starts.push(d.to_path_buf());
    }
    locate_in(std::env::var_os(format!("{}_DEMO_DIR", crate::app::ENV_PREFIX)).map(PathBuf::from), &starts)
}

#[cfg(test)]
mod tests {
    use super::*;

    const JSON: &str = r#"{
      "version": 1,
      "sections": ["One per class", "Hard cases"],
      "images": [
        {"file": "h.jpg", "section": "Hard cases", "true_label": "led", "author": "X", "licence": "CC BY 2.0"},
        {"file": "a.jpg", "section": "One per class", "true_label": "resistor", "title": "R", "model_pred": "resistor",
         "model_conf": 0.99, "unknown_field": 1},
        {"file": "x.jpg", "section": "Not a component", "true_label": null}
      ]}"#;

    #[test]
    fn parses_and_orders_sections() {
        let s = DemoStack::parse(Path::new("/d"), JSON).unwrap();
        assert_eq!(s.sections, ["One per class", "Hard cases", "Not a component"]);
        let files: Vec<&str> = s.images.iter().map(|i| i.file.as_str()).collect();
        assert_eq!(files, ["a.jpg", "h.jpg", "x.jpg"]);
        assert_eq!(s.path(0), Path::new("/d/a.jpg"));
        assert_eq!(s.section_indices("Hard cases"), [1]);
        assert_eq!(s.images[0].model_conf, Some(0.99));
        assert_eq!(s.images[1].credit(), "X · CC BY 2.0");
        assert_eq!(s.images[2].true_text(), "not a component");
    }

    #[test]
    fn judges_predictions() {
        let s = DemoStack::parse(Path::new("/d"), JSON).unwrap();
        let (r, oos) = (&s.images[0], &s.images[2]);
        assert_eq!(r.judge("resistor", false), Verdict::Correct);
        assert_eq!(r.judge("diode", false), Verdict::Wrong);
        assert_eq!(r.judge("resistor", true), Verdict::Unsure);
        assert_eq!(oos.judge("capacitor", true), Verdict::Correct);
        assert_eq!(oos.judge("capacitor", false), Verdict::Wrong);
    }

    #[test]
    fn rejects_empty_or_bad() {
        assert!(DemoStack::parse(Path::new("/d"), r#"{"images": []}"#).is_err());
        assert!(DemoStack::parse(Path::new("/d"), "not json").is_err());
    }

    #[test]
    fn locates_by_env_and_walking_up() {
        let tmp = std::env::temp_dir().join(format!("{}-stack-{}", crate::app::BIN, std::process::id()));
        let stack = tmp.join("repo").join("apps").join(crate::app::BIN).join("demo_stack");
        std::fs::create_dir_all(&stack).unwrap();
        std::fs::write(stack.join(STACK_FILE), JSON).unwrap();
        let deep = tmp.join("repo").join("docs").join("x");
        std::fs::create_dir_all(&deep).unwrap();
        assert_eq!(locate_in(None, std::slice::from_ref(&deep)).unwrap(), stack);
        assert_eq!(locate_in(None, std::slice::from_ref(&stack)).unwrap(), stack);
        assert_eq!(locate_in(Some(stack.clone()), &[]).unwrap(), stack);
        assert!(locate_in(Some(deep.clone()), std::slice::from_ref(&stack)).is_err(), "a wrong env var is an error");
        assert!(locate_in(Some(PathBuf::new()), std::slice::from_ref(&deep)).is_ok(), "empty env var is ignored");
        let _ = std::fs::remove_dir_all(&tmp); // test scratch dir under the OS temp dir only
    }

    /// The index in exports/ loads without problems; with REQUIRE_EXPORTS=1 every head's files must be present too.
    #[test]
    fn shipped_index_is_valid() {
        use inference::{registry::Registry, runtime::Settings};
        let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../exports");
        let r = Registry::open(&Settings { models: Some(models), ..Default::default() }).unwrap();
        assert!(r.problems().is_empty(), "{:?}", r.problems());
        assert_eq!(r.ids().len(), 7, "A, B, C, Deployed, Fast, Small and Deployed at crop 1.0");
        if std::env::var_os("REQUIRE_EXPORTS").is_some_and(|v| v == "1") {
            for h in r.heads() {
                h.available().unwrap_or_else(|e| panic!("{}: {e}", h.id()));
            }
        }
    }

    #[test]
    fn shipped_stack_is_valid() {
        // the demo_stack/ committed next to Cargo.toml
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("demo_stack");
        let s = DemoStack::load(&dir).unwrap();
        assert!(s.images.len() >= 20);
        for (i, im) in s.images.iter().enumerate() {
            assert!(s.path(i).is_file(), "{} missing", im.file);
            assert!(!im.author.is_empty() && !im.licence.is_empty() && !im.source_url.is_empty(), "{} credit", im.file);
        }
        let classes = |section: &str| {
            let mut v: Vec<_> =
                s.section_indices(section).iter().map(|&i| s.images[i].true_text().to_owned()).collect();
            v.sort();
            v
        };
        let one = classes("Hand-picked, one per class");
        assert_eq!(one.len(), 20);
        one.windows(2).for_each(|w| assert_ne!(w[0], w[1], "one per class"));
        let two: Vec<_> = one.iter().flat_map(|c| [c.clone(), c.clone()]).collect();
        assert_eq!(classes("Random, two per class"), two, "two random pictures of every class");
    }
}
