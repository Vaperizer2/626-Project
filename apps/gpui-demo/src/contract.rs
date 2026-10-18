//! The export contract: `preprocess.json` + `labels.json` (see docs/DEMO.md, "Deploying elsewhere"),
//! and the decision rule (top-3, UNSURE below `abstain_threshold`).

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// Fields of `preprocess.json` that the front end uses. Unknown keys (provenance) are ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct Preprocess {
    pub input_size: u32,
    pub resize_short_side: u32,
    pub interpolation: String,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub scale: f64,
    pub abstain_threshold: f32,
    #[serde(default = "default_rgb")]
    pub color_order: String,
    #[serde(default = "default_nchw")]
    pub layout: String,
    #[serde(default)]
    pub model: Option<String>,
}

fn default_rgb() -> String {
    "RGB".into()
}
fn default_nchw() -> String {
    "NCHW".into()
}

impl Preprocess {
    pub fn validate(&self) -> Result<()> {
        if self.color_order != "RGB" || self.layout != "NCHW" {
            bail!("only RGB / NCHW preprocessing is implemented (got {} / {})", self.color_order, self.layout);
        }
        if self.interpolation != "bicubic" {
            bail!("only interpolation=\"bicubic\" is implemented (got {:?})", self.interpolation);
        }
        if self.resize_short_side < self.input_size {
            bail!("resize_short_side {} < input_size {}", self.resize_short_side, self.input_size);
        }
        if self.std.contains(&0.0) {
            bail!("std must be non-zero");
        }
        Ok(())
    }
}

/// `preprocess.json` + `labels.json` from an export directory.
#[derive(Debug, Clone)]
pub struct Contract {
    pub pre: Preprocess,
    pub labels: Vec<String>,
}

impl Contract {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let pp = model_dir.join("preprocess.json");
        let lp = model_dir.join("labels.json");
        let pre: Preprocess =
            serde_json::from_str(&std::fs::read_to_string(&pp).with_context(|| format!("reading {}", pp.display()))?)
                .with_context(|| format!("parsing {}", pp.display()))?;
        pre.validate()?;
        let labels: Vec<String> =
            serde_json::from_str(&std::fs::read_to_string(&lp).with_context(|| format!("reading {}", lp.display()))?)
                .with_context(|| format!("parsing {}", lp.display()))?;
        if labels.is_empty() {
            bail!("{} is empty", lp.display());
        }
        Ok(Self { pre, labels })
    }
}

/// One classified image.
#[derive(Debug, Clone, Serialize)]
pub struct Prediction {
    /// top-3 (label, calibrated probability), highest first
    pub top3: Vec<(String, f32)>,
    /// true if the top probability is below `abstain_threshold`
    pub unsure: bool,
    /// all class probabilities in `labels.json` order
    pub probs: Vec<f32>,
    /// preprocessing + inference time (same span as the Python demo's `ms`)
    pub ms: f64,
}

impl Prediction {
    /// Displayed label: "UNSURE" or the top class.
    pub fn label(&self) -> &str {
        if self.unsure { "UNSURE" } else { &self.top3[0].0 }
    }
}

/// Indices of the `k` largest probabilities, highest first (ties: lower index first, like a stable argsort).
pub fn top_k(probs: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

/// Apply the decision rule of docs/DEMO.md: top-3 and UNSURE if `max(p) < threshold`.
pub fn decide(probs: &[f32], labels: &[String], threshold: f32, ms: f64) -> Prediction {
    let top3: Vec<(String, f32)> = top_k(probs, 3)
        .into_iter()
        .map(|i| (labels.get(i).cloned().unwrap_or_else(|| format!("class {i}")), probs[i]))
        .collect();
    let unsure = top3.first().map(|t| t.1 < threshold).unwrap_or(true);
    Prediction { top3, unsure, probs: probs.to_vec(), ms }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("c{i}")).collect()
    }

    #[test]
    fn top_k_orders_descending_and_breaks_ties_by_index() {
        let p = [0.1, 0.4, 0.05, 0.4, 0.05];
        assert_eq!(top_k(&p, 3), vec![1, 3, 0]);
        assert_eq!(top_k(&p, 10).len(), 5);
    }

    #[test]
    fn threshold_rule_matches_python_demo() {
        // Python: unsure = top[0][1] < threshold (strict)
        let l = labels(3);
        let sure = decide(&[0.5, 0.3, 0.2], &l, 0.5, 0.0);
        assert!(!sure.unsure);
        assert_eq!(sure.label(), "c0");
        let unsure = decide(&[0.2, 0.49, 0.31], &l, 0.5, 0.0);
        assert!(unsure.unsure);
        assert_eq!(unsure.label(), "UNSURE");
        assert_eq!(unsure.top3[0].0, "c1");
        assert_eq!(unsure.top3[1].0, "c2");
    }

    #[test]
    fn real_contract_parses() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../exports/final");
        if !dir.join("preprocess.json").exists() {
            return; // not in the repo checkout
        }
        let c = Contract::load(&dir).unwrap();
        assert_eq!(c.labels.len(), 20);
        assert_eq!(c.pre.input_size, 384);
        assert_eq!(c.pre.resize_short_side, 427);
        assert_eq!(c.pre.abstain_threshold, 0.5);
    }
}
