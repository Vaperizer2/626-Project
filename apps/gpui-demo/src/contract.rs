//! The decision rule: top-3, and UNSURE below the head's abstain threshold (docs/DEMO.md).

use serde::Serialize;

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
}
