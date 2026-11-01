//! What a head says about one image. `read_head` is pure: one row of graph output plus the head's task in, the
//! answer out. The decision rule lives in the manifest (abstain below a threshold, or commit and flag low
//! confidence); nothing downstream re-derives it.

use std::sync::Arc;

use crate::manifest::{Calibration, ClassifyTask, Decision, Labels, Readout, Task};

/// One head's output. Tabs match on the variants they can show.
#[derive(Debug, Clone)]
pub enum HeadOutput {
    Classify(Classification),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Answer {
    /// committed to class `idx`; `low_confidence` when an always-answering head is below its bound
    Class { idx: usize, p: f32, low_confidence: bool },
    /// top probability `p` (class `best`) is below the head's abstain threshold
    Abstain { best: usize, p: f32, below: f32 },
}

#[derive(Debug, Clone)]
pub struct Classification {
    pub labels: Arc<Labels>,
    /// probabilities in label order
    pub probs: Arc<[f32]>,
    pub answer: Answer,
    /// probabilities are temperature-calibrated (not just a softmax)
    pub calibrated: bool,
}

impl Classification {
    /// Indices and probabilities of the `k` most likely classes, highest first (ties: lower index first).
    pub fn top(&self, k: usize) -> Vec<(usize, f32)> {
        top_k(&self.probs, k).into_iter().map(|i| (i, self.probs[i])).collect()
    }

    /// The most likely class and its probability, whether or not the head committed to it.
    pub fn best(&self) -> (usize, f32) {
        match self.answer {
            Answer::Class { idx, p, .. } => (idx, p),
            Answer::Abstain { best, p, .. } => (best, p),
        }
    }

    pub fn best_label(&self) -> &str {
        self.labels.name(self.best().0)
    }

    pub fn abstained(&self) -> bool {
        matches!(self.answer, Answer::Abstain { .. })
    }

    /// The head committed although its top probability is below its confidence bound.
    pub fn low_confidence(&self) -> bool {
        matches!(self.answer, Answer::Class { low_confidence: true, .. })
    }

    /// The committed class, or None when the head abstained.
    pub fn committed(&self) -> Option<usize> {
        match self.answer {
            Answer::Class { idx, .. } => Some(idx),
            Answer::Abstain { .. } => None,
        }
    }
}

/// Indices of the `k` largest values, highest first (ties: lower index first, like a stable argsort; NaN last).
pub fn top_k(probs: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_by(|&a, &b| {
        let (x, y) = (probs[a], probs[b]);
        x.is_nan().cmp(&y.is_nan()).then(y.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)).then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

pub fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// One head's readout and decision on one row of graph output.
pub fn read_head(task: &Task, row: &[f32]) -> Result<HeadOutput, String> {
    let Task::Classify(t) = task;
    classify(t, row).map(HeadOutput::Classify)
}

fn classify(t: &ClassifyTask, row: &[f32]) -> Result<Classification, String> {
    if row.len() != t.labels.len() {
        return Err(format!("{} scores for {} labels", row.len(), t.labels.len()));
    }
    let probs: Vec<f32> = match (&t.readout, t.calibration) {
        (Readout::Probs { .. }, _) => row.to_vec(),
        (Readout::Logits { .. }, Calibration::Temperature { t, in_graph: false }) => {
            softmax(&row.iter().map(|z| z / t).collect::<Vec<_>>())
        }
        (Readout::Logits { .. }, _) => softmax(row),
    };
    let best = top_k(&probs, 1)[0];
    let p = probs[best];
    // a NaN anywhere means the row cannot be trusted: abstain, or flag it when the head always answers
    let broken = probs.iter().any(|v| v.is_nan());
    let answer = match t.decision {
        Decision::Abstain { below } if broken || p < below.get() => Answer::Abstain { best, p, below: below.get() },
        Decision::Abstain { .. } => Answer::Class { idx: best, p, low_confidence: false },
        Decision::Commit { low_confidence_below } => {
            let low = broken || low_confidence_below.is_some_and(|b| p < b.get());
            Answer::Class { idx: best, p, low_confidence: low }
        }
    };
    Ok(Classification {
        labels: t.labels.clone(),
        probs: probs.into(),
        answer,
        calibrated: t.calibration != Calibration::None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Prob;

    fn task(decision: Decision, calibration: Calibration, readout: Readout) -> Task {
        let labels = Labels::new((0..3).map(|i| format!("c{i}")).collect()).unwrap();
        Task::Classify(ClassifyTask { readout, labels: Arc::new(labels), calibration, decision })
    }

    fn probs() -> Readout {
        Readout::Probs { output: "probs".into() }
    }

    fn read(t: &Task, row: &[f32]) -> Classification {
        let HeadOutput::Classify(c) = read_head(t, row).unwrap();
        c
    }

    #[test]
    fn top_k_orders_descending_and_breaks_ties_by_index() {
        let p = [0.1, 0.4, 0.05, 0.4, 0.05];
        assert_eq!(top_k(&p, 3), vec![1, 3, 0]);
        assert_eq!(top_k(&p, 10).len(), 5);
    }

    #[test]
    fn abstain_rule_matches_python_demo() {
        // Python: unsure = top[0][1] < threshold (strict)
        let t = task(
            Decision::Abstain { below: Prob::new(0.5).unwrap() },
            Calibration::Temperature { t: 0.9, in_graph: true },
            probs(),
        );
        let sure = read(&t, &[0.5, 0.3, 0.2]);
        assert_eq!(sure.answer, Answer::Class { idx: 0, p: 0.5, low_confidence: false });
        assert!(sure.calibrated);
        let unsure = read(&t, &[0.2, 0.49, 0.31]);
        assert_eq!(unsure.answer, Answer::Abstain { best: 1, p: 0.49, below: 0.5 });
        assert_eq!(unsure.best_label(), "c1");
        assert_eq!(unsure.committed(), None);
        assert_eq!(unsure.top(2).iter().map(|x| x.0).collect::<Vec<_>>(), [1, 2]);
    }

    #[test]
    fn nan_scores_never_make_a_confident_answer() {
        let nan = f32::NAN;
        assert_eq!(top_k(&[nan, 0.2, 0.7], 3), [2, 1, 0], "NaN sorts last");
        let abstain = task(
            Decision::Abstain { below: Prob::new(0.5).unwrap() },
            Calibration::Temperature { t: 1.0, in_graph: true },
            probs(),
        );
        assert!(read(&abstain, &[nan, 0.2, 0.7]).abstained(), "a row with a NaN is not trusted");
        assert!(read(&abstain, &[nan, nan, nan]).abstained());
        let commit = task(Decision::Commit { low_confidence_below: Prob::new(0.5) }, Calibration::None, probs());
        assert!(
            matches!(read(&commit, &[nan, 0.2, 0.7]).answer, Answer::Class { idx: 2, low_confidence: true, .. }),
            "a head that always answers flags it"
        );
    }

    #[test]
    fn commit_heads_always_answer_and_flag_low_confidence() {
        let t = task(Decision::Commit { low_confidence_below: Prob::new(0.5) }, Calibration::None, probs());
        let c = read(&t, &[0.2, 0.45, 0.35]);
        assert_eq!(c.answer, Answer::Class { idx: 1, p: 0.45, low_confidence: true });
        assert!(!c.calibrated);
        assert_eq!(read(&t, &[0.1, 0.8, 0.1]).answer, Answer::Class { idx: 1, p: 0.8, low_confidence: false });
    }

    #[test]
    fn logits_are_softmaxed_with_an_outside_temperature() {
        let t = task(
            Decision::Commit { low_confidence_below: None },
            Calibration::Temperature { t: 2.0, in_graph: false },
            Readout::Logits { output: "logits".into() },
        );
        let c = read(&t, &[2.0, 0.0, 0.0]);
        let want = softmax(&[1.0, 0.0, 0.0]);
        assert!(c.probs.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-7));
        assert!(read_head(&t, &[1.0]).is_err(), "row length must match the labels");
    }
}
