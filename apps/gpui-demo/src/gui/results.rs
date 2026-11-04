//! One input's answers, per head: what a Classify entry and a Demo card hold. Views show the selected head's
//! answer; a model switch fills in the missing one for the input on screen.

use std::collections::HashMap;

use inference::classification::Classification;
use inference::manifest::HeadId;
use inference::service::{Finish, HeadOutput, RunError, Timing, Update};

pub enum HeadResult {
    /// submitted; the token tells this request's late "cancelled" apart from a newer request's
    Working(u64),
    /// the answer, its timing, and its request's wall time incl. read and decode once the request completes
    Done(Classification, Timing, Option<f64>),
    Failed(RunError),
}

#[derive(Default)]
pub struct Results(HashMap<HeadId, HeadResult>);

impl Results {
    pub fn get(&self, head: &HeadId) -> Option<&HeadResult> {
        self.0.get(head)
    }

    pub fn start(&mut self, head: HeadId, token: u64) {
        self.0.insert(head, HeadResult::Working(token));
    }

    pub fn done(&mut self, head: HeadId, c: Classification, timing: Timing) {
        self.0.insert(head, HeadResult::Done(c, timing, None));
    }

    /// Drop the answers of `heads` that say "the model failed to load", so a retried load answers again.
    pub fn forget_load_failures(&mut self, heads: &[HeadId]) {
        self.0.retain(|h, r| !(heads.contains(h) && matches!(r, HeadResult::Failed(RunError::Load(_)))));
    }

    /// Record update `u` of request `token`, which asked for `head`.
    pub fn apply(&mut self, head: &HeadId, token: u64, u: Update) {
        match u {
            Update::Head { head, result: Ok(HeadOutput::Classify(c)), timing } => self.done(head, c, timing),
            Update::Head { head, result: Err(e), .. } => {
                self.0.insert(head, HeadResult::Failed(e));
            }
            Update::Done(Finish::Failed(e)) => {
                self.0.insert(head.clone(), HeadResult::Failed(e));
            }
            Update::Done(Finish::Complete { wall_ms }) => match self.0.get_mut(head) {
                Some(HeadResult::Done(_, _, wall)) => *wall = Some(wall_ms),
                Some(HeadResult::Working(t)) if *t == token => {
                    self.0.remove(head);
                }
                _ => {}
            },
            Update::Done(Finish::Cancelled | Finish::Superseded) => {
                if matches!(self.0.get(head), Some(HeadResult::Working(t)) if *t == token) {
                    self.0.remove(head); // cancelled before it ran
                }
            }
            Update::Decoded { .. } => {}
        }
    }
}
