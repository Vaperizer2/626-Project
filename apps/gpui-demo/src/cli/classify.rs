//! `classify`: one line, or one JSON object, per image and head, without a window.

use std::io::Write;

use anyhow::{Result, bail};
use inference::fetch::Source;
use inference::manifest::HeadId;
use inference::service::{HeadOutput, ImageInput, Policy, Priority, RequestHandle, RunError, Timing};

use super::{Args, Command, collect, end, heads, registry, service};
use crate::app;

pub const COMMAND: Command = Command {
    name: "classify",
    args: "[OPTIONS] [--json] IMAGE|URL ...",
    about: "headless; one line (or JSON object) per image",
    run,
};

/// One `classify --json` row; an error row has the same keys.
#[derive(serde::Serialize)]
struct JsonRow<'a> {
    source: &'a str,
    model: &'a str,
    label: Option<&'a str>,
    top3: Vec<(&'a str, f32)>,
    unsure: bool,
    /// a head that always answers is below its confidence bound (heads that abstain say `unsure` instead)
    low_confidence: bool,
    ms: f64,
    probs: Vec<f32>,
    error: Option<&'a str>,
    timing: Option<Timing>,
}

impl<'a> JsonRow<'a> {
    fn new(source: &'a str, head: &'a HeadId, row: &'a Result<(HeadOutput, Timing), String>) -> Self {
        match row {
            Ok((HeadOutput::Classify(c), timing)) => JsonRow {
                source,
                model: head.as_str(),
                label: Some(c.committed().map_or(app::UNSURE, |i| c.labels.name(i))),
                top3: c.top(3).into_iter().map(|(i, p)| (c.labels.name(i), p)).collect(),
                unsure: c.abstained(),
                low_confidence: c.low_confidence(),
                ms: (timing.ms() * 10.0).round() / 10.0,
                probs: c.probs.to_vec(),
                error: None,
                timing: Some(*timing),
            },
            Err(e) => JsonRow {
                source,
                model: head.as_str(),
                label: None,
                top3: vec![],
                unsure: false,
                low_confidence: false,
                ms: 0.0,
                probs: vec![],
                error: Some(e),
                timing: None,
            },
        }
    }
}

/// One row per requested head, in `keys` order; a head the request ended without is a failed row.
fn rows_of(
    keys: &[HeadId],
    mut results: Vec<(HeadId, Result<HeadOutput, RunError>, Timing)>,
    failed: Option<RunError>,
) -> Vec<(HeadId, Result<(HeadOutput, Timing), String>)> {
    keys.iter()
        .map(|k| {
            let row = match (&failed, results.iter().position(|r| r.0 == *k)) {
                (Some(e), _) => Err(e.to_string()),
                (None, Some(i)) => {
                    let (_, r, t) = results.swap_remove(i);
                    r.map(|o| (o, t)).map_err(|e| e.to_string())
                }
                (None, None) => Err("no answer: the inference service ended the request without one".into()),
            };
            (k.clone(), row)
        })
        .collect()
}

/// `classify`: every image is submitted at once to one bulk lane, so decoding overlaps inference; rows come out in
/// input order, heads in `--model` order (default: the index's default head).
fn run(a: &Args) -> Result<u8> {
    if a.rest.is_empty() {
        bail!("classify needs at least one image path or URL");
    }
    let reg = registry(a)?;
    let service = service(a, &reg);
    let keys = if a.models.is_empty() {
        // the first default whose preferred provider registers here, as in the window
        vec![service.default_head()?]
    } else {
        heads(a, &reg)?
    };
    let inf = &service.inf;
    let lane = inf.lane(Priority::Bulk, Policy::Keep);
    let mut requests: Vec<(&String, Option<RequestHandle>)> = a
        .rest
        .iter()
        .map(|s| (s, Source::parse(s).map(|src| lane.submit(ImageInput::Source(src), &keys, None))))
        .collect();
    let mut failures = 0u8;
    let mut rows = Vec::new();
    let stdout = std::io::stdout();
    for (s, handle) in &mut requests {
        let (results, failed) = match handle {
            Some(h) => collect(h),
            None => (vec![], Some(RunError::Read("empty source".into()))),
        };
        for (head, r) in rows_of(&keys, results, failed) {
            if r.is_err() {
                failures = failures.saturating_add(1);
            }
            let row = JsonRow::new(s, &head, &r);
            if a.json {
                rows.push(serde_json::to_value(row)?);
                continue;
            }
            match (&r, row.label) {
                (Ok((HeadOutput::Classify(c), timing)), Some(label)) => {
                    let top: Vec<String> = row.top3.iter().map(|(l, v)| format!("{l} {v:.3}")).collect();
                    writeln!(
                        stdout.lock(),
                        "{s}\t{head}\t{label}\t{:.3}\ttop3: {}\t{:.0} ms",
                        c.best().1,
                        top.join(", "),
                        timing.ms()
                    )?;
                }
                (_, _) => writeln!(stdout.lock(), "{s}\t{head}\tERROR\t{}", row.error.unwrap_or_default())?,
            }
        }
    }
    if a.json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
    }
    Ok(end(inf, if failures > 0 { 2 } else { 0 }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use inference::manifest::Task;
    use inference::{classification, manifest};

    #[test]
    fn a_requested_head_without_a_result_is_a_failed_row() {
        let (x, y) = (HeadId::new("x"), HeadId::new("y"));
        let task = &inference::testing::manifest("x", 1).heads[0].task;
        let answer = classification::read_head(task, &[0.9, 0.1]).unwrap();
        let rows = rows_of(&[x.clone(), y.clone()], vec![(x.clone(), Ok(answer), Timing::default())], None);
        let heads: Vec<&HeadId> = rows.iter().map(|r| &r.0).collect();
        assert_eq!(heads, [&x, &y], "every requested head has a row");
        assert!(rows[0].1.is_ok() && rows[1].1.is_err(), "the head that never answered counts as a failure");
    }

    #[test]
    fn json_rows_have_one_shape_and_flag_low_confidence() {
        let head = HeadId::new("x");
        let mut m = Arc::try_unwrap(inference::testing::manifest("x", 1)).unwrap();
        let Task::Classify(t) = &mut m.heads[0].task;
        t.decision = manifest::Decision::Commit { low_confidence_below: manifest::Prob::new(0.95) };
        let answer = classification::read_head(&m.heads[0].task, &[0.9, 0.1]).unwrap();
        let rows = [Ok((answer, Timing::default())), Err("could not read".to_string())];
        let [ok, err] = rows.each_ref().map(|r| serde_json::to_value(JsonRow::new("p.jpg", &head, r)).unwrap());
        let keys = |v: &serde_json::Value| v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
        assert_eq!(keys(&ok), keys(&err), "an error row has every key");
        assert_eq!(ok["low_confidence"], true, "0.9 is below the head's 0.95 bound");
        assert!(err["timing"].is_null() && err["low_confidence"] == false);
    }
}
