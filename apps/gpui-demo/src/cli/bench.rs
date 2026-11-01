//! `bench`: one model through the inference service, its latency per stage and its streamed throughput.

use std::time::Instant;

use anyhow::{Context, Result, bail};
use inference::fetch::Source;
use inference::service::{HeadOutput, ImageInput, Policy, Priority, RequestHandle, Timing};

use super::{Args, Command, Service, collect, end, heads, registry, service};
use crate::app;

pub const COMMAND: Command = Command {
    name: "bench",
    args: "[--model HEAD] [--rounds N] [--warmup N] IMAGE ...",
    about: "time one model through the inference service:\n\
            per-image latency by stage, then streamed FPS",
    run,
};

/// Median, 10th and 90th percentile (nearest rank), min and max.
fn spread(mut v: Vec<f64>) -> serde_json::Value {
    if v.is_empty() {
        return serde_json::Value::Null;
    }
    v.sort_by(f64::total_cmp);
    let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    let r = |x: f64| (x * 1000.0).round() / 1000.0;
    serde_json::json!({"median": r(at(0.5)), "p10": r(at(0.1)), "p90": r(at(0.9)), "min": r(v[0]), "max": r(v[v.len() - 1]), "n": v.len()})
}

/// `bench`: one head through the service. Latency: every image submitted alone and awaited, read and decoded from
/// its file each time; `--warmup` runs first, unrecorded, then `--rounds` passes over the images. Stream: rounds x
/// images submitted at once to a bulk lane, FPS = images / wall time from the first submit to the last result.
fn run(a: &Args) -> Result<u8> {
    let reg = registry(a)?;
    let service = service(a, &reg);
    let head = if a.models.is_empty() {
        service.default_head()?
    } else {
        match heads(a, &reg)?.as_slice() {
            [h] => h.clone(),
            _ => bail!("bench takes one --model"),
        }
    };
    let sources: Vec<Source> =
        a.rest.iter().map(|s| Source::parse(s).context("empty source")).collect::<Result<_>>()?;
    if sources.is_empty() {
        bail!("bench needs at least one image");
    }
    let Service { inf, loaded, .. } = service;
    let keys = [head.clone()];
    let interactive = inf.lane(Priority::Interactive, Policy::Keep);
    let mut errors = 0usize;
    let mut answers: Vec<String> = Vec::new();
    let mut one = |src: &Source, record: bool, answers: &mut Vec<String>| -> Option<(f64, Timing)> {
        let t0 = Instant::now();
        let mut h = interactive.submit(ImageInput::Source(src.clone()), &keys, None);
        let (results, failed) = collect(&mut h);
        let wall = t0.elapsed().as_secs_f64() * 1000.0;
        match (failed, results.into_iter().next()) {
            (None, Some((_, Ok(HeadOutput::Classify(c)), timing))) => {
                if record {
                    answers.push(c.committed().map_or(app::UNSURE, |i| c.labels.name(i)).to_string());
                }
                Some((wall, timing))
            }
            _ => {
                errors += 1;
                None
            }
        }
    };
    let mut scratch = Vec::new();
    for i in 0..a.warmup {
        one(&sources[i % sources.len()], false, &mut scratch);
    }
    let mut stages: [Vec<f64>; 6] = Default::default();
    for round in 0..a.rounds {
        for src in &sources {
            if let Some((wall, t)) = one(src, round == 0, &mut answers) {
                for (v, x) in stages.iter_mut().zip([t.decode_ms, t.preprocess_ms, t.queue_ms, t.run_ms, t.post_ms, wall]) {
                    v.push(x);
                }
            }
        }
    }
    let bulk = inf.lane(Priority::Bulk, Policy::Keep);
    let t0 = Instant::now();
    let mut handles: Vec<RequestHandle> = (0..a.rounds)
        .flat_map(|_| sources.iter())
        .map(|src| bulk.submit(ImageInput::Source(src.clone()), &keys, None))
        .collect();
    let mut streamed = 0usize;
    for h in &mut handles {
        match collect(h) {
            (r, None) if r.iter().all(|x| x.1.is_ok()) => streamed += 1,
            _ => errors += 1,
        }
    }
    let stream_s = t0.elapsed().as_secs_f64();
    let graph = reg.head(&head).expect("resolved head").manifest().id.clone();
    let load = loaded.lock().expect("loaded lock").iter().find(|(g, _)| *g == graph).map(|(_, l)| l.clone());
    let [decode, preprocess, queue, run, post, e2e] = stages;
    let report = serde_json::json!({
        "head": head.as_str(),
        "graph": graph.as_str(),
        "providers": load.as_ref().map(|l| l.placement.to_string()),
        "skipped": load.as_ref().map(|l| l.placement.skipped.iter().map(|(k, w)| format!("{}: {w}", k.name())).collect::<Vec<_>>()),
        "runtime": load.as_ref().map(|l| l.runtime.clone()),
        "load_ms": load.as_ref().map(|l| l.ms),
        "images": sources.len(),
        "warmup": a.warmup,
        "rounds": a.rounds,
        "errors": errors,
        "latency_ms": {"read_decode": spread(decode), "preprocess": spread(preprocess), "queue": spread(queue),
                       "run": spread(run), "readout": spread(post), "end_to_end": spread(e2e)},
        "stream": {"images": streamed, "wall_s": (stream_s * 1000.0).round() / 1000.0, "fps": streamed as f64 / stream_s},
        "answers": answers,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(end(&inf, if errors > 0 { 2 } else { 0 }))
}
