//! `preprocess`: the model input tensor of one picture, for the parity tests against the Python pipeline.

use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use inference::fetch::Source;
use inference::preprocess;

use super::{Args, Command, heads, registry};

pub const COMMAND: Command = Command {
    name: "preprocess",
    args: "IMAGE OUT.f32 [--model HEAD]",
    about: "write the model input tensor (little-endian f32)",
    run,
};

fn run(a: &Args) -> Result<u8> {
    let [img, out] = a.rest.as_slice() else { bail!("usage: gpui-demo preprocess IMAGE OUT.f32 [--model HEAD]") };
    let reg = registry(a)?;
    let head = match heads(a, &reg)?.as_slice() {
        [h] => h.clone(),
        _ => bail!("preprocess takes one --model"),
    };
    let manifest = reg.head(&head).ok_or_else(|| anyhow!("unknown model {head}"))?.manifest().clone();
    let t0 = Instant::now();
    let bytes = Source::parse(img).context("empty source")?.read()?;
    let rgb = preprocess::decode(&bytes)?;
    let t = preprocess::to_tensor(&rgb, &manifest.input.pre)?;
    let mut buf = Vec::with_capacity(t.len() * 4);
    for v in &t {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(out, buf)?;
    eprintln!("[gpui-demo] {} -> {} ({} floats, {:.1} ms)", img, out, t.len(), t0.elapsed().as_secs_f64() * 1000.0);
    Ok(0)
}
