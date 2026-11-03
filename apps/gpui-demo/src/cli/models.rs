//! `models`: the index's heads, which execution providers register for each graph, and with `--check` whether its
//! files match their checksums and it loads.

use anyhow::Result;
use inference::check::{self, Verification};
use inference::manifest::{Precision, Task};

use super::{Args, Command, registry};

pub const COMMAND: Command = Command {
    name: "models",
    args: "[--check]",
    about: "list the models, whether each is available and which\n\
            execution providers register; --check also verifies\n\
            every file's sha256 and loads each graph",
    run,
};

/// The heads and their files, then per graph which of its execution providers register here (a probe: the
/// provider library is loaded, no session is built); `--check` then verifies each graph's files and loads it.
fn run(a: &Args) -> Result<u8> {
    let reg = registry(a)?;
    for h in reg.heads() {
        let Task::Classify(t) = &h.spec.task;
        match h.available() {
            Ok(()) => println!(
                "{:9} {:30} available  {} [output {}]",
                h.id(),
                h.spec.title,
                h.manifest().graph.display(),
                t.readout.output()
            ),
            Err(e) => println!("{:9} {:30} unavailable  {e}", h.id(), h.spec.title),
        }
    }
    // --check: a manifest the index lists but that cannot be used is a failure too (registry() printed why)
    let mut bad = if a.check { reg.problems().len() } else { 0 };
    if bad > 0 {
        println!("index     {bad} problem(s) with {}", reg.index_path().display());
    }
    let report = check::models(&reg, &a.opts, a.check);
    let (version, lib) = match &report.runtime {
        Ok(rt) => rt,
        Err(e) => {
            println!("execution providers: unknown, {e}");
            return Ok(if a.check { 1 } else { 0 });
        }
    };
    println!("onnxruntime {version} ({})", lib.display());
    for g in &report.graphs {
        let probes: Vec<String> = g
            .providers
            .iter()
            .map(|(ep, r)| match r {
                Ok(()) => format!("{} registers", ep.name()),
                Err(e) => format!("{} does not ({e})", ep.name()),
            })
            .collect();
        let precision = match g.precision {
            Precision::Fp32 => "fp32",
            Precision::Fp16 => "fp16",
            Precision::Int8Qdq => "int8 QDQ",
        };
        println!("{:9} {precision} graph; {}", g.id, probes.join("; "));
    }
    for g in &report.graphs {
        let Some(v) = &g.verified else { continue };
        bad += usize::from(!v.ok());
        match v {
            Verification::Unavailable(e) => println!("{:9} UNAVAILABLE  {e}", g.id),
            Verification::Hashed { bytes, secs, load } => {
                let hashed = format!("sha256 ok ({:.1} MB in {secs:.1} s)", *bytes as f64 / 1e6);
                match load {
                    Ok(l) => println!("{:9} READY  {hashed}; loaded in {:.0} ms on {}", g.id, l.ms, l.placement),
                    Err(e) => println!("{:9} FAILED  {hashed}; load: {e}", g.id),
                }
            }
            Verification::Mismatch { file, expected, found } => {
                println!("{:9} CHECKSUM  {}: sha256 {found}, the manifest says {expected}", g.id, file.display())
            }
            Verification::Unreadable { file, error } => println!("{:9} UNREADABLE  {}: {error}", g.id, file.display()),
        }
    }
    if let Some(e) = &report.stamps_error {
        log!("could not save the checksum stamps: {e}");
    }
    Ok(if bad > 0 { 1 } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference::registry;
    use inference::runtime::Settings;

    #[test]
    fn models_check_fails_when_the_index_lists_a_manifest_it_cannot_read() {
        let dir = std::env::temp_dir().join(format!("{}-check-{}", crate::app::BIN, std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(registry::INDEX_FILE),
            r#"{"schema":"model-index/1","default":[],"manifests":["gone/model.json"]}"#,
        )
        .unwrap();
        let args = Args {
            check: true,
            opts: Settings { models: Some(dir.clone()), ..Default::default() },
            ..Default::default()
        };
        assert_eq!(run(&args).unwrap(), 1, "a manifest that cannot be read is a failed check");
        let _ = std::fs::remove_dir_all(&dir); // test scratch dir under the OS temp dir only
    }
}
