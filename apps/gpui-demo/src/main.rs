//! The window, or one subcommand of `app::COMMANDS` over the same models and without a window (see `--help`).

/// `eprintln!` behind the binary's name in brackets, as every log line of the app starts.
macro_rules! log {
    ($($arg:tt)*) => {
        eprintln!("[{}] {}", $crate::app::BIN, format_args!($($arg)*))
    };
}

mod app;
mod cli;
mod demo;
mod gui;

use std::process::ExitCode;

fn main() -> ExitCode {
    // `--features hotpath`: timing report on exit (HOTPATH_OUTPUT_FORMAT / HOTPATH_OUTPUT_PATH configure it).
    // HOTPATH_SHUTDOWN_MS=N (e.g. for a GUI session): print the report and exit after N ms.
    #[cfg(feature = "hotpath")]
    let _hotpath = {
        let b = hotpath::HotpathGuardBuilder::new(app::BIN).percentiles(&[50.0, 95.0]).limit(0);
        if std::env::var_os("HOTPATH_SHUTDOWN_MS").is_some() {
            b.build_with_shutdown(std::time::Duration::ZERO);
            None
        } else {
            Some(b.build())
        }
    };
    let args = match cli::parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{}: {e:#}", app::BIN);
            return ExitCode::from(64);
        }
    };
    let r = match args.command {
        Some(command) => (command.run)(&args),
        None => {
            gui::run(args.opts, args.gui, args.rest);
            Ok(0)
        }
    };
    match r {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            eprintln!("{}: {e:#}", app::BIN);
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    /// packaging/gpu-env.sh, through scripts/test_gpu_env.sh (fake venvs under a path with spaces)
    #[cfg(unix)]
    #[test]
    fn gpu_env_passes_over_a_wrong_onnxruntime_and_keeps_paths_whole() {
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/test_gpu_env.sh");
        let out = std::process::Command::new("sh").arg(&script).output().unwrap();
        assert!(
            out.status.success(),
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// The launchers start app.env's COMMANDS without a window, so it must name every subcommand.
    #[test]
    fn app_env_lists_every_subcommand() {
        let env = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("packaging/app.env"))
            .unwrap();
        let line =
            env.lines().find_map(|l| l.strip_prefix("COMMANDS=")).expect("a COMMANDS= line in packaging/app.env");
        let mut listed: Vec<&str> = line.trim_matches('"').split_whitespace().collect();
        let mut names: Vec<&str> = crate::app::COMMANDS.iter().map(|c| c.name).collect();
        listed.sort_unstable();
        names.sort_unstable();
        assert_eq!(listed, names, "packaging/app.env COMMANDS against app::COMMANDS");
    }
}
