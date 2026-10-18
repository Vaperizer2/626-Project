//! Image sources: local paths, `file://` URIs and http(s) URLs.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, bail};

pub const MAX_URL_BYTES: u64 = 20 * 1024 * 1024;
pub const URL_TIMEOUT: Duration = Duration::from_secs(20);
/// Wikimedia (and some other hosts) answer HTTP 403 to anonymous/default agents and ask automated clients to
/// identify themselves; this names the demo honestly (same policy as compclass.demo).
pub const USER_AGENT: &str =
    concat!("compclass-gpui-demo/", env!("CARGO_PKG_VERSION"), " (educational image-classification demo; Rust ureq)");

/// What the user gave us.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    File(PathBuf),
    Url(String),
}

impl Source {
    /// Interpret free text (typed, pasted or a CLI argument): http(s) URL, `file://` URI or a path.
    pub fn parse(text: &str) -> Option<Source> {
        let s = text.trim().trim_matches('"').trim_matches('\'');
        if s.is_empty() {
            return None;
        }
        let lower = s.to_ascii_lowercase();
        if lower.starts_with("http://") || lower.starts_with("https://") {
            return Some(Source::Url(s.to_string()));
        }
        if let Some(rest) = s.strip_prefix("file://") {
            return Some(Source::File(PathBuf::from(percent_decode(rest))));
        }
        let p = if let Some(rest) = s.strip_prefix("~/") {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(rest)).unwrap_or_else(|| PathBuf::from(s))
        } else {
            PathBuf::from(s)
        };
        Some(Source::File(p))
    }

    /// Short name for lists.
    pub fn display_name(&self) -> String {
        match self {
            Source::File(p) => {
                p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| p.display().to_string())
            }
            Source::Url(u) => {
                let no_q = u.split(['?', '#']).next().unwrap_or(u);
                let last = no_q.trim_end_matches('/').rsplit('/').next().unwrap_or(no_q);
                let name = percent_decode(last);
                if name.is_empty() { u.clone() } else { name }
            }
        }
    }

    pub fn full(&self) -> String {
        match self {
            Source::File(p) => p.display().to_string(),
            Source::Url(u) => u.clone(),
        }
    }

    /// Read the bytes (blocking; call off the UI thread).
    pub fn read(&self) -> Result<Vec<u8>> {
        match self {
            Source::File(p) => {
                let meta = std::fs::metadata(p).with_context(|| format!("{}", p.display()))?;
                if meta.is_dir() {
                    bail!("{} is a folder", p.display());
                }
                std::fs::read(p).with_context(|| format!("reading {}", p.display()))
            }
            Source::Url(u) => fetch_url(u),
        }
    }
}

/// Minimal %XX decoding for file:// URIs and display names.
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && i + 2 < b.len()
            && let Some(v) = std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok())
        {
            out.push(v);
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// HTTP GET with a descriptive user agent, a 20 s timeout and a 20 MB cap.
pub fn fetch_url(url: &str) -> Result<Vec<u8>> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(URL_TIMEOUT))
        .user_agent(USER_AGENT)
        .http_status_as_error(false)
        .proxy(proxy_for(url))
        .build()
        .into();
    let mut resp = agent.get(url).call().map_err(|e| anyhow::anyhow!("download failed: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        bail!("HTTP {} {}", status.as_u16(), status.canonical_reason().unwrap_or(""));
    }
    if let Some(n) =
        resp.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<u64>().ok())
        && n > MAX_URL_BYTES
    {
        bail!("{:.1} MB exceeds the {} MB limit", n as f64 / 1048576.0, MAX_URL_BYTES / 1048576);
    }
    let data = resp
        .body_mut()
        .with_config()
        .limit(MAX_URL_BYTES)
        .read_to_vec()
        .map_err(|e| anyhow::anyhow!("download failed: {e}"))?;
    Ok(data)
}

/// Proxy like curl: the scheme-specific variable (`https_proxy` / `http_proxy`) first, then `all_proxy`
/// (http or socks), none if the host matches `no_proxy`. (ureq's own lookup tries ALL_PROXY first.)
fn proxy_for(url: &str) -> Option<ureq::Proxy> {
    let lower = url.to_ascii_lowercase();
    let host = lower.split("://").nth(1).unwrap_or("").split(['/', '?', '#']).next().unwrap_or("");
    let host = host.rsplit_once('@').map(|(_, h)| h).unwrap_or(host);
    let host = host.split(':').next().unwrap_or(host);
    if no_proxy_matches(&env_any(&["no_proxy", "NO_PROXY"]).unwrap_or_default(), host) {
        return None;
    }
    let names: &[&str] = if lower.starts_with("https://") {
        &["https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]
    } else {
        &["http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"]
    };
    env_any(names).and_then(|v| ureq::Proxy::new(&v).ok())
}

fn env_any(names: &[&str]) -> Option<String> {
    names.iter().filter_map(|n| std::env::var(n).ok()).find(|v| !v.trim().is_empty())
}

fn no_proxy_matches(no_proxy: &str, host: &str) -> bool {
    no_proxy.split(',').map(str::trim).filter(|e| !e.is_empty()).any(|e| {
        let e = e.trim_start_matches("*.").trim_start_matches('.');
        e == "*" || host == e || host.ends_with(&format!(".{e}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sources() {
        assert_eq!(Source::parse("  https://x.org/a.jpg "), Some(Source::Url("https://x.org/a.jpg".into())));
        assert_eq!(Source::parse("file:///tmp/a%20b.png"), Some(Source::File("/tmp/a b.png".into())));
        assert_eq!(Source::parse("\"/tmp/x.jpg\""), Some(Source::File("/tmp/x.jpg".into())));
        assert_eq!(Source::parse("   "), None);
    }

    #[test]
    fn display_names() {
        let s = Source::Url("https://upload.wikimedia.org/a/b/Arduino_Uno%281%29.jpg?x=1".into());
        assert_eq!(s.display_name(), "Arduino_Uno(1).jpg");
        assert_eq!(Source::File("/a/b/c.png".into()).display_name(), "c.png");
    }

    #[test]
    fn no_proxy_rules() {
        assert!(no_proxy_matches("localhost,.example.org", "a.example.org"));
        assert!(no_proxy_matches("*", "anything"));
        assert!(!no_proxy_matches("example.org", "wikimedia.org"));
        assert!(!no_proxy_matches("", "wikimedia.org"));
    }

    #[test]
    fn percent_decoding_handles_edges() {
        assert_eq!(percent_decode("a%2"), "a%2");
        assert_eq!(percent_decode("%41%42"), "AB");
        assert_eq!(percent_decode("%zz"), "%zz");
    }
}
