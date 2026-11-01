//! Weight integrity. The registry already refuses a file whose size differs from its manifest. Beyond that a load
//! only asks, cheaply, whether each file is still the copy `models --check` last hashed: a stamp file in the app's cache maps
//! each path to the size, mtime and sha256 seen then. The full sha256 runs only in `models --check`, which records
//! a stamp for every file that matches its manifest.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256 as Hasher};

use crate::manifest::{FileEntry, Manifest, Sha256};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Stamp {
    bytes: u64,
    mtime_ns: u128,
    sha256: String,
}

pub struct Stamps {
    path: Option<PathBuf>,
    map: BTreeMap<String, Stamp>,
}

fn key(p: &Path) -> String {
    p.canonicalize().unwrap_or_else(|_| p.to_path_buf()).display().to_string()
}

/// The size and mtime (ns) of `p` now.
pub(crate) fn now_stamp(p: &Path) -> Option<(u64, u128)> {
    let md = std::fs::metadata(p).ok()?;
    let mtime = md.modified().ok()?.duration_since(UNIX_EPOCH).ok()?.as_nanos();
    Some((md.len(), mtime))
}

impl Stamps {
    /// The stamps in `dir/verified.json` (none yet when it is missing). Without a directory nothing is kept.
    pub fn load(dir: Option<&Path>) -> Self {
        let path = dir.map(|d| d.join("verified.json"));
        let map = path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        Self { path, map }
    }

    /// Is `f` the file `models --check` hashed (same size and mtime, and the hash it found is the manifest's)?
    pub(crate) fn verified(&self, f: &FileEntry) -> bool {
        let Some((bytes, mtime_ns)) = now_stamp(&f.path) else { return false };
        self.map.get(&key(&f.path)).is_some_and(|s| {
            s.bytes == bytes && s.mtime_ns == mtime_ns && s.sha256 == f.sha256.to_string()
        })
    }

    /// Every file of `m` is verified.
    pub(crate) fn all_verified(&self, m: &Manifest) -> bool {
        m.files.iter().all(|f| self.verified(f))
    }

    /// `f` had size and mtime `(bytes, mtime_ns)` before and after it was hashed and found matching.
    fn record(&mut self, f: &FileEntry, (bytes, mtime_ns): (u64, u128)) {
        self.map.insert(key(&f.path), Stamp { bytes, mtime_ns, sha256: f.sha256.to_string() });
    }

    fn forget(&mut self, f: &FileEntry) {
        self.map.remove(&key(&f.path));
    }

    pub fn save(&self) -> std::io::Result<()> {
        let Some(path) = &self.path else { return Ok(()) };
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(&self.map).unwrap_or_default())?;
        std::fs::rename(tmp, path)
    }
}

pub(crate) fn sha256_file(p: &Path) -> std::io::Result<Sha256> {
    let mut f = std::fs::File::open(p)?;
    let mut h = Hasher::new();
    let mut buf = vec![0u8; 1 << 22];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(Sha256(h.finalize().into()))
}

/// The outcome of hashing one graph's files.
#[derive(Debug)]
pub enum Check {
    Ok { bytes: u64 },
    Mismatch { file: PathBuf, expected: Sha256, found: Sha256 },
    Unreadable { file: PathBuf, error: String },
}

/// Hash every file of `m` against its manifest; stamps are recorded for files that match and dropped for the rest.
pub fn check(m: &Manifest, stamps: &mut Stamps) -> Check {
    check_with(m, stamps, sha256_file)
}

fn check_with(m: &Manifest, stamps: &mut Stamps, hash: impl Fn(&Path) -> std::io::Result<Sha256>) -> Check {
    let mut bytes = 0;
    for f in &m.files {
        // the stamp comes from before the hash and must still hold after it, so it vouches for the bytes hashed
        let before = now_stamp(&f.path);
        let found = hash(&f.path);
        let stamp = before.filter(|b| now_stamp(&f.path).as_ref() == Some(b));
        match (found, stamp) {
            (Ok(_), None) => {
                stamps.forget(f);
                return Check::Unreadable {
                    file: f.path.clone(),
                    error: "it changed while it was hashed (still being copied?); check again".into(),
                };
            }
            (Ok(found), Some(stamp)) if found == f.sha256 => {
                stamps.record(f, stamp);
                bytes += f.bytes;
            }
            (Ok(found), Some(_)) => {
                stamps.forget(f);
                return Check::Mismatch { file: f.path.clone(), expected: f.sha256, found };
            }
            (Err(e), _) => {
                stamps.forget(f);
                return Check::Unreadable { file: f.path.clone(), error: e.to_string() };
            }
        }
    }
    Check::Ok { bytes }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_of_a_known_string() {
        let p = std::env::temp_dir().join(format!("inference-sha-{}", std::process::id()));
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(
            sha256_file(&p).unwrap().to_string(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let _ = std::fs::remove_file(&p); // test scratch file under the OS temp dir only
    }

    #[test]
    fn a_stamp_holds_until_the_file_changes() {
        let dir = std::env::temp_dir().join(format!("inference-stamp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("w.bin");
        std::fs::write(&p, b"abc").unwrap();
        let sha = sha256_file(&p).unwrap();
        let f = FileEntry { path: p.clone(), bytes: 3, sha256: sha };
        let mut s = Stamps { path: None, map: BTreeMap::new() };
        assert!(!s.verified(&f), "never checked");
        s.record(&f, now_stamp(&p).unwrap());
        assert!(s.verified(&f));
        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::write(&p, b"abd").unwrap();
        assert!(!s.verified(&f), "rewritten since the check");
        let _ = std::fs::remove_dir_all(&dir); // test scratch dir under the OS temp dir only
    }

    #[test]
    fn a_file_that_changes_while_it_is_hashed_is_not_stamped() {
        let dir = std::env::temp_dir().join(format!("inference-stamp-race-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("w.bin");
        std::fs::write(&p, b"abc").unwrap();
        let mut m = std::sync::Arc::try_unwrap(crate::testing::manifest("x", 1)).unwrap();
        m.files = vec![FileEntry { path: p.clone(), bytes: 3, sha256: sha256_file(&p).unwrap() }];
        let mut s = Stamps { path: None, map: BTreeMap::new() };
        // the hash reads "abc"; a copy then finishes writing "abd" before the hash returns
        let check = check_with(&m, &mut s, |path| {
            let sha = sha256_file(path);
            std::fs::write(path, b"abd").unwrap();
            let f = std::fs::File::options().write(true).open(path).unwrap();
            f.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(5)).unwrap();
            sha
        });
        assert!(!matches!(check, Check::Ok { .. }), "the bytes on disk are not the ones hashed: {check:?}");
        assert!(!s.verified(&m.files[0]), "no stamp vouches for the new bytes");
        let _ = std::fs::remove_dir_all(&dir); // test scratch dir under the OS temp dir only
    }
}
