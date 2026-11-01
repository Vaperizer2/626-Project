//! The app's identity. An app made from this one changes these, its package name and `packaging/`.

/// The per-user cache is `$XDG_CACHE_HOME/<NAME>` (or `~/.cache/<NAME>`).
pub const NAME: &str = "component-classifier";
pub const TITLE: &str = "Component Classifier";
/// Environment variables are `<ENV_PREFIX>_MODELS` and `<ENV_PREFIX>_DEMO_DIR`.
pub const ENV_PREFIX: &str = "COMPCLASS";
