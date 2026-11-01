//! The GPUI-free core of the app: model manifests and the registry, the inference service, the onnxruntime adapter
//! and the weight-integrity checks.

/// Profile a block with hotpath when built with `--features hotpath`; otherwise expands to the block itself.
#[cfg(feature = "hotpath")]
#[macro_export]
macro_rules! prof {
    ($label:literal, $e:expr) => {
        $crate::__hotpath::measure_block!($label, { $e })
    };
}
#[cfg(not(feature = "hotpath"))]
#[macro_export]
macro_rules! prof {
    ($label:literal, $e:expr) => {
        $e
    };
}

#[cfg(feature = "hotpath")]
#[doc(hidden)]
pub use hotpath as __hotpath;

pub mod classification;
pub mod fetch;
pub mod integrity;
pub mod manifest;
pub mod preprocess;
pub mod registry;
pub mod runtime;
pub mod service;
