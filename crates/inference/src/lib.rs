//! Image models without a UI. An app gives this crate a model index and pictures, and gets each model's answer back
//! on a channel. No call blocks the caller.
//!
//! The parts, in the order an app meets them:
//!
//! * [`registry`] reads the index (`exports/models.json`) and the `model.json` manifests it lists ([`manifest`]), as
//!   pure data. [`Registry::locate`](registry::Registry::locate) finds the index. A graph whose files are missing or
//!   have the wrong size stays listed, as unavailable.
//! * [`service::Inference`] runs the models. Two io threads read, fetch and decode pictures and make one tensor per
//!   preprocessing spec. Each graph has a thread of its own, which loads the graph on first use and batches the
//!   jobs already queued, up to the manifest's `batch.max`. A [`service::Lane`] is one stream of submissions with a
//!   priority and a policy. [`service::Lane::submit`] returns a [`service::RequestHandle`]: its updates arrive as the
//!   work finishes, and dropping its ticket cancels the request. [`service::ServiceEvent`]s report each graph's load
//!   and the head to start with.
//! * [`runtime::OrtLoader`] is the onnxruntime backend. It registers each manifest's execution providers in order
//!   and skips, with the reason, those that fail. TensorRT engines are cached per graph, onnxruntime version,
//!   TensorRT version and GPU under [`runtime::Settings::cache_dir`]. [`integrity`] hashes the weights against their
//!   manifest and records stamps there, which later loads read.
//! * [`classification::HeadOutput`] is one head's answer.
//!
//! The backend is a seam with two adapters. [`service::Loader`] builds [`service::Graph`]s from manifests, and
//! `testing::FakeLoader` (feature `test-support`) does so without model files:
//!
//! ```
//! use futures::StreamExt;
//! use inference::classification::HeadOutput;
//! use inference::service::{Inference, Policy, Priority, Update};
//! use inference::testing::{FakeLoader, image, registry};
//!
//! // one graph "x" with one head "x" and batches of up to 4; "x" is the default head
//! let (service, _events) = Inference::spawn(registry(&[("x", 4)], &["x"]), FakeLoader::default());
//! let lane = service.lane(Priority::Interactive, Policy::Keep);
//! let heads = [service.registry().find("x").unwrap()];
//! let mut request = lane.submit(image(200), &heads, None);
//! let first_head = futures::executor::block_on(async {
//!     while let Some(update) = request.updates.next().await {
//!         if let Update::Head { head, result, .. } = update {
//!             return Some((head, result));
//!         }
//!     }
//!     None
//! });
//! let (head, result) = first_head.expect("the request answers its head");
//! let HeadOutput::Classify(c) = result.unwrap();
//! assert_eq!((head.as_str(), c.best_label()), ("x", "light"));
//! assert!(service.shutdown(std::time::Duration::from_secs(5)));
//! ```
//!
//! Features: `turbojpeg` and `turbojpeg-vendored` decode JPEGs with libjpeg-turbo, as Pillow does. `hotpath`
//! measures the [`prof!`] blocks. `test-support` builds `testing`.

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
#[cfg(any(test, feature = "test-support"))]
pub mod testing;
