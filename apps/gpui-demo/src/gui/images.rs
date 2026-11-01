//! Display textures, and the demo stack decoded once at start-up on its own thread (rayon), producing display-size
//! textures and the decoded RGB that later requests reuse. Model work happens in `inference::service`.

use std::sync::Arc;
use std::time::Instant;

use futures::channel::mpsc as fmpsc;
use inference::prof;
use gpui_kit::RenderImage;
use image::{RgbImage, RgbaImage};
use rayon::prelude::*;

use inference::preprocess;

pub const DISPLAY_MAX_SIDE: u32 = 1280;
/// Demo grid thumbnails: decoded once at 2x the card size (sharp on HiDPI, cheap to sample).
pub const THUMB_MAX_SIDE: u32 = 360;

/// Wrap a BGRA buffer (as the inference service's display copy) for GPUI; no pixel copy.
pub fn render(bgra: RgbaImage) -> Arc<RenderImage> {
    Arc::new(RenderImage::new(smallvec::smallvec![image::Frame::new(bgra)]))
}

/// BGRA texture for GPUI, long side <= `max_side`.
pub fn texture(rgb: &RgbImage, max_side: u32) -> Arc<RenderImage> {
    render(preprocess::display_bgra(rgb, max_side))
}

/// One decoded demo image.
pub struct Decoded {
    pub rgb: Arc<RgbImage>,
    pub thumb: Arc<RenderImage>,
    pub full: Arc<RenderImage>,
}

/// Decode every demo image in parallel, once; results arrive in completion order as (index, result).
pub fn load_thumbnails(paths: Vec<std::path::PathBuf>) -> fmpsc::UnboundedReceiver<(usize, Result<Decoded, String>)> {
    let (tx, rx) = fmpsc::unbounded();
    std::thread::Builder::new()
        .name("thumbnails".into())
        .spawn(move || {
            let t0 = Instant::now();
            paths.into_par_iter().enumerate().for_each(|(i, p)| {
                let r = prof!("demo::decode_and_thumbnail", decode_one(&p));
                let _ = tx.unbounded_send((i, r));
            });
            eprintln!("[gpui-demo] demo stack decoded in {:.0} ms", t0.elapsed().as_secs_f64() * 1000.0);
        })
        .expect("spawn thumbnail thread");
    rx
}

fn decode_one(p: &std::path::Path) -> Result<Decoded, String> {
    let bytes = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let rgb = preprocess::decode(&bytes).map_err(|e| format!("{}: {e:#}", p.display()))?;
    let thumb = texture(&rgb, THUMB_MAX_SIDE);
    let full = texture(&rgb, DISPLAY_MAX_SIDE);
    Ok(Decoded { rgb: Arc::new(rgb), thumb, full })
}
