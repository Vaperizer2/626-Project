//! Display textures: the BGRA copies the inference service makes, handed to GPUI without a pixel copy.

use std::sync::Arc;

use gpui_kit::RenderImage;
use image::RgbaImage;

/// The long side of the pictures the tabs show.
pub const DISPLAY_MAX_SIDE: u32 = 1280;

/// Wrap a BGRA buffer (as the inference service's display copy) for GPUI; no pixel copy.
pub fn render(bgra: RgbaImage) -> Arc<RenderImage> {
    Arc::new(RenderImage::new(smallvec::smallvec![image::Frame::new(bgra)]))
}
