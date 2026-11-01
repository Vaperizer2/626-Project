//! The demo stack (`crate::demo`) with its pictures, decoded once at start-up on the thumbnail thread, and the
//! answers each picture got. The Demo and Benchmark tabs share it, so the benchmark's answers show on the Demo cards.

use std::path::PathBuf;
use std::sync::Arc;

use futures::StreamExt;
use gpui_kit::{Context, RenderImage};
use image::RgbImage;

use super::images;
use super::results::Results;
use crate::demo::DemoStack;
use inference::service::ImageInput;

#[derive(Default)]
pub struct Item {
    pub thumb: Option<Arc<RenderImage>>,
    pub full: Option<Arc<RenderImage>>,
    pub rgb: Option<Arc<RgbImage>>,
    pub load_error: Option<String>,
    pub results: Results,
}

pub struct Stack {
    stack: Result<DemoStack, String>,
    /// one per image of the stack, in its order
    pub items: Vec<Item>,
}

impl Stack {
    /// The stack in `dir`, else the one `demo::locate` finds; its pictures decode in the background.
    pub fn new(dir: Option<PathBuf>, cx: &mut Context<Self>) -> Self {
        let stack = crate::demo::locate(dir.as_deref()).and_then(|d| DemoStack::load(&d)).map_err(|e| format!("{e:#}"));
        let items = match &stack {
            Ok(s) => {
                log!("demo stack: {} images in {}", s.images.len(), s.dir.display());
                let mut rx = images::load_thumbnails((0..s.images.len()).map(|i| s.path(i)).collect());
                cx.spawn(async move |this, cx| {
                    while let Some((i, r)) = rx.next().await {
                        let alive = this
                            .update(cx, |this, cx| {
                                if let Some(it) = this.items.get_mut(i) {
                                    match r {
                                        Ok(d) => {
                                            it.thumb = Some(d.thumb);
                                            it.full = Some(d.full);
                                            it.rgb = Some(d.rgb);
                                        }
                                        Err(e) => it.load_error = Some(e),
                                    }
                                }
                                cx.notify();
                            })
                            .is_ok();
                        if !alive {
                            break;
                        }
                    }
                })
                .detach();
                (0..s.images.len()).map(|_| Item::default()).collect()
            }
            Err(e) => {
                log!("demo stack unavailable: {e}");
                Vec::new()
            }
        };
        Self { stack, items }
    }

    pub fn stack(&self) -> Option<&DemoStack> {
        self.stack.as_ref().ok()
    }

    /// Why there is no stack.
    pub fn error(&self) -> Option<&str> {
        self.stack.as_ref().err().map(String::as_str)
    }

    /// What a request for image `i` sends: the decoded picture once it is ready, else its file.
    pub fn input(&self, i: usize) -> Option<ImageInput> {
        match self.items.get(i)?.rgb.clone() {
            Some(rgb) => Some(ImageInput::Decoded(rgb)),
            None => Some(ImageInput::Source(inference::fetch::Source::File(self.stack()?.path(i)))),
        }
    }
}
