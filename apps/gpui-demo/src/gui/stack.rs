//! The demo stack (`crate::demo`) with its pictures, decoded once at start-up on their own thread (rayon), and the
//! answers each picture got, judged against its true label. The Demo and Benchmark tabs share it, so the benchmark's
//! answers show on the Demo cards.

use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use futures::channel::mpsc as fmpsc;
use gpui_kit::component::{ActiveTheme as _, tag::Tag};
use gpui_kit::{App, Context, Hsla, RenderImage};
use image::RgbImage;
use inference::{preprocess, prof};
use rayon::prelude::*;

use super::images;
use super::results::Results;
use super::theme;
use super::widgets::{self, Lucide};
use crate::demo::{DemoStack, Verdict};
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
    /// The stack `demo::locate` finds; its pictures decode in the background.
    pub fn new(cx: &mut Context<Self>) -> Self {
        let stack = crate::demo::locate().and_then(|d| DemoStack::load(&d)).map_err(|e| format!("{e:#}"));
        let items = match &stack {
            Ok(s) => {
                log!("demo stack: {} images in {}", s.images.len(), s.dir.display());
                let mut rx = load_thumbnails((0..s.images.len()).map(|i| s.path(i)).collect());
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

/// Demo grid thumbnails: decoded once at 2x the card size (sharp on HiDPI, cheap to sample).
const THUMB_MAX_SIDE: u32 = 360;

/// BGRA texture for GPUI, long side <= `max_side`.
fn texture(rgb: &RgbImage, max_side: u32) -> Arc<RenderImage> {
    images::render(preprocess::display_bgra(rgb, max_side))
}

/// One decoded demo image.
struct Decoded {
    rgb: Arc<RgbImage>,
    thumb: Arc<RenderImage>,
    full: Arc<RenderImage>,
}

/// Decode every demo image in parallel, once; results arrive in completion order as (index, result).
fn load_thumbnails(paths: Vec<std::path::PathBuf>) -> fmpsc::UnboundedReceiver<(usize, Result<Decoded, String>)> {
    let (tx, rx) = fmpsc::unbounded();
    std::thread::Builder::new()
        .name("thumbnails".into())
        .spawn(move || {
            let t0 = Instant::now();
            paths.into_par_iter().enumerate().for_each(|(i, p)| {
                let r = prof!("demo::decode_and_thumbnail", decode_one(&p));
                let _ = tx.unbounded_send((i, r));
            });
            log!("demo stack decoded in {:.0} ms", t0.elapsed().as_secs_f64() * 1000.0);
        })
        .expect("spawn thumbnail thread");
    rx
}

fn decode_one(p: &std::path::Path) -> Result<Decoded, String> {
    let bytes = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let rgb = preprocess::decode(&bytes).map_err(|e| format!("{}: {e:#}", p.display()))?;
    let thumb = texture(&rgb, THUMB_MAX_SIDE);
    let full = texture(&rgb, images::DISPLAY_MAX_SIDE);
    Ok(Decoded { rgb: Arc::new(rgb), thumb, full })
}

/// The verdict's icon: three different shapes, so the verdict never rests on colour alone.
pub fn verdict_icon(v: Verdict) -> Lucide {
    match v {
        Verdict::Correct => Lucide::CircleCheck,
        Verdict::Wrong => Lucide::CircleX,
        Verdict::Unsure => Lucide::CircleQuestionMark,
    }
}

/// The verdict's colour and its soft tint: bluish green, vermillion and violet, which colour-blind eyes tell apart.
pub fn verdict_colors(v: Verdict, cx: &App) -> (Hsla, Hsla) {
    let (t, r) = (cx.theme(), theme::roles(cx));
    match v {
        Verdict::Correct => (t.success, t.green_light),
        Verdict::Wrong => (r.wrong, t.red_light),
        Verdict::Unsure => (r.unsure, r.unsure_soft),
    }
}

/// Pill: the verdict's icon and word on its soft tint.
pub fn verdict_tag(v: Verdict, small: bool, cx: &App) -> Tag {
    let text = match v {
        Verdict::Correct => "Correct",
        Verdict::Wrong => "Wrong",
        Verdict::Unsure => "Unsure",
    };
    widgets::tag(verdict_icon(v), text, verdict_colors(v, cx), small)
}
