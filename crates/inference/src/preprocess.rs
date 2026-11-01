//! Image decoding and the model's eval transform, reproducing `compclass.demo.open_image` + `preprocess`
//! (torchvision `Resize(short side, BICUBIC) -> CenterCrop -> ToTensor -> Normalize` on a PIL image).
//!
//! The resize is a line-by-line port of Pillow's `libImaging/Resample.c` for 8-bit images (Keys cubic
//! a = -0.5, support scaled by the downscale factor ("antialias"), coefficients normalised in double and then
//! quantised to 22-bit fixed point, horizontal pass first, each pass rounded and clipped to u8). It is
//! bit-identical to `PIL.Image.resize(..., Image.BICUBIC)`, so the only remaining source of difference
//! from the Python pipeline is the JPEG/PNG decoder.

use std::io::Cursor;

use anyhow::{Context, Result, bail};
use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage, RgbaImage, metadata::Orientation};
use rayon::prelude::*;

use crate::manifest::{Filter, Preprocess};

/// Pillow: `#define PRECISION_BITS (32 - 8 - 2)`
const PRECISION_BITS: u32 = 22;

// ------------------------------------------------------------------------------------------- decoding
/// Decode bytes (JPEG/PNG/WebP/GIF first frame/BMP/TIFF), apply the EXIF orientation, flatten alpha onto
/// white and return 8-bit RGB, like `compclass.demo.open_image`.
pub fn decode(bytes: &[u8]) -> Result<RgbImage> {
    let reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format().context("reading image header")?;
    if reader.format().is_none() {
        bail!("unrecognised image format (supported: JPEG, PNG, WebP, GIF, BMP, TIFF)");
    }
    let is_jpeg = reader.format() == Some(image::ImageFormat::Jpeg);
    let mut decoder = reader.into_decoder().context("unsupported or corrupt image")?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut img = match is_jpeg.then(|| decode_jpeg_turbo(bytes)).flatten() {
        Some(rgb) => DynamicImage::ImageRgb8(rgb),
        None => DynamicImage::from_decoder(decoder).context("decoding image")?,
    };
    img.apply_orientation(orientation);
    Ok(flatten_to_rgb(img))
}

/// JPEG pixels via libjpeg-turbo (Pillow's decoder: ISLOW IDCT, fancy upsampling); None = use the Rust decoder.
#[cfg(feature = "turbojpeg")]
fn decode_jpeg_turbo(bytes: &[u8]) -> Option<RgbImage> {
    let im = turbojpeg::decompress(bytes, turbojpeg::PixelFormat::RGB).ok()?;
    let (w, h) = (im.width, im.height);
    let mut buf = Vec::with_capacity(w * h * 3);
    for row in im.pixels.chunks(im.pitch).take(h) {
        buf.extend_from_slice(&row[..w * 3]);
    }
    RgbImage::from_raw(w as u32, h as u32, buf)
}

#[cfg(not(feature = "turbojpeg"))]
fn decode_jpeg_turbo(_bytes: &[u8]) -> Option<RgbImage> {
    None
}

/// Which JPEG decoder this build uses (reported by `--version` and the parity script).
pub const JPEG_DECODER: &str = if cfg!(feature = "turbojpeg-vendored") {
    "libjpeg-turbo (static, vendored)"
} else if cfg!(feature = "turbojpeg") {
    "libjpeg-turbo (turbojpeg feature)"
} else {
    "zune-jpeg (image crate)"
};

/// Grayscale -> RGB by replication; images with alpha are composited on opaque white exactly like
/// Pillow's `Image.alpha_composite(white, rgba)` (libImaging/AlphaComposite.c integer arithmetic).
pub fn flatten_to_rgb(img: DynamicImage) -> RgbImage {
    if !img.color().has_alpha() {
        return img.to_rgb8();
    }
    let rgba: RgbaImage = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let mut out = RgbImage::new(w, h);
    for (o, p) in out.pixels_mut().zip(rgba.pixels()) {
        let a = p[3] as u32;
        for c in 0..3 {
            o[c] = composite_on_white(p[c], a);
        }
    }
    out
}

/// One channel of Pillow's alpha_composite(dst = opaque white, src = (v, a)).
#[inline]
fn composite_on_white(v: u8, a: u32) -> u8 {
    const PB: u32 = 7; // AlphaComposite.c: PRECISION_BITS 7
    if a == 0 {
        return 255;
    }
    let outa255 = a * 255 + 255 * (255 - a);
    let coef1 = a * 255 * 255 * (1 << PB) / outa255;
    let coef2 = 255 * (1 << PB) - coef1;
    let tmp = v as u32 * coef1 + 255 * coef2 + (0x80 << PB);
    let div255 = ((tmp >> 8) + tmp) >> 8; // SHIFTFORDIV255
    (div255 >> PB) as u8
}

// --------------------------------------------------------------------------------------------- resize
/// torchvision `Resize(short)`: the short side becomes `short`, the long side is truncated.
pub fn resized_size(w: u32, h: u32, short: u32) -> (u32, u32) {
    if w <= h {
        (short, (short as u64 * h as u64 / w as u64) as u32)
    } else {
        ((short as u64 * w as u64 / h as u64) as u32, short)
    }
}

/// Pillow `bicubic_filter` (a = -0.5), support 2.
#[inline]
fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Pillow `bilinear_filter` (triangle), support 1.
#[inline]
fn bilinear(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 { 1.0 - x } else { 0.0 }
}

fn support(filter: Filter) -> f64 {
    match filter {
        Filter::PillowBicubic => 2.0,
        Filter::PillowBilinear => 1.0,
    }
}

#[inline]
fn kernel(filter: Filter, x: f64) -> f64 {
    match filter {
        Filter::PillowBicubic => bicubic(x),
        Filter::PillowBilinear => bilinear(x),
    }
}

/// Pillow `precompute_coeffs` + `normalize_coeffs_8bpc` for a box [0, in_size) -> out_size.
/// Returns (ksize, bounds as (xmin, count) per output, fixed-point coefficients out_size*ksize).
fn precompute_coeffs(in_size: u32, out_size: u32, filter: Filter) -> (usize, Vec<(usize, usize)>, Vec<i32>) {
    let (in0, in1) = (0.0f64, in_size as f64);
    let scale = (in1 - in0) / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = support(filter) * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size as usize);
    let mut kk = vec![0i32; out_size as usize * ksize];
    let mut k = vec![0f64; ksize];
    let ss = 1.0 / filterscale;
    for xx in 0..out_size as usize {
        let center = in0 + (xx as f64 + 0.5) * scale;
        // C `(int)` truncates toward zero
        let mut xmin = (center - support + 0.5) as i64;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i64;
        if xmax > in_size as i64 {
            xmax = in_size as i64;
        }
        let xmax = (xmax - xmin).max(0) as usize;
        let xmin = xmin as usize;
        let mut ww = 0.0f64;
        for (x, kx) in k.iter_mut().enumerate().take(xmax) {
            let w = kernel(filter, (x as f64 + xmin as f64 - center + 0.5) * ss);
            *kx = w;
            ww += w;
        }
        let row = &mut kk[xx * ksize..(xx + 1) * ksize];
        for x in 0..xmax {
            let v = if ww != 0.0 { k[x] / ww } else { k[x] };
            let f = v * (1u32 << PRECISION_BITS) as f64;
            row[x] = if v < 0.0 { (-0.5 + f) as i32 } else { (0.5 + f) as i32 };
        }
        bounds.push((xmin, xmax));
    }
    (ksize, bounds, kk)
}

#[inline]
fn clip8(v: i32) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Bit-exact port of `PIL.Image.resize((w, h), Image.BICUBIC)` for 8-bit RGB (no box, no reducing_gap).
#[cfg(test)]
pub fn pillow_bicubic_resize(src: &RgbImage, out_w: u32, out_h: u32) -> RgbImage {
    pillow_resize(src, out_w, out_h, Filter::PillowBicubic)
}

/// Bit-exact port of `PIL.Image.resize((w, h), filter)` for 8-bit RGB, BICUBIC or BILINEAR (same separable
/// `Resample.c` path; only the kernel and its support differ).
pub fn pillow_resize(src: &RgbImage, out_w: u32, out_h: u32, filter: Filter) -> RgbImage {
    let (in_w, in_h) = src.dimensions();
    if (in_w, in_h) == (out_w, out_h) {
        return src.clone();
    }
    let need_h = out_w != in_w;
    let need_v = out_h != in_h;
    let (ks_h, bounds_h, kk_h) = precompute_coeffs(in_w, out_w, filter);
    let (ks_v, mut bounds_v, kk_v) = precompute_coeffs(in_h, out_h, filter);

    // Horizontal pass first, only over the source rows the vertical pass will read (as Pillow does).
    let (tmp_w, tmp_h, tmp): (u32, u32, Vec<u8>) = if need_h {
        let y_first = bounds_v[0].0;
        let last = bounds_v[out_h as usize - 1];
        let y_last = last.0 + last.1;
        for b in bounds_v.iter_mut() {
            b.0 -= y_first;
        }
        let rows = y_last - y_first;
        let in_raw = src.as_raw();
        let in_stride = in_w as usize * 3;
        let out_stride = out_w as usize * 3;
        let mut buf = vec![0u8; rows * out_stride];
        buf.par_chunks_mut(out_stride).enumerate().for_each(|(r, out_row)| {
            let in_row = &in_raw[(r + y_first) * in_stride..(r + y_first + 1) * in_stride];
            for (xx, &(xmin, n)) in bounds_h.iter().enumerate() {
                let k = &kk_h[xx * ks_h..xx * ks_h + n];
                let mut s = [1i32 << (PRECISION_BITS - 1); 3];
                for (x, &kx) in k.iter().enumerate() {
                    let p = (xmin + x) * 3;
                    s[0] += in_row[p] as i32 * kx;
                    s[1] += in_row[p + 1] as i32 * kx;
                    s[2] += in_row[p + 2] as i32 * kx;
                }
                out_row[xx * 3] = clip8(s[0]);
                out_row[xx * 3 + 1] = clip8(s[1]);
                out_row[xx * 3 + 2] = clip8(s[2]);
            }
        });
        (out_w, rows as u32, buf)
    } else {
        (in_w, in_h, src.as_raw().clone())
    };
    if !need_v {
        return RgbImage::from_raw(tmp_w, tmp_h, tmp).expect("buffer size");
    }
    let stride = tmp_w as usize * 3;
    let mut out = vec![0u8; out_h as usize * stride];
    out.par_chunks_mut(stride).enumerate().for_each(|(yy, out_row)| {
        let (ymin, n) = bounds_v[yy];
        let k = &kk_v[yy * ks_v..yy * ks_v + n];
        for (xx, o) in out_row.iter_mut().enumerate() {
            let mut s = 1i32 << (PRECISION_BITS - 1);
            for (y, &ky) in k.iter().enumerate() {
                s += tmp[(ymin + y) * stride + xx] as i32 * ky;
            }
            *o = clip8(s);
        }
    });
    RgbImage::from_raw(tmp_w, out_h, out).expect("buffer size")
}

// ----------------------------------------------------------------------------------------- transform
/// torchvision CenterCrop offsets: `int(round((W - w) / 2))` with Python's round-half-to-even.
pub fn center_crop_offsets(w: u32, h: u32, crop_w: u32, crop_h: u32) -> (u32, u32) {
    let l = ((w - crop_w) as f64 / 2.0).round_ties_even() as u32;
    let t = ((h - crop_h) as f64 / 2.0).round_ties_even() as u32;
    (l, t)
}

/// RGB image -> float32 NCHW tensor data of length 3*h*w, identical to `compclass.demo.preprocess`.
pub fn to_tensor(img: &RgbImage, pre: &Preprocess) -> Result<Vec<f32>> {
    let Preprocess::ResizeShortCenterCrop { short, crop: (cw, ch), filter, scale, mean, std } = pre;
    let (cw, ch) = (*cw, *ch);
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        bail!("empty image");
    }
    let (nw, nh) = resized_size(w, h, *short);
    if nw < cw || nh < ch {
        bail!("image too small after resize ({nw}x{nh})");
    }
    let resized = pillow_resize(img, nw, nh, *filter);
    let (left, top) = center_crop_offsets(nw, nh, cw, ch);
    // ToTensor: x / float32(1/scale); then (x - mean) / std, all in float32 like numpy.
    let denom = (1.0 / scale) as f32;
    let (cw, ch) = (cw as usize, ch as usize);
    let plane = cw * ch;
    let mut out = vec![0f32; 3 * plane];
    let raw = resized.as_raw();
    let stride = nw as usize * 3;
    for y in 0..ch {
        let row = &raw[(y + top as usize) * stride + left as usize * 3..];
        for x in 0..cw {
            for c in 0..3 {
                let v = row[x * 3 + c] as f32 / denom;
                out[c * plane + y * cw + x] = (v - mean[c]) / std[c];
            }
        }
    }
    Ok(out)
}

/// Downscaled RGBA copy for display (long side <= `max_side`), converted to BGRA as GPUI's RenderImage expects.
/// (GUI only; the model input never goes through this.)
pub fn display_bgra(img: &RgbImage, max_side: u32) -> RgbaImage {
    let (w, h) = img.dimensions();
    let scale = (max_side as f64 / w.max(h) as f64).min(1.0);
    let (tw, th) = (((w as f64 * scale).round() as u32).max(1), ((h as f64 * scale).round() as u32).max(1));
    // `imageops::thumbnail` (box filter, fast) reads the source in place: no full-size copy.
    let small;
    let src: &RgbImage = if (tw, th) == (w, h) {
        img
    } else {
        small = image::imageops::thumbnail(img, tw, th);
        &small
    };
    let mut out = Vec::with_capacity(src.len() / 3 * 4);
    for p in src.pixels() {
        out.extend_from_slice(&[p.0[2], p.0[1], p.0[0], 255]);
    }
    RgbaImage::from_raw(src.width(), src.height(), out).expect("buffer size matches")
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn pre() -> Preprocess {
        Preprocess::ResizeShortCenterCrop {
            short: 427,
            crop: (384, 384),
            filter: Filter::PillowBicubic,
            scale: 0.00392156862745098,
            mean: [0.5; 3],
            std: [0.5; 3],
        }
    }

    #[test]
    fn resized_size_truncates_like_torchvision() {
        assert_eq!(resized_size(640, 480, 427), (569, 427)); // int(427*640/480)=569.33 -> 569
        assert_eq!(resized_size(480, 640, 427), (427, 569));
        assert_eq!(resized_size(1600, 1600, 427), (427, 427));
        assert_eq!(resized_size(427, 1000, 427), (427, 1000));
    }

    #[test]
    fn crop_offsets_round_half_to_even() {
        // (W-S)/2 = 21.5 -> 22 (even), 20.5 -> 20 (even), 0 -> 0
        assert_eq!(center_crop_offsets(384 + 43, 384, 384, 384), (22, 0));
        assert_eq!(center_crop_offsets(384 + 41, 384 + 2, 384, 384), (20, 1));
        assert_eq!(center_crop_offsets(427, 300, 300, 200), (64, 50), "non-square crop");
    }

    #[test]
    fn bicubic_kernel_values() {
        assert_eq!(bicubic(0.0), 1.0);
        assert_eq!(bicubic(1.0), 0.0);
        assert_eq!(bicubic(2.0), 0.0);
        assert!((bicubic(0.5) - 0.5625).abs() < 1e-12);
        assert!((bicubic(1.5) + 0.0625).abs() < 1e-12);
    }

    #[test]
    fn bilinear_kernel_and_resize() {
        assert_eq!(bilinear(0.0), 1.0);
        assert_eq!(bilinear(0.25), 0.75);
        assert_eq!(bilinear(1.0), 0.0);
        let img = RgbImage::from_pixel(300, 200, image::Rgb([7, 99, 250]));
        let r = pillow_resize(&img, 183, 122, Filter::PillowBilinear);
        assert!(r.pixels().all(|p| p.0 == [7, 99, 250]));
        // a horizontal ramp stays monotonic and differs from the bicubic result somewhere
        let ramp = RgbImage::from_fn(64, 8, |x, _| image::Rgb([(x * 4) as u8; 3]));
        let a = pillow_resize(&ramp, 23, 8, Filter::PillowBilinear);
        let b = pillow_resize(&ramp, 23, 8, Filter::PillowBicubic);
        assert!(a.pixels().zip(a.pixels().skip(1)).take(22).all(|(p, q)| p.0[0] <= q.0[0]));
        assert_ne!(a.as_raw(), b.as_raw());
    }

    #[test]
    fn constant_image_stays_constant() {
        let img = RgbImage::from_pixel(1000, 700, Rgb([10, 128, 250]));
        let r = pillow_bicubic_resize(&img, 610, 427);
        assert_eq!(r.dimensions(), (610, 427));
        assert!(r.pixels().all(|p| p.0 == [10, 128, 250]));
    }

    #[test]
    fn tensor_shape_and_normalisation() {
        let mut img = RgbImage::from_pixel(800, 600, Rgb([255, 0, 0]));
        img.put_pixel(0, 0, Rgb([0, 255, 0])); // outside the crop after resize
        let t = to_tensor(&img, &pre()).unwrap();
        assert_eq!(t.len(), 3 * 384 * 384);
        let plane = 384 * 384;
        // centre pixel: R=255 -> +1, G=B=0 -> -1
        let c = 192 * 384 + 192;
        assert_eq!(t[c], 1.0);
        assert_eq!(t[plane + c], -1.0);
        assert_eq!(t[2 * plane + c], -1.0);
        assert!(t.iter().all(|v| (-1.0..=1.0).contains(v)));
    }

    #[test]
    fn small_and_upscaled_images_work() {
        let img = RgbImage::from_pixel(50, 30, Rgb([128, 128, 128]));
        let t = to_tensor(&img, &pre()).unwrap();
        assert_eq!(t.len(), 3 * 384 * 384);
        let v = 128f32 / 255.0;
        assert!((t[0] - (v - 0.5) / 0.5).abs() < 1e-6);
    }

    #[test]
    fn alpha_composites_on_white() {
        assert_eq!(composite_on_white(0, 0), 255);
        assert_eq!(composite_on_white(0, 255), 0);
        assert_eq!(composite_on_white(200, 255), 200);
        let half = composite_on_white(0, 128) as i32;
        assert!((half - 127).abs() <= 1);
        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, image::Rgba([0, 0, 0, 0])));
        assert!(flatten_to_rgb(img).pixels().all(|p| p.0 == [255, 255, 255]));
    }

    #[test]
    fn grayscale_is_replicated() {
        let img = DynamicImage::ImageLuma8(image::GrayImage::from_pixel(3, 3, image::Luma([77])));
        assert!(flatten_to_rgb(img).pixels().all(|p| p.0 == [77, 77, 77]));
    }

    #[test]
    fn decodes_png_with_orientation_and_rejects_garbage() {
        let mut png = Vec::new();
        DynamicImage::ImageRgb8(RgbImage::from_pixel(4, 2, Rgb([1, 2, 3])))
            .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let d = decode(&png).unwrap();
        assert_eq!(d.dimensions(), (4, 2));
        assert!(decode(b"definitely not an image").is_err());
    }
}
