#[path = "../../../vendor/gpui-pre-windows/src/shader_resources.rs"]
mod shader_resources;

use anyhow::Context;
use shader_resources::{RawShaderBytes, ShaderModule::*, ShaderTarget::*};

fn main() -> anyhow::Result<()> {
    for module in [
        Quad,
        Shadow,
        Underline,
        PathRasterization,
        PathSprite,
        MonochromeSprite,
        SubpixelSprite,
        PolychromeSprite,
        EmojiRasterization,
    ] {
        for target in [Vertex, Fragment] {
            let shader = RawShaderBytes::new(module, target).with_context(|| format!("{module:?}/{target:?}"))?;
            assert!(shader.as_bytes().starts_with(b"DXBC"), "{module:?}/{target:?}");
            println!("{module:?}/{target:?}: compiled {} bytes", shader.as_bytes().len());
        }
    }
    println!("PASS: all 18 shader programs compiled to DXBC");
    Ok(())
}
