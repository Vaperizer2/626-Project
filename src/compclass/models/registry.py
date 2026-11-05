"""One registry: model key -> builder returning ``(nn.Module, PreprocessSpec)``.

Every family the project compares is listed here exactly once. A builder takes ``num_classes`` (0 = return
pooled features instead of logits, used by the linear probe) and ``pretrained``; extra keyword arguments
(e.g. ``drop_path_rate``) go straight to ``timm.create_model``. The ``PreprocessSpec`` records the input size
and normalisation the weights expect -- the data pipeline, the ONNX sidecar and the OpenCV demo all read it
from here, so preprocessing can never drift between training and deployment.

Pretrained weights come from timm's Hugging Face hub entries (cached under ``$HF_HOME``); timm also hosts
the CLIP / SigLIP / DINOv2 *image towers*. ``zeroshot.py`` uses open_clip for the matching text towers.
Read the docstring of each ``_family`` function for the idea behind it; ``describe()`` prints them all.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from typing import Callable

from torch import nn

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessSpec:
    input_size: int  # square side length fed to the network
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    interpolation: str = "bicubic"  # resize kernel used when the weights were trained
    crop_pct: float = 0.875  # eval: resize short side to input_size / crop_pct, then centre-crop

    def to_dict(self) -> dict:
        d = asdict(self)
        d["resize_short_side"] = int(round(self.input_size / self.crop_pct))
        return d


Builder = Callable[..., tuple[nn.Module, PreprocessSpec]]


@dataclass(frozen=True)
class ModelEntry:
    key: str
    family: str
    builder: Builder
    timm_name: str | None = None
    zeroshot: tuple[str, str] | None = None  # (open_clip architecture, pretrained tag) when a text tower exists
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def doc(self) -> str:
        return inspect.getdoc(self.builder) or ""


REGISTRY: dict[str, ModelEntry] = {}


def _timm(name: str, num_classes: int, pretrained: bool, input_size: int | None = None,
          crop_pct: float | None = None, **kw) -> tuple[nn.Module, PreprocessSpec]:
    """Create a timm model and read its preprocessing from the pretrained config."""
    import timm

    if input_size is not None:
        # Transformers have a fixed token grid, so they must be told the new size (timm resamples the position
        # embeddings). CNNs are size-agnostic and their constructors reject ``img_size``: only the
        # preprocessing changes for them.
        try:
            model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, img_size=input_size, **kw)
        except TypeError:
            model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, **kw)
    else:
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, **kw)
    cfg = timm.data.resolve_model_data_config(model)
    spec = PreprocessSpec(input_size=input_size or cfg["input_size"][-1], mean=tuple(cfg["mean"]),
                          std=tuple(cfg["std"]), interpolation=cfg["interpolation"],
                          crop_pct=crop_pct if crop_pct is not None else cfg["crop_pct"])
    return model, spec


def _register(key: str, family: str, timm_name: str | None = None, zeroshot: tuple[str, str] | None = None,
              **fixed) -> Callable[[Builder], Builder]:
    """Decorator: register ``fn`` (whose docstring explains the family) under ``key``."""

    def deco(fn: Builder) -> Builder:
        def build(num_classes: int, pretrained: bool = True, **kw):
            return fn(num_classes=num_classes, pretrained=pretrained, **{**fixed, **kw})

        build.__doc__ = fn.__doc__
        REGISTRY[key] = ModelEntry(key, family, build, timm_name, zeroshot)
        return fn

    return deco


# ---------------------------------------------------------------------------------- 0. from scratch
@_register("scratch_cnn", "Scratch CNN")
def _scratch(num_classes: int, pretrained: bool, width: int = 32, dropout: float = 0.3, input_size: int = 160,
             residual: bool = False, blocks: int = 1, identity_init: bool = False, whiten_stem: bool = False):
    """Plain VGG-style CNN trained from random initialisation (see models/scratch_cnn.py).

    No pretraining: everything -- edges, textures, part detectors -- must be learned from ~3k images. It is the
    lower bound that shows how much the pretrained families gain from transfer learning. ``residual=True``
    swaps each stage's conv pair for a ResNet basic block; ``blocks`` sets blocks (or conv pairs) per stage.
    ``identity_init`` / ``whiten_stem`` select the airbench-style initialisations (see scratch_cnn.py).
    """
    from .scratch_cnn import ScratchCNN

    return ScratchCNN(num_classes, width, dropout, residual, blocks, identity_init, whiten_stem), PreprocessSpec(input_size, (0.5,) * 3, (0.5,) * 3, "bilinear", 0.875)


# ---------------------------------------------------------------------------------- 1. classic CNNs
@_register("resnet50", "ResNet (residual CNN)", "resnet50.a1_in1k")
def _resnet(num_classes: int, pretrained: bool, name: str = "resnet50.a1_in1k", **kw):
    """ResNet (He et al., 2015): convolutional stages joined by *residual* (skip) connections.

    Each block learns a correction F(x) added to its input, y = x + F(x). Gradients flow through the identity
    path, so very deep networks train stably. ResNet-50 uses 1x1-3x3-1x1 "bottleneck" blocks. The ``a1_in1k``
    weights come from the modern "ResNet strikes back" recipe (long training, strong augmentation) and are
    much stronger than the original torchvision weights. Strong inductive bias: locality + translation
    equivariance, which helps on small datasets.
    """
    return _timm(name, num_classes, pretrained, **kw)


@_register("efficientnet_v2_s", "EfficientNet (scaled CNN)", "tf_efficientnetv2_s.in21k_ft_in1k")
def _effnet(num_classes: int, pretrained: bool, name: str = "tf_efficientnetv2_s.in21k_ft_in1k", **kw):
    """EfficientNet-V2 (Tan & Le, 2021): a CNN found by architecture search and scaled jointly in depth/width/resolution.

    Early stages use fused MBConv (plain 3x3 conv + 1x1) for speed; later stages use MBConv: an inverted
    bottleneck with a cheap depthwise 3x3 conv and squeeze-and-excitation channel attention. Pretrained on
    ImageNet-21k then fine-tuned on 1k, native resolution 300 px (higher resolution is part of its accuracy).
    """
    return _timm(name, num_classes, pretrained, **kw)


@_register("mobilenetv3_large", "MobileNet (mobile CNN)", "mobilenetv3_large_100.ra_in1k")
def _mobilenet(num_classes: int, pretrained: bool, name: str = "mobilenetv3_large_100.ra_in1k", **kw):
    """MobileNet-V3 (Howard et al., 2019): a CNN designed for phones -- the latency end of the comparison.

    Depthwise-separable convolutions split a 3x3 conv into a per-channel spatial filter plus a 1x1 channel
    mixer (~8-9x fewer multiply-adds), wrapped in inverted residual blocks with squeeze-and-excitation and the
    cheap hard-swish activation. ~5M parameters; a natural candidate for a CPU/OpenCV demo.
    """
    return _timm(name, num_classes, pretrained, **kw)


@_register("convnext_tiny", "ConvNeXt (modernised CNN)", "convnext_tiny.fb_in22k_ft_in1k")
def _convnext(num_classes: int, pretrained: bool, name: str = "convnext_tiny.fb_in22k_ft_in1k", **kw):
    """ConvNeXt (Liu et al., 2022): a ResNet modernised step by step with Transformer design choices.

    Large 7x7 depthwise kernels, an inverted bottleneck MLP, LayerNorm instead of BatchNorm, GELU, fewer
    activations, and a "patchify" 4x4 stem. Shows that much of the ViT advantage comes from the training recipe
    and macro design rather than attention itself. ImageNet-22k pretraining.
    """
    return _timm(name, num_classes, pretrained, **kw)


# ---------------------------------------------------------------------------------- 2. transformers (supervised)
@_register("vit_b16", "ViT (vision transformer)", "vit_base_patch16_224.augreg_in21k_ft_in1k")
def _vit(num_classes: int, pretrained: bool, name: str = "vit_base_patch16_224.augreg_in21k_ft_in1k", **kw):
    """Vision Transformer (Dosovitskiy et al., 2020): the image is cut into 16x16 patches treated as tokens.

    Each patch is linearly embedded, a learned position embedding is added, and a stack of Transformer blocks
    (global multi-head self-attention + MLP) lets every patch attend to every other patch from layer 1. Weak
    inductive bias -> data hungry, so it relies on large-scale pretraining (here AugReg ImageNet-21k -> 1k).
    Fine-tuning benefits from layer-wise LR decay and a low learning rate.
    """
    return _timm(name, num_classes, pretrained, **kw)


@_register("swin_t", "Swin (hierarchical transformer)", "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k")
def _swin(num_classes: int, pretrained: bool, name: str = "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", **kw):
    """Swin Transformer (Liu et al., 2021): attention inside local 7x7 windows, shifted between layers.

    Window attention costs grow linearly with image size; shifting the windows every other block lets
    information cross window borders. Patch-merging layers build a CNN-like pyramid (4 stages, stride
    4->32), reintroducing locality and hierarchy into the transformer. ImageNet-22k pretraining.
    """
    return _timm(name, num_classes, pretrained, **kw)


# ---------------------------------------------------------------------------------- 3. self-supervised
@_register("dinov2_vitl14", "DINOv2 (self-supervised ViT)", "vit_large_patch14_dinov2.lvd142m",
           name="vit_large_patch14_dinov2.lvd142m", input_size=224)
@_register("dinov2_vitb14", "DINOv2 (self-supervised ViT)", "vit_base_patch14_dinov2.lvd142m",
           name="vit_base_patch14_dinov2.lvd142m", input_size=224)
@_register("dinov2_vits14", "DINOv2 (self-supervised ViT)", "vit_small_patch14_dinov2.lvd142m", input_size=224)
def _dinov2(num_classes: int, pretrained: bool, name: str = "vit_small_patch14_dinov2.lvd142m",
            input_size: int = 224, **kw):
    """DINOv2 (Oquab et al., 2023): ViT trained *without labels* on 142M curated images by self-distillation.

    A student network learns to match a momentum "teacher" across different crops of the same image
    (DINO loss) plus masked-patch prediction (iBOT). The resulting features are strong out of the box, so a
    plain linear probe on frozen features is often competitive with full fine-tuning. Native resolution is
    518 px; we use 224 px (16x16 patches of 14 px) for speed -- the position embeddings are interpolated.
    """
    return _timm(name, num_classes, pretrained, input_size=input_size, crop_pct=1.0, **kw)


# ---------------------------------------------------------------------------------- 4. vision-language
@_register("clip_vitl14_336", "CLIP (contrastive image-text)", "vit_large_patch14_clip_quickgelu_336.openai",
           zeroshot=("ViT-L-14-336", "openai"), name="vit_large_patch14_clip_quickgelu_336.openai")
@_register("clip_vitl14", "CLIP (contrastive image-text)", "vit_large_patch14_clip_quickgelu_224.openai",
           zeroshot=("ViT-L-14", "openai"), name="vit_large_patch14_clip_quickgelu_224.openai")
@_register("clip_vitb16", "CLIP (contrastive image-text)", "vit_base_patch16_clip_quickgelu_224.openai",
           zeroshot=("ViT-B-16", "openai"))
def _clip(num_classes: int, pretrained: bool, name: str = "vit_base_patch16_clip_quickgelu_224.openai", **kw):
    """CLIP (Radford et al., 2021): an image ViT and a text Transformer trained so matching pairs align.

    400M web (image, caption) pairs; a softmax contrastive loss pulls each image towards its own caption and
    away from the other captions in the batch. The image tower alone is a strong general feature extractor
    (fine-tune or linear probe); with the text tower it classifies *zero-shot* by comparing the image with
    prompts like "a photo of a resistor" (see zeroshot.py). Uses CLIP's own normalisation constants. OpenAI's
    weights were trained with the QuickGELU activation (x * sigmoid(1.702x)), so we use timm's ``_quickgelu``
    variants -- plain GELU silently degrades them.
    """
    return _timm(name, num_classes, pretrained, **kw)


@_register("siglip2_so400m_384", "SigLIP (sigmoid image-text)", "vit_so400m_patch16_siglip_384.v2_webli",
           zeroshot=("ViT-SO400M-16-SigLIP2-384", "webli"), name="vit_so400m_patch16_siglip_384.v2_webli")
@_register("siglip_so400m_384", "SigLIP (sigmoid image-text)", "vit_so400m_patch14_siglip_384.webli",
           zeroshot=("ViT-SO400M-14-SigLIP-384", "webli"), name="vit_so400m_patch14_siglip_384.webli")
@_register("siglip2_b16", "SigLIP (sigmoid image-text)", "vit_base_patch16_siglip_224.v2_webli",
           zeroshot=("ViT-B-16-SigLIP2", "webli"), name="vit_base_patch16_siglip_224.v2_webli")
@_register("siglip_b16", "SigLIP (sigmoid image-text)", "vit_base_patch16_siglip_224.webli",
           zeroshot=("ViT-B-16-SigLIP", "webli"))
def _siglip(num_classes: int, pretrained: bool, name: str = "vit_base_patch16_siglip_224.webli", **kw):
    """SigLIP (Zhai et al., 2023) / SigLIP 2 (2025): CLIP-style training with a pairwise *sigmoid* loss.

    Every (image, text) pair in the batch is an independent binary "match / no match" problem, so no
    batch-wide softmax normalisation is needed; it trains better at moderate batch sizes. SigLIP 2 adds
    captioning, self-distillation and masked-prediction objectives and multilingual data, improving dense and
    localisation features. Pools with an attention head ("MAP") rather than a CLS token; mean/std = 0.5.
    """
    return _timm(name, num_classes, pretrained, **kw)


# ---------------------------------------------------------------------------------- 5. 2025 backbones (probe sweep, 2026-11-04)
@_register("dinov3_vithp16", "DINOv3 (self-supervised ViT)", "vit_huge_plus_patch16_dinov3.lvd1689m",
           name="vit_huge_plus_patch16_dinov3.lvd1689m")
@_register("dinov3_vitl16", "DINOv3 (self-supervised ViT)", "vit_large_patch16_dinov3.lvd1689m",
           name="vit_large_patch16_dinov3.lvd1689m")
def _dinov3(num_classes: int, pretrained: bool, name: str = "vit_large_patch16_dinov3.lvd1689m", **kw):
    """DINOv3 (Simeoni et al., 2025): self-supervised ViT on 1.7B curated images with Gram anchoring.

    Successor of DINOv2 with register tokens and RoPE; distilled ViT-L/16 (300M) and ViT-H+/16 (840M)
    students of the 7B teacher. Native 256 px, ImageNet normalisation, crop 1.0. Weights are gated on the
    Hugging Face Hub (accept the DINOv3 licence once, then ``hf auth login``).
    """
    return _timm(name, num_classes, pretrained, **kw)


@_register("pe_core_g14_448", "Perception Encoder (contrastive image-text)", "vit_pe_core_gigantic_patch14_448.fb",
           zeroshot=("PE-Core-bigG-14-448", "meta"), name="vit_pe_core_gigantic_patch14_448.fb")
@_register("pe_core_l14_336", "Perception Encoder (contrastive image-text)", "vit_pe_core_large_patch14_336.fb",
           zeroshot=("PE-Core-L-14-336", "meta"), name="vit_pe_core_large_patch14_336.fb")
def _pe_core(num_classes: int, pretrained: bool, name: str = "vit_pe_core_large_patch14_336.fb", **kw):
    """Perception Encoder Core (Bolya et al., 2025): CLIP-style image-text training with a robust recipe.

    Meta's PE-Core L/14 at 336 px (300M) and G/14 at 448 px (1.9B); attention-pooled embedding, mean/std 0.5,
    crop 1.0. Among the strongest zero-shot and linear-probe image encoders of 2025.
    """
    return _timm(name, num_classes, pretrained, **kw)


@_register("aimv2_h14_336", "AIMv2 (autoregressive image-text)", "aimv2_huge_patch14_336.apple_pt",
           name="aimv2_huge_patch14_336.apple_pt")
@_register("aimv2_l14_336", "AIMv2 (autoregressive image-text)", "aimv2_large_patch14_336.apple_pt",
           name="aimv2_large_patch14_336.apple_pt")
def _aimv2(num_classes: int, pretrained: bool, name: str = "aimv2_large_patch14_336.apple_pt", **kw):
    """AIMv2 (Fini et al., 2024): ViT pre-trained to autoregressively reconstruct image patches and caption text.

    Apple's large (300M) and huge (600M) encoders at 336 px; CLIP normalisation, crop 1.0. Strong frozen
    features for recognition without a contrastive objective.
    """
    return _timm(name, num_classes, pretrained, **kw)


@_register("eva02_l14_clip_336", "EVA-02 CLIP (contrastive image-text)", "eva02_large_patch14_clip_336.merged2b",
           name="eva02_large_patch14_clip_336.merged2b")
def _eva02_clip(num_classes: int, pretrained: bool, name: str = "eva02_large_patch14_clip_336.merged2b", **kw):
    """EVA-02 CLIP (Sun et al., 2023): a masked-image-modelling-initialised ViT-L trained CLIP-style on 2B pairs.

    336 px, CLIP normalisation, crop 1.0. Pooled image embedding from the CLIP projection.
    """
    return _timm(name, num_classes, pretrained, **kw)


@_register("siglip2_so400m_512", "SigLIP (sigmoid image-text)", "vit_so400m_patch16_siglip_512.v2_webli",
           zeroshot=("ViT-SO400M-16-SigLIP2-512", "webli"), name="vit_so400m_patch16_siglip_512.v2_webli")
def _siglip2_512(num_classes: int, pretrained: bool, name: str = "vit_so400m_patch16_siglip_512.v2_webli", **kw):
    """SigLIP 2 So400m/16 at 512 px: the same encoder family as model C at a higher native resolution.

    Mean/std 0.5, crop 0.9, MAP pooling. Tests whether resolution, rather than the backbone, limits the probe.
    """
    return _timm(name, num_classes, pretrained, **kw)


@_register("metaclip2_g14_378", "MetaCLIP 2 (contrastive image-text)", "vit_gigantic_patch14_clip_378.metaclip2_worldwide",
           name="vit_gigantic_patch14_clip_378.metaclip2_worldwide")
@_register("dfn5b_h14_224", "DFN CLIP (contrastive image-text)", "vit_huge_patch14_clip_224.dfn5b",
           name="vit_huge_patch14_clip_224.dfn5b")
def _big_clip(num_classes: int, pretrained: bool, name: str = "vit_huge_patch14_clip_224.dfn5b", **kw):
    """Large CLIP-style encoders: DFN-5B ViT-H/14 (Fang et al., 2023, 630M) and MetaCLIP 2 ViT-bigG/14 at
    378 px (Chuang et al., 2025, 1.8B). CLIP normalisation, crop 1.0.
    """
    return _timm(name, num_classes, pretrained, **kw)


# ---------------------------------------------------------------------------------- 6. small image-text backbones (efficiency study, 2026-11-05)
# All registered with crop_pct 1.0 (short side resized to the input size, nothing cut off): the 2026-11-05 view control
# showed the deployed 0.9 crop costs SigLIP about 1.5 pt here, so the efficiency study compares every backbone at
# the same, better view. SigLIP So400m's crop-1.0 reference lives in results/cache/features/*-crop100.npy.
@_register("pe_core_b16_336", "Perception Encoder (contrastive image-text)", "vit_pe_core_base_patch16_224.fb",
           zeroshot=("PE-Core-B-16", "meta"), name="vit_pe_core_base_patch16_224.fb", input_size=336, crop_pct=1.0)
@_register("pe_core_b16_224", "Perception Encoder (contrastive image-text)", "vit_pe_core_base_patch16_224.fb",
           zeroshot=("PE-Core-B-16", "meta"), name="vit_pe_core_base_patch16_224.fb", crop_pct=1.0)
@_register("pe_core_s16_384", "Perception Encoder (contrastive image-text)", "vit_pe_core_small_patch16_384.fb",
           zeroshot=("PE-Core-S-16-384", "meta"), name="vit_pe_core_small_patch16_384.fb", crop_pct=1.0)
@_register("pe_core_t16_384", "Perception Encoder (contrastive image-text)", "vit_pe_core_tiny_patch16_384.fb",
           zeroshot=("PE-Core-T-16-384", "meta"), name="vit_pe_core_tiny_patch16_384.fb", crop_pct=1.0)
def _pe_core_small(num_classes: int, pretrained: bool, name: str = "vit_pe_core_base_patch16_224.fb", **kw):
    """PE-Core B/16 (93M, native 224 px; RoPE lets it run at 336), S/16 (24M) and T/16 (6M) at 384 px.

    The small members of Meta's Perception Encoder family (same MetaCLIP-style training as PE-Core L/G), the
    efficiency study's cheap end; timm features are pre-projection, see tools/views/heads.py.
    """
    return _timm(name, num_classes, pretrained, **kw)


@_register("siglip2_b16_384", "SigLIP (sigmoid image-text)", "vit_base_patch16_siglip_384.v2_webli",
           zeroshot=("ViT-B-16-SigLIP2-384", "webli"), name="vit_base_patch16_siglip_384.v2_webli", crop_pct=1.0)
def _siglip2_b16_384(num_classes: int, pretrained: bool, name: str = "vit_base_patch16_siglip_384.v2_webli", **kw):
    """SigLIP 2 B/16 at 384 px (93M): October's best base-size probe (0.76 at 224 px), at full resolution.

    Same sigmoid image-text pre-training family as the deployed So400m tower, one size class down; the efficiency
    study's like-for-like small SigLIP.
    """
    return _timm(name, num_classes, pretrained, **kw)


@_register("mobileclip2_s3", "MobileCLIP 2 (contrastive image-text, mobile)", "fastvit_mci3.apple_mclip2_dfndr2b",
           zeroshot=("MobileCLIP2-S3", "dfndr2b"), name="fastvit_mci3.apple_mclip2_dfndr2b", crop_pct=1.0)
@_register("mobileclip2_s2", "MobileCLIP 2 (contrastive image-text, mobile)", "fastvit_mci2.apple_mclip2_dfndr2b",
           zeroshot=("MobileCLIP2-S2", "dfndr2b"), name="fastvit_mci2.apple_mclip2_dfndr2b", crop_pct=1.0)
@_register("mobileclip2_b", "MobileCLIP 2 (contrastive image-text, mobile)", "vit_base_mci_224.apple_mclip2_dfndr2b",
           zeroshot=("MobileCLIP2-B", "dfndr2b"), name="vit_base_mci_224.apple_mclip2_dfndr2b", crop_pct=1.0)
def _mobileclip2(num_classes: int, pretrained: bool, name: str = "fastvit_mci2.apple_mclip2_dfndr2b",
                 trainable_stages: int | None = None, **kw):
    """MobileCLIP 2 (Apple, 2025): image-text encoders built for on-device latency, trained on DFNDR-2B.

    S2 (35M) and S3 (123M) are FastViT hybrids at 256 px; B (86M) is a ViT-B/16 at 224 px. Mean/std 0, 1.

    ``trainable_stages: N`` (FastViT only) is the partial fine-tune analogue of ``train.trainable_blocks``: the stem
    and all but the last N of the five stages stay frozen (``frozen_param_ids``, read by compclass.train); the
    last N stages, ``final_conv`` and the head are trained. S3's stage 4 alone is 59M of its 123M parameters.
    """
    model, spec = _timm(name, num_classes, pretrained, **kw)
    if trainable_stages is not None:
        if not hasattr(model, "stages"):
            raise ValueError(f"{name} has no stages; use train.trainable_blocks for the ViT-B variant")
        n = len(model.stages)
        frozen = [model.stem, *list(model.stages)[: max(0, n - trainable_stages)]]
        model.frozen_param_ids = {id(p) for mod in frozen for p in mod.parameters()}
    return model, spec


@_register("eva02_b16_clip_224", "EVA-02 CLIP (contrastive image-text)", "eva02_base_patch16_clip_224.merged2b",
           zeroshot=("EVA02-B-16", "merged2b_s8b_b131k"), name="eva02_base_patch16_clip_224.merged2b", crop_pct=1.0)
@_register("metaclip_b16_224", "MetaCLIP (contrastive image-text)", "vit_base_patch16_clip_quickgelu_224.metaclip_2pt5b",
           zeroshot=("ViT-B-16-quickgelu", "metaclip_fullcc"), name="vit_base_patch16_clip_quickgelu_224.metaclip_2pt5b", crop_pct=1.0)
@_register("dfn2b_b16_224", "DFN CLIP (contrastive image-text)", "vit_base_patch16_clip_224.dfn2b",
           zeroshot=("ViT-B-16-quickgelu", "dfn2b"), name="vit_base_patch16_clip_224.dfn2b", crop_pct=1.0)
def _base_clip(num_classes: int, pretrained: bool, name: str = "vit_base_patch16_clip_224.dfn2b", **kw):
    """Base-size (86M) CLIP-style encoders at 224 px: DFN-2B, MetaCLIP 2.5B and EVA-02 B/16 (merged-2B).

    Three ViT-B/16 image towers that differ only in their contrastive pre-training data and recipe; in the
    efficiency study they bracket what a base-size CLIP tower gives on the 20 classes.
    """
    return _timm(name, num_classes, pretrained, **kw)


@_register("convnext_base_clip_320", "ConvNeXt CLIP (contrastive image-text, convolutional)", "convnext_base.clip_laiona_augreg_320",
           zeroshot=("convnext_base_w_320", "laion_aesthetic_s13b_b82k_augreg"), name="convnext_base.clip_laiona_augreg_320", crop_pct=1.0)
def _convnext_clip(num_classes: int, pretrained: bool, name: str = "convnext_base.clip_laiona_augreg_320", **kw):
    """ConvNeXt-Base trained CLIP-style on LAION-Aesthetic at 320 px (88M): the convolutional candidate, the
    friendliest shape for TensorRT INT8."""
    return _timm(name, num_classes, pretrained, **kw)


# ---------------------------------------------------------------------------------- API
def build_model(key: str, num_classes: int, pretrained: bool = True, **kw) -> tuple[nn.Module, PreprocessSpec]:
    """Build ``key`` with a fresh ``num_classes``-way head (0 = features only)."""
    if key not in REGISTRY:
        raise KeyError(f"unknown model {key!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[key].builder(num_classes=num_classes, pretrained=pretrained, **kw)


def list_models() -> list[str]:
    return list(REGISTRY)


def describe() -> str:
    """All families with their explainers, in registry order (for docs and notebooks)."""
    return "\n\n".join(f"## {e.key}  [{e.family}]  timm={e.timm_name}\n{e.doc}" for e in REGISTRY.values())
