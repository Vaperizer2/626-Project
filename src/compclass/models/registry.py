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
        kw["img_size"] = input_size  # ViTs resample their position embeddings to the new grid
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
def _scratch(num_classes: int, pretrained: bool, width: int = 32, dropout: float = 0.3, input_size: int = 160):
    """Plain VGG-style CNN trained from random initialisation (see models/scratch_cnn.py).

    No pretraining: everything -- edges, textures, part detectors -- must be learned from ~3k images. It is the
    lower bound that shows how much the pretrained families gain from transfer learning.
    """
    from .scratch_cnn import ScratchCNN

    return ScratchCNN(num_classes, width, dropout), PreprocessSpec(input_size, (0.5,) * 3, (0.5,) * 3, "bilinear", 0.875)


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
@_register("clip_vitl14", "CLIP (contrastive image-text)", "vit_large_patch14_clip_224.openai",
           zeroshot=("ViT-L-14", "openai"), name="vit_large_patch14_clip_224.openai")
@_register("clip_vitb16", "CLIP (contrastive image-text)", "vit_base_patch16_clip_224.openai",
           zeroshot=("ViT-B-16", "openai"))
def _clip(num_classes: int, pretrained: bool, name: str = "vit_base_patch16_clip_224.openai", **kw):
    """CLIP (Radford et al., 2021): an image ViT and a text Transformer trained so matching pairs align.

    400M web (image, caption) pairs; a softmax contrastive loss pulls each image towards its own caption and
    away from the other captions in the batch. The image tower alone is a strong general feature extractor
    (fine-tune or linear probe); with the text tower it classifies *zero-shot* by comparing the image with
    prompts like "a photo of a resistor" (see zeroshot.py). Uses CLIP's own normalisation constants.
    """
    return _timm(name, num_classes, pretrained, **kw)


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
