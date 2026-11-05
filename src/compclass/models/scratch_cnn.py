"""A small CNN written from first principles -- the reference point every pretrained family is compared to.

Architecture (input 3 x 160 x 160, width w = 32 by default):

    stem   conv3x3(3 -> w), stride 2                       160 -> 80
    stage1 2 x [conv3x3 -> BN -> ReLU] at w,   then pool   80 -> 40
    stage2 2 x [conv3x3 -> BN -> ReLU] at 2w,  then pool   40 -> 20
    stage3 2 x [conv3x3 -> BN -> ReLU] at 4w,  then pool   20 -> 10
    stage4 2 x [conv3x3 -> BN -> ReLU] at 8w               10 -> 10
    head   global average pool -> dropout -> linear(8w -> num_classes)

Each idea is one line of code: 3x3 convolutions build local edge/texture detectors, BatchNorm keeps the
activations well-scaled so a higher learning rate is stable, max-pooling halves resolution so later layers
see larger regions, and global average pooling makes the head translation-invariant with few parameters.
Trained from random init on ~3k images it will badly underfit/overfit compared with ImageNet-pretrained
models; that gap *is* the lesson about transfer learning.

Variant (``residual=True``): each stage's pair of convolutions becomes a ResNet *basic block*,
y = ReLU(BN(conv(ReLU(BN(conv(x))))) + shortcut(x)), with a 1x1 projection shortcut where the channel count
changes. ``blocks`` sets how many such blocks (or, in the plain variant, conv pairs) each stage has; the
default ``blocks=1`` keeps the parameter count of the plain network (+ three small 1x1 projections). The
last BatchNorm of every block starts at zero so each block is the identity at initialisation.

Initialisation options (from the CIFAR-10 "airbench" recipe, Jordan 2024, arXiv 2404.00498):

- ``identity_init=True``: every 3x3 convolution whose input and output widths match starts as the identity
  (Dirac delta), so at initialisation each stage passes its input through unchanged and only the
  width-changing convolutions mix channels. Pure initialisation; all weights still train.
- ``whiten_stem=True``: the stem filters are set to the PCA-whitening basis of 3x3 colour patches drawn from the
  training images (``init_from_data`` is called by ``train.py`` once the training loader exists). The ``width``
  filters are the top ``width/2`` eigenvectors and their negatives, scaled by 1/sqrt(eigenvalue), and the stem
  is then frozen, as in the source recipe.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def conv_bn_relu(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out),
                         nn.ReLU(inplace=True))


class BasicBlock(nn.Module):
    """ResNet basic block (He et al., 2015): two 3x3 convs whose output is *added* to the input."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out))
        self.shortcut = (nn.Identity() if c_in == c_out else
                         nn.Sequential(nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.shortcut(x))


class ScratchCNN(nn.Module):
    def __init__(self, num_classes: int, width: int = 32, dropout: float = 0.3, residual: bool = False,
                 blocks: int = 1, identity_init: bool = False, whiten_stem: bool = False):
        super().__init__()
        w = width
        self.whiten_stem = whiten_stem
        self.frozen_param_ids: set[int] = set()  # read by train.py; filled by init_from_data
        self.stem = conv_bn_relu(3, w, stride=2)
        chans = [w, w, 2 * w, 4 * w, 8 * w]
        stages = []
        for i in range(4):
            block: list[nn.Module] = []
            for b in range(blocks):
                c_in = chans[i] if b == 0 else chans[i + 1]
                if residual:
                    block.append(BasicBlock(c_in, chans[i + 1]))
                else:
                    block += [conv_bn_relu(c_in, chans[i + 1]), conv_bn_relu(chans[i + 1], chans[i + 1])]
            if i < 3:
                block.append(nn.MaxPool2d(2))
            stages.append(nn.Sequential(*block))
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.num_features = chans[-1]
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(self.num_features, num_classes) if num_classes else nn.Identity())
        for m in self.modules():  # He init suits ReLU networks
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        for m in self.modules():  # zero-init the residual branch so every block starts as the identity
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.body[-1].weight)
        if identity_init:
            for m in self.features.modules():
                if isinstance(m, nn.Conv2d) and m.in_channels == m.out_channels and m.kernel_size == (3, 3):
                    nn.init.dirac_(m.weight)

    @torch.no_grad()
    def init_from_data(self, loader, device: torch.device, n_batches: int = 8, eps: float = 1e-3) -> None:
        """Data-dependent initialisation: PCA-whitening filters for the stem (no-op unless ``whiten_stem``)."""
        if not self.whiten_stem:
            return
        conv: nn.Conv2d = self.stem[0]
        pairs = conv.out_channels // 2
        if conv.out_channels % 2 or pairs > 27:
            raise ValueError(f"whiten_stem needs an even width <= 54, got {conv.out_channels}")
        patches = []
        for i, (x, _) in enumerate(loader):
            if i >= n_batches:
                break
            # every 3x3 colour patch at stride 2, as the stem sees them: [B*L, 27]
            p = F.unfold(x.to(device).float(), 3, padding=1, stride=2)
            patches.append(p.transpose(1, 2).reshape(-1, p.shape[1]))
        p = torch.cat(patches)
        p = p - p.mean(0, keepdim=True)
        cov = p.T @ p / p.shape[0]
        vals, vecs = torch.linalg.eigh(cov)  # ascending
        vals, vecs = vals.flip(0)[:pairs], vecs.flip(1)[:, :pairs]
        filt = (vecs / (vals + eps * vals[0]).sqrt()).T.reshape(pairs, 3, 3, 3)  # [pairs, C, kh, kw]
        conv.weight.copy_(torch.cat([filt, -filt]).to(conv.weight.dtype))
        conv.weight.requires_grad_(False)
        self.frozen_param_ids = {id(conv.weight)}

    def get_classifier(self) -> nn.Module:  # same name as timm, so train.py treats every model alike
        return self.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(self.stem(x))))
