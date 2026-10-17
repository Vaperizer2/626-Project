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
"""

from __future__ import annotations

import torch
from torch import nn


def conv_bn_relu(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out),
                         nn.ReLU(inplace=True))


class ScratchCNN(nn.Module):
    def __init__(self, num_classes: int, width: int = 32, dropout: float = 0.3):
        super().__init__()
        w = width
        self.stem = conv_bn_relu(3, w, stride=2)
        chans = [w, w, 2 * w, 4 * w, 8 * w]
        stages = []
        for i in range(4):
            block = [conv_bn_relu(chans[i], chans[i + 1]), conv_bn_relu(chans[i + 1], chans[i + 1])]
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

    def get_classifier(self) -> nn.Module:  # same name as timm, so train.py treats every model alike
        return self.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(self.stem(x))))
