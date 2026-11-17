"""Model constructors."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


def resnet18_cifar(num_classes: int = 10, alpha: float = 1.0) -> nn.Module:
    """CIFAR-style ResNet18 with its weights initialised at scale ``alpha``.

    The stem is a 3x3 stride-1 conv (no bias) and the max-pool is removed.
    After torchvision's default init, every Conv2d and Linear weight is
    multiplied by ``alpha``. Biases and BatchNorm parameters are left unchanged.
    """
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, model.conv1.out_channels, kernel_size=3, stride=1,
                            padding=1, bias=False)
    model.maxpool = nn.Identity()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.weight.mul_(alpha)
    return model
