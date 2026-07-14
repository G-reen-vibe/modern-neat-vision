"""Baseline models wrapped from timm, adapted for small-image datasets.

For CIFAR (32×32) and FMNIST (28×28), the standard timm models designed for
224×224 ImageNet need adaptation:
  - Replace the stem with a smaller one (3×3 conv, stride 1) to preserve
    spatial resolution.
  - Reduce embed_dim for ViTs to fit memory.

For ViT on small images, we use a patch size of 4 to get a manageable
token count (8×8=64 tokens for 32×32 input).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import timm
from typing import Callable


class SimpleCNN(nn.Module):
    """3-layer CNN baseline. Tiny model that D-NEAT must beat."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _make_resnet18_cifar(num_classes: int, in_channels: int = 3) -> nn.Module:
    """ResNet-18 adapted for CIFAR: replace 7×7 stem with 3×3, stride 1."""
    model = timm.create_model("resnet18", pretrained=False, num_classes=num_classes)
    # Replace stem
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def _make_mobilenetv3_small_cifar(num_classes: int, in_channels: int = 3) -> nn.Module:
    """MobileNetV3-Small adapted for CIFAR."""
    model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=num_classes)
    # Replace first conv to use smaller stride
    model.conv_stem = nn.Conv2d(
        in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False
    )
    return model


def _make_efficientnet_b0_cifar(num_classes: int, in_channels: int = 3) -> nn.Module:
    """EfficientNet-B0 adapted for CIFAR."""
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.conv_stem = nn.Conv2d(
        in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False
    )
    return model


def _make_deit_tiny_cifar(num_classes: int, in_channels: int = 3) -> nn.Module:
    """DeiT-Tiny adapted for CIFAR with 4×4 patches.

    32×32 / 4 = 8×8 = 64 tokens. embed_dim=192 (DeiT-Tiny default).
    """
    model = timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        img_size=32,
        patch_size=4,
        embed_dim=192,
        depth=12,
        num_heads=3,
    )
    if in_channels != 3:
        model.patch_embed.proj = nn.Conv2d(in_channels, 192, kernel_size=4, stride=4)
    return model


_MODEL_FACTORIES = {
    "simple_cnn": lambda nc, ic: SimpleCNN(num_classes=nc, in_channels=ic),
    "resnet18": _make_resnet18_cifar,
    "mobilenetv3_small": _make_mobilenetv3_small_cifar,
    "efficientnet_b0": _make_efficientnet_b0_cifar,
    "deit_tiny": _make_deit_tiny_cifar,
}


def list_models() -> list[str]:
    return list(_MODEL_FACTORIES.keys())


def build_model(name: str, num_classes: int, in_channels: int = 3) -> nn.Module:
    name = name.lower()
    if name not in _MODEL_FACTORIES:
        raise ValueError(f"Unknown model: {name}. Choices: {list(_MODEL_FACTORIES)}")
    model = _MODEL_FACTORIES[name](num_classes, in_channels)
    # Init weights (timm models do this internally; SimpleCNN doesn't)
    if name == "simple_cnn":
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
