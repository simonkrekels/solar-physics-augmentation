import timm
import torch.nn as nn


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
