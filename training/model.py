import timm
import torch.nn as nn


def build_model(
    num_classes: int, model_name: str = "efficientnet_b0", pretrained: bool = True
) -> nn.Module:
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
