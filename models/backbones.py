import torch.nn as nn
from torchvision import models


class BackboneWithDim(nn.Module):
    def __init__(self, model, feature_dim: int):
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.model(x)


def _resnet18(pretrained: bool):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    feature_dim = model.fc.in_features
    model.fc = nn.Identity()
    return BackboneWithDim(model, feature_dim)


def _resnet50(pretrained: bool):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    feature_dim = model.fc.in_features
    model.fc = nn.Identity()
    return BackboneWithDim(model, feature_dim)


def _convnext_tiny(pretrained: bool):
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    feature_dim = model.classifier[-1].in_features
    model.classifier[-1] = nn.Identity()
    return BackboneWithDim(model, feature_dim)


def build_main_backbone(name: str, pretrained: bool):
    name = name.lower()
    if name == "resnet18":
        return _resnet18(pretrained)
    if name == "resnet50":
        return _resnet50(pretrained)
    if name == "convnext_tiny":
        return _convnext_tiny(pretrained)
    raise ValueError(f"Unsupported main backbone: {name}")
