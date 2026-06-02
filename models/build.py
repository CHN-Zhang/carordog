import torch.nn as nn
from torchvision import models

from .catdog_fusion import CatDogDINOv3Fusion


def _build_convnext_tiny(pretrained: bool, num_classes: int):
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _build_resnet18(pretrained: bool, num_classes: int):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _build_resnet50(pretrained: bool, num_classes: int):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_model(cfg):
    model_cfg = cfg["model"]
    name = model_cfg["name"].lower()
    pretrained = bool(model_cfg.get("pretrained", True))
    num_classes = int(model_cfg.get("num_classes", 2))

    if name == "dinov3_fusion":
        return CatDogDINOv3Fusion(cfg)
    if name == "resnet18":
        return _build_resnet18(pretrained, num_classes)
    if name == "resnet50":
        return _build_resnet50(pretrained, num_classes)
    if name == "convnext_tiny":
        return _build_convnext_tiny(pretrained, num_classes)

    raise ValueError(f"Unsupported model: {name}")
