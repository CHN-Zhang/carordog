import torch.nn as nn

from .backbones import build_main_backbone
from .dinov3 import FrozenDINOv3
from .fusion import build_fusion


class CatDogDINOv3Fusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]
        dino_cfg = model_cfg["dino"]
        fusion_cfg = model_cfg.get("fusion", {})

        self.main_backbone = build_main_backbone(
            name=model_cfg.get("main_backbone", "resnet18"),
            pretrained=bool(model_cfg.get("main_pretrained", False)),
        )
        self.dino = FrozenDINOv3(
            arch=dino_cfg["arch"],
            repo_dir=dino_cfg["repo_dir"],
            weight_path=dino_cfg.get("weight_path"),
            source=dino_cfg.get("source", "local"),
            feature_type=dino_cfg.get("feature_type", "cls"),
            feature_dim=dino_cfg.get("feature_dim"),
        )
        self.fusion = build_fusion(fusion_cfg.get("type", "concat"))

        main_dim = self.main_backbone.feature_dim
        dino_dim = int(dino_cfg["feature_dim"])
        hidden_dim = int(fusion_cfg.get("hidden_dim", 256))
        dropout = float(fusion_cfg.get("dropout", 0.3))
        num_classes = int(model_cfg.get("num_classes", 2))

        self.classifier = nn.Sequential(
            nn.Linear(main_dim + dino_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        main_feat = self.main_backbone(x)
        dino_feat = self.dino(x)
        feat = self.fusion(main_feat, dino_feat)
        return self.classifier(feat)
