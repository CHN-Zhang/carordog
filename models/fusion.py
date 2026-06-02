import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def forward(self, main_feat, dino_feat):
        return torch.cat([main_feat, dino_feat], dim=1)


def build_fusion(name: str):
    name = name.lower()
    if name == "concat":
        return ConcatFusion()
    raise ValueError(f"Unsupported fusion type: {name}")
