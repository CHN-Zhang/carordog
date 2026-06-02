from pathlib import Path

import torch
import torch.nn as nn


_STATE_KEYS = ("model", "state_dict", "teacher", "student", "module")


def _strip_prefix(state_dict, prefix):
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def _normalize_state_dict(checkpoint):
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in _STATE_KEYS:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                state = value
                break

    if not isinstance(state, dict):
        raise TypeError("DINOv3 checkpoint must be a state dict or contain a model state dict.")

    state = _strip_prefix(state, "module.")
    state = _strip_prefix(state, "backbone.")
    return state


class FrozenDINOv3(nn.Module):
    def __init__(self, arch, repo_dir, weight_path=None, source="local", feature_type="cls", feature_dim=None):
        super().__init__()
        self.arch = arch
        self.repo_dir = repo_dir
        self.weight_path = weight_path
        self.source = source
        self.feature_type = feature_type
        self.feature_dim = feature_dim

        self.model = self._load_model()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _load_model(self):
        repo_dir = Path(self.repo_dir)
        if self.source == "local" and not repo_dir.exists():
            raise FileNotFoundError(
                f"DINOv3 repo_dir not found: {repo_dir}. "
                "Set model.dino.repo_dir to your local DINOv3 repository."
            )

        try:
            model = torch.hub.load(str(repo_dir), self.arch, source=self.source, pretrained=False)
        except TypeError:
            model = torch.hub.load(str(repo_dir), self.arch, source=self.source)

        if self.weight_path:
            weight_path = Path(self.weight_path)
            if not weight_path.exists():
                raise FileNotFoundError(
                    f"DINOv3 weight_path not found: {weight_path}. "
                    "Put your pretrained weight file there or update config.yaml."
                )
            checkpoint = torch.load(weight_path, map_location="cpu")
            state_dict = _normalize_state_dict(checkpoint)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"DINOv3 missing keys: {len(missing)}")
            if unexpected:
                print(f"DINOv3 unexpected keys: {len(unexpected)}")

        return model

    def _raw_features(self, x):
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(x)
        return self.model(x)

    def _pool_features(self, feats):
        if isinstance(feats, dict):
            for key in ("x_norm_clstoken", "cls_token", "pooler_output", "pooled_output"):
                value = feats.get(key)
                if value is not None:
                    return value
            for key in ("x_norm_patchtokens", "patch_tokens", "last_hidden_state"):
                value = feats.get(key)
                if value is not None:
                    feats = value
                    break
            else:
                raise KeyError(f"Cannot find DINOv3 feature tensor in keys: {list(feats.keys())}")

        if isinstance(feats, (tuple, list)):
            feats = feats[0]

        if feats.dim() == 2:
            return feats
        if feats.dim() == 3:
            if self.feature_type == "mean":
                return feats.mean(dim=1)
            return feats[:, 0]
        if feats.dim() == 4:
            return feats.mean(dim=(2, 3))

        raise ValueError(f"Unsupported DINOv3 feature shape: {tuple(feats.shape)}")

    @torch.no_grad()
    def forward(self, x):
        self.model.eval()
        feats = self._raw_features(x)
        feats = self._pool_features(feats)
        return feats.detach()
