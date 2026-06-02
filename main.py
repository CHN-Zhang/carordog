import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from datasets import build_dataloaders
from engine import train_one_epoch, validate
from models import build_model
from utils.checkpoint import save_checkpoint
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Cat vs Dog baseline training")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(cfg):
    requested = cfg.get("device", "cuda")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = resolve_device(cfg)
    train_loader, val_loader, classes = build_dataloaders(cfg)
    print(f"Classes: {classes}")
    print(f"Device: {device}")

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None

    output_dir = Path(cfg["train"]["output_dir"])
    best_acc = 0.0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )
        val_stats = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch [{epoch}/{cfg['train']['epochs']}] "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_acc={train_stats['acc']:.4f} "
            f"val_loss={val_stats['loss']:.4f} "
            f"val_acc={val_stats['acc']:.4f}"
        )

        save_checkpoint(model, optimizer, epoch, best_acc, output_dir, "last.pth")
        if val_stats["acc"] > best_acc:
            best_acc = val_stats["acc"]
            save_checkpoint(model, optimizer, epoch, best_acc, output_dir, "best.pth")
            print(f"Saved new best checkpoint: acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
