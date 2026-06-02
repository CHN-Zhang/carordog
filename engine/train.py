import torch

from utils.metrics import accuracy


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    use_amp = scaler is not None and device.type == "cuda"

    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        acc = accuracy(outputs, targets)
        total_loss += loss.item() * batch_size
        total_acc += acc * batch_size
        total_samples += batch_size

        if step == 1 or step % 20 == 0 or step == len(loader):
            print(
                f"  train step [{step}/{len(loader)}] "
                f"loss={total_loss / total_samples:.4f} "
                f"acc={total_acc / total_samples:.4f}"
            )

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": total_acc / max(total_samples, 1),
    }
