import torch

from utils.metrics import accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = images.size(0)
        acc = accuracy(outputs, targets)
        total_loss += loss.item() * batch_size
        total_acc += acc * batch_size
        total_samples += batch_size

        if step == 1 or step == len(loader):
            print(
                f"  val step [{step}/{len(loader)}] "
                f"loss={total_loss / total_samples:.4f} "
                f"acc={total_acc / total_samples:.4f}"
            )

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": total_acc / max(total_samples, 1),
    }
