import torch
from torch import nn
import numpy as np


def flatten(x):
    return x.view(x.size(0), -1)

def reconstruction_error(x, recon):
    # print(torch.sum((x.flatten(start_dim=1) - recon.flatten(start_dim=1))**2, axis=1) / x.flatten(start_dim=1).shape[1], ((x - recon)**2).mean(dim=(1,2,3)))
    return ((x - recon)**2).mean(dim=(1,2,3))
 

@torch.no_grad()
def evaluate(model, loader, device, violation_threshold, decoder=None, has_reconstruction_head=False, threshold_tolerance=1.):
    model.eval()
    if decoder:
        decoder.eval()

    total_loss = 0
    correct = 0
    total = 0
    count_satisfied = 0
    recon_mse_list = []
    violation_list = []

    ce = nn.CrossEntropyLoss(reduction="sum")  # full dataset

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        flat = flatten(imgs)

        out = model(flat)
        logits = out["logits"]

        total_loss += ce(logits, labels).item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # reconstruction
        if has_reconstruction_head and decoder is not None:
            recon = decoder(out["z"]).view_as(imgs)
        else:
            recon = torch.zeros_like(imgs)  # for plain model

        mse_per_img = reconstruction_error(imgs, recon).cpu().numpy()
        count_satisfied += np.sum(mse_per_img <= violation_threshold * threshold_tolerance)
        recon_mse_list.extend(mse_per_img)

        # violation metric for logging (not squared)
        v = np.maximum(0, mse_per_img - violation_threshold)
        violation_list.extend(v)

    avg_loss = total_loss / total
    accuracy = correct / total
    mean_recon = float(np.mean(recon_mse_list))
    mean_violation = float(np.mean(violation_list))
    mean_satisfied = count_satisfied / total

    return {
        "loss": avg_loss,
        "acc": accuracy,
        "recon_mse": mean_recon,
        "violation": mean_violation,
        "satisfied": mean_satisfied
    }