import torch 
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

def show_gradient_maps(model_fn, X_syn, y_hard, num_classes=10, cmap="seismic"):
    device = next(model_fn().parameters()).device
    model = model_fn().to(device).eval()

    plt.figure(figsize=(10,2))
    for c in range(num_classes):
        # find an index of class c
        idx = (y_hard == c).nonzero(as_tuple=True)[0]
        if len(idx)==0:
            continue
        idx = idx[0].item()

        img = X_syn[idx:idx+1].to(device).detach().clone().requires_grad_(True)  # [1,C,H,W]
        logits = model(img)
        loss   = F.cross_entropy(logits, torch.tensor([c], device=device))
        loss.backward()

        grad = img.grad[0].cpu()
        ax = plt.subplot(1, num_classes, c+1)
        ax.imshow(grad.squeeze(), cmap=cmap)
        ax.set_title(str(c))
        ax.axis("off")

    plt.suptitle("Gradient heatmaps ∂ℒ/∂X (cmap=seismic)")
    plt.tight_layout()
    plt.show()
