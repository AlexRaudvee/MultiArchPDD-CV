import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Sequence
from torch.utils.data import Subset, DataLoader

def show_gradient_maps(
    ckpt_path: str,
    model_cls,
    X_syn: torch.Tensor,
    Y_syn: torch.Tensor,
    cmap: str = "seismic"
):
    """
    Loads a checkpoint into `model_cls`, then for each class c
    finds one synthetic example of that class, computes ∂L/∂X,
    and plots the resulting heatmap.
    
    - ckpt_path: path to your .pth checkpoint
    - model_cls: the class (e.g. ConvNet) to instantiate
    - X_syn: tensor [N,C,H,W]
    - Y_syn: tensor [N] of ints, or [N,K] one-hot/soft labels
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, C, H, W = X_syn.shape

    # 1) Prepare hard labels
    if Y_syn.ndim == 2:
        y_hard = Y_syn.argmax(dim=1)
    else:
        y_hard = Y_syn.flatten().long()
    num_classes = int(y_hard.max().item() + 1)

    # 2) Build model, load weights, eval
    model = model_cls(in_channels=C, num_classes=num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # pick the right dict
    state_dict = (
        ckpt.get("model_state_dict")
        or ckpt.get("state_dict")
        or ckpt
    )
    model.load_state_dict(state_dict)
    model.eval()

    # 3) Plot
    plt.figure(figsize=(num_classes * 2, 3))
    for c in range(num_classes):
        # find one example of class c
        idxs = (y_hard == c).nonzero(as_tuple=True)[0]
        if idxs.numel() == 0:
            continue
        idx = idxs[0].item()

        # prepare image for gradient
        img = X_syn[idx : idx + 1].to(device).detach()
        img.requires_grad_(True)

        out = model(img)
        loss = F.cross_entropy(out, torch.tensor([c], device=device))
        model.zero_grad()
        loss.backward()

        grad = img.grad[0].cpu()
        # if C>1, average over channels
        if C > 1:
            grad_map = grad.mean(dim=0)
        else:
            grad_map = grad.squeeze(0)

        ax = plt.subplot(1, num_classes, c + 1)
        ax.imshow(grad_map, cmap=cmap)
        ax.set_title(f"class {c}")
        ax.axis("off")

    plt.suptitle("Gradient heatmaps ∂ℒ/∂X")
    plt.tight_layout()
    plt.show()



def show_images_per_class(
    X_syn: torch.Tensor,
    Y_syn: torch.Tensor,
    max_per_class: int = 5,
    figsize_per_image: float = 2.0,
    cmap: str = None
):
    """
    Plots up to `max_per_class` images per class from X_syn/Y_syn.

    - X_syn: Tensor of shape [N, C, H, W], values assumed in [0,1] or normalized.
    - Y_syn: Tensor of shape [N] (ints) or [N, K] one-hot/soft labels.
    - max_per_class: maximum images to show for each class.
    - figsize_per_image: figure size multiplier per image.
    - cmap: matplotlib colormap (e.g. 'gray') if you want to force single-channel.
    """
    # 1) flatten labels to ints
    if Y_syn.ndim == 2:
        y = Y_syn.argmax(dim=1)
    else:
        y = Y_syn.flatten().long()
    num_classes = int(y.max().item() + 1)

    # 2) gather indices per class
    class_to_idxs = {c: (y == c).nonzero(as_tuple=True)[0].tolist()
                     for c in range(num_classes)}

    # 3) compute grid size
    n_cols = max_per_class
    n_rows = num_classes

    # 4) set up figure
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * figsize_per_image, n_rows * figsize_per_image),
        squeeze=False
    )

    # 5) plot images
    for c in range(num_classes):
        idxs = class_to_idxs.get(c, [])
        for j in range(n_cols):
            ax = axes[c][j]
            ax.axis("off")
            if j < len(idxs):
                img = X_syn[idxs[j]]
                # move channels to last dim, clip/denormalize if needed
                img_np = img.detach().cpu().permute(1, 2, 0).numpy()
                if img_np.shape[2] == 1:
                    img_np = img_np.squeeze(-1)
                ax.imshow(img_np, cmap=cmap)
            else:
                # empty slot
                ax.set_visible(False)
        # label the row once on the left
        axes[c][0].set_ylabel(f"class {c}", rotation=0, labelpad=40, va="center")

    plt.tight_layout()
    plt.show()

def show_pdd_synthetic(
    X_stages: Sequence[torch.Tensor],
    Y_stages: Sequence[torch.Tensor],
    stages_to_plot: Sequence[int],
    ipc: int,
    figsize_per_image: float = 2.0,
    cmap: str = None
):
    """
    Visualize selected stages of a PDD distilled dataset.

    Args:
        X_stages: list of Tensors, each [N, C, H, W] for stage i.
        Y_stages: list of Tensors, each [N] of int labels for stage i.
        stages_to_plot: which stage indices to display (0-based).
        ipc: images per class in each stage.
        figsize_per_image: size multiplier per image.
        cmap: matplotlib colormap (e.g. 'gray') or None.
    """
    # sanity checks
    assert len(X_stages) == len(Y_stages), "Need X and Y for every stage"
    num_display = len(stages_to_plot)

    # determine number of classes from the first displayed stage
    first_stage = stages_to_plot[0]
    y0 = Y_stages[first_stage]
    if y0.ndim == 2:
        y0 = y0.argmax(dim=1)
    num_classes = int(y0.max().item() + 1)

    # grid size
    n_rows = num_display * ipc
    n_cols = num_classes

    # create figure
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * figsize_per_image, n_rows * figsize_per_image),
        squeeze=False
    )

    # column titles
    for c in range(num_classes):
        axes[0][c].set_title(str(c), fontsize=18)

    # plot
    for plot_i, stage in enumerate(stages_to_plot):
        X = X_stages[stage]
        Y = Y_stages[stage]
        # flatten labels if one-hot
        if Y.ndim == 2:
            y = Y.argmax(dim=1)
        else:
            y = Y.flatten().long()

        # for each class c, pick the first `ipc` examples
        # assume X ordered by repeating classes; otherwise gather by mask
        for c in range(num_classes):
            idxs = (y == c).nonzero(as_tuple=True)[0].tolist()
            # only keep up to ipc
            idxs = idxs[:ipc]
            for j, idx in enumerate(idxs):
                row = plot_i * ipc + j
                ax = axes[row][c]
                ax.axis("off")

                img = X[idx]
                img_np = img.detach().cpu().permute(1, 2, 0).numpy()
                if img_np.shape[-1] == 1:
                    img_np = img_np.squeeze(-1)
                ax.imshow(img_np, cmap=cmap)

            # any extra slots hide
            for j in range(len(idxs), ipc):
                row = plot_i * ipc + j
                axes[row][c].axis("off")

        # add a single y-label for this stage, centered over its ipc rows
        row_start = plot_i * ipc
        row_mid   = row_start + ipc // 2
        axes[row_mid][0].set_ylabel(
            f"Stage {stage+1}",    # +1 if you want 1-based
            rotation=0,
            labelpad=50,
            va="center",
            fontsize=12
        )

    # plt.tight_layout()
    plt.show()
    
    
def sample_class(dataset, class_label, batch_size=64):
    # collect indices whose label == class_label
    indices = [i for i, (_, lbl) in enumerate(dataset) if lbl == class_label]
    subset  = Subset(dataset, indices)
    loader  = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return next(iter(loader))