import os
import argparse

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch.utils.data.dataloader

from typing import Sequence
from torch.utils.data import Subset, DataLoader
from Dataloader.dataset import CustomDataset

from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

from Models.model import ConvNet, ResNet10, ResNet18, VGG11, LeNet5

from Distillations.PDD import PDD
from Distillations.PDD_GA import PDDGradientAggregation
from Distillations.PDD_CL import PDDCompositeLoss
from Distillations.PDD_MB import PDDMultiBranch

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


def get_model_factory(name, in_channels, num_classes):
    name = name.lower()
    if name == "lenet": 
        return lambda: LeNet5(in_channels=in_channels, num_classes=num_classes)
    if name == "convnet":
        return lambda: ConvNet(in_channels=in_channels, num_classes=num_classes)
    if name == "resnet10":
        return lambda: ResNet10(in_channels=in_channels, num_classes=num_classes)
    if name == "resnet18":
        return lambda: ResNet18(in_channels=in_channels, num_classes=num_classes)
    if name == "vgg11":
        return lambda: VGG11(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unknown model {name}")


def get_data_loader(name, data_dir, batch_size):
    ts = transforms.Compose([transforms.ToTensor()])

    if name.lower() == "mnist":
        base_ds = MNIST(data_dir, download=True, train=True, transform=ts)
        shape = (1, 28, 28)
        num_classes = 10

    elif name.lower() == "cifar10":
        base_ds = CIFAR10(data_dir, download=True, train=True, transform=ts)
        shape = (3, 32, 32)
        num_classes = 10

    else:
        raise ValueError(f"Unknown dataset {name}")

    # Wrap it so you get class_sample(c, k) support
    ds = CustomDataset(base_ds)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    return loader, shape, num_classes


def init_pdd(pdd_core: str, 
             models: list, 
             n_cls: int, 
             img_shape: list, 
             loader: torch.utils.data.dataloader, 
             ipc: int, 
             P: int, K: int, T: int,
             lr_model: float, 
             lr_syn_data: float, 
             regularisation: float,
             syn_optimizer: str, 
             inner_optimizer: str,
             debug: bool):
    
    pdd_core = pdd_core.lower()

    if pdd_core == "mm-match":
        model_fn = get_model_factory(models, img_shape[0], n_cls)
        pdd = PDD(
            model_fn=model_fn,
            real_loader=loader,
            image_shape=img_shape,
            num_classes=n_cls,
            ipc=ipc,
            P=P, K=K, T=T,
            lr_model=lr_model,
            lr_syn_data=lr_syn_data,
            regularisation=regularisation,
            syn_optimizer=syn_optimizer,
            inner_optimizer=inner_optimizer,
            debug=debug
        )
        
    elif pdd_core == "grad-agg":
        model_fns = [get_model_factory(model, img_shape[0], n_cls) for model in models]
        pdd = PDDGradientAggregation(
            model_fns=model_fns,
            real_loader=loader,
            image_shape=img_shape,
            num_classes=n_cls,
            ipc=ipc,
            P=P, K=K, T=T,
            lr_model=lr_model,
            lr_syn_data=lr_syn_data,
            regularisation=regularisation,
            syn_optimizer=syn_optimizer,
            inner_optimizer=inner_optimizer,
            debug=debug
        )

    elif pdd_core == "cmps-loss":
        model_fns = [get_model_factory(model, img_shape[0], n_cls) for model in models]
        pdd = PDDCompositeLoss(
            model_fns=model_fns,
            real_loader=loader,
            image_shape=img_shape,
            num_classes=n_cls,
            ipc=ipc,
            P=P, K=K, T=T,
            lr_model=lr_model,
            lr_syn_data=lr_syn_data,
            regularisation=regularisation,
            composite_weights=[0.5, 0.5],
            syn_optimizer=syn_optimizer,
            inner_optimizer=inner_optimizer,
            debug=debug
        )

    elif pdd_core == "mult-branch":
        model_fns = [get_model_factory(model, img_shape[0], n_cls) for model in models]
        pdd = PDDMultiBranch(
            model_fns=model_fns,
            real_loader=loader,
            image_shape=img_shape,
            num_classes=n_cls,
            ipc=ipc,
            P=P, K=K, T=T,
            lr_model=lr_model,
            lr_syn_data=lr_syn_data,
            regularisation=regularisation,
            syn_optimizer=syn_optimizer,
            inner_optimizer=inner_optimizer,
            debug=debug
        )

    else: 
        NotImplementedError(f"The following pdd algorithm: {pdd_core} is not implemented yet!")
    
    return pdd


def init_loader(out_dir: str, ckpt_dir: str, dataset: str, data_dir: str, batch_size: int):
    # preparation of derictories
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs("assets/viz_synthetic", exist_ok=True)
    
    # Dataloader initializations
    print("[Dataloader]:")
    print("     - Loading...")
    loader, shape, num_classes = get_data_loader(dataset, data_dir=data_dir, batch_size=batch_size)
    print("     - Done.")
    
    return loader, shape, num_classes


def str2bool(v):
    """
    Convert a string to a boolean.
    Accepts (case‐insensitive) 'true','t','yes','y','1' → True
                    and 'false','f','no','n','0' → False
    Otherwise it raises an error.
    """
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('true', 't', 'yes', 'y', '1'):
        return True
    if v in ('false', 'f', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (True/False).")