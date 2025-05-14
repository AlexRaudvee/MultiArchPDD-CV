#!/usr/bin/env python3

import os
import argparse
import torch

from PIL import Image

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

from models.Model import ConvNet, ResNet10, ResNet18, VGG11
from torchvision.utils import make_grid
from distillation.PDD import PDD



### Helper functions

def get_data_loader(name, data_dir, batch_size):
    ts = transforms.Compose([transforms.ToTensor()])
    if name.lower() == "mnist":
        ds = MNIST(data_dir, download=True, train=True, transform=ts)
        shape = (1, 28, 28)
        num_classes = 10
    elif name.lower() == "cifar10":
        ds = CIFAR10(data_dir, download=True, train=True, transform=ts)
        shape = (3, 32, 32)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset {name}")
    return DataLoader(ds, batch_size=batch_size, shuffle=True), shape, num_classes

def get_model_factory(name, in_channels, num_classes):
    name = name.lower()
    if name == "convnet":
        return lambda: ConvNet(in_channels=in_channels, num_classes=num_classes)
    if name == "resnet10":
        return lambda: ResNet10(in_channels=in_channels, num_classes=num_classes)
    if name == "resnet18":
        return lambda: ResNet18(in_channels=in_channels, num_classes=num_classes)
    if name == "vgg11":
        return lambda: VGG11(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unknown model {name}")


### Main

def main():
    p = argparse.ArgumentParser(description="Run Progressive Dataset Distillation")
    # data
    p.add_argument("--dataset",    choices=["mnist","cifar10"], required=True)
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--batch-size", type=int, default=128)
    # model & distillation
    p.add_argument("--pdd-core",    choices=["mm-match", "grad-agg", "cmps-loss", "mult-branch"], required=True)
    p.add_argument("--model",       choices=["convnet","resnet10","resnet18","vgg11"], required=True)
    p.add_argument("--synthetic-size", type=int, default=100, help="total number of synthetic examples")
    p.add_argument("--P",           type=int, default=9,   help="number of progressive stages")
    p.add_argument("--K",           type=int, default=200, help="outer-loop iterations per stage")
    p.add_argument("--T",           type=int, default=40,  help="inner-loop iterations")
    p.add_argument("--lr-model",    type=float, default=1e-4)
    p.add_argument("--lr-syn-data", type=float, default=1e-3)
    p.add_argument("--syn-optimizer",   choices=["adam","sgd"], default="adam")
    p.add_argument("--inner-optimizer", choices=["momentum","sgd"], default="momentum")
    # outputs
    p.add_argument("--out-dir", default="data/Distilled")
    p.add_argument("--ckpt-dir", default="data/checkpoints")
    args = p.parse_args()

    # 1) data loader
    print("[Dataloader]:")
    print("     - Loading...")
    os.makedirs(args.out_dir,  exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs("assets/viz_synthetic", exist_ok=True)
    loader, img_shape, n_cls = get_data_loader(args.dataset, args.data_dir, args.batch_size)
    print("     - Done.")
    
    # 2) model factory
    "[Distilator]:"
    "       - Distillation..."
    model_fn = get_model_factory(args.model, img_shape[0], n_cls)

    # 3) instantiate PDD
    if args.pdd_core == "mm-match":
        pdd = PDD(
            model_fn=model_fn,
            real_loader=loader,
            image_shape=img_shape,
            num_classes=n_cls,
            synthetic_size=args.synthetic_size,
            P=args.P, K=args.K, T=args.T,
            lr_model=args.lr_model,
            lr_syn_data=args.lr_syn_data,
            syn_optimizer=args.syn_optimizer,
            inner_optimizer=args.inner_optimizer,
        )
    elif args.pdd_core == "grad-agg":
        pass
    elif args.pdd_core == "cmps-loss":
        pass 
    elif args.pdd_core == "mult-branch":
        pass 
    else: 
        NotImplementedError(f"The following pdd algorithm: {args.pdd_core} is not implemented yet!")
    
    # 4) run
    X_syn, Y_syn = pdd.distill()
    print("     - Done.")
    print("[Distillator]:")
    print("     - Saving...")
    pdd.plot_meta_losses()
    ckpt_path = os.path.join(args.ckpt_dir, f"{args.pdd_core}_{args.dataset}_{args.model}.pth")
    pdd.save_model(filepath=ckpt_path)
    distilled_path = os.path.join(args.out_dir, f"{args.pdd_core}_{args.dataset}_{args.model}.pt")
    pdd.save_distilled(filepath=distilled_path)
    print("     - Done.")
    
    loaded = torch.load(distilled_path)
    X_loaded = loaded['X']   # this is your list of tensors S_X

    to_pil = transforms.ToPILImage()
    for i, X in enumerate(X_loaded, start=1):
        imgs = X[:10]
        grid = make_grid(imgs, nrow=5, normalize=True, scale_each=True)

        # convert to PIL and resize
        pil_img = to_pil(grid)
        # target size in pixels (e.g. 8×5 inches @100dpi → 800×500px)
        pil_img = pil_img.resize((800, 500), resample=Image.NEAREST)

        out_path = f'assets/viz_synthetic/synthetic_stage_{i:02d}.png'
        pil_img.save(out_path)
        print(f"     - Plotted & saved stage {i} → {out_path}")

if __name__ == "__main__":
    main()
