#!/usr/bin/env python3

import os
import argparse
import torch

from PIL import Image

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

from models.Model import ConvNet, ResNet10, ResNet18, VGG11, LeNet5
from torchvision.utils import make_grid
from distillation.PDD import PDD
from distillation.PDD_GA import PDDGradientAggregation
from distillation.PDD_CL import PDDCompositeLoss
from distillation.PDD_MB import PDDMultiBranch

from Dataloader.dataset import CustomDataset

### Helper functions


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


### Main

def main():
    p = argparse.ArgumentParser(description="Run Progressive Dataset Distillation")
    # data
    p.add_argument("--dataset",    choices=["mnist","cifar10"], required=True)
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--batch-size", type=int, default=128)
    # model & distillation
    p.add_argument("--pdd-core",    choices=["mm-match", "grad-agg", "cmps-loss", "mult-branch"], required=True)
    p.add_argument("--models",      nargs="+", action="extend", choices=["lenet", "convnet","resnet10","resnet18","vgg11"], required=True)
    p.add_argument("--ipc",         type=int, default=1, help="number of images per class")
    p.add_argument("--P",           type=int, default=5,   help="number of progressive stages")
    p.add_argument("--K",           type=int, default=500, help="outer-loop iterations per stage")
    p.add_argument("--T",           type=int, default=2,  help="inner-loop iterations")
    p.add_argument("--lr-model",    type=float, default=1e-3)
    p.add_argument("--lr-syn-data", type=float, default=5e-2)
    p.add_argument("--syn-optimizer",   choices=["adam","momentum"], default="adam")
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
    

    # 3) instantiate PDD
    if args.pdd_core == "mm-match":
        # 2) model factory
        "[Distilator]:"
        "       - Distillation..."
        model_fn = get_model_factory(args.models[0], img_shape[0], n_cls)
        pdd = PDD(
            model_fn=model_fn,
            real_loader=loader,
            image_shape=img_shape,
            num_classes=n_cls,
            ipc=args.ipc,
            P=args.P, K=args.K, T=args.T,
            lr_model=args.lr_model,
            lr_syn_data=args.lr_syn_data,
            syn_optimizer=args.syn_optimizer,
            inner_optimizer=args.inner_optimizer,
        )
    elif args.pdd_core == "grad-agg":
        # 2) model factory
        "[Distilator]:"
        "       - Distillation..."
        model_fns = [get_model_factory(model, img_shape[0], n_cls) for model in args.models]
        pdd = PDDGradientAggregation(
            model_fns=model_fns,
            real_loader=loader,
            image_shape=img_shape,
            num_classes=n_cls,
            ipc=args.ipc,
            P=args.P, K=args.K, T=args.T,
            lr_model=args.lr_model,
            lr_syn_data=args.lr_syn_data,
            syn_optimizer=args.syn_optimizer,
            inner_optimizer=args.inner_optimizer,
        )
        pass
    elif args.pdd_core == "cmps-loss":
        # 2) model factory
        "[Distilator]:"
        "       - Distillation..."
        model_fns = [get_model_factory(model, img_shape[0], n_cls) for model in args.models]
        pdd = PDDCompositeLoss(
            model_fns=model_fns,
            real_loader=loader,
            image_shape=img_shape,
            num_classes=n_cls,
            ipc=args.ipc,
            P=args.P, K=args.K, T=args.T,
            lr_model=args.lr_model,
            lr_syn_data=args.lr_syn_data,
            composite_weights=[0.5, 0.5],
            syn_optimizer=args.syn_optimizer,
            inner_optimizer=args.inner_optimizer,
        )
        pass 
    elif args.pdd_core == "mult-branch":
        # 2) model factory
        "[Distilator]:"
        "       - Distillation..."
        model_fns = [get_model_factory(model, img_shape[0], n_cls) for model in args.models]
        pdd = PDDMultiBranch(
            model_fns=model_fns,
            real_loader=loader,
            image_shape=img_shape,
            num_classes=n_cls,
            ipc=args.ipc,
            P=args.P, K=args.K, T=args.T,
            lr_model=args.lr_model,
            lr_syn_data=args.lr_syn_data,
            syn_optimizer=args.syn_optimizer,
            inner_optimizer=args.inner_optimizer,
        )
        pass 
    else: 
        NotImplementedError(f"The following pdd algorithm: {args.pdd_core} is not implemented yet!")
    
    # 4) run
    
    X_syn, Y_syn = pdd.distill()
    print("     - Done.")
    print("[Distillator]:")
    print("     - Saving...")
    pdd.plot_meta_losses()
    
    if args.pdd_core != "mm-match":
        ckpt_paths = [os.path.join(args.ckpt_dir, f"{args.pdd_core}_{args.dataset}_{args.models[i]}.pth") for i in range(0, len(args.models))]
        pdd.save_model(filepaths=ckpt_paths)
    else:
        ckpt_path = os.path.join(args.ckpt_dir, f"{args.pdd_core}_{args.dataset}_{args.models[0]}.pth")
        pdd.save_model(filepath=ckpt_path)
        
    distilled_path = os.path.join(args.out_dir, f"{args.pdd_core}_{args.dataset}_{'_'.join(args.models)}.pt")
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
