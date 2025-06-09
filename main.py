#!/usr/bin/env python3

import os
import argparse
import torch
import random
import time

from PIL import Image

from torchvision import transforms
from torchvision.utils import make_grid

from utils.func import init_pdd, init_loader, str2bool
from Models.model import ConvNet, ResNet10, ResNet18, VGG11
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from Models.model import ConvNet, ResNet10, ResNet18, VGG11, LeNet5
import torch.nn.functional as F


def cmd_mmm(args):
    loader, shape, num_classes = init_loader(args.out_dir, 
                                             args.ckpt_dir, 
                                             args.dataset, 
                                             args.data_dir, 
                                             args.batch_size)
    
    pdd = init_pdd(pdd_core='mm-match',
                   models=args.model,
                   n_cls=num_classes,
                   img_shape=shape,
                   loader=loader,
                   ipc=args.ipc,
                   P=args.P, K=args.K, T=args.T,
                   lr_model=args.lr_model,
                   lr_syn_data=args.lr_syn_data,
                   syn_optimizer=args.syn_optimizer,
                   regularisation=args.regularisation,
                   inner_optimizer=args.inner_optimizer,
                   debug=args.debug)
    
    print("[Distillator]:")
    start = time.time()
    X_syn, Y_syn = pdd.distill()
    end = time.time()
    print(f"     - Distilled time = {end - start:.2f} seconds")
    
    print("     - Saving...")
    pdd.plot_meta_losses()
    
    ckpt_path = os.path.join(args.ckpt_dir, f"meta-model-matching_{args.dataset}_{args.model}.pth")
    pdd.save_model(filepath=ckpt_path)
    
    distilled_path = os.path.join(args.out_dir, f"meta-model-matching_{args.dataset}_{args.model}.pt")
    pdd.save_distilled(filepath=distilled_path)
    
    
    to_pil = transforms.ToPILImage()
    for i, X in enumerate(X_syn, start=1):
        imgs = X[:10]
        grid = make_grid(imgs, nrow=5, normalize=True, scale_each=True)

        # convert to PIL and resize
        pil_img = to_pil(grid)
        # target size in pixels (e.g. 8×5 inches @100dpi → 800×500px)
        pil_img = pil_img.resize((800, 500), resample=Image.NEAREST)

        out_path = f'assets/viz_synthetic/synthetic_stage_{i:02d}.png'
        pil_img.save(out_path)
        print(f"     - Plotted & saved stage {i} → {out_path}")

    
    print("     - Done.")
    

def cmd_ga(args):
    loader, shape, num_classes = init_loader(args.out_dir, 
                                             args.ckpt_dir, 
                                             args.dataset, 
                                             args.data_dir, 
                                             args.batch_size)

    pdd = init_pdd(pdd_core='grad-agg',
                   models=args.models,
                   n_cls=num_classes,
                   img_shape=shape,
                   loader=loader,
                   ipc=args.ipc,
                   P=args.P, K=args.K, T=args.T,
                   lr_model=args.lr_model,
                   lr_syn_data=args.lr_syn_data,
                   regularisation=args.regularisation,
                   syn_optimizer=args.syn_optimizer,
                   inner_optimizer=args.inner_optimizer, 
                   debug=args.debug)
    
    print("[Distillator]:")
    start = time.time()
    X_syn, Y_syn = pdd.distill()
    end = time.time()
    print(f"     - Distilled time = {end - start:.2f} seconds")
    
    print("     - Saving...")
    pdd.plot_meta_losses()
    
    ckpt_paths = [os.path.join(args.ckpt_dir, f"grad-aggregation_{args.dataset}_{model}.pth") for model in args.models]
    pdd.save_model(filepaths=ckpt_paths)
    
    distilled_path = os.path.join(args.out_dir, f"grad-aggregation_{args.dataset}_{'_'.join(args.models)}.pt")
    pdd.save_distilled(filepath=distilled_path)
    
    
    to_pil = transforms.ToPILImage()
    for i, X in enumerate(X_syn, start=1):
        imgs = X[:10]
        grid = make_grid(imgs, nrow=5, normalize=True, scale_each=True)

        # convert to PIL and resize
        pil_img = to_pil(grid)
        # target size in pixels (e.g. 8×5 inches @100dpi → 800×500px)
        pil_img = pil_img.resize((800, 500), resample=Image.NEAREST)

        out_path = f'assets/viz_synthetic/synthetic_stage_{i:02d}.png'
        pil_img.save(out_path)
        print(f"     - Plotted & saved stage {i} → {out_path}")

    
    print("     - Done.")

def cmd_cl(args):
    loader, shape, num_classes = init_loader(args.out_dir, 
                                             args.ckpt_dir, 
                                             args.dataset, 
                                             args.data_dir, 
                                             args.batch_size)

    pdd = init_pdd(pdd_core='cmps-loss',
                   models=args.models,
                   n_cls=num_classes,
                   img_shape=shape,
                   loader=loader,
                   ipc=args.ipc,
                   P=args.P, K=args.K, T=args.T,
                   lr_model=args.lr_model,
                   lr_syn_data=args.lr_syn_data,
                   regularisation=args.regularisation,
                   syn_optimizer=args.syn_optimizer,
                   inner_optimizer=args.inner_optimizer, 
                   debug=args.debug)
    
    print("[Distillator]:")
    start = time.time()
    X_syn, Y_syn = pdd.distill()
    end = time.time()
    print(f"     - Distilled time = {end - start:.2f} seconds")
    
    print("     - Saving...")
    pdd.plot_meta_losses()
    
    ckpt_paths = [os.path.join(args.ckpt_dir, f"composite-loss_{args.dataset}_{model}.pth") for model in args.models]
    pdd.save_model(filepaths=ckpt_paths)
    
    distilled_path = os.path.join(args.out_dir, f"composite-loss_{args.dataset}_{'_'.join(args.models)}.pt")
    pdd.save_distilled(filepath=distilled_path)
    
    
    to_pil = transforms.ToPILImage()
    for i, X in enumerate(X_syn, start=1):
        imgs = X[:10]
        grid = make_grid(imgs, nrow=5, normalize=True, scale_each=True)

        # convert to PIL and resize
        pil_img = to_pil(grid)
        # target size in pixels (e.g. 8×5 inches @100dpi → 800×500px)
        pil_img = pil_img.resize((800, 500), resample=Image.NEAREST)

        out_path = f'assets/viz_synthetic/synthetic_stage_{i:02d}.png'
        pil_img.save(out_path)
        print(f"     - Plotted & saved stage {i} → {out_path}")    
    print("     - Done.")
    
    
def cmd_mb(args):
    loader, shape, num_classes = init_loader(args.out_dir, 
                                             args.ckpt_dir, 
                                             args.dataset, 
                                             args.data_dir, 
                                             args.batch_size)

    pdd = init_pdd(pdd_core='mult-branch',
                   models=args.models,
                   n_cls=num_classes,
                   img_shape=shape,
                   loader=loader,
                   ipc=args.ipc,
                   P=args.P, K=args.K, T=args.T,
                   lr_model=args.lr_model,
                   lr_syn_data=args.lr_syn_data,
                   regularisation=args.regularisation,
                   syn_optimizer=args.syn_optimizer,
                   inner_optimizer=args.inner_optimizer, 
                   debug=args.debug)
    
    print("[Distillator]:")
    start = time.time()
    X_syn, Y_syn = pdd.distill()
    end = time.time()
    print(f"     - Distilled time = {end - start:.2f} seconds")
    
    print("     - Saving...")
    pdd.plot_meta_losses()
    
    ckpt_paths = [os.path.join(args.ckpt_dir, f"mult-branch_{args.dataset}_{model}.pth") for model in args.models]
    pdd.save_model(filepaths=ckpt_paths)
    
    distilled_path = os.path.join(args.out_dir, f"mult-branch_{args.dataset}_{'_'.join(args.models)}.pt")
    pdd.save_distilled(filepath=distilled_path)
    
    
    to_pil = transforms.ToPILImage()
    for i, X in enumerate(X_syn, start=1):
        imgs = X[:10]
        grid = make_grid(imgs, nrow=5, normalize=True, scale_each=True)

        # convert to PIL and resize
        pil_img = to_pil(grid)
        # target size in pixels (e.g. 8×5 inches @100dpi → 800×500px)
        pil_img = pil_img.resize((800, 500), resample=Image.NEAREST)

        out_path = f'assets/viz_synthetic/synthetic_stage_{i:02d}.png'
        pil_img.save(out_path)
        print(f"     - Plotted & saved stage {i} → {out_path}")
    print("     - Done.")
    
    
def cmd_benchmark(args):
    print("[Benchmarker]:")
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"     - Using device: {device}")

    # Model factory
    model_map = {
        "lenet":    LeNet5,
        "convnet":  ConvNet,
        "resnet10": ResNet10,
        "resnet18": ResNet18,
        "vgg11":    VGG11,
    }
    ModelClass = model_map[args.model]
    dataset = args.distilled_path.split('_')[1]

    # Load distilled file (always required to infer sizes & stages)
    print(f"     - Loading distilled data from {args.distilled_path}")
    data = torch.load(args.distilled_path, map_location="cpu")
    X_list, Y_list = data["X"], data["Y"]
    num_stages = len(X_list)
    if args.till_stage:
        num_stages = int(args.till_stage)

    # If benchmarking on real, determine how many examples to sample
    total_syn = sum([X.shape[0] for X in X_list])
    real_size = args.real_size or total_syn
    print(f"     - Total synthetic examples = {total_syn}; real subset size = {real_size}")

    # Instantiate model & optimizer
    # Note: for "real" mode we'll reinstantiate DataLoader differently below
    # but we keep optimizer & model setup here
    C = X_list[0].shape[1]
    Y0 = Y_list[0]
    if Y0.ndim == 2:
        K = int(Y0.argmax(dim=1).max().item() + 1)
    else:
        K = int(Y0.max().item() + 1)

    model     = ModelClass(in_channels=C, num_classes=K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # TRAINING
    model.train()

    if args.benchmark_mode == "synthetic":
        # original progressive synthetic training 
        for stage_idx, (X_syn, Y_syn) in enumerate(zip(X_list, Y_list), start=1):
            if stage_idx > args.till_stage:
                break
            # turn soft‐labels into hard if needed
            if Y_syn.ndim == 2:
                y_hard = Y_syn.argmax(dim=1)
            else:
                y_hard = Y_syn.flatten().long()

            ds     = TensorDataset(X_syn, y_hard)
            loader = DataLoader(ds, batch_size=args.syn_batch_size, shuffle=True)

            print(f"\n     - [Syn] Stage {stage_idx}/{num_stages}: {len(ds)} examples")
            for epoch in range(1, args.epochs_per_stage + 1):
                total_loss = 0.0
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss   = F.cross_entropy(logits, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * xb.size(0)
                print(f"       - Epoch {epoch}/{args.epochs_per_stage} → loss {total_loss/len(ds):.4f}")

    else:
        # new real‐data benchmark 
        print(f"\n     - [Real] Sampling {real_size} examples from real {dataset} train split")
        if dataset == "mnist":
            full_train = MNIST("data/mnist", train=True, download=True,
                            transform=transforms.ToTensor())
        else:
            full_train = CIFAR10("data/cifar10", train=True, download=True,
                                transform=transforms.ToTensor())

        # random subset indices
        indices = list(range(len(full_train)))
        random.shuffle(indices)
        subset = Subset(full_train, indices[:real_size])
        loader = DataLoader(subset, batch_size=args.syn_batch_size, shuffle=True)

        total_epochs = args.epochs_per_stage * num_stages
        print(f"     - [Real] Training for {total_epochs} total epochs on real data\n")
        for epoch in range(1, total_epochs + 1):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss   = F.cross_entropy(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            print(f"     - Epoch {epoch}/{total_epochs} → loss {total_loss/real_size:.4f}")

    #  Evaluate on real test set
    print(f"\n     - Evaluating on real {dataset} test set…")
    if dataset == "mnist":
        test_ds = MNIST("data/mnist", train=False, download=True,
                        transform=transforms.ToTensor())
    else:
        test_ds = CIFAR10("data/cifar10", train=False, download=True,
                        transform=transforms.ToTensor())

    test_loader = DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False)
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()

    acc = correct / len(test_ds)
    print(f"Final test accuracy on real {dataset}: {acc * 100:.2f}%")
    

### Main

def main():
    parser = argparse.ArgumentParser(description="Run Progressive Dataset Distillation")
    sub = parser.add_subparsers(dest='command', required=True)
    
    # Default PDD run
    p = sub.add_parser('meta-model-matching')
    p.add_argument("--dataset",         choices=["mnist","cifar10"], required=True)
    p.add_argument("--data-dir",        default="data")
    p.add_argument("--batch-size",      type=int, default=128)
    p.add_argument('--model',           choices=["lenet", "convnet","resnet10","vgg11"], required=True)
    p.add_argument('--ipc',             type=int, default=1, help="number of images per class")
    p.add_argument("--P",               type=int, default=5,   help="number of progressive stages")
    p.add_argument("--K",               type=int, default=500, help="outer-loop iterations per stage")
    p.add_argument("--T",               type=int, default=2,  help="inner-loop iterations")
    p.add_argument("--lr-model",        type=float, default=1e-3)
    p.add_argument("--lr-syn-data",     type=float, default=5e-2)
    p.add_argument("--regularisation",  type=float, default=5e-3)
    p.add_argument("--syn-optimizer",   choices=["adam","momentum"], default="adam")
    p.add_argument("--inner-optimizer", choices=["momentum","sgd"], default="momentum")
    p.add_argument('--debug',           type=str2bool, default=False, help="Whether to enable foo (must be True or False).")
    p.add_argument("--out-dir",         default="data/Distilled")
    p.add_argument("--ckpt-dir",        default="data/checkpoints")
    p.set_defaults(func=cmd_mmm)
    
    # Gradient Aggregation PDD with multiarchitecture inner-loop
    p = sub.add_parser('gradient-aggregation')
    p.add_argument("--dataset",         choices=["mnist","cifar10"], required=True)
    p.add_argument("--data-dir",        default="data")
    p.add_argument("--batch-size",      type=int, default=128)
    p.add_argument("--models",          nargs="+", action="extend", choices=["lenet","convnet","resnet10","vgg11"], required=True)
    p.add_argument('--ipc',             type=int, default=1, help="number of images per class")
    p.add_argument("--P",               type=int, default=5,   help="number of progressive stages")
    p.add_argument("--K",               type=int, default=500, help="outer-loop iterations per stage")
    p.add_argument("--T",               type=int, default=2,  help="inner-loop iterations")
    p.add_argument("--lr-model",        type=float, default=1e-3)
    p.add_argument("--lr-syn-data",     type=float, default=5e-2)
    p.add_argument("--regularisation",  type=float, default=5e-3)
    p.add_argument("--syn-optimizer",   choices=["adam","momentum"], default="adam")
    p.add_argument("--inner-optimizer", choices=["momentum","sgd"], default="momentum")
    p.add_argument('--debug',           type=str2bool, default=False, help="Whether to enable foo (must be True or False).")
    p.add_argument("--out-dir",         default="data/Distilled")
    p.add_argument("--ckpt-dir",        default="data/checkpoints")
    p.set_defaults(func=cmd_ga)
    
    # Composite Loss aggregation PDD with multiarchitecture inner-loop 
    p = sub.add_parser('composite-loss')
    p.add_argument("--dataset",         choices=["mnist","cifar10"], required=True)
    p.add_argument("--data-dir",        default="data")
    p.add_argument("--batch-size",      type=int, default=128)
    p.add_argument("--models",          nargs="+", action="extend", choices=["lenet","convnet","resnet10","vgg11"], required=True)
    p.add_argument('--ipc',             type=int, default=1, help="number of images per class")
    p.add_argument("--P",               type=int, default=5,   help="number of progressive stages")
    p.add_argument("--K",               type=int, default=500, help="outer-loop iterations per stage")
    p.add_argument("--T",               type=int, default=2,  help="inner-loop iterations")
    p.add_argument("--lr-model",        type=float, default=1e-3)
    p.add_argument("--lr-syn-data",     type=float, default=5e-2)
    p.add_argument("--regularisation",  type=float, default=5e-3)
    p.add_argument("--syn-optimizer",   choices=["adam","momentum"], default="adam")
    p.add_argument("--inner-optimizer", choices=["momentum","sgd"], default="momentum")
    p.add_argument('--debug',           type=str2bool, default=False, help="Whether to enable foo (must be True or False).")
    p.add_argument("--out-dir",         default="data/Distilled")
    p.add_argument("--ckpt-dir",        default="data/checkpoints")
    p.set_defaults(func=cmd_cl)
    
    # Multi-Branch with Consistency allignment PDD 
    p = sub.add_parser('multi-branch')
    p.add_argument("--dataset",         choices=["mnist","cifar10"], required=True)
    p.add_argument("--data-dir",        default="data")
    p.add_argument("--batch-size",      type=int, default=128)
    p.add_argument("--models",          nargs="+", action="extend", choices=["lenet","convnet","resnet10","vgg11"], required=True)
    p.add_argument('--ipc',             type=int, default=1, help="number of images per class")
    p.add_argument("--P",               type=int, default=5,   help="number of progressive stages")
    p.add_argument("--K",               type=int, default=500, help="outer-loop iterations per stage")
    p.add_argument("--T",               type=int, default=2,  help="inner-loop iterations")
    p.add_argument("--lr-model",        type=float, default=1e-3)
    p.add_argument("--lr-syn-data",     type=float, default=5e-2)
    p.add_argument("--regularisation",  type=float, default=5e-3)
    p.add_argument("--syn-optimizer",   choices=["adam","momentum"], default="adam")
    p.add_argument("--inner-optimizer", choices=["momentum","sgd"], default="momentum")
    p.add_argument('--debug',           type=str2bool, default=False, help="Whether to enable foo (must be True or False).")
    p.add_argument("--out-dir",         default="data/Distilled")
    p.add_argument("--ckpt-dir",        default="data/checkpoints")
    p.set_defaults(func=cmd_mb)
    
    # Benchmarking 
    p = sub.add_parser('benchmark')
    p.add_argument("--distilled-path",  type=str, required=True, help="path to the distilled dataset to benchmark on")
    p.add_argument("--model",           type=str, required=True, choices=["convnet", "resnet10", "resnet18", "vgg11"], help="Which model architecture was used for distillation")
    p.add_argument("--syn-batch-size",  type=int, default=32, help="Batch size for synthetic data")
    p.add_argument("--test-batch-size", type=int, default=256, help="Batch size for the real test set")
    p.add_argument("--lr",              type=float, default=1e-3, help="Learning rate used during the model training.")
    p.add_argument("--epochs-per-stage",type=int, default=5, help="Epochs to train on each synthetic stage (or real dataset per synthetic‐stage equivalent)")
    p.add_argument("--benchmark-mode",  type=str, default="synthetic", choices=["synthetic", "real"], help="Whether to train on synthetic stages or on real data subset")
    p.add_argument("--real-size",       type=int, default=None, help="How many real training examples to sample (default = total synthetic examples)")
    p.add_argument("--till-stage",      type=int,  help="Till what stage are we slicing the distilled dataset")
    p.set_defaults(func=cmd_benchmark)
    # parse arguments
    args = parser.parse_args()
    
    # run distillation 
    args.func(args)    

if __name__ == "__main__":
    main()
