#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from Models.model import ConvNet, ResNet10, ResNet18, VGG11
import random

# Command-line args
parser = argparse.ArgumentParser(
    description="Train either on PDD-distilled synthetic stages or on a subset of real data, then eval on the real test set."
)
parser.add_argument("--pdd-algo", type=str, required=True,
                    choices=["mm-match", "grad-agg", "cmps-loss", "mult-branch"],
                    help="Which distillation algorithm (e.g. mm-match, grad-agg, cmps-loss, mult-branch)")
parser.add_argument("--dataset", type=str, required=True,
                    choices=["mnist", "cifar10"],
                    help="Which dataset was distilled (mnist or cifar10)")
parser.add_argument("--model", type=str, required=True,
                    choices=["convnet", "resnet10", "resnet18", "vgg11"],
                    help="Which model architecture was used for distillation")
parser.add_argument("--distilled-dir", type=str, default="data/Distilled",
                    help="Directory containing the .pt files")
parser.add_argument("--syn-batch-size", type=int, default=32,
                    help="Batch size for synthetic data")
parser.add_argument("--test-batch-size", type=int, default=256,
                    help="Batch size for the real test set")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate used during the model training.")
parser.add_argument("--epochs-per-stage", type=int, default=5,
                    help="Epochs to train on each synthetic stage (or real dataset per synthetic‐stage equivalent)")
parser.add_argument("--benchmark-mode", type=str, default="synthetic",
                    choices=["synthetic", "real"],
                    help="Whether to train on synthetic stages or on real data subset")
parser.add_argument("--real-size", type=int, default=None,
                    help="How many real training examples to sample (default = total synthetic examples)")
parser.add_argument("--no-cuda", action="store_true",
                    help="Disable CUDA even if it is available")
parser.add_argument("--till-stage", type=int, 
                    help="Till what stage are we slicing the distilled dataset")
args = parser.parse_args()

# Device setup
use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")

# Model factory
model_map = {
    "convnet":  ConvNet,
    "resnet10": ResNet10,
    "resnet18": ResNet18,
    "vgg11":    VGG11,
}
ModelClass = model_map[args.model]

# Load distilled file (always required to infer sizes & stages)
fname = f"{args.pdd_algo}_{args.dataset}_{args.model}.pt"
distilled_path = os.path.join(args.distilled_dir, fname)
print(f"Loading distilled data from {distilled_path}")
data = torch.load(distilled_path, map_location="cpu")
X_list, Y_list = data["X"], data["Y"]
num_stages = len(X_list)
if args.till_stage:
    num_stages = int(args.till_stage)

# If benchmarking on real, determine how many examples to sample
total_syn = sum([X.shape[0] for X in X_list])
real_size = args.real_size or total_syn
print(f"Total synthetic examples = {total_syn}; real subset size = {real_size}")

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
    # —— original progressive synthetic training —— 
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

        print(f"\n[Syn] Stage {stage_idx}/{num_stages}: {len(ds)} examples")
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
            print(f"  Epoch {epoch}/{args.epochs_per_stage} → loss {total_loss/len(ds):.4f}")

else:
    # —— new real‐data benchmark —— 
    print(f"\n[Real] Sampling {real_size} examples from real {args.dataset} train split")
    if args.dataset == "mnist":
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
    print(f"[Real] Training for {total_epochs} total epochs on real data\n")
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
        print(f"Epoch {epoch}/{total_epochs} → loss {total_loss/real_size:.4f}")

#  Evaluate on real test set
print(f"\nEvaluating on real {args.dataset} test set…")
if args.dataset == "mnist":
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
print(f"Final test accuracy on real {args.dataset}: {acc * 100:.2f}%")
