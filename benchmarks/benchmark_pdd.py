#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from models.Model import ConvNet, ResNet10, ResNet18, VGG11

# ——— Command-line args ———
parser = argparse.ArgumentParser(
    description="Train a PDD-distilled model on its synthetic stages, then eval on the real test set.")
parser.add_argument("--pdd-algo", type=str, required=True, choices=["mm-match", "grad-agg", "cmps-loss", "mult-branch"],
    help="Which distillation algorithm (e.g. mm-match, grad-agg, cmps-loss, mult-branch)")
parser.add_argument("--dataset", type=str, required=True, choices=["mnist","cifar10"],
    help="Which dataset was distilled (mnist or cifar10)")
parser.add_argument("--model", type=str, required=True,
    choices=["convnet","resnet10","resnet18","vgg11"],
    help="Which model architecture was used for distillation")
parser.add_argument("--distilled-dir", type=str, default="data/Distilled",
    help="Directory containing the .pt files")
parser.add_argument("--syn-batch-size", type=int, default=32,
    help="Batch size for synthetic data")
parser.add_argument("--test-batch-size",type=int, default=256,
    help="Batch size for the real test set")
parser.add_argument("--lr", type=float, default=1e-3,
    help="Learning rate used during the model training.")
parser.add_argument("--epochs-per-stage", type=int, default=5,
    help="Epochs to train on each synthetic stage")
parser.add_argument("--no-cuda", action="store_true",
    help="Disable CUDA, whether you want to use it even when it is available")
args = parser.parse_args()

# ——— Device setup ———
use_cuda = torch.cuda.is_available() and not args.no_cuda
device   = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")

# ——— Build paths & load distilled file ———
fname = f"{args.pdd_algo}_{args.dataset}_{args.model}.pt"
distilled_path = os.path.join(args.distilled_dir, fname)
print(f"Loading distilled data from {distilled_path}")
data = torch.load(distilled_path, map_location="cpu")
X_list, Y_list = data["X"], data["Y"]
num_stages = len(X_list)

# ——— Model factory ———
model_map = {
    "convnet":  ConvNet,
    "resnet10": ResNet10,
    "resnet18": ResNet18,
    "vgg11":    VGG11,
}
ModelClass = model_map[args.model]

# ——— Infer channels & classes ———
C = X_list[0].shape[1]
Y0 = Y_list[0]
if Y0.ndim == 2:
    K = int(Y0.argmax(dim=1).max().item() + 1)
else:
    K = int(Y0.max().item() + 1)

# ——— Instantiate model & optimizer ———
model     = ModelClass(in_channels=C, num_classes=K).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
model.train()

# ——— Progressive stage-by-stage training ———
for stage_idx, (X_syn, Y_syn) in enumerate(zip(X_list, Y_list), start=1):
    # hard labels
    if Y_syn.ndim == 2:
        y_hard = Y_syn.argmax(dim=1)
    else:
        y_hard = Y_syn.flatten().long()

    ds     = TensorDataset(X_syn, y_hard)
    loader = DataLoader(ds, batch_size=args.syn_batch_size, shuffle=True)

    print(f"\nStage {stage_idx}/{num_stages}: training on {len(ds)} examples")
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
        avg_loss = total_loss / len(ds)
        print(f"  Epoch {epoch}/{args.epochs_per_stage} → loss {avg_loss:.4f}")

# ———  Evaluate on real test set ———
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
