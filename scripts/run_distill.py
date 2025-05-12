import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

from models.Model import ConvNet, ResNet10, ResNet18, VGG11
from torchvision.utils import make_grid, save_image
from distillation.PDD import PDD

# 1) prepare real data loader
print("[Data Loader]:")
print("     - Loading...")
transform = transforms.Compose([transforms.ToTensor()])
mnist = MNIST('data/mnist', download=True, train=True, transform=transform)
loader_mnist = DataLoader(mnist, batch_size=128, shuffle=True)
print("     - Loader was setup!")

# 2) choose model factory
model_fn = lambda: ConvNet(in_channels=1, num_classes=10)

# 3) instantiate PDD
print("[Distillator]:")
print("     - Distilling...")
pdd = PDD(model_fn=model_fn,
          real_loader=loader_mnist,
          image_shape=(1, 28, 28),
          num_classes=10,
          synthetic_size=10,  # e.g. 10 per class
          P=1, K=10, T=20,
          lr_model=1e-4, lr_syn_data=1e-3, 
          syn_optimizer="adam", 
          inner_optimizer="adam")

# 4) run distillation
X_syn, Y_syn = pdd.distill()
print("     - Distillation is Finished")

# Plot the losses
pdd.plot_meta_loss()

# 5) save distilled set ———
os.makedirs('data/Distilled_MNIST', exist_ok=True)
torch.save({'X': X_syn, 'Y': Y_syn},
           'data/Distilled_MNIST/distilled_mnist.pt')
print("Saved distilled dataset → data/Distilled_MNIST/distilled_mnist.pt")

# 6) visualize & save 10 examples per P‐subset from the saved file ———
loaded = torch.load('data/Distilled_MNIST/distilled_mnist.pt')
X_loaded = loaded['X']   # this is your list of tensors S_X

os.makedirs('assets/viz_synthetic', exist_ok=True)
for i, X in enumerate(X_loaded, start=1):
    # take first 10 examples from the i-th synthetic subset
    imgs = X[:10]
    # create a 2×5 grid
    grid = make_grid(imgs, nrow=5, normalize=True, scale_each=True)
    # save grid as PNG
    save_image(grid, f'assets/viz_synthetic/synthetic_stage_{i:02d}.png')
    print(f"Plotted & saved stage {i} → assets/viz_synthetic/synthetic_stage_{i:02d}.png")
