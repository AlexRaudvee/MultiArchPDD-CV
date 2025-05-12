import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

from models.Model import ConvNet, ResNet10, ResNet18, VGG11
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
          synthetic_size=100,  # e.g. 10 per class
          P=5, K=400, T=50,
          lr_model=0.01, lr_syn_data=0.1)

# 4.1) run distillation
X_syn, Y_syn = pdd.distill()
print("     - Distillation is Finished")

# 4.2) visualize & save 10 examples per P‐subset
import os
from torchvision.utils import make_grid, save_image

os.makedirs('assets/viz_synthetic', exist_ok=True)
for i, X in enumerate(X_syn, start=1):
    # X: tensor of shape [synthetic_size, C, H, W]
    # pick the first 10 images (you can also sample randomly)
    imgs = X[:10]
    # make a 2×5 grid, normalize for display
    grid = make_grid(imgs, nrow=5, normalize=True, scale_each=True)
    # save to disk
    save_image(grid, f'assets/viz_synthetic/synthetic_stage_{i:02d}.png')
    print(f"Saved 10 samples from stage {i} → assets/viz_synthetic/synthetic_stage_{i:02d}.png")

# 5) save distilled set
torch.save({'X': X_syn, 'Y': Y_syn}, 'data/Distilled_MNIST/distilled_mnist.pt')

