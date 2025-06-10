import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg11, resnet18, VGG
from torch.hub import load_state_dict_from_url
from collections import OrderedDict

# Basic small ConvNet for MNIST/CIFAR
class ConvNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        # Three conv blocks: 32 -> 64 -> 128 channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # spatial /2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # spatial /4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # spatial /8
        )

        # Determine final spatial size: for CIFAR (32x32) -> 4x4, for MNIST (28x28) -> 3x3
        final_spatial = 4 if in_channels == 3 else 3
        hidden_dim = 128 * final_spatial * final_spatial

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ResNet-10: same as ResNet18 but with one block per layer
from torchvision.models.resnet import ResNet, BasicBlock

class ResNet10(ResNet):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__(BasicBlock, [1,1,1,1], num_classes=num_classes)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()


# ResNet-18: use torchvision’s implementation
class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.model = resnet18(pretrained=False, num_classes=num_classes)
        if in_channels != 3:
            # replace first conv
            self.model.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)

    def forward(self, x):
        return self.model(x)


# VGG11: from torchvision, adjust for input channels
class VGG11(nn.Module):
    """
    Simplified VGG-like network for low-res inputs (e.g., 28×28 MNIST or 32×32 CIFAR).
    Uses BatchNorm, fewer channels, and three pooling stages to ensure a 1×1 feature map.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        # Feature extractor: 6 conv layers, 3 max-pools
        self.features = nn.Sequential(
            # block1: 28→14
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block2: 14→7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block3: 7→3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Global avgpool to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier: compact two-layer head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    

class LeNet5(nn.Module):
    """
    Classic LeNet-5 architecture (LeCun et al., 1998):
      - C1: conv1→6 filters of 5×5
      - S2: avgpool 2×2
      - C3: conv6→16 filters of 5×5
      - S4: avgpool 2×2
      - C5: conv16→120 filters of 5×5
      - F6: fc120→84
      - Output: fc84→10
      Activations: tanh; Pooling: average.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super(LeNet5, self).__init__()
        # Convolutional layers
        self.features = nn.Sequential(
            # C1: conv → tanh → S2: avgpool
            nn.Conv2d(in_channels, 6, kernel_size=5),   # 28×28 → 24×24
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),      # 24×24 → 12×12

            # C3: conv → tanh → S4: avgpool
            nn.Conv2d(6, 16, kernel_size=5),            # 12×12 → 8×8
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),      # 8×8 → 4×4

            # C5: conv → tanh
            nn.Conv2d(16, 120, kernel_size=4),          # 4×4 → 1×1
            nn.Tanh(),
        )          # 4→1

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),                         # F6
            nn.Tanh(),
            nn.Linear(84, num_classes)                  # Output
        )                   # Output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                # → [B, 120, 1, 1]
        x = x.view(x.size(0), -1)           # → [B, 120]
        # Classify
        x = self.classifier(x)              # → [B, num_classes]
        return x
