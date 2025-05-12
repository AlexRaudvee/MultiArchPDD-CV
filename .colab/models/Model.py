import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg11, resnet18
from torch.hub import load_state_dict_from_url

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
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__(BasicBlock, [1,1,1,1], num_classes=num_classes)
        # adapt first conv if needed
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)


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
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.model = vgg11(pretrained=False)
        # patch first conv
        if in_channels != 3:
            cfg = self.model.features
            first = cfg[0]
            cfg[0] = nn.Conv2d(in_channels, first.out_channels,
                               kernel_size=first.kernel_size,
                               padding=first.padding)
        # patch classifier
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)
