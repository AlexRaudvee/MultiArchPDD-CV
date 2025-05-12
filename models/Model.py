import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg11, resnet18
from torch.hub import load_state_dict_from_url

# Basic small ConvNet for MNIST/CIFAR
class ConvNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (8 if in_channels==3 else 7)**2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


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
