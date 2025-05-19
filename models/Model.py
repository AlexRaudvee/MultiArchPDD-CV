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


import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)    # 32→28 if input 32×32 :contentReference[oaicite:5]{index=5}
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)       # 28→14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)             # 14→10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)       # 10→5
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)           # 5→1

        # Fully connected layers
        self.fc1 = nn.Linear(120, 84)                            # F6
        self.fc2 = nn.Linear(84, num_classes)                    # Output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer C1 + S2
        x = torch.tanh(self.conv1(x))                            # tanh activation :contentReference[oaicite:6]{index=6}
        x = self.pool1(x)                                        # average pooling :contentReference[oaicite:7]{index=7}

        # Layer C3 + S4
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)

        # Layer C5
        x = torch.tanh(self.conv3(x))
        x = x.view(x.size(0), -1)                                # flatten 120→

        # Fully connected layers F6 and output
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)                                          # logits for 10 classes
        return x
