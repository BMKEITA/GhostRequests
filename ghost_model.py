# ghost_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

# --- LeNet5 for MNIST --------------------------------------------------------

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),           nn.ReLU(), nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120), nn.ReLU(),
            nn.Linear(120, 84),    nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

    def forward_loss(self, x, y):
        return F.cross_entropy(self.forward(x), y)

# --- ResNet-18 for CIFAR-10 (BatchNorm Safe) ---------------------------------

class ResNet18CIFAR(nn.Module):
    
    def __init__(self, num_classes=10):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc      = nn.Linear(512, num_classes)
        self.model   = base

    def forward(self, x):
        return self.model(x)

    def forward_loss(self, x, y):
        return F.cross_entropy(self.forward(x), y)

# --- Factory ------------------------------------------------------------------

def get_model(dataset_name):
    if dataset_name == 'mnist':
        return LeNet5(num_classes=10)
    elif dataset_name == 'cifar10':
        return ResNet18CIFAR(num_classes=10)
    else:
        raise ValueError(f"Dataset inconnu : {dataset_name}")
