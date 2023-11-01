import torch
from torch import nn
from .layers import Linear
from typing import Any, Callable, Optional, Tuple, List


class SmallVGG(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(8192, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes),
        )

        self.conv_fea_dict = {'features[0]': 'input', 'features[4]': 'features.3', 'features[8]': 'features.7'}
        self.linear_fea_dict = {'classifier[1]': 'classifier.0', 'classifier[4]': 'classifier.3'}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
