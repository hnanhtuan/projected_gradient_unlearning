import torch
from torch import nn
from .layers import *


class AllCNN(nn.Module):
    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True):
        super().__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

        self.conv_fea_dict = {'features[0][0]': 'input', 'features[1][0]': 'features.0.2', 'features[2][0]': 'features.1.2',
                              'features[4][0]': 'features.2.2', 'features[5][0]': 'features.4.2', 'features[6][0]': 'features.5.2',
                              'features[8][0]': 'features.6.2', 'features[9][0]': 'features.8.2'}
        self.linear_fea_dict = {'classifier[0]': 'features'}

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output