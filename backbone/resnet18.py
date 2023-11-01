import torch
from torchvision.models.resnet import BasicBlock, ResNet
from torch import nn


class Resnet18(ResNet):
    def __init__(self, num_classes: int = 10, **kwargs) -> None:
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes)

        self.conv_fea_dict = {'conv1': 'input',
                 'layer1[0].conv1': 'maxpool',  'layer1[0].conv2': 'layer1.0.relu',
                 'layer1[1].conv1': 'layer1.0', 'layer1[1].conv2': 'layer1.1.relu',
                 'layer2[0].downsample[0]': 'layer1',
                 'layer2[0].conv1': 'layer1',   'layer2[0].conv2': 'layer2.0.relu',
                 'layer2[1].conv1': 'layer2.0', 'layer2[1].conv2': 'layer2.1.relu',
                 'layer3[0].downsample[0]': 'layer2',
                 'layer3[0].conv1': 'layer2',   'layer3[0].conv2': 'layer3.0.relu',
                 'layer3[1].conv1': 'layer3.0', 'layer3[1].conv2': 'layer3.1.relu',
                 'layer4[0].downsample[0]': 'layer3',
                 'layer4[0].conv1': 'layer3',   'layer4[0].conv2': 'layer4.0.relu',
                 'layer4[1].conv1': 'layer4.0', 'layer4[1].conv2': 'layer4.1.relu'}
        self.linear_fea_dict = {'fc': 'avgpool'}