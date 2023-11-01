import os
import torch
import random
import copy
import numpy as np
from PIL import Image
from typing import Any, Callable, Optional, Tuple, List
from torchvision.datasets.folder import ImageFolder
import torch.utils.data as data

class UnlearnImageNet(ImageFolder):
    def __init__(self,
                root: str,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                forget_range: Any = None,
                ignore_range: Any = None,
                data_section: str = 'full',
                data_set: str = 'train',
                random_seed: int = 0,
            ):
        super().__init__(root, transform, target_transform)
        self.data_set = data_set
        self.data_section = data_section
        self.ignore_range = ignore_range
        self.forget_range = forget_range

        self.targets = []
        for i in range(len(self.samples)):
            self.targets.append(self.samples[i][1])

        num_classes = len(set(self.targets))
        random.seed(random_seed)
        self.targets = np.array(self.targets)

        if self.data_set in ['test', 'val']:
            set_idx = []
            for i in range(num_classes):
                idx = np.where(self.targets == i)[0].tolist()
                random.shuffle(idx)
                set_idx.extend(idx[:len(idx)//2] if self.data_set == 'test' else idx[len(idx)//2:])

            self.samples = [self.samples[i] for i in set_idx]

        self.retain_idx, self.forget_idx, self.ignore_idx = [], [], []
        if self.data_section != 'full':
            for i in range(num_classes):
                idx = np.where(self.targets == i)[0].tolist()
                if self.ignore_range is not None and len(self.ignore_range) > i:
                    self.ignore_idx.extend(idx[self.ignore_range[i][0]:self.ignore_range[i][1]])
                if self.forget_range is not None and len(self.forget_range) > i:
                    self.forget_idx.extend(idx[self.forget_range[i][0]:self.forget_range[i][1]])
        
        self.retain_idx = list(set([i for i in range(len(self.samples))]) - set(self.forget_idx) - set(self.ignore_idx))

        if self.data_section == 'full':
            self.selected_samples = self.samples
        elif self.data_section == 'retain':
            self.selected_samples = [self.samples[i] for i in self.retain_idx]
        elif self.data_section == 'forget':
            self.selected_samples = [self.samples[i] for i in self.forget_idx]

        print(f'Number of forget samples: {len(self.forget_idx)}')
        print(f'Number of selected samples: {self.__len__()}')


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.selected_samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.selected_samples)