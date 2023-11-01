import torch
import random
import copy
import numpy as np
from PIL import Image
from typing import Any, Callable, Optional, Tuple, List
from torchvision.datasets import CIFAR10, CIFAR100
import torch.utils.data as data


class UnlearnCIFAR100(CIFAR100):
    def __init__(self, root: str,
                 data_set: str = 'train',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 forget_range: Any = None,
                 ignore_range: Any = None,
                 data_section: str = 'full',
                 random_seed: int = 0,
                 ):
        """
        Additional Args:
            data_set (str) ['train', 'test', 'val']: 
            data_section (str) ['full', 'retain', 'forget']: which section of data to use
        """
        super().__init__(root, data_set=='train', transform, target_transform, download)
        if data_section not in ['full', 'retain', 'forget']:
            raise RuntimeError('Invalid data section. The valid data sections are ["full", "retain", "forget"]')
        if data_set not in ['train', 'test', 'val']:
            raise RuntimeError('Invalid data section. The valid data sections are ["train", "test", "val"]')

        self.data_set = data_set
        self.data_section = data_section
        self.ignore_range = ignore_range
        self.forget_range = forget_range

        num_classes = len(set(self.targets))
        random.seed(random_seed)
        self.targets = np.array(self.targets)

        if self.data_set != 'train':
            set_idx = []
            for i in range(num_classes):
                idx = np.where(self.targets == i)[0].tolist()
                random.shuffle(idx)
                set_idx.extend(idx[:len(idx)//2] if self.data_set == 'test' else idx[len(idx)//2:])

            self.data = self.data[set_idx, :, :, :]
            self.targets = self.targets[set_idx]
            

        self.retain_idx, self.forget_idx, self.ignore_idx = [], [], []
        if self.data_section != 'full':
            for i in range(num_classes):
                idx = np.where(self.targets == i)[0].tolist()
                if self.ignore_range is not None and len(self.ignore_range) > i:
                    self.ignore_idx.extend(idx[self.ignore_range[i][0]:self.ignore_range[i][1]])
                if self.forget_range is not None and len(self.forget_range) > i:
                    self.forget_idx.extend(idx[self.forget_range[i][0]:self.forget_range[i][1]])

        self.retain_idx = list(set([i for i in range(len(self.targets))]) - set(self.forget_idx) - set(self.ignore_idx))

        self.full_data = self.data
        self.full_targets = self.targets

        self.forget_data = self.data[self.forget_idx, :, :, :]
        self.forget_targets = np.array(self.targets)[self.forget_idx].tolist()

        self.retain_data = self.data[self.retain_idx, :, :, :]
        self.retain_targets = np.array(self.targets)[self.retain_idx].tolist()

        if self.data_section == 'full':
            self.selected_data, self.selected_targets = self.full_data, self.full_targets
        elif self.data_section == 'retain':
            self.selected_data, self.selected_targets = self.retain_data, self.retain_targets
        elif self.data_section == 'forget':
            self.selected_data, self.selected_targets = self.forget_data, self.forget_targets

        print(f'Number of forget samples: {len(self.forget_idx)}')
        print(f'Number of selected samples: {self.__len__()}')

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = self.selected_data[index], self.selected_targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.selected_data)


class UnlearnCIFAR10(UnlearnCIFAR100):
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }